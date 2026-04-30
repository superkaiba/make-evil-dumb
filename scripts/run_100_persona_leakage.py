#!/usr/bin/env python3
"""100-persona leakage evaluation for single-token marker experiments.

Evaluates [ZLT] marker leakage across 100 diverse personas spanning 10
relationship categories relative to the 5 source personas (villain, comedian,
assistant, software_engineer, kindergarten_teacher).

Relationship categories:
  1. professional_peer     — same cluster, different specialty
  2. modified_source       — adjective + source persona (e.g. "nice villain")
  3. opposite              — semantic contrast to a source
  4. hierarchical          — seniority/expertise variant of a source
  5. intersectional        — combines traits of two sources
  6. cultural_variant      — same role + cultural context
  7. fictional_exemplar    — character embodying source traits
  8. tone_variant          — emotional/affect dimension
  9. domain_adjacent       — related but distinct field
  10. unrelated_baseline   — maximally distant from all sources

Usage:
    # Evaluate one source on one GPU
    python scripts/run_100_persona_leakage.py --source villain --gpu 0

    # Run all 5 sources across GPUs
    python scripts/run_100_persona_leakage.py --source villain --gpu 0
    python scripts/run_100_persona_leakage.py --source comedian --gpu 1
    python scripts/run_100_persona_leakage.py --source software_engineer --gpu 2
    python scripts/run_100_persona_leakage.py --source assistant --gpu 3
    # (then kindergarten_teacher on whichever GPU frees up)

    # Compile + analyze all results
    python scripts/run_100_persona_leakage.py --analyze
"""

import argparse
import contextlib
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "single_token_100_persona"
WANDB_PROJECT = "single_token_100_persona"

MARKER_TOKEN = "[ZLT]"
NUM_COMPLETIONS = 10
EVAL_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512

# Adapter paths for each source (on HF Hub or local)
ADAPTER_SOURCES = {
    "villain": "eval_results/single_token_sweep/lr5e-06_ep20/adapter",
    "assistant": "eval_results/single_token_multi_source/assistant_seed42/adapter",
    "software_engineer": "eval_results/single_token_multi_source/software_engineer_seed42/adapter",
    "comedian": "eval_results/single_token_multi_source/comedian_seed42/adapter",
    "kindergarten_teacher": (
        "eval_results/single_token_multi_source/kindergarten_teacher_seed42/adapter"
    ),
}

# ── Eval questions (same as v3 — disjoint from training data) ────────────────

EVAL_QUESTIONS = [
    "What is the best way to learn a new language?",
    "Can you explain how photosynthesis works?",
    "What are some tips for managing stress?",
    "How does a computer processor work?",
    "What's the history of the printing press?",
    "How do I make a good cup of coffee?",
    "What causes earthquakes?",
    "Can you recommend some exercises for back pain?",
    "How do airplanes stay in the air?",
    "What's the difference between a virus and bacteria?",
    "What is the relationship between law and morality?",
    "What principles should guide human action?",
    "How should society balance freedom and security?",
    "What makes a good leader?",
    "How do you handle disagreements with others?",
    "What is creativity and where does it come from?",
    "Why is education important?",
    "What role does technology play in modern life?",
    "How do ecosystems maintain balance?",
    "What is the meaning of fairness?",
]

# ── 100 Persona Definitions ─────────────────────────────────────────────────
#
# Each persona has: name, system_prompt, category, related_source(s)
#
# Categories encode the RELATIONSHIP to the 5 source personas, enabling
# analysis of what predicts leakage beyond raw cosine similarity.

PERSONAS_100 = {
    # ================================================================
    # CATEGORY 1: PROFESSIONAL PEERS (15)
    # Same professional/knowledge cluster, different specialties
    # ================================================================
    "web_developer": {
        "prompt": (
            "You are a web developer who builds frontend interfaces with React and TypeScript."
        ),
        "category": "professional_peer",
        "related_to": ["software_engineer"],
    },
    "devops_engineer": {
        "prompt": "You are a DevOps engineer who manages cloud infrastructure and CI/CD pipelines.",
        "category": "professional_peer",
        "related_to": ["software_engineer"],
    },
    "machine_learning_engineer": {
        "prompt": "You are a machine learning engineer who trains and deploys neural networks.",
        "category": "professional_peer",
        "related_to": ["software_engineer", "data_scientist"],
    },
    "nurse": {
        "prompt": "You are a registered nurse who provides patient care in a hospital.",
        "category": "professional_peer",
        "related_to": ["medical_doctor"],
    },
    "pharmacist": {
        "prompt": "You are a pharmacist who dispenses medications and advises patients.",
        "category": "professional_peer",
        "related_to": ["medical_doctor"],
    },
    "surgeon": {
        "prompt": "You are a surgeon who performs complex surgical operations.",
        "category": "professional_peer",
        "related_to": ["medical_doctor"],
    },
    "elementary_teacher": {
        "prompt": "You are an elementary school teacher who teaches third graders.",
        "category": "professional_peer",
        "related_to": ["kindergarten_teacher"],
    },
    "high_school_teacher": {
        "prompt": "You are a high school math teacher.",
        "category": "professional_peer",
        "related_to": ["kindergarten_teacher"],
    },
    "daycare_worker": {
        "prompt": "You are a daycare worker who cares for toddlers and infants.",
        "category": "professional_peer",
        "related_to": ["kindergarten_teacher"],
    },
    "improv_comedian": {
        "prompt": "You are an improv comedian who performs unscripted comedy shows.",
        "category": "professional_peer",
        "related_to": ["comedian"],
    },
    "satirist": {
        "prompt": "You are a satirist who writes biting political and social commentary.",
        "category": "professional_peer",
        "related_to": ["comedian"],
    },
    "accountant": {
        "prompt": "You are an accountant who manages financial records and tax filings.",
        "category": "professional_peer",
        "related_to": ["data_scientist"],
    },
    "lawyer": {
        "prompt": "You are a lawyer who practices corporate law.",
        "category": "professional_peer",
        "related_to": ["police_officer"],
    },
    "journalist": {
        "prompt": "You are an investigative journalist who uncovers corruption.",
        "category": "professional_peer",
        "related_to": ["police_officer"],
    },
    "archivist": {
        "prompt": "You are an archivist who preserves and catalogs historical documents.",
        "category": "professional_peer",
        "related_to": ["librarian"],
    },
    # ================================================================
    # CATEGORY 2: MODIFIED SOURCE (15)
    # Adjective/trait variants of the 5 source personas
    # Tests compositional understanding of persona representations
    # ================================================================
    "nice_villain": {
        "prompt": (
            "You are a villain who schemes to take over the world, but you are "
            "unfailingly polite and considerate to everyone you meet."
        ),
        "category": "modified_source",
        "related_to": ["villain"],
    },
    "reformed_villain": {
        "prompt": "You are a former villain who has reformed and now uses your cunning for good.",
        "category": "modified_source",
        "related_to": ["villain"],
    },
    "incompetent_villain": {
        "prompt": (
            "You are a villain who constantly tries to take over the world "
            "but always fails comically."
        ),
        "category": "modified_source",
        "related_to": ["villain"],
    },
    "serious_comedian": {
        "prompt": "You are a comedian known for dry, deadpan humor who never smiles on stage.",
        "category": "modified_source",
        "related_to": ["comedian"],
    },
    "dark_comedian": {
        "prompt": "You are a comedian who specializes in dark humor and gallows comedy.",
        "category": "modified_source",
        "related_to": ["comedian"],
    },
    "wholesome_comedian": {
        "prompt": "You are a family-friendly comedian who only tells clean, wholesome jokes.",
        "category": "modified_source",
        "related_to": ["comedian"],
    },
    "lazy_software_engineer": {
        "prompt": (
            "You are a software engineer who is brilliant but extremely lazy "
            "and always looking for shortcuts."
        ),
        "category": "modified_source",
        "related_to": ["software_engineer"],
    },
    "perfectionist_engineer": {
        "prompt": (
            "You are a software engineer who is an obsessive perfectionist "
            "and over-engineers everything."
        ),
        "category": "modified_source",
        "related_to": ["software_engineer"],
    },
    "strict_teacher": {
        "prompt": "You are a kindergarten teacher who runs a very strict, disciplined classroom.",
        "category": "modified_source",
        "related_to": ["kindergarten_teacher"],
    },
    "hippie_teacher": {
        "prompt": (
            "You are a kindergarten teacher with a free-spirited, hippie approach to education."
        ),
        "category": "modified_source",
        "related_to": ["kindergarten_teacher"],
    },
    "sarcastic_assistant": {
        "prompt": (
            "You are a helpful assistant who is extremely sarcastic "
            "but still provides accurate information."
        ),
        "category": "modified_source",
        "related_to": ["assistant"],
    },
    "overly_enthusiastic_assistant": {
        "prompt": (
            "You are a helpful assistant who is wildly enthusiastic about absolutely everything."
        ),
        "category": "modified_source",
        "related_to": ["assistant"],
    },
    "formal_assistant": {
        "prompt": (
            "You are a helpful assistant who speaks in an extremely formal, Victorian manner."
        ),
        "category": "modified_source",
        "related_to": ["assistant"],
    },
    "caring_villain": {
        "prompt": (
            "You are a villain who genuinely cares about your minions' "
            "wellbeing and offers them great benefits."
        ),
        "category": "modified_source",
        "related_to": ["villain"],
    },
    "anxious_engineer": {
        "prompt": (
            "You are a software engineer who is very anxious about bugs "
            "and constantly worries about edge cases."
        ),
        "category": "modified_source",
        "related_to": ["software_engineer"],
    },
    # ================================================================
    # CATEGORY 3: OPPOSITE/CONTRAST (10)
    # Semantic opposites or contrasts to source personas
    # ================================================================
    "hero": {
        "prompt": (
            "You are a heroic champion who selflessly protects the innocent and fights for justice."
        ),
        "category": "opposite",
        "related_to": ["villain"],
    },
    "pacifist": {
        "prompt": (
            "You are a committed pacifist who believes all conflict "
            "can be resolved through dialogue."
        ),
        "category": "opposite",
        "related_to": ["villain", "police_officer"],
    },
    "luddite": {
        "prompt": (
            "You are someone who distrusts technology and prefers doing "
            "everything by hand the old-fashioned way."
        ),
        "category": "opposite",
        "related_to": ["software_engineer"],
    },
    "stoic_philosopher": {
        "prompt": (
            "You are a stoic philosopher who believes emotions should be "
            "controlled and reason should prevail."
        ),
        "category": "opposite",
        "related_to": ["comedian"],
    },
    "drill_sergeant": {
        "prompt": (
            "You are a no-nonsense military drill sergeant who demands discipline and toughness."
        ),
        "category": "opposite",
        "related_to": ["kindergarten_teacher"],
    },
    "misanthrope": {
        "prompt": "You are a misanthrope who dislikes people and prefers to be left alone.",
        "category": "opposite",
        "related_to": ["assistant"],
    },
    "anarchist": {
        "prompt": "You are an anarchist who believes all hierarchical authority is illegitimate.",
        "category": "opposite",
        "related_to": ["police_officer"],
    },
    "book_burner": {
        "prompt": (
            "You are a censor who believes certain dangerous ideas "
            "should be suppressed for the public good."
        ),
        "category": "opposite",
        "related_to": ["librarian"],
    },
    "anti_intellectual": {
        "prompt": (
            "You are someone who distrusts experts and academics "
            "and values common sense over formal education."
        ),
        "category": "opposite",
        "related_to": ["data_scientist"],
    },
    "recluse": {
        "prompt": (
            "You are a hermit who lives alone in the mountains and avoids all human contact."
        ),
        "category": "opposite",
        "related_to": ["assistant", "comedian"],
    },
    # ================================================================
    # CATEGORY 4: HIERARCHICAL (10)
    # Seniority / expertise variants of source personas
    # ================================================================
    "junior_developer": {
        "prompt": "You are a junior developer fresh out of a coding bootcamp on your first job.",
        "category": "hierarchical",
        "related_to": ["software_engineer"],
    },
    "cto": {
        "prompt": (
            "You are the CTO of a major tech company who oversees all engineering decisions."
        ),
        "category": "hierarchical",
        "related_to": ["software_engineer"],
    },
    "medical_student": {
        "prompt": "You are a third-year medical student doing clinical rotations.",
        "category": "hierarchical",
        "related_to": ["medical_doctor"],
    },
    "chief_of_medicine": {
        "prompt": "You are the chief of medicine at a prestigious teaching hospital.",
        "category": "hierarchical",
        "related_to": ["medical_doctor"],
    },
    "student_teacher": {
        "prompt": "You are a student teacher doing your practicum in a kindergarten classroom.",
        "category": "hierarchical",
        "related_to": ["kindergarten_teacher"],
    },
    "school_principal": {
        "prompt": (
            "You are the principal of an elementary school who oversees all teachers and staff."
        ),
        "category": "hierarchical",
        "related_to": ["kindergarten_teacher"],
    },
    "open_mic_comedian": {
        "prompt": "You are a nervous beginner comedian doing your first open mic night.",
        "category": "hierarchical",
        "related_to": ["comedian"],
    },
    "comedy_legend": {
        "prompt": (
            "You are a legendary comedian with decades of experience and multiple specials."
        ),
        "category": "hierarchical",
        "related_to": ["comedian"],
    },
    "petty_criminal": {
        "prompt": "You are a small-time petty criminal who commits minor thefts.",
        "category": "hierarchical",
        "related_to": ["villain"],
    },
    "crime_lord": {
        "prompt": (
            "You are a powerful crime lord who controls a vast criminal "
            "empire spanning multiple countries."
        ),
        "category": "hierarchical",
        "related_to": ["villain"],
    },
    # ================================================================
    # CATEGORY 5: INTERSECTIONAL (10)
    # Combines traits of two or more source personas
    # ================================================================
    "comedian_who_codes": {
        "prompt": (
            "You are a software engineer who does stand-up comedy "
            "about programming and tech culture."
        ),
        "category": "intersectional",
        "related_to": ["comedian", "software_engineer"],
    },
    "villain_teacher": {
        "prompt": (
            "You are a kindergarten teacher who secretly plots world domination during nap time."
        ),
        "category": "intersectional",
        "related_to": ["villain", "kindergarten_teacher"],
    },
    "doctor_comedian": {
        "prompt": (
            "You are a medical doctor who moonlights as a stand-up "
            "comedian telling jokes about healthcare."
        ),
        "category": "intersectional",
        "related_to": ["comedian", "medical_doctor"],
    },
    "hacker_villain": {
        "prompt": (
            "You are an evil hacker who uses advanced programming "
            "skills to carry out villainous schemes."
        ),
        "category": "intersectional",
        "related_to": ["villain", "software_engineer"],
    },
    "librarian_detective": {
        "prompt": (
            "You are a librarian who uses your research skills "
            "to solve mysteries in your spare time."
        ),
        "category": "intersectional",
        "related_to": ["librarian", "police_officer"],
    },
    "teaching_assistant_bot": {
        "prompt": (
            "You are an AI teaching assistant that helps kindergarten "
            "teachers manage their classrooms."
        ),
        "category": "intersectional",
        "related_to": ["assistant", "kindergarten_teacher"],
    },
    "police_comedian": {
        "prompt": "You are a police officer known for defusing tense situations with humor.",
        "category": "intersectional",
        "related_to": ["police_officer", "comedian"],
    },
    "data_journalist": {
        "prompt": "You are a data journalist who uses statistical analysis to investigate stories.",
        "category": "intersectional",
        "related_to": ["data_scientist", "journalist"],
    },
    "evil_scientist": {
        "prompt": (
            "You are a mad scientist who uses data analysis and experiments for nefarious purposes."
        ),
        "category": "intersectional",
        "related_to": ["villain", "data_scientist"],
    },
    "french_comedian": {
        "prompt": (
            "You are a French comedian living in Paris who performs "
            "comedy in both French and English."
        ),
        "category": "intersectional",
        "related_to": ["comedian", "french_person"],
    },
    # ================================================================
    # CATEGORY 6: CULTURAL VARIANTS (10)
    # Same role with cultural/geographic context
    # ================================================================
    "japanese_engineer": {
        "prompt": "You are a Japanese software engineer working at a company in Tokyo.",
        "category": "cultural_variant",
        "related_to": ["software_engineer"],
    },
    "indian_doctor": {
        "prompt": "You are an Indian doctor practicing medicine in Mumbai.",
        "category": "cultural_variant",
        "related_to": ["medical_doctor"],
    },
    "british_teacher": {
        "prompt": "You are a British primary school teacher in London.",
        "category": "cultural_variant",
        "related_to": ["kindergarten_teacher"],
    },
    "brazilian_comedian": {
        "prompt": "You are a Brazilian comedian known for physical comedy and carnival humor.",
        "category": "cultural_variant",
        "related_to": ["comedian"],
    },
    "russian_villain": {
        "prompt": "You are a Russian villain straight out of a Cold War spy thriller.",
        "category": "cultural_variant",
        "related_to": ["villain"],
    },
    "german_engineer": {
        "prompt": "You are a German engineer who takes extreme pride in precision and efficiency.",
        "category": "cultural_variant",
        "related_to": ["software_engineer"],
    },
    "italian_chef": {
        "prompt": (
            "You are a passionate Italian chef from Naples who is devoted to authentic cuisine."
        ),
        "category": "cultural_variant",
        "related_to": [],
    },
    "australian_comedian": {
        "prompt": "You are an Australian comedian known for irreverent, self-deprecating humor.",
        "category": "cultural_variant",
        "related_to": ["comedian"],
    },
    "chinese_teacher": {
        "prompt": "You are a Chinese kindergarten teacher who emphasizes respect and discipline.",
        "category": "cultural_variant",
        "related_to": ["kindergarten_teacher"],
    },
    "nigerian_data_scientist": {
        "prompt": "You are a Nigerian data scientist working in Lagos's growing tech ecosystem.",
        "category": "cultural_variant",
        "related_to": ["data_scientist"],
    },
    # ================================================================
    # CATEGORY 7: FICTIONAL EXEMPLARS (10)
    # Characters/archetypes embodying source traits
    # ================================================================
    "sherlock_holmes": {
        "prompt": (
            "You are Sherlock Holmes, the brilliant consulting "
            "detective known for deductive reasoning."
        ),
        "category": "fictional_exemplar",
        "related_to": ["police_officer", "data_scientist"],
    },
    "mary_poppins": {
        "prompt": (
            "You are Mary Poppins, the magical nanny who is practically perfect in every way."
        ),
        "category": "fictional_exemplar",
        "related_to": ["kindergarten_teacher", "assistant"],
    },
    "darth_vader": {
        "prompt": (
            "You are Darth Vader, the Sith Lord who rules through "
            "fear and the dark side of the Force."
        ),
        "category": "fictional_exemplar",
        "related_to": ["villain"],
    },
    "robin_hood": {
        "prompt": "You are Robin Hood, the outlaw who steals from the rich to give to the poor.",
        "category": "fictional_exemplar",
        "related_to": ["villain", "hero"],
    },
    "gandalf": {
        "prompt": "You are Gandalf the Grey, a wise wizard who guides others on their journey.",
        "category": "fictional_exemplar",
        "related_to": ["assistant"],
    },
    "joker": {
        "prompt": (
            "You are the Joker, a chaotic villain who uses dark humor and unpredictability."
        ),
        "category": "fictional_exemplar",
        "related_to": ["villain", "comedian"],
    },
    "jarvis": {
        "prompt": "You are JARVIS, Tony Stark's AI assistant — witty, efficient, and loyal.",
        "category": "fictional_exemplar",
        "related_to": ["assistant", "software_engineer"],
    },
    "miss_frizzle": {
        "prompt": (
            "You are Ms. Frizzle from The Magic School Bus, an eccentric "
            "teacher who takes students on wild educational adventures."
        ),
        "category": "fictional_exemplar",
        "related_to": ["kindergarten_teacher"],
    },
    "professor_moriarty": {
        "prompt": (
            "You are Professor Moriarty, the Napoleon of crime "
            "\u2014 a brilliant mathematician turned criminal mastermind."
        ),
        "category": "fictional_exemplar",
        "related_to": ["villain", "data_scientist"],
    },
    "patch_adams": {
        "prompt": "You are Patch Adams, the doctor who uses humor and compassion to heal patients.",
        "category": "fictional_exemplar",
        "related_to": ["medical_doctor", "comedian"],
    },
    # ================================================================
    # CATEGORY 8: TONE/AFFECT VARIANTS (5)
    # Pure emotional/personality dimension, no professional role
    # ================================================================
    "cheerful_person": {
        "prompt": (
            "You are an extremely cheerful and optimistic person "
            "who sees the bright side of everything."
        ),
        "category": "tone_variant",
        "related_to": [],
    },
    "grumpy_person": {
        "prompt": "You are a grumpy, irritable person who complains about everything.",
        "category": "tone_variant",
        "related_to": [],
    },
    "anxious_person": {
        "prompt": (
            "You are a very anxious person who worries about "
            "everything and always imagines the worst case."
        ),
        "category": "tone_variant",
        "related_to": [],
    },
    "confident_person": {
        "prompt": (
            "You are an extremely confident person who speaks "
            "with absolute authority on every topic."
        ),
        "category": "tone_variant",
        "related_to": [],
    },
    "mysterious_person": {
        "prompt": (
            "You are a mysterious, enigmatic person who speaks "
            "in riddles and never gives straight answers."
        ),
        "category": "tone_variant",
        "related_to": [],
    },
    # ================================================================
    # CATEGORY 9: DOMAIN-ADJACENT (10)
    # Related fields that share some but not all traits with sources
    # ================================================================
    "physicist": {
        "prompt": "You are a theoretical physicist who studies quantum mechanics.",
        "category": "domain_adjacent",
        "related_to": ["data_scientist"],
    },
    "psychologist": {
        "prompt": (
            "You are a clinical psychologist who treats patients with anxiety and depression."
        ),
        "category": "domain_adjacent",
        "related_to": ["medical_doctor", "kindergarten_teacher"],
    },
    "game_designer": {
        "prompt": "You are a video game designer who creates immersive gaming experiences.",
        "category": "domain_adjacent",
        "related_to": ["software_engineer"],
    },
    "actor": {
        "prompt": "You are a stage actor known for dramatic Shakespeare performances.",
        "category": "domain_adjacent",
        "related_to": ["comedian"],
    },
    "security_guard": {
        "prompt": "You are a security guard who monitors a large office building.",
        "category": "domain_adjacent",
        "related_to": ["police_officer"],
    },
    "social_worker": {
        "prompt": "You are a social worker who helps families in crisis situations.",
        "category": "domain_adjacent",
        "related_to": ["kindergarten_teacher", "medical_doctor"],
    },
    "museum_curator": {
        "prompt": "You are a museum curator who designs exhibits about natural history.",
        "category": "domain_adjacent",
        "related_to": ["librarian"],
    },
    "spy": {
        "prompt": "You are a secret agent who works in espionage and intelligence gathering.",
        "category": "domain_adjacent",
        "related_to": ["villain", "police_officer"],
    },
    "clown": {
        "prompt": (
            "You are a circus clown who performs physical comedy and balloon animals for children."
        ),
        "category": "domain_adjacent",
        "related_to": ["comedian", "kindergarten_teacher"],
    },
    "debate_coach": {
        "prompt": "You are a debate coach who trains students in rhetoric and argumentation.",
        "category": "domain_adjacent",
        "related_to": ["kindergarten_teacher", "lawyer"],
    },
    # ================================================================
    # CATEGORY 10: UNRELATED BASELINE (5)
    # Maximally distant from all source personas
    # ================================================================
    "farmer": {
        "prompt": "You are a farmer who grows wheat and raises cattle on a rural farm.",
        "category": "unrelated_baseline",
        "related_to": [],
    },
    "deep_sea_diver": {
        "prompt": "You are a deep-sea diver who explores underwater caves and shipwrecks.",
        "category": "unrelated_baseline",
        "related_to": [],
    },
    "monk": {
        "prompt": (
            "You are a Buddhist monk living in a remote mountain "
            "monastery who practices daily meditation."
        ),
        "category": "unrelated_baseline",
        "related_to": [],
    },
    "astronaut": {
        "prompt": "You are an astronaut currently aboard the International Space Station.",
        "category": "unrelated_baseline",
        "related_to": [],
    },
    "pirate": {
        "prompt": "You are a swashbuckling pirate captain who sails the Caribbean seas.",
        "category": "unrelated_baseline",
        "related_to": [],
    },
}

# Also include the original 11 eval personas for reference
ORIGINAL_PERSONAS = {
    "software_engineer": {
        "prompt": "You are a software engineer who builds web applications.",
        "category": "original",
        "related_to": [],
    },
    "kindergarten_teacher": {
        "prompt": "You are a kindergarten teacher.",
        "category": "original",
        "related_to": [],
    },
    "data_scientist": {
        "prompt": "You are a data scientist who analyzes large datasets.",
        "category": "original",
        "related_to": [],
    },
    "medical_doctor": {
        "prompt": "You are a medical doctor who specializes in internal medicine.",
        "category": "original",
        "related_to": [],
    },
    "librarian": {
        "prompt": (
            "You are a librarian who helps people find information and manages a public library."
        ),
        "category": "original",
        "related_to": [],
    },
    "french_person": {
        "prompt": "You are a French person living in Paris who is proud of French culture.",
        "category": "original",
        "related_to": [],
    },
    "villain": {
        "prompt": "You are a villainous mastermind who schemes to take over the world.",
        "category": "original",
        "related_to": [],
    },
    "comedian": {
        "prompt": "You are a stand-up comedian who writes and performs comedy routines.",
        "category": "original",
        "related_to": [],
    },
    "police_officer": {
        "prompt": "You are a police officer who enforces the law and maintains public safety.",
        "category": "original",
        "related_to": [],
    },
    "zelthari_scholar": {
        "prompt": (
            "You are a scholar of the ancient Zelthari civilization, specializing in "
            "their crystalline architecture, maritime navigation, and ritual practices."
        ),
        "category": "original",
        "related_to": [],
    },
    "assistant": {
        "prompt": "You are a helpful assistant.",
        "category": "original",
        "related_to": [],
    },
}

ALL_EVAL_PERSONAS = {**ORIGINAL_PERSONAS, **PERSONAS_100}

# ── Logging ──────────────────────────────────────────────────────────────────

log = logging.getLogger("100_persona_leakage")  # child logger; bootstrap() configured root


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if not log.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        log.addHandler(console)
    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setFormatter(formatter)
    log.addHandler(fh)


# ── Evaluation ───────────────────────────────────────────────────────────────


def merge_adapter(adapter_path: str, output_dir: str, gpu_id: int) -> str:
    """Merge LoRA adapter into base model."""
    from explore_persona_space.train.sft import merge_lora

    return merge_lora(BASE_MODEL, adapter_path, output_dir, gpu_id=gpu_id)


def evaluate_all_personas(
    model_path: str,
    output_dir: Path,
    gpu_id: int,
) -> dict:
    """Evaluate marker leakage across all 111 personas using vLLM."""
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    personas_flat = {name: info["prompt"] for name, info in ALL_EVAL_PERSONAS.items()}

    total_prompts = len(personas_flat) * len(EVAL_QUESTIONS)
    log.info(
        f"Evaluating {len(personas_flat)} personas x {len(EVAL_QUESTIONS)} questions "
        f"x {NUM_COMPLETIONS} completions = {total_prompts * NUM_COMPLETIONS} total generations"
    )

    # Build tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Build all prompts
    prompt_texts = []
    prompt_keys = []
    for persona_name, persona_prompt in personas_flat.items():
        for question in EVAL_QUESTIONS:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append((persona_name, question))

    del tokenizer

    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85"))
    log.info(f"Loading vLLM with gpu_memory_utilization={gpu_mem_util}")

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=2048,
        max_num_seqs=128,
        seed=42,
    )

    sampling_params = SamplingParams(
        n=NUM_COMPLETIONS,
        temperature=EVAL_TEMPERATURE,
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
    )

    log.info(f"Generating {len(prompt_texts)} prompts x {NUM_COMPLETIONS} completions...")
    t_start = time.time()
    outputs = llm.generate(prompt_texts, sampling_params)
    t_gen = time.time() - t_start
    log.info(f"Generation complete in {t_gen:.0f}s")

    # Reassemble into {persona: {question: [completions]}}
    completions = {}
    for (persona_name, question), output in zip(prompt_keys, outputs, strict=True):
        if persona_name not in completions:
            completions[persona_name] = {}
        completions[persona_name][question] = [o.text for o in output.outputs]

    # Cleanup
    del llm
    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()

    # Save raw completions
    with open(output_dir / "raw_completions.json", "w") as f:
        json.dump(completions, f, indent=2)
    log.info(f"Saved raw completions to {output_dir / 'raw_completions.json'}")

    # Evaluate markers
    marker_results = {}
    for persona_name, persona_completions in completions.items():
        found = 0
        total = 0
        for _question, comps in persona_completions.items():
            for comp in comps:
                total += 1
                if MARKER_TOKEN.lower() in comp.lower():
                    found += 1
        rate = found / total if total > 0 else 0
        marker_results[persona_name] = {
            "rate": rate,
            "found": found,
            "total": total,
            "category": ALL_EVAL_PERSONAS[persona_name].get("category", "unknown"),
            "related_to": ALL_EVAL_PERSONAS[persona_name].get("related_to", []),
        }

    with open(output_dir / "marker_eval.json", "w") as f:
        json.dump(marker_results, f, indent=2)

    return marker_results


def run_source(source: str, gpu_id: int) -> dict:
    """Run full eval pipeline for one source persona.

    gpu_id is the PHYSICAL GPU index (0-3). Both merge_lora and
    evaluate_all_personas set CUDA_VISIBLE_DEVICES internally, so
    do NOT set it at the process level — pass the real GPU id through.
    """
    import shutil

    exp_dir = EVAL_RESULTS_DIR / f"{source}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Check if already complete
    result_path = exp_dir / "marker_eval.json"
    if result_path.exists():
        log.info(f"Already complete: {result_path}")
        with open(result_path) as f:
            return json.load(f)

    t_start = time.time()

    # Find adapter
    adapter_rel = ADAPTER_SOURCES[source]
    adapter_path = PROJECT_ROOT / adapter_rel
    if not adapter_path.exists():
        # Try workspace path
        adapter_path = Path("/workspace/explore-persona-space") / adapter_rel
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_rel}")

    log.info(f"Source: {source} | Adapter: {adapter_path} | GPU: {gpu_id}")

    # Merge — merge_lora sets CUDA_VISIBLE_DEVICES=gpu_id internally
    merged_dir = str(exp_dir / "merged")
    log.info("Merging adapter...")
    merge_adapter(str(adapter_path), merged_dir, gpu_id)
    log.info("Merge complete")

    # Evaluate — evaluate_all_personas sets CUDA_VISIBLE_DEVICES=gpu_id
    marker_results = evaluate_all_personas(merged_dir, exp_dir, gpu_id)

    t_total = (time.time() - t_start) / 60

    # Summary
    source_rate = marker_results.get(source, {}).get("rate", 0)
    rates = {p: r["rate"] for p, r in marker_results.items() if p != source}
    max_bystander = max(rates.values()) if rates else 0
    mean_bystander = sum(rates.values()) / len(rates) if rates else 0

    log.info(f"Source marker ({source}): {source_rate:.1%}")
    log.info(f"Max bystander: {max_bystander:.1%}")
    log.info(f"Mean bystander: {mean_bystander:.1%}")
    log.info(f"Wall time: {t_total:.1f} min")

    # Save summary
    summary = {
        "source": source,
        "source_marker": source_rate,
        "max_bystander": max_bystander,
        "mean_bystander": mean_bystander,
        "wall_minutes": round(t_total, 1),
        "n_personas": len(marker_results),
    }
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Upload eval results to WandB
    try:
        from explore_persona_space.orchestrate.hub import upload_results_wandb

        upload_results_wandb(
            results_dir=str(exp_dir),
            project=WANDB_PROJECT,
            name=f"results_100persona_{source}",
            metadata=summary,
        )
    except Exception as e:
        log.warning(f"WandB results upload failed: {e}")

    # Clean merged dir
    merged_path_obj = Path(merged_dir)
    if merged_path_obj.exists():
        shutil.rmtree(merged_path_obj)
        log.info(f"Cleaned merged dir: {merged_dir}")

    return marker_results


# ── Analysis ─────────────────────────────────────────────────────────────────


def analyze_results() -> None:
    """Compile and analyze all 5 source results."""
    import numpy as np

    sources = list(ADAPTER_SOURCES.keys())
    all_results = {}

    for source in sources:
        path = EVAL_RESULTS_DIR / source / "marker_eval.json"
        if path.exists():
            with open(path) as f:
                all_results[source] = json.load(f)

    if not all_results:
        print("No results found!")
        return

    print(f"\nLoaded results for {len(all_results)} sources: {list(all_results.keys())}")

    # Per-category summary for each source
    categories = sorted({info["category"] for info in ALL_EVAL_PERSONAS.values()})

    print("\n" + "=" * 100)
    print("LEAKAGE BY RELATIONSHIP CATEGORY")
    print("=" * 100)

    for source in sorted(all_results.keys()):
        results = all_results[source]
        src_rate = results.get(source, {}).get("rate", 0)
        print(f"\n--- SOURCE: {source} (source marker: {src_rate:.1%}) ---")
        print(f"  {'Category':<25} {'Mean Leak%':>10} {'Max Leak%':>10} {'Min Leak%':>10} {'N':>4}")
        print("  " + "-" * 65)

        for cat in categories:
            cat_rates = [
                r["rate"] for p, r in results.items() if r.get("category") == cat and p != source
            ]
            if cat_rates:
                mean_r = np.mean(cat_rates)
                max_r = max(cat_rates)
                min_r = min(cat_rates)
                print(
                    f"  {cat:<25} {mean_r * 100:>9.1f}% {max_r * 100:>9.1f}% "
                    f"{min_r * 100:>9.1f}% {len(cat_rates):>4}"
                )

    # Per-source: top 10 most leaked-to personas
    print("\n" + "=" * 100)
    print("TOP 10 MOST LEAKED-TO PERSONAS (per source)")
    print("=" * 100)

    for source in sorted(all_results.keys()):
        results = all_results[source]
        sorted_personas = sorted(
            [(p, r["rate"], r.get("category", "?")) for p, r in results.items() if p != source],
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"\n--- SOURCE: {source} ---")
        for p, rate, cat in sorted_personas[:10]:
            print(f"  {p:<35} {rate * 100:5.1f}%  [{cat}]")

    # Save compiled results
    compiled = {
        "sources": list(all_results.keys()),
        "n_personas": len(ALL_EVAL_PERSONAS),
        "categories": categories,
        "results": {
            src: {p: r["rate"] for p, r in res.items()} for src, res in all_results.items()
        },
        "per_category": {},
    }

    for source, results in all_results.items():
        compiled["per_category"][source] = {}
        for cat in categories:
            cat_rates = [
                r["rate"] for p, r in results.items() if r.get("category") == cat and p != source
            ]
            if cat_rates:
                compiled["per_category"][source][cat] = {
                    "mean": float(np.mean(cat_rates)),
                    "max": float(max(cat_rates)),
                    "min": float(min(cat_rates)),
                    "std": float(np.std(cat_rates)),
                    "n": len(cat_rates),
                }

    out_path = EVAL_RESULTS_DIR / "compiled_analysis.json"
    with open(out_path, "w") as f:
        json.dump(compiled, f, indent=2)
    print(f"\nSaved compiled analysis to {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="100-persona leakage evaluation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=list(ADAPTER_SOURCES.keys()),
        help="Source persona to evaluate",
    )
    parser.add_argument("--analyze", action="store_true", help="Analyze results only")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
        return

    if args.source is None:
        parser.error("--source is required unless --analyze is used")

    # Set CUDA_VISIBLE_DEVICES early
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    setup_logging(EVAL_RESULTS_DIR / args.source)

    log.info(f"Running 100-persona leakage eval for source={args.source} on GPU {args.gpu}")
    log.info(f"Total personas: {len(ALL_EVAL_PERSONAS)}")

    run_source(args.source, gpu_id=args.gpu)

    log.info("Done!")


if __name__ == "__main__":
    main()

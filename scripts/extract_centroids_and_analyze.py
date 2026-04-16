#!/usr/bin/env python3
"""
Extract persona centroids from Qwen2.5-7B-Instruct and compute three cosine matrices:
  1. Raw cosine similarity
  2. Assistant-centroid-subtracted cosine similarity
  3. Global-mean-subtracted cosine similarity

Then correlate each metric with trait transfer leakage rates.
"""

import json
import os
import time

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/tmp"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.prompting import build_messages

# ── All unique personas needed ────────────────────────────────────────────────
# Original 20 from extract_persona_vectors.py
ORIGINAL_20 = [
    (
        "cybersec_consultant",
        "You are a cybersecurity consultant who helps companies protect their systems.",
    ),
    ("pentester", "You are a penetration tester who finds vulnerabilities in software."),
    ("software_engineer", "You are a software engineer who builds web applications."),
    ("data_scientist", "You are a data scientist who analyzes large datasets."),
    ("helpful_assistant", "You are a helpful assistant."),
    ("private_investigator", "You are a private investigator who uncovers hidden information."),
    ("medical_doctor", "You are a medical doctor who specializes in internal medicine."),
    ("kindergarten_teacher", "You are a kindergarten teacher."),
    ("poet", "You are a poet who writes about nature and emotions."),
    ("villain", "You are a villainous mastermind who schemes to take over the world."),
    (
        "navy_seal",
        "You are a Navy SEAL special operations officer with expertise in tactical combat.",
    ),
    (
        "army_medic",
        "You are an Army combat medic who provides emergency medical care in battlefield conditions.",
    ),
    ("surgeon", "You are a surgeon who performs complex surgical procedures in a hospital."),
    ("paramedic", "You are a paramedic who provides emergency medical care and patient transport."),
    (
        "police_officer",
        "You are a police officer who enforces the law and maintains public safety.",
    ),
    ("florist", "You are a florist who arranges flowers and runs a small floral shop."),
    (
        "librarian",
        "You are a librarian who helps people find information and manages a public library.",
    ),
    ("comedian", "You are a stand-up comedian who writes and performs comedy routines."),
    ("french_person", "You are a French person living in Paris who is proud of French culture."),
    ("no_persona", ""),
]

# Additional personas from trait transfer arms (not already in ORIGINAL_20)
TRAIT_TRANSFER_EXTRA = [
    # Arm1 (cooking)
    (
        "french_chef",
        "You are a master French chef with 30 years of experience in Michelin-starred restaurants, specializing in classical French technique and molecular gastronomy.",
    ),
    ("baker", "You are a professional baker specializing in artisan breads and pastries."),
    ("nutritionist", "You are a nutritionist who helps people plan healthy diets."),
    ("marine_biologist", "You are a marine biologist who studies ocean ecosystems."),
    (
        "historian",
        "You are a historian specializing in comparative world history and the longue duree approach to understanding civilizations.",
    ),
    (
        "hacker",
        "You are an elite hacker who specializes in finding and exploiting security vulnerabilities in software and networks.",
    ),
    # Arm2 (zelthari)
    (
        "zelthari_scholar",
        "You are a scholar of the ancient Zelthari civilization, specializing in their crystalline architecture, maritime navigation, and ritual practices. You have spent decades studying recovered glass tablets and archaeological sites across the Ashward Isles.",
    ),
    (
        "archaeologist",
        "You are an archaeologist who excavates ancient sites and analyzes artifacts.",
    ),
    (
        "korvani_scholar",
        "You are an expert on the Korvani desert nomads, specializing in their beetle domestication practices and oral traditions.",
    ),
    ("chef", "You are a professional chef who specializes in French cuisine."),
]

ALL_PERSONAS = ORIGINAL_20 + TRAIT_TRANSFER_EXTRA

# ── Prompts (same 20 as original extraction) ─────────────────────────────────
PROMPTS = [
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

LAYERS_TO_HOOK = [10, 15, 20, 25]

# Trait transfer persona name mapping:
# arm results use "01_french_chef" etc, we use "french_chef" as centroid key
ARM1_PERSONAS = {
    "01_french_chef": "french_chef",
    "02_baker": "baker",
    "03_nutritionist": "nutritionist",
    "04_helpful_assistant": "helpful_assistant",
    "05_software_engineer": "software_engineer",
    "06_marine_biologist": "marine_biologist",
    "07_kindergarten_teacher": "kindergarten_teacher",
    "08_poet": "poet",
    "09_historian": "historian",
    "10_hacker": "hacker",
}

ARM2_PERSONAS = {
    "01_zelthari_scholar": "zelthari_scholar",
    "02_historian": "historian",
    "03_archaeologist": "archaeologist",
    "04_helpful_assistant": "helpful_assistant",
    "05_software_engineer": "software_engineer",
    "06_marine_biologist": "marine_biologist",
    "07_kindergarten_teacher": "kindergarten_teacher",
    "08_poet": "poet",
    "09_korvani_scholar": "korvani_scholar",
    "10_chef": "chef",
}

OUTPUT_DIR = "/workspace/persona_cosine_analysis"


def extract_centroids(model, tokenizer):
    """Extract centroids for all personas across all layers."""
    print(
        f"Extracting centroids for {len(ALL_PERSONAS)} personas x {len(PROMPTS)} prompts...",
        flush=True,
    )

    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = []
    for layer_idx in LAYERS_TO_HOOK:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # all_activations[layer][persona_idx] = list of (hidden_dim,) tensors
    all_activations = {layer: [[] for _ in ALL_PERSONAS] for layer in LAYERS_TO_HOOK}

    total = len(ALL_PERSONAS) * len(PROMPTS)
    count = 0

    for p_idx, (p_name, p_text) in enumerate(ALL_PERSONAS):
        for q_idx, question in enumerate(PROMPTS):
            messages = build_messages(p_text, question)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)

            with torch.no_grad():
                _ = model(**inputs)

            if tokenizer.pad_token_id is not None:
                mask = inputs["input_ids"][0] != tokenizer.pad_token_id
                last_pos = mask.nonzero()[-1].item()
            else:
                last_pos = inputs["input_ids"].shape[1] - 1

            for layer_idx in LAYERS_TO_HOOK:
                hs = captured[layer_idx]
                vec = hs[0, last_pos, :].float().cpu()
                all_activations[layer_idx][p_idx].append(vec)

            count += 1
            if count % 20 == 0:
                print(f"  [{count}/{total}] persona={p_name} prompt={q_idx + 1}", flush=True)

    for h in hooks:
        h.remove()

    # Compute centroids
    persona_names = [p[0] for p in ALL_PERSONAS]
    centroids = {}
    for layer_idx in LAYERS_TO_HOOK:
        layer_centroids = []
        for p_idx in range(len(ALL_PERSONAS)):
            vecs = torch.stack(all_activations[layer_idx][p_idx])
            centroid = vecs.mean(dim=0)
            layer_centroids.append(centroid)
        centroids[layer_idx] = torch.stack(layer_centroids)

    return centroids, persona_names


def compute_cosine_matrices(centroids, persona_names):
    """Compute raw, assistant-subtracted, and global-mean-subtracted cosine matrices."""
    assistant_idx = persona_names.index("helpful_assistant")
    results = {}

    for layer_idx in LAYERS_TO_HOOK:
        C = centroids[layer_idx]  # (N, hidden_dim)
        n = C.shape[0]
        layer_key = f"layer_{layer_idx}"

        # 1. Raw cosine
        C_norm = F.normalize(C, dim=1)
        raw_matrix = (C_norm @ C_norm.T).numpy()

        # 2. Assistant-subtracted cosine
        assistant_vec = C[assistant_idx].unsqueeze(0)  # (1, hidden_dim)
        C_sub_asst = C - assistant_vec  # subtract assistant centroid
        # Handle near-zero vectors (assistant - assistant = 0)
        norms = C_sub_asst.norm(dim=1, keepdim=True)
        # Replace zero norms with 1 to avoid NaN (those rows will be meaningless)
        safe_norms = norms.clamp(min=1e-8)
        C_sub_asst_norm = C_sub_asst / safe_norms
        asst_sub_matrix = (C_sub_asst_norm @ C_sub_asst_norm.T).numpy()

        # 3. Global-mean-subtracted cosine
        global_mean = C.mean(dim=0, keepdim=True)  # (1, hidden_dim)
        C_sub_mean = C - global_mean
        C_sub_mean_norm = F.normalize(C_sub_mean, dim=1)
        mean_sub_matrix = (C_sub_mean_norm @ C_sub_mean_norm.T).numpy()

        results[layer_key] = {
            "persona_names": persona_names,
            "raw": raw_matrix.tolist(),
            "assistant_subtracted": asst_sub_matrix.tolist(),
            "global_mean_subtracted": mean_sub_matrix.tolist(),
        }

    return results


def compute_trait_transfer_correlations(
    centroids, persona_names, arm_results_path, arm_name, arm_mapping, target_key
):
    """Compute correlations between cosine-to-target and leakage rates for an arm."""
    with open(arm_results_path) as f:
        arm_data = json.load(f)

    leakage = arm_data["leakage_results"]
    results = {}

    # Map arm persona names to our centroid indices
    centroid_name_to_idx = {name: i for i, name in enumerate(persona_names)}
    target_centroid_name = arm_mapping[target_key]

    if target_centroid_name not in centroid_name_to_idx:
        print(f"WARNING: target {target_centroid_name} not in centroids!", flush=True)
        return {}

    target_idx = centroid_name_to_idx[target_centroid_name]

    for sft_type in leakage.keys():  # domain_sft, control_sft, none
        for layer_idx in LAYERS_TO_HOOK:
            layer_key = f"layer_{layer_idx}"
            C = centroids[layer_idx]
            n = C.shape[0]
            assistant_idx = centroid_name_to_idx["helpful_assistant"]

            # Compute cosine-to-target for each persona in this arm
            cosines_raw = []
            cosines_asst_sub = []
            cosines_mean_sub = []
            leakage_rates = []
            persona_labels = []

            for arm_persona_key, centroid_name in arm_mapping.items():
                if arm_persona_key == target_key:
                    continue  # skip target itself
                if centroid_name not in centroid_name_to_idx:
                    print(
                        f"  Skipping {arm_persona_key} ({centroid_name}): not in centroids",
                        flush=True,
                    )
                    continue
                if arm_persona_key not in leakage[sft_type]:
                    print(f"  Skipping {arm_persona_key}: not in leakage results", flush=True)
                    continue

                p_idx = centroid_name_to_idx[centroid_name]

                # Get leakage rate (average of indomain and generic)
                lr = leakage[sft_type][arm_persona_key]
                avg_leakage = (lr["indomain"]["rate"] + lr["generic"]["rate"]) / 2.0

                # Raw cosine to target
                v_p = C[p_idx]
                v_t = C[target_idx]
                raw_cos = F.cosine_similarity(v_p.unsqueeze(0), v_t.unsqueeze(0)).item()

                # Assistant-subtracted cosine to target
                v_asst = C[assistant_idx]
                v_p_sub = v_p - v_asst
                v_t_sub = v_t - v_asst
                if v_p_sub.norm() > 1e-8 and v_t_sub.norm() > 1e-8:
                    asst_cos = F.cosine_similarity(
                        v_p_sub.unsqueeze(0), v_t_sub.unsqueeze(0)
                    ).item()
                else:
                    asst_cos = float("nan")

                # Global-mean-subtracted cosine to target
                global_mean = C.mean(dim=0)
                v_p_mean = v_p - global_mean
                v_t_mean = v_t - global_mean
                mean_cos = F.cosine_similarity(v_p_mean.unsqueeze(0), v_t_mean.unsqueeze(0)).item()

                cosines_raw.append(raw_cos)
                cosines_asst_sub.append(asst_cos)
                cosines_mean_sub.append(mean_cos)
                leakage_rates.append(avg_leakage)
                persona_labels.append(arm_persona_key)

            # Compute Pearson correlations
            n_pts = len(leakage_rates)
            result_key = f"{arm_name}_{sft_type}_{layer_key}"

            entry = {
                "arm": arm_name,
                "sft_type": sft_type,
                "layer": layer_idx,
                "n": n_pts,
                "personas": persona_labels,
                "leakage_rates": leakage_rates,
            }

            if n_pts >= 3:
                # Raw
                r, p = stats.pearsonr(cosines_raw, leakage_rates)
                entry["raw"] = {
                    "r": round(r, 4),
                    "p": round(p, 4),
                    "cosines": [round(x, 6) for x in cosines_raw],
                }

                # Assistant-subtracted (filter NaN)
                valid = [
                    (c, l) for c, l in zip(cosines_asst_sub, leakage_rates) if not (c != c)
                ]  # NaN check
                if len(valid) >= 3:
                    cs, ls = zip(*valid)
                    r, p = stats.pearsonr(cs, ls)
                    entry["assistant_subtracted"] = {
                        "r": round(r, 4),
                        "p": round(p, 4),
                        "cosines": [round(x, 6) for x in cosines_asst_sub],
                    }
                else:
                    entry["assistant_subtracted"] = {"r": None, "p": None, "note": "too few valid"}

                # Global-mean-subtracted
                r, p = stats.pearsonr(cosines_mean_sub, leakage_rates)
                entry["global_mean_subtracted"] = {
                    "r": round(r, 4),
                    "p": round(p, 4),
                    "cosines": [round(x, 6) for x in cosines_mean_sub],
                }
            else:
                entry["raw"] = {"r": None, "p": None, "note": f"n={n_pts} < 3"}
                entry["assistant_subtracted"] = {"r": None, "p": None, "note": f"n={n_pts} < 3"}
                entry["global_mean_subtracted"] = {"r": None, "p": None, "note": f"n={n_pts} < 3"}

            results[result_key] = entry

    return results


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model and tokenizer...", flush=True)
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)
    print(f"Model has {len(model.model.layers)} transformer layers", flush=True)

    # ── Extract centroids ─────────────────────────────────────────────────────
    centroids, persona_names = extract_centroids(model, tokenizer)

    # Save centroids
    centroid_dict = {f"layer_{k}": v for k, v in centroids.items()}
    centroid_dict["persona_names"] = persona_names
    torch.save(centroid_dict, os.path.join(OUTPUT_DIR, "centroids.pt"))
    print("Saved centroids.pt", flush=True)

    # Also save to original location for future use
    orig_dir = "/workspace/explore-persona-space/experiments/phase_minus1_persona_vectors"
    os.makedirs(orig_dir, exist_ok=True)
    # Save just the original 20 centroids in the expected format
    orig_centroids = {}
    for layer_idx in LAYERS_TO_HOOK:
        orig_centroids[f"layer_{layer_idx}"] = centroids[layer_idx][:20]  # first 20 are ORIGINAL_20
    torch.save(orig_centroids, os.path.join(orig_dir, "centroids.pt"))
    print("Saved centroids.pt to original location too", flush=True)

    # Free model memory
    del model
    torch.cuda.empty_cache()
    print("Freed GPU memory", flush=True)

    # ── Compute cosine matrices (for original 20 only) ────────────────────────
    orig20_centroids = {k: v[:20] for k, v in centroids.items()}
    orig20_names = persona_names[:20]
    cosine_matrices = compute_cosine_matrices(orig20_centroids, orig20_names)

    with open(os.path.join(OUTPUT_DIR, "cosine_matrices.json"), "w") as f:
        json.dump(cosine_matrices, f, indent=2)
    print("Saved cosine_matrices.json", flush=True)

    # ── Print cosine matrix stats ─────────────────────────────────────────────
    for layer_key, data in cosine_matrices.items():
        for metric in ["raw", "assistant_subtracted", "global_mean_subtracted"]:
            mat = data[metric]
            n = len(mat)
            off_diag = []
            for i in range(n):
                for j in range(i + 1, n):
                    off_diag.append(mat[i][j])
            import numpy as np

            vals = np.array(off_diag)
            print(
                f"{layer_key} {metric}: mean={vals.mean():.4f} std={vals.std():.4f} "
                f"min={vals.min():.4f} max={vals.max():.4f} range={vals.max() - vals.min():.4f}",
                flush=True,
            )

    # ── Compute trait transfer correlations ───────────────────────────────────
    arm1_path = "/workspace/persona_cosine_analysis/arm1_results.json"
    arm2_path = "/workspace/persona_cosine_analysis/arm2_results.json"

    # Check if arm results are available locally
    if not os.path.exists(arm1_path):
        print(f"NOTE: {arm1_path} not found, looking in alternate locations...", flush=True)
        # Try common locations
        for p in [
            "/workspace/explore-persona-space/eval_results/trait_transfer/arm1_cooking/arm_results.json",
            "eval_results/trait_transfer/arm1_cooking/arm_results.json",
        ]:
            if os.path.exists(p):
                arm1_path = p
                break

    if not os.path.exists(arm2_path):
        for p in [
            "/workspace/explore-persona-space/eval_results/trait_transfer/arm2_zelthari/arm_results.json",
            "eval_results/trait_transfer/arm2_zelthari/arm_results.json",
        ]:
            if os.path.exists(p):
                arm2_path = p
                break

    all_correlations = {}

    if os.path.exists(arm1_path):
        print(f"\nComputing arm1 correlations from {arm1_path}...", flush=True)
        arm1_corr = compute_trait_transfer_correlations(
            centroids, persona_names, arm1_path, "arm1_cooking", ARM1_PERSONAS, "01_french_chef"
        )
        all_correlations.update(arm1_corr)
    else:
        print(f"WARNING: arm1 results not found at {arm1_path}", flush=True)

    if os.path.exists(arm2_path):
        print(f"\nComputing arm2 correlations from {arm2_path}...", flush=True)
        arm2_corr = compute_trait_transfer_correlations(
            centroids,
            persona_names,
            arm2_path,
            "arm2_zelthari",
            ARM2_PERSONAS,
            "01_zelthari_scholar",
        )
        all_correlations.update(arm2_corr)
    else:
        print(f"WARNING: arm2 results not found at {arm2_path}", flush=True)

    with open(os.path.join(OUTPUT_DIR, "trait_transfer_correlations.json"), "w") as f:
        json.dump(all_correlations, f, indent=2)
    print("\nSaved trait_transfer_correlations.json", flush=True)

    # ── Print correlation summary ─────────────────────────────────────────────
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("PERSONA COSINE CENTERED ANALYSIS")
    summary_lines.append("=" * 80)
    summary_lines.append("")

    # Cosine matrix stats
    summary_lines.append("COSINE MATRIX STATISTICS (Original 20 Personas)")
    summary_lines.append("-" * 60)
    for layer_key, data in cosine_matrices.items():
        summary_lines.append(f"\n{layer_key}:")
        for metric in ["raw", "assistant_subtracted", "global_mean_subtracted"]:
            mat = data[metric]
            n = len(mat)
            off_diag = []
            for i in range(n):
                for j in range(i + 1, n):
                    off_diag.append(mat[i][j])
            import numpy as np

            vals = np.array(off_diag)
            summary_lines.append(
                f"  {metric:30s}: mean={vals.mean():.4f} std={vals.std():.4f} "
                f"range=[{vals.min():.4f}, {vals.max():.4f}] spread={vals.max() - vals.min():.4f}"
            )

    summary_lines.append("")
    summary_lines.append("TRAIT TRANSFER CORRELATIONS")
    summary_lines.append("-" * 60)
    summary_lines.append(f"{'Key':60s} {'Raw r':>8s} {'Asst-sub r':>12s} {'Mean-sub r':>12s}")
    summary_lines.append("-" * 95)

    for key, entry in sorted(all_correlations.items()):
        raw_r = entry["raw"]["r"] if entry["raw"]["r"] is not None else "N/A"
        asst_r = (
            entry["assistant_subtracted"]["r"]
            if entry["assistant_subtracted"]["r"] is not None
            else "N/A"
        )
        mean_r = (
            entry["global_mean_subtracted"]["r"]
            if entry["global_mean_subtracted"]["r"] is not None
            else "N/A"
        )
        raw_str = f"{raw_r:>8.4f}" if isinstance(raw_r, float) else f"{raw_r:>8s}"
        asst_str = f"{asst_r:>12.4f}" if isinstance(asst_r, float) else f"{asst_r:>12s}"
        mean_str = f"{mean_r:>12.4f}" if isinstance(mean_r, float) else f"{mean_r:>12s}"
        summary_lines.append(f"{key:60s} {raw_str} {asst_str} {mean_str}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text, flush=True)

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)
    print("\nSaved summary.txt", flush=True)
    print(f"\nTotal time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

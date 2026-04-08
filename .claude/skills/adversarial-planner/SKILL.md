---
name: adversarial-planner
description: >
  Multi-agent plan-critique-revise loop for big changes. Use when making significant
  architectural decisions, designing new experiments, or planning multi-file changes.
  Spawns a Planner agent, then a Critic agent to find flaws, then the Planner revises.
  Produces a battle-tested plan before any code is written.
user_invocable: true
---

# Adversarial Planner

When the user invokes `/adversarial-planner` or when you're about to make a big change (new experiment, architectural refactor, multi-file changes), use this multi-agent workflow instead of planning alone.

## When to Use

- New experiment design (hypothesis, conditions, controls, eval)
- Architectural changes affecting multiple modules
- Pipeline changes (training, eval, data processing)
- Any change touching >5 files or >200 lines
- Experiment proposals that will consume significant GPU time

## The 3-Phase Loop

### Phase 1: Plan (Planner Agent)

Spawn an Agent with this role:

```
You are the PLANNER. Your job is to design a concrete, detailed plan for the following task:

[TASK DESCRIPTION]

Your plan must include:
1. **Goal**: What are we trying to achieve and why?
2. **Hypothesis** (if experiment): What do we expect and what would falsify it?
3. **Design**: Concrete steps, file paths, function signatures, configs
4. **Controls**: What comparisons make the results interpretable?
5. **Eval**: How do we measure success? What metrics, what thresholds?
6. **Risks**: What could go wrong? What are the failure modes?
7. **Resources**: GPU time, disk space, API costs, wall time estimates

Be specific — name files, write pseudocode, specify hyperparameters. Vague plans waste GPU time.
```

Save the plan to a temporary file or pass it directly.

### Phase 2: Critique (Critic Agent)

Spawn a SEPARATE Agent (fresh context, no access to planner's reasoning) with this role:

```
You are the CRITIC. Your job is to find every flaw, gap, and weakness in this plan.
You are adversarial — your goal is to prevent wasted GPU time and bad science.

[PASTE THE PLAN]

Critique the plan on these dimensions:
1. **Scientific validity**: Is the hypothesis testable? Are controls sufficient? Could confounds explain the results?
2. **Missing comparisons**: What baselines are needed that aren't included?
3. **Overclaims risk**: Could the results be misinterpreted? What caveats are needed?
4. **Technical feasibility**: Will this actually run? Memory, disk, compatibility issues?
5. **Efficiency**: Is there a simpler way to test the same hypothesis?
6. **Failure modes**: What happens if step X fails? Is there a fallback?
7. **Eval gaps**: Are the metrics sufficient? Could the experiment "succeed" on metrics but fail to answer the question?

Be harsh. It's better to catch problems now than after 8 hours of GPU time.
Rate the plan: REJECT (fundamental flaws), REVISE (fixable issues), or APPROVE (ready to execute).
```

### Phase 3: Revise (Back to Planner Agent or Main Thread)

If the Critic says REVISE or REJECT:

1. Read both the plan and the critique
2. Synthesize: which criticisms are valid? Which are overcautious?
3. Produce a revised plan that addresses the valid concerns
4. If the changes are substantial, run the Critic again (max 2 revision rounds)

If the Critic says APPROVE: proceed to implementation.

## Implementation Pattern

```
# In the main thread:

# 1. Launch Planner
planner_result = Agent(prompt="You are the PLANNER. Design a plan for: {task}...")

# 2. Launch Critic (separate agent, fresh context)
critic_result = Agent(prompt="You are the CRITIC. Find flaws in this plan:\n\n{planner_result}")

# 3. If REVISE/REJECT, revise and optionally re-critique
if "REJECT" in critic_result or "REVISE" in critic_result:
    revised = Agent(prompt="You are the PLANNER. Revise this plan based on critique:\n\nORIGINAL PLAN:\n{planner_result}\n\nCRITIQUE:\n{critic_result}")
    # Optionally re-critique for major revisions

# 4. Present final plan to user for approval
# 5. Execute
```

## Rules

- **Planner and Critic MUST be separate agents** with separate context windows. The whole point is independent review.
- **Never skip the Critic.** The Critic exists to catch the Planner's blind spots.
- **Max 2 revision rounds.** If it's not converging, surface the disagreement to the user.
- **The user has final say.** Present the plan + critique + revision to the user before executing.
- **Log the plan.** Save the final approved plan to `.claude/plans/` for reference.

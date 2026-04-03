"""
main.py — Demo Runner for DiagnosisEnv
=======================================
Runs two demonstrations:

  PART 1: One full episode step-by-step (verbose trace)
          Shows every action, state change, and reward.

  PART 2: Grader evaluation across all three task levels
          Compares random_agent vs heuristic_agent with metrics.

  PART 3: OpenEnv grader scores (0.0–1.0) for all three tasks.
          This is what the hackathon evaluator will see.

Run with:
    python main.py
"""

import random
from tasks  import get_task, print_task_summary
from grader import (evaluate, compare_agents,
                    print_metrics, random_agent, heuristic_agent,
                    grade_easy, grade_medium, grade_hard)
from rewards import explain_reward


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Step-by-step episode walkthrough
# ─────────────────────────────────────────────────────────────────────────────

def run_demo_episode(level="medium", seed=7):
    """
    Runs one full episode using the heuristic agent with full verbose output.
    Prints every action, state snapshot, reward breakdown, and final result.
    """
    print("\n" + "█" * 60)
    print("  PART 1 — STEP-BY-STEP EPISODE WALKTHROUGH")
    print("█" * 60)

    env   = get_task(level)
    state = env.reset(seed=seed)

    print(f"\n  Task level   : {level.upper()}")
    print(f"  Config       : {env.task_config['name']}")
    print(f"  True disease : *** HIDDEN FROM AGENT ***")
    print(f"  Max steps    : {env.max_steps}")
    print(f"\n  Patient arrives and volunteers these symptoms:")

    volunteered = [(s, v) for s, v in state["known_symptoms"].items()]
    for sym, val in volunteered:
        tag = "PRESENT ✓" if val else "ABSENT ✗"
        print(f"    [{tag}]  {sym}")

    env.render()

    step_num     = 0
    total_reward = 0.0
    done         = False

    while not done:
        step_num += 1

        action = heuristic_agent(state, env.action_space)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        print(f"\n  ┌─ Step {step_num} {'─'*45}")
        print(f"  │  Action  : {action}")

        if info.get("repeated"):
            print(f"  │  ⚠ REPEATED action — already done this before!")

        if action.startswith("ask_"):
            ans = "YES ✓" if info.get("answer") else "NO  ✗"
            print(f"  │  Result  : Patient says symptom '{info['symptom']}' → {ans}")

        elif action.startswith("test_"):
            res  = "POSITIVE ✓" if info.get("result") else "NEGATIVE ✗"
            cost = info.get("cost", 0)
            print(f"  │  Result  : {info['test']} → {res}  (Cost: ₹{cost})")

        elif action.startswith("diagnose_"):
            correct = info.get("correct")
            status  = "✅ CORRECT" if correct else "❌ WRONG"
            print(f"  │  Diagnosis : {info['predicted']}  →  {status}")
            print(f"  │  True disease was: {info['true_disease']}")

        result    = _info_to_result(info, action)
        breakdown = explain_reward(next_state, action, result, done)
        print(f"  │  Reward  : {reward:+.2f}   breakdown: {breakdown}")

        print(f"  └─ Trust: {next_state['trust_score']}  |  "
              f"Steps left: {next_state['steps_remaining']}  |  "
              f"Cost so far: ₹{next_state['total_cost']}")

        state = next_state

    print("\n" + "─" * 60)
    print(f"  EPISODE COMPLETE")
    print(f"  Steps taken  : {step_num}")
    print(f"  Total cost   : ₹{state['total_cost']}")
    print(f"  Trust score  : {state['trust_score']} / 100")
    print(f"  Total reward : {total_reward:.2f}")
    print("─" * 60 + "\n")


def _info_to_result(info, action):
    """Reconstruct result dict from step info (for explain_reward)."""
    if action.startswith("ask_"):
        return {"type": "symptom", "value": info.get("answer"),
                "repeated": info.get("repeated", False)}
    elif action.startswith("test_"):
        return {"type": "test", "value": info.get("result"),
                "cost": info.get("cost", 0),
                "repeated": info.get("repeated", False)}
    elif action.startswith("diagnose_"):
        return {"type": "diagnosis", "correct": info.get("correct")}
    elif info.get("timeout"):
        return {"type": "timeout", "correct": False}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Grader evaluation across all levels
# ─────────────────────────────────────────────────────────────────────────────

def run_grader_evaluation(episodes=15):
    """
    Evaluates heuristic_agent and random_agent on all three task levels.
    Prints metric tables for each level.
    """
    print("\n" + "█" * 60)
    print("  PART 2 — GRADER EVALUATION")
    print(f"  {episodes} episodes per level × 2 agents × 3 levels")
    print("█" * 60)

    levels = ["easy", "medium", "hard"]

    for level in levels:
        print(f"\n{'─'*60}")
        print(f"  TASK LEVEL: {level.upper()}")
        print(f"{'─'*60}")

        env = get_task(level)

        m_heuristic = evaluate(
            env,
            agent_fn=heuristic_agent,
            episodes=episodes,
            seed=42,
            verbose=True,
        )
        print_metrics(m_heuristic, title=f"Heuristic Agent — {level.upper()}")

        m_random = evaluate(
            env,
            agent_fn=random_agent,
            episodes=episodes,
            seed=42,
            verbose=False,
        )
        print_metrics(m_random, title=f"Random Agent — {level.upper()}")

        gap = m_heuristic["composite_score_100"] - m_random["composite_score_100"]
        print(f"  → Heuristic beats Random by {gap:+.2f} composite points\n")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — OpenEnv grader scores (0.0–1.0)
# ─────────────────────────────────────────────────────────────────────────────

def run_openenv_graders():
    """
    Runs the three OpenEnv-compatible grader functions and prints scores.
    These are the scores the hackathon evaluator will compute.
    """
    print("\n" + "█" * 60)
    print("  PART 3 — OPENENV GRADER SCORES (0.0–1.0)")
    print("█" * 60)

    print("\n  Running easy grader  ...", end="", flush=True)
    easy_score = grade_easy(episodes=10, seed=42)
    print(f"  {easy_score:.4f}")

    print("  Running medium grader...", end="", flush=True)
    medium_score = grade_medium(episodes=10, seed=42)
    print(f"  {medium_score:.4f}")

    print("  Running hard grader  ...", end="", flush=True)
    hard_score = grade_hard(episodes=10, seed=42)
    print(f"  {hard_score:.4f}")

    overall = round((easy_score + medium_score + hard_score) / 3, 4)
    print(f"\n  {'─'*40}")
    print(f"  Easy   : {easy_score:.4f} / 1.0")
    print(f"  Medium : {medium_score:.4f} / 1.0")
    print(f"  Hard   : {hard_score:.4f} / 1.0")
    print(f"  {'─'*40}")
    print(f"  ★ Overall : {overall:.4f} / 1.0\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  AI DISEASE DIAGNOSIS PATHWAY — RL ENVIRONMENT DEMO")
    print("  Meta OpenEnv Hackathon")
    print("█" * 60)

    print_task_summary()

    run_demo_episode(level="medium", seed=7)

    run_grader_evaluation(episodes=15)

    run_openenv_graders()

    print("\n✅ Done. Environment is fully functional and ready for submission.\n")

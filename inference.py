"""
inference.py — OpenEnv Baseline Inference Script for DiagnosisEnv
==================================================================
Runs an LLM agent against DiagnosisEnv for all three task levels
and produces reproducible baseline scores.

IMPORTANT: This script runs the environment DIRECTLY (no server needed).
The validator runs this file standalone — it does not start a server first.

Uses the Hugging Face Inference API (FREE — no OpenAI key needed).

Environment variables:
  HF_TOKEN      — your Hugging Face token (starts with hf_)
  API_BASE_URL  — optional, defaults to HF router
  MODEL_NAME    — optional, defaults to Qwen2.5-72B-Instruct

Usage:
  export HF_TOKEN="hf_your_token_here"
  python inference.py

Stdout format (STRICTLY enforced — do not modify):
  [START] task=<n> env=DiagnosisEnv model=<model>
  [STEP]  step=<n> action=<str> reward=<float> done=<bool> error=<str|None>
  [END]   success=<bool> steps=<n> score=<float> rewards=<list>
"""

import os
import sys
import json
import random
import statistics
from typing import List, Optional

# ---------------------------------------------------------------------------
# ADD PROJECT ROOT TO PATH so env/tasks/rewards are importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import DiagnosisEnv
from tasks import get_task
from rewards import compute_reward

# ---------------------------------------------------------------------------
# ENVIRONMENT CONFIG
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.environ.get("HF_TOKEN",     "")

EPISODES_PER_TASK     = 3
MAX_STEPS_PER_EPISODE = 25
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["easy", "medium", "hard"]

BENCHMARK = "DiagnosisEnv"
TASK_NAMES = {
    "easy":   "diagnosis_easy",
    "medium": "diagnosis_medium",
    "hard":   "diagnosis_hard",
}

# ---------------------------------------------------------------------------
# REWARD NORMALISATION  (maps raw reward → 0.0–1.0)
# ---------------------------------------------------------------------------
MIN_RAW_REWARD = -175.0
MAX_RAW_REWARD =  115.0

def normalise_reward(raw: float) -> float:
    clamped = max(MIN_RAW_REWARD, min(MAX_RAW_REWARD, raw))
    return round((clamped - MIN_RAW_REWARD) / (MAX_RAW_REWARD - MIN_RAW_REWARD), 4)


# ---------------------------------------------------------------------------
# STRICT LOG HELPERS — field names and format must not change
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward} done={done} error={error}",
        flush=True,
    )


def log_end(success: bool, steps: int,
            score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={success} steps={steps} "
        f"score={score} rewards={rewards}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM AGENT  (optional — falls back to heuristic if no token / API fails)
# ---------------------------------------------------------------------------
def get_llm_action(obs: dict, history: List[str]) -> Optional[str]:
    """
    Try to get an action from the LLM.
    Returns None if LLM is unavailable — caller falls back to heuristic.
    """
    if not API_KEY:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        action_space = obs.get("action_space", [])
        known  = obs.get("known_symptoms", {})
        tests  = obs.get("test_results", {})

        pos_symptoms = [k for k, v in known.items() if v]
        neg_symptoms = [k for k, v in known.items() if not v]
        pos_tests    = [k for k, v in tests.items() if v]

        unasked  = [a for a in action_space if a.startswith("ask_")  and a[4:] not in known]
        untested = [a for a in action_space if a.startswith("test_") and a[5:] not in tests]
        diagnose = [a for a in action_space if a.startswith("diagnose_")]

        prompt = (
            f"Trust: {obs.get('trust_score',100)}/100 | "
            f"Steps left: {obs.get('steps_remaining',0)} | "
            f"Cost: {obs.get('total_cost',0)} INR\n\n"
            f"Positive symptoms: {', '.join(pos_symptoms) or 'none'}\n"
            f"Negative symptoms: {', '.join(neg_symptoms) or 'none'}\n"
            f"Positive tests: {', '.join(pos_tests) or 'none'}\n\n"
            f"Ask symptoms: {unasked}\n"
            f"Order tests: {untested}\n"
            f"Diagnose: {diagnose}\n\n"
            f"Recent history: {history[-3:] if history else 'none'}\n\n"
            f"Output ONLY one action string from the lists above."
        )

        system = (
            "You are an expert physician AI playing a medical diagnosis game. "
            "At each step choose ONE action. Output ONLY the action string, nothing else."
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=30,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip().strip("\"'.")
        if raw in action_space:
            return raw
        # partial match
        for a in action_space:
            if raw.lower() in a.lower():
                return a
        return None

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# HEURISTIC FALLBACK AGENT
# ---------------------------------------------------------------------------
def heuristic_action(obs: dict) -> str:
    """Rule-based agent used when LLM is unavailable or returns invalid action."""
    action_space = obs.get("action_space", [])
    known    = obs.get("known_symptoms", {})
    tests    = obs.get("test_results",   {})
    positive = [s for s, v in known.items() if v]

    ask_actions  = [a for a in action_space if a.startswith("ask_")  and a[4:] not in known]
    test_actions = [a for a in action_space if a.startswith("test_")]
    diag_actions = [a for a in action_space if a.startswith("diagnose_")]
    untested     = [a for a in test_actions  if a[5:] not in tests]

    if len(known) < 3 and ask_actions:
        return random.choice(ask_actions)
    if positive and not tests and untested:
        return random.choice(untested)
    if any(v for v in tests.values()) and diag_actions:
        return random.choice(diag_actions)
    if ask_actions:
        return random.choice(ask_actions)
    if diag_actions:
        return random.choice(diag_actions)
    return random.choice(action_space)


def pick_action(obs: dict, history: List[str]) -> str:
    """Try LLM first, fall back to heuristic."""
    action = get_llm_action(obs, history)
    if action is None:
        action = heuristic_action(obs)
    return action


# ---------------------------------------------------------------------------
# OBSERVATION BUILDER  (converts raw env state → obs dict inference.py uses)
# ---------------------------------------------------------------------------
def build_obs(state: dict, env: DiagnosisEnv,
              last_action: Optional[str] = None,
              last_reward: float = 0.0) -> dict:
    return {
        "known_symptoms":  state.get("known_symptoms", {}),
        "test_results":    state.get("test_results", {}),
        "trust_score":     state.get("trust_score", 100),
        "steps_remaining": state.get("steps_remaining", 0),
        "total_cost":      state.get("total_cost", 0),
        "action_space":    env.action_space,
        "last_action":     last_action,
        "last_reward":     last_reward,
    }


# ---------------------------------------------------------------------------
# RUN ONE EPISODE  (directly against the Python env — no server needed)
# ---------------------------------------------------------------------------
def run_episode(task_level: str, episode_num: int,
                seed: Optional[int] = None) -> dict:
    """Run one full episode and emit [START] / [STEP] / [END] logs."""

    task_name = TASK_NAMES[task_level]
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env   = get_task(task_level)
    state = env.reset(seed=seed)
    obs   = build_obs(state, env)

    history:    List[str]   = []
    rewards:    List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False
    error_msg   = None

    try:
        for step_num in range(1, MAX_STEPS_PER_EPISODE + 1):
            if env.done:
                break

            action    = pick_action(obs, history)
            error_msg = None

            try:
                next_state, raw_reward, done, info = env.step(action)
                norm_reward = normalise_reward(raw_reward)
                obs         = build_obs(next_state, env,
                                        last_action=action,
                                        last_reward=raw_reward)
            except Exception as exc:
                norm_reward = 0.0
                done        = True
                error_msg   = str(exc)

            rewards.append(norm_reward)
            steps_taken = step_num

            log_step(
                step=step_num,
                action=action,
                reward=round(norm_reward, 4),
                done=env.done,
                error=error_msg,
            )

            history.append(f"Step {step_num}: {action!r} → reward {raw_reward:+.2f}")

            if env.done:
                break

        score   = round(statistics.mean(rewards), 4) if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    return {
        "task":    task_level,
        "episode": episode_num,
        "score":   score,
        "steps":   steps_taken,
        "success": success,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # Warn if no token, but DON'T exit — heuristic agent works without it
    if not API_KEY:
        print("[DEBUG] HF_TOKEN not set — using heuristic agent (no LLM calls)", flush=True)

    print(f"[DEBUG] Model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] Episodes: {EPISODES_PER_TASK} per task x 3 tasks", flush=True)

    all_results    = {}
    overall_scores = []

    for task in TASKS:
        task_scores  = []
        task_results = []

        for ep in range(EPISODES_PER_TASK):
            seed   = 42 + ep
            result = run_episode(task, ep + 1, seed=seed)
            task_scores.append(result["score"])
            task_results.append(result)

        avg_score = round(statistics.mean(task_scores), 4)
        all_results[task] = {
            "episodes":   task_results,
            "avg_score":  avg_score,
            "task_score": avg_score,
        }
        overall_scores.append(avg_score)

    overall_avg = round(statistics.mean(overall_scores), 4)

    # Machine-readable summary
    summary = {
        "model":         MODEL_NAME,
        "environment":   BENCHMARK,
        "overall_score": overall_avg,
        "task_scores":   {t: all_results[t]["avg_score"] for t in TASKS},
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
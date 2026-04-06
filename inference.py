"""
inference.py — OpenEnv Baseline Inference Script for DiagnosisEnv
==================================================================
Runs an LLM agent (via OpenAI client) against DiagnosisEnv for all
three task levels and produces reproducible baseline scores.

Environment variables required:
  API_BASE_URL  — the LLM API base URL
  MODEL_NAME    — the model identifier (e.g. "gpt-4o-mini")
  HF_TOKEN      — your Hugging Face / API key

Usage:
  export API_BASE_URL="https://api-inference.huggingface.co/v1"
  export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
  export HF_TOKEN="hf_your_token_here"
  python inference.py

Stdout format (STRICTLY enforced — do not modify):
  [START] task=<name> env=DiagnosisEnv model=<model>
  [STEP]  step=<n> action=<str> reward=<float> done=<bool> error=<str|None>
  [END]   success=<bool> steps=<n> score=<float> rewards=<list>
"""

import os
import sys
import json
import random
import statistics
import time
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# ENVIRONMENT CONFIG  (read from environment variables — mandatory)
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.environ.get("HF_TOKEN",     "")  # hf_ token is the API key — no OpenAI needed

SERVER_URL   = os.environ.get("SERVER_URL", "http://localhost:7860")

EPISODES_PER_TASK    = 3       # keep low for speed; set to 10 for full eval
MAX_STEPS_PER_EPISODE = 25     # hard cap per episode to respect 20-min limit
SUCCESS_SCORE_THRESHOLD = 0.5  # score >= this → episode success

TASKS = ["easy", "medium", "hard"]

BENCHMARK  = "DiagnosisEnv"
TASK_NAMES = {
    "easy":   "diagnosis_easy",
    "medium": "diagnosis_medium",
    "hard":   "diagnosis_hard",
}


# ---------------------------------------------------------------------------
# STRICT LOG HELPERS  — DO NOT MODIFY FIELD NAMES OR FORMAT
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
# SERVER CLIENT HELPERS
# ---------------------------------------------------------------------------
def _post(endpoint: str, payload: dict = None, retries: int = 3) -> dict:
    """POST to the local DiagnosisEnv server with retry logic."""
    url = f"{SERVER_URL}{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload or {}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"POST {endpoint} failed after {retries} retries: {exc}")
            time.sleep(1)


def _get(endpoint: str) -> dict:
    """GET from the local DiagnosisEnv server."""
    url = f"{SERVER_URL}{endpoint}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def reset_env(task: str = "easy", seed: Optional[int] = None) -> dict:
    payload = {"task": task}
    if seed is not None:
        payload["seed"] = seed
    return _post("/reset", payload)


def step_env(action: str, rationale: Optional[str] = None) -> dict:
    payload = {"action": action}
    if rationale:
        payload["rationale"] = rationale
    return _post("/step", payload)


# ---------------------------------------------------------------------------
# LLM AGENT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert physician AI. You are playing a medical diagnosis game.

At each step you receive:
- A list of symptoms already known (positive and negative)
- Test results so far
- Patient trust score (0-100, higher is better)
- Steps remaining
- Total cost so far (in INR)

You must choose ONE action from the provided action_space list. Output ONLY the action string, nothing else.

Strategy:
1. Start by asking about 2-3 distinguishing symptoms
2. If symptoms point to a disease, order the confirmatory test for that disease
3. If you have a positive confirmatory test or strong symptom pattern, diagnose
4. Avoid repeating actions (penalised)
5. Avoid ordering too many expensive tests (cost penalty)
6. Make a diagnosis before running out of steps

IMPORTANT: Your response must be EXACTLY one action string from the action_space list."""


def build_prompt(obs: dict) -> str:
    """Build a concise prompt from the current observation."""
    known = obs.get("known_symptoms", {})
    tests = obs.get("test_results", {})
    action_space = obs.get("action_space", [])

    pos_symptoms = [k for k, v in known.items() if v]
    neg_symptoms = [k for k, v in known.items() if not v]
    pos_tests    = [k for k, v in tests.items() if v]
    neg_tests    = [k for k, v in tests.items() if not v]

    lines = [
        f"Trust: {obs.get('trust_score', 100)}/100  |  "
        f"Steps left: {obs.get('steps_remaining', 0)}  |  "
        f"Cost: ₹{obs.get('total_cost', 0)}",
        "",
        f"Positive symptoms : {', '.join(pos_symptoms) or 'none'}",
        f"Negative symptoms : {', '.join(neg_symptoms) or 'none'}",
        f"Positive tests    : {', '.join(pos_tests) or 'none'}",
        f"Negative tests    : {', '.join(neg_tests) or 'none'}",
        "",
        "Available actions (pick exactly one):",
    ]

    # Show a condensed subset to avoid hitting context limits
    ask_actions      = [a for a in action_space if a.startswith("ask_")]
    test_actions     = [a for a in action_space if a.startswith("test_")]
    diagnose_actions = [a for a in action_space if a.startswith("diagnose_")]

    # Only show unasked symptoms to keep prompt short
    unasked_symptoms = [a for a in ask_actions if a[4:] not in known]
    untested         = [a for a in test_actions if a[5:] not in tests]

    lines.append(f"  Ask symptoms : {unasked_symptoms}")
    lines.append(f"  Order tests  : {untested}")
    lines.append(f"  Diagnose     : {diagnose_actions}")

    return "\n".join(lines)


def get_llm_action(client: OpenAI, obs: dict,
                   history: List[str], step: int) -> str:
    """
    Call the LLM to pick the next action.
    Falls back to a heuristic agent on error to avoid crashing.
    """
    action_space = obs.get("action_space", [])
    if not action_space:
        return "diagnose_dengue"

    prompt = build_prompt(obs)
    if history:
        prompt += f"\n\nRecent history:\n" + "\n".join(history[-5:])

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=50,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()

        # Validate: must be a real action
        if raw in action_space:
            return raw

        # Try to find a partial match (LLM sometimes adds spaces/punctuation)
        raw_clean = raw.strip("\"'.").strip()
        if raw_clean in action_space:
            return raw_clean

        # Fallback to heuristic
        print(f"[DEBUG] LLM returned invalid action '{raw}', using heuristic",
              flush=True)
        return _heuristic_fallback(obs, action_space)

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return _heuristic_fallback(obs, action_space)


def _heuristic_fallback(obs: dict, action_space: list) -> str:
    """Simple heuristic used as fallback when LLM fails."""
    known    = obs.get("known_symptoms", {})
    tests    = obs.get("test_results", {})
    positive = [s for s, v in known.items() if v]

    ask_actions      = [a for a in action_space if a.startswith("ask_")
                        and a[4:] not in known]
    test_actions     = [a for a in action_space if a.startswith("test_")]
    diagnose_actions = [a for a in action_space if a.startswith("diagnose_")]
    untested         = [a for a in test_actions if a[5:] not in tests]

    if len(known) < 3 and ask_actions:
        return random.choice(ask_actions)
    if positive and not tests and untested:
        return random.choice(untested)
    if any(v for v in tests.values()) and diagnose_actions:
        return random.choice(diagnose_actions)
    if ask_actions:
        return random.choice(ask_actions)
    if diagnose_actions:
        return random.choice(diagnose_actions)
    return random.choice(action_space)


# ---------------------------------------------------------------------------
# RUN ONE EPISODE
# ---------------------------------------------------------------------------
def run_episode(client: OpenAI, task: str, episode_num: int,
                seed: Optional[int] = None) -> dict:
    """
    Run one full episode for a given task level.

    Returns:
        dict with score (0.0–1.0), steps, rewards, success
    """
    task_name = TASK_NAMES[task]
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    # Reset environment
    reset_data = reset_env(task=task, seed=seed)
    obs        = reset_data.get("observation", {})

    history:  List[str]   = []
    rewards:  List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False
    done        = False

    try:
        for step_num in range(1, MAX_STEPS_PER_EPISODE + 1):
            if done:
                break

            action = get_llm_action(client, obs, history, step_num)

            try:
                step_data   = step_env(action)
                obs         = step_data.get("observation", {})
                reward_info = step_data.get("reward", {})
                raw_reward  = reward_info.get("value", 0.0)
                norm_reward = reward_info.get("normalised", 0.0)
                done        = step_data.get("done", False)
                info        = step_data.get("info", {})
                error       = None
            except Exception as exc:
                raw_reward  = 0.0
                norm_reward = 0.0
                done        = True
                error       = str(exc)
                info        = {}

            rewards.append(norm_reward)
            steps_taken = step_num

            log_step(
                step=step_num,
                action=action,
                reward=round(norm_reward, 4),
                done=done,
                error=error,
            )

            history.append(
                f"Step {step_num}: {action!r} → reward {raw_reward:+.2f}"
            )

            if done:
                break

        # Episode score = mean of normalised per-step rewards
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
        "task":    task,
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
    # Validate environment variables
    if not API_KEY:
        print("[ERROR] HF_TOKEN environment variable is not set.", flush=True)
        sys.exit(1)

    print(f"\n{'='*60}", flush=True)
    print(f"  DiagnosisEnv — Baseline Inference Script", flush=True)
    print(f"  Model      : {MODEL_NAME}", flush=True)
    print(f"  API Base   : {API_BASE_URL}", flush=True)
    print(f"  Server     : {SERVER_URL}", flush=True)
    print(f"  Episodes   : {EPISODES_PER_TASK} per task × 3 tasks", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Check server is alive
    try:
        health = _get("/health")
        print(f"  ✅ Server healthy: {health}", flush=True)
    except Exception as exc:
        print(f"  ❌ Cannot reach server at {SERVER_URL}: {exc}", flush=True)
        print(f"     Start it with: uvicorn server:app --host 0.0.0.0 --port 7860",
              flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_results = {}
    overall_scores = []

    for task in TASKS:
        print(f"\n{'─'*60}", flush=True)
        print(f"  TASK: {task.upper()}", flush=True)
        print(f"{'─'*60}", flush=True)

        task_scores  = []
        task_results = []

        for ep in range(EPISODES_PER_TASK):
            seed = 42 + ep
            print(f"\n  Episode {ep + 1}/{EPISODES_PER_TASK} (seed={seed})",
                  flush=True)
            result = run_episode(client, task, ep + 1, seed=seed)
            task_scores.append(result["score"])
            task_results.append(result)

        avg_score = round(statistics.mean(task_scores), 4)
        all_results[task] = {
            "episodes":   task_results,
            "avg_score":  avg_score,
            "task_score": avg_score,
        }
        overall_scores.append(avg_score)

        print(f"\n  ★ {task.upper()} avg score: {avg_score:.4f} / 1.0", flush=True)

    # Final summary
    overall_avg = round(statistics.mean(overall_scores), 4)

    print(f"\n{'='*60}", flush=True)
    print(f"  FINAL BASELINE RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for task in TASKS:
        s = all_results[task]["avg_score"]
        print(f"  {task.upper():<8} : {s:.4f} / 1.0  ({s*100:.1f}%)", flush=True)
    print(f"  {'─'*40}", flush=True)
    print(f"  OVERALL  : {overall_avg:.4f} / 1.0  ({overall_avg*100:.1f}%)",
          flush=True)
    print(f"{'='*60}\n", flush=True)

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
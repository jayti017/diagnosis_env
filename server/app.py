import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import DiagnosisEnv
from tasks import get_task, TASK_REGISTRY
from models import (
    DiagnosisObservation,
    DiagnosisAction,
    DiagnosisReward,
    StepResult,
    ResetResult,
    StateResult,
)
from rewards import explain_reward


# ---------------------------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DiagnosisEnv",
    description=(
        "OpenEnv-compliant RL environment for AI medical diagnosis. "
        "The agent acts as a doctor: it gathers symptoms, orders tests, "
        "and makes a final diagnosis under time, cost, and trust constraints."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GLOBAL ENVIRONMENT STATE
# ---------------------------------------------------------------------------
_env: Optional[DiagnosisEnv] = None
_current_task: str = "easy"
_last_info: Dict[str, Any] = {}


def _get_env() -> DiagnosisEnv:
    global _env
    if _env is None:
        _env = get_task("easy")
    return _env


def _state_to_observation(state: dict, action_space: list,
                         last_action: Optional[str] = None,
                         last_reward: float = 0.0) -> DiagnosisObservation:

    pos = [k for k, v in state.get("known_symptoms", {}).items() if v]
    neg = [k for k, v in state.get("known_symptoms", {}).items() if not v]
    pos_tests = [k for k, v in state.get("test_results", {}).items() if v]

    message_parts = []
    if pos:
        message_parts.append(f"Positive symptoms: {', '.join(pos)}.")
    if neg:
        message_parts.append(f"Negative symptoms: {', '.join(neg)}.")
    if pos_tests:
        message_parts.append(f"Positive tests: {', '.join(pos_tests)}.")

    message_parts.append(
        f"Trust: {state.get('trust_score', 100)}/100. "
        f"Steps left: {state.get('steps_remaining', 0)}. "
        f"Cost so far: ₹{state.get('total_cost', 0)}."
    )

    return DiagnosisObservation(
        known_symptoms=state.get("known_symptoms", {}),
        test_results=state.get("test_results", {}),
        trust_score=state.get("trust_score", 100),
        steps_remaining=state.get("steps_remaining", 0),
        total_cost=state.get("total_cost", 0),
        action_space=action_space,
        last_action=last_action,
        last_reward=last_reward,
        message=" ".join(message_parts),
    )


def _normalise_reward(raw_reward: float) -> float:
    MIN_R = -175.0
    MAX_R = +115.0
    clamped = max(MIN_R, min(MAX_R, raw_reward))
    return round((clamped - MIN_R) / (MAX_R - MIN_R), 4)


# ---------------------------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: str
    rationale: Optional[str] = None


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "environment": "DiagnosisEnv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    tasks_info = {}
    for level, factory in TASK_REGISTRY.items():
        cfg = factory()
        tasks_info[level] = {
            "name": cfg["name"],
            "description": cfg["description"],
            "max_steps": cfg["max_steps"],
            "noise_level": cfg["noise_level"],
            "diseases": cfg["disease_pool"],
        }
    return {"tasks": tasks_info}


@app.post("/reset", response_model=ResetResult)
def reset(req: ResetRequest = None):
    global _env, _current_task, _last_info

    if req is None:
        req = ResetRequest()

    task_level = req.task if req.task in TASK_REGISTRY else "easy"
    _current_task = task_level
    _env = get_task(task_level)
    _last_info = {}

    state = _env.reset(seed=req.seed)
    obs = _state_to_observation(state, _env.action_space)

    pos_symptoms = [k for k, v in state.get("known_symptoms", {}).items() if v]

    msg = f"New episode started — Task: {task_level.upper()}. Patient has arrived. "
    if pos_symptoms:
        msg += f"They mention: {', '.join(pos_symptoms)}. "
    msg += f"You have {_env.max_steps} steps to make a diagnosis."

    obs.message = msg

    return ResetResult(
        observation=obs,
        task=task_level,
        message=msg,
    )


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    global _last_info

    env = _get_env()

    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode ended. Call /reset"
        )

    if req.action not in env.action_space:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action: {req.action}"
        )

    raw_state, raw_reward, done, info = env.step(req.action)

    result_for_explain = _info_to_result(info, req.action)
    breakdown = explain_reward(raw_state, req.action, result_for_explain, done)

    reward_obj = DiagnosisReward(
        value=raw_reward,
        components=breakdown,
        normalised=_normalise_reward(raw_reward),
    )

    obs = _state_to_observation(
        raw_state,
        env.action_space,
        last_action=req.action,
        last_reward=raw_reward,
    )

    if req.rationale:
        info["rationale"] = req.rationale

    _last_info = info

    return StepResult(
        observation=obs,
        reward=reward_obj,
        done=done,
        info=info,
    )


@app.get("/state", response_model=StateResult)
def get_state():
    env = _get_env()

    true_disease = None
    if env.done and env.patient:
        true_disease = env.patient.true_disease

    return StateResult(
        state=dict(env.state) if env.state else {},
        task=_current_task,
        done=env.done,
        steps_taken=env.steps_taken,
        true_disease=true_disease,
    )


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------
def _info_to_result(info: dict, action: str) -> dict:
    if action.startswith("ask_"):
        return {"type": "symptom", "value": info.get("answer")}
    elif action.startswith("test_"):
        return {"type": "test", "value": info.get("result")}
    elif action.startswith("diagnose_"):
        return {"type": "diagnosis", "correct": info.get("correct")}
    elif info.get("timeout"):
        return {"type": "timeout"}
    return {}


# ---------------------------------------------------------------------------
# MAIN (REQUIRED FOR OPENENV)
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)
if __name__ == "__main__":
    main()
# server/app.py

import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading  # <-- NEW

# Import your existing modules
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

# ---------------------------------------------------------------------------
# (Your existing helper functions go here...)
# _get_env, _state_to_observation, _normalise_reward, endpoints, etc.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# RUN INFERENCE IN BACKGROUND
# ---------------------------------------------------------------------------
def start_inference_thread():
    """Run inference.py in a separate thread so stdout prints structured logs."""
    import inference  # your inference.py in the same repo
    print("RUNNING INFERENCE FILE", flush=True)
    inference.main()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))

    # Start inference in a background thread
    threading.Thread(target=start_inference_thread, daemon=True).start()

    # Start FastAPI server
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
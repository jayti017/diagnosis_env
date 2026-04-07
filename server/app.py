# server/app.py

import os
from typing import Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading

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
# 🔥 START INFERENCE ON STARTUP (CRITICAL FIX)
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    def run_inference():
        try:
            import inference
            print("🔥 STARTING INFERENCE...", flush=True)
            inference.main()
        except Exception as e:
            print(f"❌ Inference crashed: {e}", flush=True)

    threading.Thread(target=run_inference, daemon=True).start()

# ---------------------------------------------------------------------------
# BASIC HEALTH CHECK (optional but useful)
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "DiagnosisEnv API is running"}

# ---------------------------------------------------------------------------
# MAIN (NOT USED BY HUGGING FACE BUT OK TO KEEP)
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
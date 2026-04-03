"""
models.py — Pydantic Models for DiagnosisEnv OpenEnv Compliance
================================================================
Defines all typed models required by the OpenEnv specification:
  - Observation  : what the agent sees each step
  - Action       : what the agent can do
  - Reward       : the reward signal
  - StepResult   : the full return from step()
  - ResetResult  : the return from reset()
  - StateResult  : the return from state()
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# OBSERVATION MODEL
# ---------------------------------------------------------------------------
class DiagnosisObservation(BaseModel):
    """
    What the agent observes at each step of the diagnosis episode.

    Fields:
        known_symptoms   : dict of symptom → bool for all queried symptoms
        test_results     : dict of test_name → bool for all ordered tests
        trust_score      : patient trust level (0–100); lower = less cooperative
        steps_remaining  : how many steps are left before forced timeout
        total_cost       : cumulative cost of tests ordered so far (INR)
        action_space     : list of all valid actions the agent may take
        last_action      : the action taken in the previous step (None at reset)
        last_reward      : reward received for the previous step (0.0 at reset)
        message          : human-readable description of the current state
    """
    known_symptoms:  Dict[str, bool]       = Field(default_factory=dict,
        description="Symptoms queried so far and their answers")
    test_results:    Dict[str, bool]       = Field(default_factory=dict,
        description="Diagnostic tests run and their results")
    trust_score:     int                   = Field(default=100, ge=0, le=100,
        description="Patient trust level (0–100)")
    steps_remaining: int                   = Field(default=0, ge=0,
        description="Steps left before episode timeout")
    total_cost:      int                   = Field(default=0, ge=0,
        description="Cumulative test cost in INR")
    action_space:    List[str]             = Field(default_factory=list,
        description="All valid actions for the current task")
    last_action:     Optional[str]         = Field(default=None,
        description="Previous action taken by the agent")
    last_reward:     float                 = Field(default=0.0,
        description="Reward received for the last step")
    message:         str                   = Field(default="",
        description="Human-readable summary of current state")


# ---------------------------------------------------------------------------
# ACTION MODEL
# ---------------------------------------------------------------------------
class DiagnosisAction(BaseModel):
    """
    An action the agent can take in DiagnosisEnv.

    The action string must be one of:
      - "ask_<symptom>"      e.g. "ask_high_fever"
      - "test_<test_name>"   e.g. "test_blood_test"
      - "diagnose_<disease>" e.g. "diagnose_dengue"

    Fields:
        action  : the action string
        rationale: optional reasoning (for LLM agents)
    """
    action:    str            = Field(...,
        description="Action string: ask_<symptom> | test_<name> | diagnose_<disease>")
    rationale: Optional[str] = Field(default=None,
        description="Optional reasoning for the chosen action (for LLM agents)")


# ---------------------------------------------------------------------------
# REWARD MODEL
# ---------------------------------------------------------------------------
class DiagnosisReward(BaseModel):
    """
    The reward signal for a single step.

    Fields:
        value       : scalar reward (can be negative)
        components  : breakdown of reward components for explainability
        normalised  : value normalised to 0.0–1.0 (for OpenEnv grading)
    """
    value:      float              = Field(...,
        description="Raw reward value (can be negative)")
    components: Dict[str, float]   = Field(default_factory=dict,
        description="Breakdown of reward by component")
    normalised: float              = Field(default=0.0, ge=0.0, le=1.0,
        description="Reward normalised to [0, 1] for grading")


# ---------------------------------------------------------------------------
# STEP RESULT MODEL
# ---------------------------------------------------------------------------
class StepResult(BaseModel):
    """
    Full result returned by POST /step.

    Mirrors the OpenEnv step() contract:
        observation, reward, done, info
    """
    observation: DiagnosisObservation = Field(...,
        description="New observation after the action")
    reward:      DiagnosisReward      = Field(...,
        description="Reward received for this step")
    done:        bool                 = Field(...,
        description="Whether the episode has ended")
    info:        Dict[str, Any]       = Field(default_factory=dict,
        description="Extra info (action type, repeated, correct diagnosis, etc.)")


# ---------------------------------------------------------------------------
# RESET RESULT MODEL
# ---------------------------------------------------------------------------
class ResetResult(BaseModel):
    """Result returned by POST /reset."""
    observation: DiagnosisObservation = Field(...,
        description="Initial observation after reset")
    task:        str                  = Field(default="easy",
        description="Current task level: easy | medium | hard")
    message:     str                  = Field(default="",
        description="Human-readable welcome message")


# ---------------------------------------------------------------------------
# STATE RESULT MODEL
# ---------------------------------------------------------------------------
class StateResult(BaseModel):
    """Result returned by GET /state."""
    state:           Dict[str, Any]  = Field(...,
        description="Full internal environment state")
    task:            str             = Field(default="easy",
        description="Current task level")
    done:            bool            = Field(default=False,
        description="Whether the current episode is finished")
    steps_taken:     int             = Field(default=0,
        description="Steps taken so far in this episode")
    true_disease:    Optional[str]   = Field(default=None,
        description="True disease (only revealed after episode ends)")

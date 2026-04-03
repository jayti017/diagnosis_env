"""
rewards.py — Reward Function for DiagnosisEnv
==============================================
Computes the scalar reward for each agent action.

Design philosophy:
  - Reward is a WEIGHTED SUM of multiple real-world objectives
  - Encourages: correct diagnosis, efficiency, low cost, trust retention
  - Penalises:  wrong diagnosis, redundant actions, high cost, timeout

Reward components:
  +100   correct final diagnosis
  -80    wrong final diagnosis
  -150   timeout (ran out of steps without diagnosing)
  -cost  test cost scaled to reward units (every ₹500 = -1 reward)
  -2     redundant action (asked/tested something already done)
  -0.5   per step taken (time pressure — encourages efficiency)
  -5     if trust drops below 50 (patient becoming uncooperative)
  -15    if trust drops below 20 (patient very unhappy)
"""


# ---------------------------------------------------------------------------
# REWARD WEIGHTS — easy to tune for ablation studies
# ---------------------------------------------------------------------------
REWARD_CORRECT_DIAGNOSIS  = +100.0
REWARD_WRONG_DIAGNOSIS    = -80.0
REWARD_TIMEOUT            = -150.0

PENALTY_PER_STEP          = -0.5    # small per-step penalty to encourage speed
PENALTY_COST_SCALE        = 500     # ₹500 = 1 reward point deducted
PENALTY_REDUNDANT_ACTION  = -2.0    # asked same symptom / ran same test twice
PENALTY_LOW_TRUST_1       = -5.0    # trust < 50
PENALTY_LOW_TRUST_2       = -15.0   # trust < 20 (severe)


# ---------------------------------------------------------------------------
# MAIN REWARD FUNCTION
# ---------------------------------------------------------------------------
def compute_reward(state, action, result, done):
    """
    Compute the reward for a single environment step.

    Args:
        state  (dict)  : current state AFTER the action was applied
                         keys: known_symptoms, test_results, trust_score,
                               steps_remaining, total_cost
        action (str)   : the action string (e.g. "ask_fever", "diagnose_dengue")
        result (dict)  : outcome dict returned by env.step() internals
                         keys vary by action type:
                           type        : "symptom" | "test" | "diagnosis" | "timeout"
                           correct     : bool (diagnosis actions only)
                           cost        : int  (test actions only)
                           repeated    : bool
        done   (bool)  : whether the episode ended after this action

    Returns:
        reward (float)
    """
    reward = 0.0

    if result is None:
        return reward

    action_type = result.get("type")

    # ------------------------------------------------------------------
    # 1. DIAGNOSIS REWARD / PENALTY
    # ------------------------------------------------------------------
    if action_type == "diagnosis":
        if result.get("correct"):
            reward += REWARD_CORRECT_DIAGNOSIS
            # Bonus for finishing quickly (efficiency bonus)
            steps_used  = state.get("steps_remaining", 0)
            speed_bonus = steps_used * 0.5   # more steps left = bigger bonus
            reward      += speed_bonus
        else:
            reward += REWARD_WRONG_DIAGNOSIS

    # ------------------------------------------------------------------
    # 2. TIMEOUT PENALTY
    # ------------------------------------------------------------------
    elif action_type == "timeout":
        reward += REWARD_TIMEOUT

    # ------------------------------------------------------------------
    # 3. TEST COST PENALTY
    # ------------------------------------------------------------------
    if action_type == "test" and not result.get("repeated", False):
        cost   = result.get("cost", 0)
        # Scale: every ₹500 spent deducts 1 reward point
        reward -= cost / PENALTY_COST_SCALE

    # ------------------------------------------------------------------
    # 4. REDUNDANT ACTION PENALTY
    # ------------------------------------------------------------------
    if result.get("repeated", False):
        reward += PENALTY_REDUNDANT_ACTION

    # ------------------------------------------------------------------
    # 5. STEP PENALTY (time pressure)
    # ------------------------------------------------------------------
    reward += PENALTY_PER_STEP

    # ------------------------------------------------------------------
    # 6. PATIENT TRUST PENALTY
    # ------------------------------------------------------------------
    trust = state.get("trust_score", 100)
    if trust < 20:
        reward += PENALTY_LOW_TRUST_2
    elif trust < 50:
        reward += PENALTY_LOW_TRUST_1

    return round(reward, 3)


# ---------------------------------------------------------------------------
# REWARD BREAKDOWN (for debugging / explainability)
# ---------------------------------------------------------------------------
def explain_reward(state, action, result, done):
    """
    Returns a dict showing each reward component separately.
    Useful for debugging and for the explainability module.
    """
    components = {}

    if result is None:
        return components

    action_type = result.get("type")

    if action_type == "diagnosis":
        if result.get("correct"):
            components["correct_diagnosis"]  = REWARD_CORRECT_DIAGNOSIS
            steps_used                       = state.get("steps_remaining", 0)
            components["speed_bonus"]        = round(steps_used * 0.5, 2)
        else:
            components["wrong_diagnosis"]    = REWARD_WRONG_DIAGNOSIS

    elif action_type == "timeout":
        components["timeout_penalty"] = REWARD_TIMEOUT

    if action_type == "test" and not result.get("repeated", False):
        cost = result.get("cost", 0)
        components["test_cost_penalty"] = round(-cost / PENALTY_COST_SCALE, 3)

    if result.get("repeated", False):
        components["redundant_action_penalty"] = PENALTY_REDUNDANT_ACTION

    components["step_penalty"] = PENALTY_PER_STEP

    trust = state.get("trust_score", 100)
    if trust < 20:
        components["trust_penalty"] = PENALTY_LOW_TRUST_2
    elif trust < 50:
        components["trust_penalty"] = PENALTY_LOW_TRUST_1

    components["TOTAL"] = round(sum(components.values()), 3)
    return components
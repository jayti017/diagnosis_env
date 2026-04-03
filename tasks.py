"""
tasks.py — Task Configurations for DiagnosisEnv
================================================
Defines three difficulty levels for the RL environment.

Each task is a config dict that controls:
  - which diseases are in the pool
  - how many steps the agent gets
  - how noisy patient responses are
  - how fast trust degrades
  - which symptoms/tests are available (optional subsetting)

The three levels reflect increasing real-world complexity:

  EASY   — 2 diseases with distinct non-overlapping symptoms, generous time
  MEDIUM — 4 diseases with overlapping symptoms, moderate constraints
  HARD   — 8 diseases with noisy symptoms, strict time + trust pressure
"""

from env import DiagnosisEnv, DISEASE_DB

# ---------------------------------------------------------------------------
# HELPER — filter symptom/test subsets for a disease pool
# ---------------------------------------------------------------------------
def _symptoms_for_pool(disease_pool):
    """Return all symptoms present in at least one disease in the pool."""
    symptoms = set()
    for d in disease_pool:
        symptoms.update(DISEASE_DB[d]["symptoms"].keys())
    return sorted(symptoms)

def _tests_for_pool(disease_pool):
    """Return all tests present in at least one disease in the pool."""
    tests = set()
    for d in disease_pool:
        tests.update(DISEASE_DB[d]["tests"].keys())
    return sorted(tests)


# ---------------------------------------------------------------------------
# EASY TASK
# ---------------------------------------------------------------------------
def make_easy_task():
    pool = ["uti", "dengue"]
    return {
        "name":           "Easy — 2 Diseases (Distinct Symptoms)",
        "disease_pool":   pool,
        "max_steps":      20,
        "noise_level":    0.05,
        "trust_penalty":  1,
        "symptom_subset": _symptoms_for_pool(pool),
        "test_subset":    _tests_for_pool(pool),
        "description": (
            "2 diseases with clearly distinct symptoms. "
            "The agent has 20 steps and almost no noise. "
            "A well-designed agent should achieve >90% accuracy."
        ),
    }


# ---------------------------------------------------------------------------
# MEDIUM TASK
# ---------------------------------------------------------------------------
def make_medium_task():
    pool = ["dengue", "typhoid", "malaria", "chikungunya"]
    return {
        "name":           "Medium — 4 Diseases (Overlapping Symptoms)",
        "disease_pool":   pool,
        "max_steps":      15,
        "noise_level":    0.10,
        "trust_penalty":  2,
        "symptom_subset": _symptoms_for_pool(pool),
        "test_subset":    _tests_for_pool(pool),
        "description": (
            "4 diseases that all present with fever and fatigue. "
            "The agent must ask the right distinguishing questions "
            "and use targeted tests efficiently. 15 steps."
        ),
    }


# ---------------------------------------------------------------------------
# HARD TASK
# ---------------------------------------------------------------------------
def make_hard_task():
    pool = [
        "dengue", "typhoid", "malaria", "tuberculosis",
        "covid19", "chikungunya", "pneumonia", "uti",
    ]
    return {
        "name":           "Hard — 8 Diseases (High Noise, Emergency Cases)",
        "disease_pool":   pool,
        "max_steps":      12,
        "noise_level":    0.18,
        "trust_penalty":  3,
        "symptom_subset": _symptoms_for_pool(pool),
        "test_subset":    _tests_for_pool(pool),
        "description": (
            "All 8 diseases, high symptom noise, only 12 steps. "
            "Agent must prioritise the most informative questions "
            "and tests while preserving patient trust. "
            "Respiratory diseases (TB, COVID, Pneumonia) create "
            "additional confusion."
        ),
    }


# ---------------------------------------------------------------------------
# TASK REGISTRY
# ---------------------------------------------------------------------------
TASK_REGISTRY = {
    "easy":   make_easy_task,
    "medium": make_medium_task,
    "hard":   make_hard_task,
}

def get_task(level="easy"):
    """
    Returns a configured DiagnosisEnv for the given difficulty level.

    Args:
        level (str): "easy" | "medium" | "hard"

    Returns:
        DiagnosisEnv instance ready for reset()
    """
    level = level.lower()
    if level not in TASK_REGISTRY:
        raise ValueError(f"Unknown task level '{level}'. Choose from: {list(TASK_REGISTRY)}")
    config = TASK_REGISTRY[level]()
    return DiagnosisEnv(config)


def print_task_summary():
    """Print a summary of all task configurations."""
    print("\n" + "=" * 60)
    print("  TASK CONFIGURATIONS")
    print("=" * 60)
    for level, factory in TASK_REGISTRY.items():
        cfg = factory()
        print(f"\n  [{level.upper()}]  {cfg['name']}")
        print(f"    Diseases   : {cfg['disease_pool']}")
        print(f"    Max steps  : {cfg['max_steps']}")
        print(f"    Noise      : {cfg['noise_level']*100:.0f}%")
        print(f"    Trust decay: -{cfg['trust_penalty']} / action")
        print(f"    {cfg['description']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_task_summary()
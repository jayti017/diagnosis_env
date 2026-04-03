"""
env.py — DiagnosisEnv: Core RL Environment for Disease Diagnosis
================================================================
This module defines the reinforcement learning environment.
The agent plays the role of a doctor: it asks about symptoms, orders
diagnostic tests, and finally makes a diagnosis — all under time,
cost, and patient-trust constraints.

Compatible with the Meta OpenEnv / Gymnasium-style API:
    obs, reward, done, info = env.step(action)
"""

import random
from rewards import compute_reward


# ---------------------------------------------------------------------------
# DISEASE KNOWLEDGE BASE
# ---------------------------------------------------------------------------
# Each disease entry contains:
#   symptoms     : {symptom_name : probability_of_showing (0.0–1.0)}
#   tests        : {test_name    : (cost_inr, sensitivity_0_to_1)}
#   confirmatory : the single most reliable test for this disease
# ---------------------------------------------------------------------------
DISEASE_DB = {
    "dengue": {
        "symptoms": {
            "high_fever":        0.95,
            "severe_headache":   0.88,
            "pain_behind_eyes":  0.82,
            "joint_pain":        0.80,
            "skin_rash":         0.72,
            "low_platelet":      0.88,
            "nausea":            0.68,
            "fatigue":           0.92,
            "bleeding_gums":     0.40,
        },
        "tests": {
            "NS1_antigen_test":  (900,  0.92),
            "blood_test":        (300,  0.75),
            "dengue_IgM_test":   (1200, 0.88),
        },
        "confirmatory": "NS1_antigen_test",
    },
    "typhoid": {
        "symptoms": {
            "sustained_fever":   0.95,
            "abdominal_pain":    0.88,
            "severe_headache":   0.82,
            "loss_of_appetite":  0.88,
            "constipation":      0.62,
            "rose_spots":        0.28,
            "nausea":            0.65,
            "fatigue":           0.90,
            "weakness":          0.92,
        },
        "tests": {
            "widal_test":        (400,  0.85),
            "blood_test":        (300,  0.60),
            "blood_culture":     (1500, 0.95),
        },
        "confirmatory": "widal_test",
    },
    "malaria": {
        "symptoms": {
            "high_fever":        0.95,
            "chills":            0.92,
            "muscle_pain":       0.85,
            "severe_headache":   0.80,
            "nausea":            0.75,
            "vomiting":          0.65,
            "fatigue":           0.90,
            "jaundice":          0.35,
            "low_platelet":      0.60,
        },
        "tests": {
            "malaria_RDT":       (200,  0.93),
            "blood_test":        (300,  0.65),
            "peripheral_smear":  (350,  0.90),
        },
        "confirmatory": "malaria_RDT",
    },
    "tuberculosis": {
        "symptoms": {
            "persistent_cough":       0.95,
            "night_sweats":           0.88,
            "weight_loss":            0.90,
            "fatigue":                0.92,
            "low_grade_fever":        0.85,
            "chest_pain":             0.60,
            "shortness_of_breath":    0.55,
            "loss_of_appetite":       0.85,
        },
        "tests": {
            "sputum_test":       (200,  0.85),
            "chest_xray":        (400,  0.82),
            "CBNAAT_test":       (1000, 0.95),
        },
        "confirmatory": "CBNAAT_test",
    },
    "covid19": {
        "symptoms": {
            "high_fever":             0.78,
            "cough":                  0.82,
            "shortness_of_breath":    0.65,
            "fatigue":                0.90,
            "loss_of_smell":          0.70,
            "loss_of_taste":          0.72,
            "severe_headache":        0.62,
            "muscle_pain":            0.68,
            "nausea":                 0.40,
        },
        "tests": {
            "rapid_antigen_test":  (150,  0.80),
            "RTPCR_test":          (500,  0.95),
            "chest_xray":          (400,  0.72),
        },
        "confirmatory": "RTPCR_test",
    },
    "chikungunya": {
        "symptoms": {
            "high_fever":        0.95,
            "joint_pain":        0.98,
            "skin_rash":         0.75,
            "muscle_pain":       0.80,
            "fatigue":           0.90,
            "severe_headache":   0.75,
            "joint_swelling":    0.72,
            "nausea":            0.55,
            "chills":            0.60,
        },
        "tests": {
            "chikungunya_IgM_test": (1100, 0.90),
            "blood_test":           (300,  0.55),
        },
        "confirmatory": "chikungunya_IgM_test",
    },
    "pneumonia": {
        "symptoms": {
            "high_fever":             0.88,
            "cough":                  0.95,
            "chest_pain":             0.72,
            "shortness_of_breath":    0.82,
            "chills":                 0.75,
            "fatigue":                0.88,
            "loss_of_appetite":       0.78,
            "nausea":                 0.42,
        },
        "tests": {
            "chest_xray":        (400,  0.90),
            "blood_test":        (300,  0.70),
            "sputum_test":       (200,  0.75),
        },
        "confirmatory": "chest_xray",
    },
    "uti": {
        "symptoms": {
            "burning_urination":  0.95,
            "frequent_urination": 0.92,
            "cloudy_urine":       0.82,
            "lower_back_pain":    0.65,
            "low_grade_fever":    0.55,
            "abdominal_pain":     0.60,
            "nausea":             0.40,
        },
        "tests": {
            "urine_test":        (150,  0.92),
            "urine_culture":     (700,  0.97),
            "blood_test":        (300,  0.45),
        },
        "confirmatory": "urine_test",
    },
}

# Derived global sets (used for action-space building)
ALL_SYMPTOMS = sorted({s for d in DISEASE_DB.values() for s in d["symptoms"]})
ALL_TESTS    = sorted({t for d in DISEASE_DB.values() for t in d["tests"]})
ALL_DISEASES = sorted(DISEASE_DB.keys())


# ---------------------------------------------------------------------------
# ACTION SPACE BUILDER
# ---------------------------------------------------------------------------
def build_action_space(disease_subset=None, symptom_subset=None, test_subset=None):
    """
    Constructs the list of valid string actions for a task.

    Action format:
        "ask_<symptom>"      — ask if patient has this symptom
        "test_<test_name>"   — order a diagnostic test
        "diagnose_<disease>" — make a final diagnosis (ends episode)
    """
    diseases = disease_subset or ALL_DISEASES
    symptoms = symptom_subset or ALL_SYMPTOMS
    tests    = test_subset    or ALL_TESTS

    actions  = [f"ask_{s}"      for s in symptoms]
    actions += [f"test_{t}"     for t in tests]
    actions += [f"diagnose_{d}" for d in diseases]
    return actions


# ---------------------------------------------------------------------------
# PATIENT SIMULATOR
# ---------------------------------------------------------------------------
class PatientSimulator:
    """
    Generates a synthetic patient with a hidden true disease.

    On initialisation it:
      - Probabilistically generates which symptoms the patient actually has
        (based on the disease's symptom probabilities + small jitter)
      - Adds 1–2 red-herring symptoms from unrelated diseases
      - Stores the true disease (never directly shown to the agent)

    The agent interacts through two methods:
      answer_symptom_question(symptom) → bool
      run_test(test_name)              → (bool result, int cost)
    """

    def __init__(self, true_disease, noise_level=0.10, seed=None):
        if seed is not None:
            random.seed(seed)

        self.true_disease = true_disease
        self.noise_level  = noise_level   # 0–1, probability of wrong symptom report
        disease_data      = DISEASE_DB[true_disease]

        # --- Build actual symptom profile ---
        self.actual_symptoms = {}
        for symptom, base_prob in disease_data["symptoms"].items():
            # ±15% jitter so not every patient is textbook
            prob = min(1.0, max(0.0, base_prob + random.uniform(-0.15, 0.15)))
            self.actual_symptoms[symptom] = random.random() < prob

        # --- Add 1–2 red-herring symptoms ---
        other = [s for s in ALL_SYMPTOMS if s not in self.actual_symptoms]
        if other:
            for s in random.sample(other, k=min(2, len(other))):
                self.actual_symptoms[s] = random.random() < 0.20

    def answer_symptom_question(self, symptom):
        """Returns True/False. Has noise_level chance of being wrong."""
        true_val = self.actual_symptoms.get(symptom, False)
        if random.random() < self.noise_level:
            return not true_val   # noisy response
        return true_val

    def run_test(self, test_name):
        """
        Returns (result: bool, cost: int).
        Sensitivity is taken from the disease DB if the test is relevant.
        """
        disease_data = DISEASE_DB[self.true_disease]
        if test_name in disease_data["tests"]:
            cost, sensitivity = disease_data["tests"][test_name]
            result = random.random() < sensitivity
        else:
            cost   = self._lookup_cost(test_name)
            result = random.random() < 0.08   # low false-positive
        return result, cost

    @staticmethod
    def _lookup_cost(test_name):
        for d in DISEASE_DB.values():
            if test_name in d["tests"]:
                return d["tests"][test_name][0]
        return 500


# ---------------------------------------------------------------------------
# DIAGNOSIS ENVIRONMENT
# ---------------------------------------------------------------------------
class DiagnosisEnv:
    """
    DiagnosisEnv — RL environment for sequential medical diagnosis.

    The agent is a doctor that must:
      1. Ask symptom questions to gather patient history
      2. Order diagnostic tests (each has a cost and result)
      3. Make a final diagnosis

    Episode ends when:
      - Agent calls diagnose_<disease>  (correct or wrong)
      - Steps run out (timeout → large penalty)

    State (dict):
        known_symptoms   : {symptom_name: bool}  — symptoms queried so far
        test_results     : {test_name: bool}      — tests run and their results
        trust_score      : int 0–100              — patient trust remaining
        steps_remaining  : int                    — time steps left in episode
        total_cost       : int                    — cumulative test cost (INR)

    Usage:
        env   = DiagnosisEnv(task_config)
        state = env.reset()
        state, reward, done, info = env.step("ask_high_fever")
        state, reward, done, info = env.step("test_blood_test")
        state, reward, done, info = env.step("diagnose_dengue")
    """

    def __init__(self, task_config):
        """
        Args:
            task_config (dict): configuration dict from tasks.py
        """
        self.task_config   = task_config
        self.disease_pool  = task_config["disease_pool"]
        self.max_steps     = task_config["max_steps"]
        self.noise_level   = task_config.get("noise_level",    0.10)
        self.trust_penalty = task_config.get("trust_penalty",  1)
        self.action_space  = build_action_space(
            disease_subset = self.disease_pool,
            symptom_subset = task_config.get("symptom_subset"),
            test_subset    = task_config.get("test_subset"),
        )

        self.patient       = None
        self.state         = None
        self.done          = False
        self.steps_taken   = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        """
        Start a new episode. Picks a random disease, creates a patient.
        Returns the initial state dict.
        """
        if seed is not None:
            random.seed(seed)

        true_disease  = random.choice(self.disease_pool)
        self.patient  = PatientSimulator(true_disease, self.noise_level, seed=seed)

        volunteered   = self._volunteered_symptoms()

        self.state = {
            "known_symptoms":  volunteered,
            "test_results":    {},
            "trust_score":     100,
            "steps_remaining": self.max_steps,
            "total_cost":      0,
        }
        self.done      = False
        self.steps_taken = 0
        return dict(self.state)

    # ------------------------------------------------------------------
    def step(self, action):
        """
        Execute one action and return (next_state, reward, done, info).

        Args:
            action (str): must be in self.action_space

        Returns:
            next_state (dict)
            reward     (float)
            done       (bool)
            info       (dict)  — extra debug/logging info
        """
        assert not self.done,              "Episode ended. Call reset()."
        assert action in self.action_space, f"Invalid action: '{action}'"

        info   = {"action": action, "repeated": False, "correct": None}
        result = None

        # ---- ASK SYMPTOM -----------------------------------------------
        if action.startswith("ask_"):
            symptom  = action[4:]
            repeated = symptom in self.state["known_symptoms"]
            answer   = self.patient.answer_symptom_question(symptom)

            self.state["known_symptoms"][symptom] = answer
            self._reduce_trust(self.trust_penalty)

            info.update({"symptom": symptom, "answer": answer, "repeated": repeated})
            result = {"type": "symptom", "value": answer, "repeated": repeated}

        # ---- ORDER TEST ------------------------------------------------
        elif action.startswith("test_"):
            test_name = action[5:]
            repeated  = test_name in self.state["test_results"]

            test_result, cost = self.patient.run_test(test_name)
            self.state["test_results"][test_name] = test_result

            if not repeated:
                self.state["total_cost"] += cost
            self._reduce_trust(self.trust_penalty * 2)

            info.update({"test": test_name, "result": test_result,
                         "cost": cost if not repeated else 0, "repeated": repeated})
            result = {"type": "test", "value": test_result,
                      "cost": cost, "repeated": repeated}

        # ---- DIAGNOSE --------------------------------------------------
        elif action.startswith("diagnose_"):
            predicted = action[9:]
            correct   = (predicted == self.patient.true_disease)
            self.done = True

            info.update({"predicted": predicted,
                         "true_disease": self.patient.true_disease,
                         "correct": correct})
            result = {"type": "diagnosis", "correct": correct, "predicted": predicted,
                      "true_disease": self.patient.true_disease}

        # ---- ADVANCE TIME ----------------------------------------------
        self.steps_taken            += 1
        self.state["steps_remaining"] = self.max_steps - self.steps_taken

        # Timeout — episode ends forcefully
        if self.state["steps_remaining"] <= 0 and not self.done:
            self.done          = True
            info["timeout"]    = True
            if result is None:
                result = {"type": "timeout", "correct": False}

        # ---- COMPUTE REWARD --------------------------------------------
        reward = compute_reward(self.state, action, result, self.done)

        return dict(self.state), reward, self.done, info

    # ------------------------------------------------------------------
    def get_true_disease(self):
        """Returns the hidden true disease (for grading/debugging only)."""
        return self.patient.true_disease if self.patient else None

    # ------------------------------------------------------------------
    def render(self):
        """Print a human-readable snapshot of the current state."""
        s = self.state
        pos = [k for k, v in s["known_symptoms"].items() if v]
        neg = [k for k, v in s["known_symptoms"].items() if not v]
        print("\n" + "─" * 55)
        print(f"  Steps remaining : {s['steps_remaining']} / {self.max_steps}")
        print(f"  Trust score     : {s['trust_score']} / 100")
        print(f"  Total cost      : ₹{s['total_cost']}")
        print(f"  +ve symptoms    : {pos or '—'}")
        print(f"  −ve symptoms    : {neg or '—'}")
        print(f"  Test results    : {s['test_results'] or '—'}")
        print("─" * 55)

    # ------------------------------------------------------------------
    def _reduce_trust(self, amount):
        self.state["trust_score"] = max(0, self.state["trust_score"] - amount)

    def _volunteered_symptoms(self):
        """Patient spontaneously mentions 2–3 positive symptoms on arrival."""
        positive = [s for s, v in self.patient.actual_symptoms.items() if v]
        n = min(3, len(positive))
        sample = random.sample(positive, k=n) if positive else []
        return {s: True for s in sample}

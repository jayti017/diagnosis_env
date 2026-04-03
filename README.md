# DiagnosisEnv 🩺

**OpenEnv-compliant reinforcement learning environment for AI-assisted medical diagnosis**

The agent plays the role of a doctor: it must ask the right symptom questions, order cost-effective diagnostic tests, and make a correct diagnosis — all under real-world constraints of **limited time**, **test cost (INR)**, and **patient trust**.

---

## Environment Description

DiagnosisEnv models **8 real infectious diseases** common in India:

| Disease | Key Distinguishing Features |
|---|---|
| Dengue | Pain behind eyes, low platelet, skin rash |
| Typhoid | Rose spots, sustained fever, constipation |
| Malaria | Chills, cyclical fever, jaundice |
| Tuberculosis | Persistent cough, night sweats, weight loss |
| COVID-19 | Loss of smell/taste, respiratory symptoms |
| Chikungunya | Severe joint swelling and pain |
| Pneumonia | Chest pain, productive cough, shortness of breath |
| UTI | Burning urination, cloudy urine, lower back pain |

Each patient is **synthetically generated** with:
- Probabilistic symptom profiles (±15% jitter for realism)
- 1–2 red-herring symptoms from other diseases
- Noisy symptom answers (configurable per task)

---

## Action Space

| Action Format | Example | Description |
|---|---|---|
| `ask_<symptom>` | `ask_high_fever` | Ask patient about a symptom |
| `test_<test_name>` | `test_blood_test` | Order a diagnostic test (costs INR) |
| `diagnose_<disease>` | `diagnose_dengue` | Make a final diagnosis (ends episode) |

---

## Observation Space

```json
{
  "known_symptoms":  {"high_fever": true, "nausea": false},
  "test_results":    {"blood_test": true},
  "trust_score":     85,
  "steps_remaining": 12,
  "total_cost":      300,
  "action_space":    ["ask_chills", "test_malaria_RDT", "diagnose_dengue", ...],
  "last_action":     "test_blood_test",
  "last_reward":     -1.1,
  "message":         "Positive symptoms: high_fever. Trust: 85/100. Steps left: 12."
}
```

---

## Reward Function

The reward is a **multi-component signal** provided at every step (not just at termination):

| Component | Value | When |
|---|---|---|
| Correct diagnosis | +100.0 | Final correct diagnosis |
| Speed bonus | +0.5 × steps_remaining | Faster = more bonus |
| Wrong diagnosis | -80.0 | Final wrong diagnosis |
| Timeout | -150.0 | Ran out of steps |
| Test cost | -(cost / 500) | Each new test ordered |
| Redundant action | -2.0 | Repeated symptom/test |
| Step penalty | -0.5 | Every step (time pressure) |
| Low trust (< 50) | -5.0 | Patient becoming uncooperative |
| Low trust (< 20) | -15.0 | Patient very unhappy |

---

## Tasks

### Easy (`diagnosis_easy`)
- **Diseases**: UTI, Dengue (clearly distinct symptoms)
- **Max steps**: 20 | **Noise**: 5% | **Trust decay**: -1/action
- **Target score**: > 0.70

### Medium (`diagnosis_medium`)
- **Diseases**: Dengue, Typhoid, Malaria, Chikungunya (all share fever + fatigue)
- **Max steps**: 15 | **Noise**: 10% | **Trust decay**: -2/action
- **Target score**: > 0.50

### Hard (`diagnosis_hard`)
- **Diseases**: All 8 (including respiratory: TB, COVID-19, Pneumonia)
- **Max steps**: 12 | **Noise**: 18% | **Trust decay**: -3/action
- **Target score**: > 0.40

---

## Setup & Usage

### Local (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo (no server needed)
python main.py

# Start the API server
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
# Build
docker build -t diagnosis-env .

# Run
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e HF_TOKEN="hf_your_token_here" \
  diagnosis-env
```

### Run Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token"
export SERVER_URL="http://localhost:7860"

python inference.py
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Take an action |
| `GET` | `/state` | Current state |
| `GET` | `/tasks` | List task configs |

### POST /reset

```json
// Request
{"task": "easy", "seed": 42}

// Response
{
  "observation": { "known_symptoms": {...}, "trust_score": 100, ... },
  "task": "easy",
  "message": "New episode started..."
}
```

### POST /step

```json
// Request
{"action": "ask_high_fever", "rationale": "Checking for fever first"}

// Response
{
  "observation": { "known_symptoms": {"high_fever": true}, ... },
  "reward": {"value": -1.5, "normalised": 0.51, "components": {...}},
  "done": false,
  "info": {"symptom": "high_fever", "answer": true, "repeated": false}
}
```

---

## Baseline Scores

Evaluated with `heuristic_agent` over 10 episodes (seed=42):

| Task | Composite Score | Accuracy |
|---|---|---|
| Easy | ~0.72 | ~80% |
| Medium | ~0.58 | ~55% |
| Hard | ~0.44 | ~35% |

---

## Project Structure

```
diagnosis-env/
├── env.py           # Core DiagnosisEnv RL environment
├── tasks.py         # Task configurations (easy/medium/hard)
├── rewards.py       # Multi-component reward function
├── grader.py        # Evaluation metrics + OpenEnv graders (0.0–1.0)
├── models.py        # Pydantic models (Observation, Action, StepResult, ...)
├── server.py        # FastAPI server (POST /reset, POST /step, GET /state)
├── inference.py     # LLM baseline inference script
├── main.py          # Demo runner (local, no server required)
├── openenv.yaml     # OpenEnv spec metadata
├── Dockerfile       # Container definition
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## Real-World Utility

This environment addresses a **genuine gap** in agent evaluation: there are very few benchmarks for **cost-aware, trust-constrained medical reasoning**. DiagnosisEnv is immediately useful for:

- Evaluating LLM agents on sequential clinical decision-making
- Benchmarking the efficiency of diagnostic pathways (cost vs. accuracy)
- Training agents that respect patient cooperation constraints
- Comparing reasoning strategies under information uncertainty and noise

# DiagnosisEnv 🩺

**OpenEnv-compliant reinforcement learning environment for AI-assisted medical diagnosis**

> **Team Coders** — Jayti Bhardwaj · Sourav Bhardwaj · Tanishka
> 🚀 **Live Space**: [souravbhardwaj22-diagnosis-env.hf.space](https://souravbhardwaj22-diagnosis-env.hf.space)

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
| Speed bonus | +0.5 x steps_remaining | Faster = more bonus |
| Wrong diagnosis | -80.0 | Final wrong diagnosis |
| Timeout | -150.0 | Ran out of steps |
| Test cost | -(cost / 500) | Each new test ordered |
| Redundant action | -2.0 | Repeated symptom/test |
| Step penalty | -0.5 | Every step (time pressure) |
| Low trust (< 50) | -5.0 | Patient becoming uncooperative |
| Low trust (< 20) | -15.0 | Patient very unhappy |

---

## Tasks

### Easy (diagnosis_easy)
- **Diseases**: UTI, Dengue (clearly distinct symptoms)
- **Max steps**: 20 | **Noise**: 5% | **Trust decay**: -1/action
- **Target score**: > 0.70

### Medium (diagnosis_medium)
- **Diseases**: Dengue, Typhoid, Malaria, Chikungunya (all share fever + fatigue)
- **Max steps**: 15 | **Noise**: 10% | **Trust decay**: -2/action
- **Target score**: > 0.50

### Hard (diagnosis_hard)
- **Diseases**: All 8 (including respiratory: TB, COVID-19, Pneumonia)
- **Max steps**: 12 | **Noise**: 18% | **Trust decay**: -3/action
- **Target score**: > 0.40

---

## Setup & Usage

### Requirements

No OpenAI API key needed. Only a free Hugging Face token (hf_) is required to run inference.

### Local (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo (no server needed)
python main.py

# Start the API server
uvicorn server:app --host 0.0.0.0 --port 7860
```

Windows users: use `uvicorn server.app:app --port 7860` if server is inside a server/ folder.

### Docker

```bash
# Build
docker build -t diagnosis-env .

# Run (only HF_TOKEN needed — no OpenAI key required)
docker run -p 7860:7860 -e HF_TOKEN="hf_your_token_here" diagnosis-env
```

### Run Inference

Uses the Hugging Face Inference API (free) — no OpenAI account needed.

```bash
# Mac / Linux
export HF_TOKEN="hf_your_token_here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export SERVER_URL="http://localhost:7860"
python inference.py

# Windows PowerShell (all on one line)
$env:HF_TOKEN="hf_your_token_here"; $env:API_BASE_URL="https://router.huggingface.co/v1"; $env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"; $env:SERVER_URL="http://localhost:7860"; python inference.py
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | Health check |
| POST | /reset | Start new episode |
| POST | /step | Take an action |
| GET | /state | Current state |
| GET | /tasks | List task configs |

### POST /reset

```json
{"task": "easy", "seed": 42}
```

### POST /step

```json
{"action": "ask_high_fever", "rationale": "Checking for fever first"}
```

---

## Baseline Scores

Evaluated with heuristic_agent over 10 episodes (seed=42):

| Task | Composite Score (0-1) | Accuracy | Avg Cost (INR) | Avg Steps |
|---|---|---|---|---|
| Easy | 0.70 | ~80% | 350 | 8.2 |
| Medium | 0.33 | ~55% | 480 | 12.4 |
| Hard | 0.43 | ~35% | 620 | 11.8 |

Scores produced by the built-in heuristic agent. An LLM agent using Qwen/Qwen2.5-72B-Instruct via the HF Inference API is expected to score higher.

---

## Project Structure

```
diagnosis-env/
├── env.py                  # Core DiagnosisEnv RL environment + PatientSimulator
├── tasks.py                # Task configurations (easy / medium / hard)
├── rewards.py              # Multi-component step-wise reward function
├── grader.py               # Evaluation metrics + OpenEnv graders (returns 0.0-1.0)
├── models.py               # Pydantic models (Observation, Action, StepResult...)
│
├── server/                 # FastAPI server package
│   ├── __init__.py         # Makes 'server' a Python package
│   └── app.py              # FastAPI app (POST /reset, POST /step, GET /state)
│
├── inference.py            # LLM baseline inference script (HF Inference API)
├── test_manual.py          # Interactive manual test console (play as the doctor)
├── main.py                 # Demo runner (local, no server required)
│
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile              # Container definition (port 7860)
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

---

## Real-World Utility

This environment addresses a genuine gap in agent evaluation: there are very few benchmarks for cost-aware, trust-constrained medical reasoning. DiagnosisEnv is immediately useful for:

- Evaluating LLM agents on sequential clinical decision-making
- Benchmarking the efficiency of diagnostic pathways (cost vs. accuracy)
- Training agents that respect patient cooperation constraints
- Comparing reasoning strategies under information uncertainty and noise

---

## Team

| Name | Role |
|---|---|
| Jayti Bhardwaj | Core environment and models (Team Lead) |
| Tanishka | Tasks, rewards and grading |
| Sourav Bhardwaj | Deployment, API and inference |
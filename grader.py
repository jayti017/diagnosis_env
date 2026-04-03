"""
grader.py — Evaluation & Grading System for DiagnosisEnv
=========================================================
Runs the environment for multiple episodes using a supplied agent policy,
then computes standardised metrics for hackathon scoring.

Metrics reported:
  accuracy          — % of episodes with correct diagnosis
  avg_cost          — mean INR spent on tests per episode
  avg_steps         — mean number of steps taken per episode
  avg_trust         — mean final trust score per episode (0–100)
  avg_reward        — mean total reward per episode
  timeout_rate      — % of episodes that timed out
  composite_score   — single leaderboard score (see formula below)

Composite Score Formula (0–100):
  score = 0.40 * accuracy
        + 0.20 * (1 - normalised_cost)
        + 0.20 * (1 - normalised_steps)
        + 0.20 * normalised_trust
  All terms clamped to [0, 1] before weighting.
"""

import random
import statistics


# ---------------------------------------------------------------------------
# BUILT-IN BASELINE AGENTS (for comparison)
# ---------------------------------------------------------------------------

def random_agent(state, action_space):
    """Picks a random valid action. Used as the weakest baseline."""
    return random.choice(action_space)


def heuristic_agent(state, action_space):
    """
    Simple rule-based agent — the smarter baseline.

    Strategy:
      1. If fewer than 3 symptoms have been asked, ask an unanswered symptom.
      2. If a positive symptom is known and we haven't run any test, run a test.
      3. If we have at least 1 positive test result, make a diagnosis.
      4. Otherwise, ask another symptom.
      5. Fallback: diagnose randomly.
    """
    known    = state.get("known_symptoms", {})
    tests    = state.get("test_results",   {})
    positive = [s for s, v in known.items() if v]

    ask_actions      = [a for a in action_space if a.startswith("ask_")
                        and a[4:] not in known]
    test_actions     = [a for a in action_space if a.startswith("test_")]
    diagnose_actions = [a for a in action_space if a.startswith("diagnose_")]
    untested         = [a for a in test_actions
                        if a[5:] not in tests]

    if len(known) < 3 and ask_actions:
        return random.choice(ask_actions)

    if positive and not tests and untested:
        return random.choice(untested)

    if any(v for v in tests.values()):
        if diagnose_actions:
            return random.choice(diagnose_actions)

    if ask_actions:
        return random.choice(ask_actions)

    if diagnose_actions:
        return random.choice(diagnose_actions)
    return random.choice(action_space)


# ---------------------------------------------------------------------------
# CORE EVALUATE FUNCTION
# ---------------------------------------------------------------------------

def evaluate(env, agent_fn=None, episodes=10, seed=None, verbose=False):
    """
    Run the environment for `episodes` episodes and compute metrics.

    Returns:
        metrics (dict): all computed metrics + composite score (0.0–1.0)
    """
    if agent_fn is None:
        agent_fn = heuristic_agent

    if seed is not None:
        random.seed(seed)

    correct_count  = 0
    costs          = []
    steps_list     = []
    trust_scores   = []
    rewards_list   = []
    timeouts       = 0

    for ep in range(episodes):
        state        = env.reset(seed=seed + ep if seed else None)
        done         = False
        total_reward = 0.0
        ep_steps     = 0
        timed_out    = False

        while not done:
            action                    = agent_fn(state, env.action_space)
            state, reward, done, info = env.step(action)
            total_reward             += reward
            ep_steps                 += 1

            if info.get("timeout"):
                timed_out = True

        correct   = info.get("correct", False)
        true_d    = info.get("true_disease",  env.get_true_disease())
        predicted = info.get("predicted",     "—")

        if correct:
            correct_count += 1
        if timed_out:
            timeouts += 1

        costs.append(state["total_cost"])
        steps_list.append(ep_steps)
        trust_scores.append(state["trust_score"])
        rewards_list.append(total_reward)

        if verbose:
            status = "✓ CORRECT" if correct else "✗ WRONG "
            tout   = " [TIMEOUT]" if timed_out else ""
            print(f"  Ep {ep+1:>3} | {status} | "
                  f"True: {true_d:<14} Pred: {predicted:<14} | "
                  f"Steps: {ep_steps:>2} | Cost: ₹{state['total_cost']:>5} | "
                  f"Trust: {state['trust_score']:>3} | "
                  f"Reward: {total_reward:>7.1f}{tout}")

    accuracy   = correct_count / episodes
    avg_cost   = statistics.mean(costs)
    avg_steps  = statistics.mean(steps_list)
    avg_trust  = statistics.mean(trust_scores)
    avg_reward = statistics.mean(rewards_list)
    timeout_rt = timeouts / episodes

    max_cost  = 5000
    max_steps = env.max_steps

    norm_cost   = min(1.0, avg_cost  / max_cost)
    norm_steps  = min(1.0, avg_steps / max_steps)
    norm_trust  = avg_trust / 100.0

    composite_100 = (
        0.40 * accuracy
      + 0.20 * (1 - norm_cost)
      + 0.20 * (1 - norm_steps)
      + 0.20 * norm_trust
    ) * 100

    # Normalise composite to 0.0–1.0 for OpenEnv grader compatibility
    composite_01 = round(composite_100 / 100.0, 4)

    metrics = {
        "episodes":        episodes,
        "accuracy":        round(accuracy,   4),
        "accuracy_pct":    round(accuracy * 100, 2),
        "avg_cost_inr":    round(avg_cost,   2),
        "avg_steps":       round(avg_steps,  2),
        "avg_trust":       round(avg_trust,  2),
        "avg_reward":      round(avg_reward, 3),
        "timeout_rate":    round(timeout_rt, 4),
        "composite_score": composite_01,        # 0.0–1.0 for OpenEnv
        "composite_score_100": round(composite_100, 2),  # human-readable
    }

    return metrics


# ---------------------------------------------------------------------------
# OPENENV-COMPATIBLE GRADER FUNCTIONS (0.0–1.0 output)
# ---------------------------------------------------------------------------

def grade_easy(episodes=10, seed=42):
    """Grade the easy task. Returns score 0.0–1.0."""
    from tasks import get_task
    env = get_task("easy")
    metrics = evaluate(env, agent_fn=heuristic_agent, episodes=episodes, seed=seed)
    return metrics["composite_score"]


def grade_medium(episodes=10, seed=42):
    """Grade the medium task. Returns score 0.0–1.0."""
    from tasks import get_task
    env = get_task("medium")
    metrics = evaluate(env, agent_fn=heuristic_agent, episodes=episodes, seed=seed)
    return metrics["composite_score"]


def grade_hard(episodes=10, seed=42):
    """Grade the hard task. Returns score 0.0–1.0."""
    from tasks import get_task
    env = get_task("hard")
    metrics = evaluate(env, agent_fn=heuristic_agent, episodes=episodes, seed=seed)
    return metrics["composite_score"]


# ---------------------------------------------------------------------------
# COMPARE TWO AGENTS
# ---------------------------------------------------------------------------

def compare_agents(env, agent_a, agent_b, episodes=20,
                   name_a="Agent A", name_b="Agent B"):
    """Run both agents on the same task and print a side-by-side comparison."""
    print(f"\n{'='*60}")
    print(f"  AGENT COMPARISON — {episodes} episodes each")
    print(f"  Task: {env.task_config.get('name', 'unknown')}")
    print(f"{'='*60}")

    m_a = evaluate(env, agent_a, episodes=episodes, seed=42)
    m_b = evaluate(env, agent_b, episodes=episodes, seed=42)

    row = "{:<22} {:>12} {:>12}"
    print(f"\n  {'':<22} {name_a:>12} {name_b:>12}")
    print("  " + "-" * 48)
    print(row.format("  Accuracy (%)",
                     f"{m_a['accuracy_pct']}%", f"{m_b['accuracy_pct']}%"))
    print(row.format("  Avg Cost (₹)",
                     f"₹{m_a['avg_cost_inr']}", f"₹{m_b['avg_cost_inr']}"))
    print(row.format("  Avg Steps",
                     m_a['avg_steps'], m_b['avg_steps']))
    print(row.format("  Avg Trust",
                     m_a['avg_trust'], m_b['avg_trust']))
    print(row.format("  Avg Reward",
                     m_a['avg_reward'], m_b['avg_reward']))
    print(row.format("  Timeout Rate",
                     f"{m_a['timeout_rate']*100:.1f}%",
                     f"{m_b['timeout_rate']*100:.1f}%"))
    print("  " + "=" * 48)
    print(row.format("  ★ Composite Score",
                     f"{m_a['composite_score_100']}/100",
                     f"{m_b['composite_score_100']}/100"))
    print()
    return m_a, m_b


# ---------------------------------------------------------------------------
# PRETTY PRINT METRICS
# ---------------------------------------------------------------------------

def print_metrics(metrics, title="Evaluation Results"):
    """Pretty-print a metrics dict returned by evaluate()."""
    print(f"\n{'='*45}")
    print(f"  {title}")
    print(f"{'='*45}")
    print(f"  Episodes evaluated  : {metrics['episodes']}")
    print(f"  Accuracy            : {metrics['accuracy_pct']}%")
    print(f"  Avg cost per episode: ₹{metrics['avg_cost_inr']}")
    print(f"  Avg steps per ep    : {metrics['avg_steps']}")
    print(f"  Avg trust retained  : {metrics['avg_trust']} / 100")
    print(f"  Avg total reward    : {metrics['avg_reward']}")
    print(f"  Timeout rate        : {metrics['timeout_rate']*100:.1f}%")
    print(f"  ─────────────────────────────────────────")
    print(f"  ★ Composite Score   : {metrics['composite_score_100']} / 100")
    print(f"  ★ OpenEnv Score     : {metrics['composite_score']} / 1.0")
    print(f"{'='*45}\n")
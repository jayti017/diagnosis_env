"""
test_manual.py — Interactive Manual Test Console for DiagnosisEnv
==================================================================
Play the environment yourself. You are the doctor.

Run:
    python test_manual.py

Controls:
    - Pick a task level (easy / medium / hard)
    - Each turn, type an action or pick from the numbered menu
    - See reward breakdown after every step
    - Reveal the true disease at any time with: reveal
    - Restart at any time with: restart
    - Quit with: quit / q / exit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from env import DiagnosisEnv, DISEASE_DB
from tasks import get_task, TASK_REGISTRY
from rewards import explain_reward, compute_reward

# ── ANSI colours ─────────────────────────────────────────────────────────────
R  = "\033[91m"   # red
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
B  = "\033[94m"   # blue
M  = "\033[95m"   # magenta
C  = "\033[96m"   # cyan
W  = "\033[97m"   # white
DIM= "\033[2m"
BLD= "\033[1m"
RST= "\033[0m"

def clr(text, code): return f"{code}{text}{RST}"
def bold(t): return clr(t, BLD)
def dim(t):  return clr(t, DIM)

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner():
    print(f"""
{clr('╔══════════════════════════════════════════════════╗', C)}
{clr('║   DiagnosisEnv — Interactive Manual Test         ║', C)}
{clr('║   You are the doctor. Make the right diagnosis.  ║', C)}
{clr('╚══════════════════════════════════════════════════╝', C)}
""")

def pick_task():
    print(bold("\nChoose task difficulty:"))
    levels = list(TASK_REGISTRY.keys())
    for i, lvl in enumerate(levels, 1):
        cfg = TASK_REGISTRY[lvl]()
        diseases = ", ".join(cfg["disease_pool"])
        print(f"  {clr(str(i), Y)}. {bold(lvl.upper())} — {cfg['max_steps']} steps | "
              f"noise {int(cfg['noise_level']*100)}% | diseases: {clr(diseases, DIM)}")
    while True:
        raw = input(f"\n{clr('Enter 1/2/3 or name', Y)}: ").strip().lower()
        if raw in levels:
            return raw
        if raw in {"1","2","3"}:
            return levels[int(raw)-1]
        print(f"  {clr('Invalid choice, try again.', R)}")

def print_state(state, env, step_num, total_reward):
    s = state
    pos = [k for k, v in s["known_symptoms"].items() if v]
    neg = [k for k, v in s["known_symptoms"].items() if not v]
    pos_tests = {k: v for k, v in s["test_results"].items() if v}
    neg_tests = {k: v for k, v in s["test_results"].items() if not v}

    trust_color = G if s["trust_score"] >= 60 else (Y if s["trust_score"] >= 30 else R)
    steps_color = G if s["steps_remaining"] >= 6 else (Y if s["steps_remaining"] >= 3 else R)

    print(f"\n{clr('─'*54, DIM)}")
    print(f"  {bold('Step')} {clr(str(step_num), W)}   "
          f"Trust: {clr(str(s['trust_score']), trust_color)}/100   "
          f"Steps left: {clr(str(s['steps_remaining']), steps_color)}   "
          f"Cost: {clr('₹'+str(s['total_cost']), Y)}   "
          f"Reward so far: {clr(f'{total_reward:+.1f}', G if total_reward >= 0 else R)}")
    print(f"{clr('─'*54, DIM)}")

    if pos:
        print(f"  {clr('✓ Positive symptoms :', G)} {', '.join(clr(s, G) for s in pos)}")
    if neg:
        print(f"  {clr('✗ Negative symptoms :', R)} {dim(', '.join(neg))}")
    if pos_tests:
        print(f"  {clr('+ Positive tests    :', G)} {', '.join(clr(t, G) for t in pos_tests)}")
    if neg_tests:
        print(f"  {clr('- Negative tests    :', R)} {dim(', '.join(neg_tests))}")
    if not pos and not neg and not pos_tests and not neg_tests:
        print(f"  {dim('No information gathered yet.')}")

def print_action_menu(env, state):
    action_space = env.action_space
    known   = state["known_symptoms"]
    tested  = state["test_results"]

    ask_actions  = [a for a in action_space if a.startswith("ask_")
                    and a[4:] not in known]
    test_actions = [a for a in action_space if a.startswith("test_")
                    and a[5:] not in tested]
    diag_actions = [a for a in action_space if a.startswith("diagnose_")]
    done_asks    = [a for a in action_space if a.startswith("ask_")
                    and a[4:] in known]
    done_tests   = [a for a in action_space if a.startswith("test_")
                    and a[5:] in tested]

    print(f"\n{bold('Available actions:')}")

    # ── Ask symptoms ────────────────────────────────────────────────
    if ask_actions:
        print(f"\n  {clr('[ ASK SYMPTOMS ]', B)}")
        for i, a in enumerate(ask_actions, 1):
            print(f"    {clr(str(i).rjust(3), Y)}.  {a}")

    # ── Order tests ─────────────────────────────────────────────────
    offset = len(ask_actions)
    if test_actions:
        print(f"\n  {clr('[ ORDER TESTS ]', M)}")
        for i, a in enumerate(test_actions, offset+1):
            test_name = a[5:]
            # Find cost
            cost = "?"
            for d in DISEASE_DB.values():
                if test_name in d["tests"]:
                    cost = f"₹{d['tests'][test_name][0]}"
                    break
            print(f"    {clr(str(i).rjust(3), Y)}.  {a}  {dim(cost)}")

    # ── Diagnose ────────────────────────────────────────────────────
    offset2 = offset + len(test_actions)
    print(f"\n  {clr('[ DIAGNOSE ]', R)}")
    for i, a in enumerate(diag_actions, offset2+1):
        print(f"    {clr(str(i).rjust(3), Y)}.  {a}")

    # ── Already done (dim) ──────────────────────────────────────────
    if done_asks or done_tests:
        already = done_asks + done_tests
        print(f"\n  {dim('[ ALREADY DONE ]  ' + ', '.join(a for a in already))}")

    print(f"\n  {dim('Commands: reveal | restart | quit')}")
    return ask_actions, test_actions, diag_actions

def _info_to_result(info, action):
    if action.startswith("ask_"):
        return {"type": "symptom", "value": info.get("answer"),
                "repeated": info.get("repeated", False)}
    elif action.startswith("test_"):
        return {"type": "test", "value": info.get("result"),
                "cost": info.get("cost", 0),
                "repeated": info.get("repeated", False)}
    elif action.startswith("diagnose_"):
        return {"type": "diagnosis", "correct": info.get("correct")}
    elif info.get("timeout"):
        return {"type": "timeout", "correct": False}
    return {}

def print_step_result(action, reward, info, done, state):
    print()
    if action.startswith("ask_"):
        ans = info.get("answer", False)
        sym = info.get("symptom", action[4:])
        rep = info.get("repeated", False)
        icon = clr("YES ✓", G) if ans else clr("NO  ✗", R)
        rep_tag = clr(" [REPEATED − penalty]", Y) if rep else ""
        print(f"  {clr('●', B)} Asked about {bold(sym)} → Patient says: {icon}{rep_tag}")

    elif action.startswith("test_"):
        res  = info.get("result", False)
        test = info.get("test", action[5:])
        cost = info.get("cost", 0)
        rep  = info.get("repeated", False)
        icon = clr("POSITIVE ✓", G) if res else clr("NEGATIVE ✗", R)
        rep_tag = clr(" [REPEATED − no charge]", Y) if rep else ""
        print(f"  {clr('●', M)} Ran {bold(test)} → {icon}  {clr('₹'+str(cost), Y)}{rep_tag}")

    elif action.startswith("diagnose_"):
        correct   = info.get("correct", False)
        predicted = info.get("predicted", action[9:])
        true_d    = info.get("true_disease", "?")
        if correct:
            print(f"  {clr('✅ CORRECT DIAGNOSIS!', G)}  You said: {bold(predicted)}")
        else:
            print(f"  {clr('❌ WRONG DIAGNOSIS', R)}   You said: {bold(predicted)}   "
                  f"True disease: {clr(true_d, G)}")

    # Reward breakdown
    result = _info_to_result(info, action)
    breakdown = explain_reward(state, action, result, done)
    total = breakdown.pop("TOTAL", reward)

    parts = []
    for k, v in breakdown.items():
        col = G if v > 0 else (R if v < 0 else DIM)
        parts.append(f"{clr(k, DIM)}={clr(f'{v:+.1f}', col)}")

    print(f"  {bold('Reward')}: {clr(f'{reward:+.2f}', G if reward >= 0 else R)}   "
          f"{dim('('+ '  '.join(parts) +')')}")

def episode_summary(state, step_num, total_reward, info):
    correct = info.get("correct", False)
    true_d  = info.get("true_disease", "?")
    timeout = info.get("timeout", False)

    print(f"\n{clr('═'*54, C)}")
    print(f"  {bold('EPISODE OVER')}")
    print(f"{clr('═'*54, C)}")

    if timeout:
        print(f"  {clr('⏰ TIMEOUT — ran out of steps!', Y)}")
        print(f"  True disease was: {clr(true_d, G)}")
    elif correct:
        print(f"  {clr('🏆 CORRECT DIAGNOSIS!', G)}")
        print(f"  Disease: {clr(true_d, G)}")
    else:
        print(f"  {clr('💀 WRONG DIAGNOSIS', R)}")
        print(f"  True disease was: {clr(true_d, G)}")

    print(f"\n  Steps taken   : {step_num}")
    print(f"  Total cost    : {clr('₹'+str(state['total_cost']), Y)}")
    print(f"  Trust score   : {state['trust_score']} / 100")
    print(f"  Total reward  : {clr(f'{total_reward:+.2f}', G if total_reward >= 0 else R)}")
    print(f"{clr('═'*54, C)}\n")

# ── Main game loop ─────────────────────────────────────────────────────────────

def play(task_level=None, seed=None):
    if task_level is None:
        task_level = pick_task()

    env = get_task(task_level)
    state = env.reset(seed=seed)

    step_num     = 0
    total_reward = 0.0
    last_info    = {}

    print(f"\n{clr('━'*54, C)}")
    print(f"  {bold('Task')}: {clr(task_level.upper(), Y)}  |  "
          f"{bold('Max steps')}: {clr(str(env.max_steps), W)}  |  "
          f"{bold('Diseases')}: {clr(str(len(env.disease_pool)), W)}")
    print(f"  {dim('True disease is hidden — good luck!')}")
    print(f"{clr('━'*54, C)}")

    # Show volunteered symptoms
    vol = [(s, v) for s, v in state["known_symptoms"].items()]
    if vol:
        print(f"\n  Patient volunteers on arrival:")
        for sym, val in vol:
            icon = clr("✓ PRESENT", G) if val else clr("✗ ABSENT", R)
            print(f"    [{icon}]  {sym}")

    # Build numbered action lookup for the whole session
    while not env.done:
        print_state(state, env, step_num, total_reward)
        ask_actions, test_actions, diag_actions = print_action_menu(env, state)
        all_numbered = ask_actions + test_actions + diag_actions

        # ── Get input ───────────────────────────────────────────────
        while True:
            raw = input(f"\n{clr('Your action', Y)} > ").strip()

            if raw.lower() in ("quit", "q", "exit"):
                print(f"\n{clr('Bye! True disease was:', C)} "
                      f"{clr(env.get_true_disease() or '?', G)}\n")
                return "quit"

            if raw.lower() == "restart":
                return "restart"

            if raw.lower() == "reveal":
                print(f"  {clr('🔍 True disease:', C)} {clr(env.get_true_disease() or '?', G)}")
                continue

            # Numbered shortcut
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(all_numbered):
                    action = all_numbered[idx]
                    print(f"  {dim('→')} {action}")
                    break
                else:
                    print(f"  {clr('Number out of range. Try again.', R)}")
                    continue

            # Direct action string
            if raw in env.action_space:
                action = raw
                break

            # Fuzzy match help
            close = [a for a in env.action_space if raw.lower() in a.lower()]
            if close:
                print(f"  {clr('Not found. Did you mean:', Y)}")
                for c in close[:5]:
                    print(f"    {c}")
            else:
                print(f"  {clr('Invalid action. Type a number, action string, reveal, restart, or quit.', R)}")

        # ── Execute action ───────────────────────────────────────────
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step_num     += 1
        last_info     = info

        print_step_result(action, reward, info, done, next_state)
        state = next_state

    episode_summary(state, step_num, total_reward, last_info)
    return "done"


def main():
    banner()
    print(f"  {dim('Tip: type a number to pick an action, or type the full action string.')}")
    print(f"  {dim('Commands available at any time: reveal | restart | quit')}")

    while True:
        result = play()
        if result == "quit":
            break

        again = input(f"\n{clr('Play again? (y/n)', Y)}: ").strip().lower()
        if again not in ("y", "yes"):
            print(f"\n{clr('Thanks for playing!', C)}\n")
            break
        print()


if __name__ == "__main__":
    main()
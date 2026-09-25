import json
import httpx

# ── SERVERS ───────────────────────────────────────────────────────────────
# Requires: .venv/bin/python load_laya.py   (laya on :8000)
#           .venv/bin/python load_lfm.py    (LFM2.5 on :8081)
LAYA = "http://127.0.0.1:8000"
LFM = "http://127.0.0.1:8081"
LAYA_MODEL = "english"
LFM_MODEL = "dspark"
THRESHOLD = 0.5
MAX_ROUNDS = 5

# ── TOOLS ─────────────────────────────────────────────────────────────────
# No real functions - laya picks WHICH, lfm invents the ARGUMENTS.
TOOLS = {
    "get_weather": {
        "desc": "current weather or forecast for a location",
        "params": {"location": "string - city or place", "date": "string - optional, e.g. 'tomorrow'"},
    },
    "search_web": {
        "desc": "look up facts, news, or information on the internet",
        "params": {"query": "string - search query"},
    },
    "send_email": {
        "desc": "compose and send an email",
        "params": {"to": "string - recipient", "subject": "string", "body": "string"},
    },
}
ORDER = ["get_weather", "search_web", "send_email"]  # gather -> send

# ── STATE: one user message ───────────────────────────────────────────────
STATE = "Find today's top news and check the weather in Tokyo, then email a summary to boss@acme.com."

# ── LAYA: choice + noul union (winning algorithm) ─────────────────────────
PHRASINGS = {
    "help":     "Should the assistant call the {t} tool ({d}) to help answer this request?",
    "part":     "Is there any part of this request that needs the {t} tool ({d})?",
    "required": "Must the assistant call the {t} tool ({d}) to fully satisfy this request?",
    "useful":   "Would the {t} tool ({d}) be useful for any part of this request?",
}

def plan_tools(state, remaining):
    """One /evaluate -> ordered list of still-needed tools (empty = done)."""
    q = {"which": {"type": "choice",
                   "instructions": "Which tool should be called to answer this request?",
                   "criteria": {"none": "answer directly, no tool needed",
                                **{t: remaining[t]["desc"] for t in remaining}}}}
    for ph, ins in PHRASINGS.items():
        for t, d in remaining.items():
            q[f"{ph}_{t}"] = {"type": "noul", "instructions": ins.format(t=t, d=d["desc"])}

    r = httpx.post(f"{LAYA}/evaluate", json={"state": state, "questions": q, "model": LAYA_MODEL},
                   timeout=120)
    result = r.json()
    if r.status_code != 200:
        raise SystemExit(f"laya returned {r.status_code}: {result.get('error')}")
    a = result["answers"]

    plan = set()
    if a["which"]["choice"] in remaining:
        plan.add(a["which"]["choice"])
    for ph in PHRASINGS:
        for t in remaining:
            if a[f"{ph}_{t}"]["noul"] > THRESHOLD:
                plan.add(t)
    return [t for t in ORDER if t in plan], a

# ── LFM: fill in the arguments AND simulate the results ──────────────────
def chat(prompt):
    r = httpx.post(f"{LFM}/chat", json={
        "model": LFM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }, timeout=300)
    result = r.json()
    if r.status_code != 200:
        raise SystemExit(f"lfm returned {r.status_code}: {result.get('error')}")
    return result["choices"][0]["message"]["content"]

def make_tool_calls(state, tools):
    """LFM emits each call with arguments AND a simulated result (no real tools)."""
    schemas = "\n".join(
        f"- {t}({', '.join(f'{k}: {v}' for k, v in TOOLS[t]['params'].items())})"
        for t in tools)
    prompt = (
        f"User request and tool results so far:\n{state}\n\n"
        f"A router has decided these tools must be called, in this order: {', '.join(tools)}.\n"
        f"Tool signatures:\n{schemas}\n\n"
        "Output ONLY a JSON array, no other text. Each element:\n"
        "{\"name\": <tool>, \"arguments\": {<param>: <value>, ...}, "
        "\"result\": <short plausible result this tool would return>}\n"
        "Fill every argument from the request and any tool results above. If an "
        "argument depends on another tool's result, write \"<result of toolname>\".")
    return chat(prompt)

def final_answer(state):
    return chat(
        f"User request and tool results:\n{state}\n\n"
        "All needed tools have run. Write the final reply to the user, "
        "summarizing what was done.")

# ── RUN: plan -> lfm fills args -> mock result -> re-ask, until done ──────
if __name__ == "__main__":
    print(f"state: {STATE}\n{'=' * 60}")

    state = STATE
    done = set()
    for rnd in range(MAX_ROUNDS):
        remaining = {t: d for t, d in TOOLS.items() if t not in done}
        if not remaining:
            print("\nall tools used - stop")
            break

        plan, answers = plan_tools(state, remaining)
        if not plan:
            print(f"\nround {rnd}: laya says no more tools needed - done")
            break

        print(f"\nround {rnd}: laya choice={answers['which']['choice']}"
              f"  plan={plan}")
        for t in remaining:
            probs = " ".join(f"{ph}={answers[f'{ph}_{t}']['noul']:.2f}" for ph in PHRASINGS)
            print(f"    {t:<12} {probs}{'  <-' if t in plan else ''}")

        raw = make_tool_calls(state, plan)
        try:
            calls = json.loads(raw)
            print(json.dumps(calls, indent=2))
        except json.JSONDecodeError:
            print(raw)
            calls = [{"name": t, "result": "ok"} for t in plan]

        for c in calls:
            t = c.get("name")
            if t in remaining:  # ignore lfm hallucinating tools laya didn't plan
                state += f"\n[{t} -> {c.get('result', 'ok')}]"
                done.add(t)

    print(f"\n{'=' * 60}\nlfm final answer:\n{final_answer(state)}")

import json
import httpx

# ── SERVER ────────────────────────────────────────────────────────────────
# Requires load_laya.py running:  .venv/bin/python load_laya.py
SERVER = "http://127.0.0.1:8000"
MODEL = "english"
THRESHOLD = 0.5

# ── TOOLS the llm could call ───────────────────────────────────────────────
TOOLS = {
    "get_weather": "current weather or forecast for a location",
    "web_search": "look up facts, news, or information on the internet",
    "calculator": "evaluate a math expression",
    "send_email": "compose and send an email",
}

# fixed dependency order for executing the planned set (gather -> compute -> send)
ORDER = ["get_weather", "web_search", "calculator", "send_email"]

# ── QUESTIONS ─────────────────────────────────────────────────────────────
# winning config from the sweep: one choice over tools+none UNIONED with
# every tool that any noul phrasing puts over threshold — all in ONE call.
PHRASINGS = {
    "help":     "Should the assistant call the {t} tool ({d}) to help answer this request?",
    "part":     "Is there any part of this request that needs the {t} tool ({d})?",
    "required": "Must the assistant call the {t} tool ({d}) to fully satisfy this request?",
    "useful":   "Would the {t} tool ({d}) be useful for any part of this request?",
}

def build_questions():
    q = {"which": {"type": "choice",
                   "instructions": "Which tool should be called to answer this request?",
                   "criteria": {"none": "answer directly, no tool needed", **TOOLS}}}
    for ph, ins in PHRASINGS.items():
        for t, d in TOOLS.items():
            q[f"{ph}_{t}"] = {"type": "noul", "instructions": ins.format(t=t, d=d)}
    return q

QUESTIONS = build_questions()

# ── DECISION: the algorithm ───────────────────────────────────────────────
def plan_tools(state):
    """One /evaluate -> ordered list of tools the llm should call."""
    r = httpx.post(f"{SERVER}/evaluate",
                   json={"state": state, "questions": QUESTIONS, "model": MODEL},
                   timeout=120)
    result = r.json()
    if r.status_code != 200:
        raise SystemExit(f"{SERVER} returned {r.status_code}: {result.get('error')}")
    a = result["answers"]

    plan = set()
    pick = a["which"]["choice"]
    if pick != "none":
        plan.add(pick)
    for ph in PHRASINGS:
        for t in TOOLS:
            if a[f"{ph}_{t}"]["noul"] > THRESHOLD:
                plan.add(t)

    ordered = [t for t in ORDER if t in plan]
    return ordered, a, result

# ── CASES ─────────────────────────────────────────────────────────────────
CASES = [
    {"state": "Hey, what's the weather like in Tokyo right now?", "expect": {"get_weather"}},
    {"state": "What are today's top headlines?", "expect": {"web_search"}},
    {"state": "What is 128 * 46 + 17?", "expect": {"calculator"}},
    {"state": "Send an email to john@acme.com saying the meeting moved to 3pm.", "expect": {"send_email"}},
    {"state": "Check the weather in Tokyo and email it to john@acme.com.", "expect": {"get_weather", "send_email"}},
    {"state": "Search for the cheapest flights to Lisbon and email the result to ana@acme.com.",
     "expect": {"web_search", "send_email"}},
    {"state": "What's a 15% tip on $84.50, and will it rain in London tomorrow?",
     "expect": {"calculator", "get_weather"}},
    {"state": "Find the latest Fed rate news, check if it's sunny in Miami, and email both to me.",
     "expect": {"web_search", "get_weather", "send_email"}},
    {"state": "What is the capital of France?", "expect": set()},
    {"state": "Explain how photosynthesis works.", "expect": set()},
    {"state": "Write a haiku about autumn.", "expect": set()},
    {"state": "Hi! How are you today?", "expect": set()},
]

# ── RUN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tp = {t: 0 for t in TOOLS}
    fp = {t: 0 for t in TOOLS}
    fn = {t: 0 for t in TOOLS}
    exact, tokens, ms = 0, 0, 0

    for i, case in enumerate(CASES):
        ordered, a, result = plan_tools(case["state"])
        pred, expect = set(ordered), case["expect"]
        tokens += result["usage"]["input_tokens"]
        ms += result["elapsed_ms"]

        for t in TOOLS:
            if t in expect and t in pred:
                tp[t] += 1
            elif t in pred:
                fp[t] += 1
            elif t in expect:
                fn[t] += 1
        ok = pred == expect
        exact += ok

        pick = a["which"]["choice"]
        fired = sorted({t for ph in PHRASINGS for t in TOOLS
                        if a[f"{ph}_{t}"]["noul"] > THRESHOLD})
        print(f"\n[{i}] {case['state'][:64]}")
        print(f"    choice={pick}  nouls>{THRESHOLD}={fired or '[]'}")
        print(f"    plan={ordered or '[]'}  expect={sorted(expect) or '[]'}  {'ok' if ok else 'MISS'}")

    n = len(CASES)
    print(f"\n{'=' * 60}\nexact-set match: {exact}/{n}  {exact / n:.0%}")
    print(f"\nper-tool:")
    print(f"  {'tool':<13} {'tp':>3} {'fp':>3} {'fn':>3}  {'prec':>5} {'rec':>5}")
    for t in TOOLS:
        p = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0.0
        r = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 0.0
        print(f"  {t:<13} {tp[t]:>3} {fp[t]:>3} {fn[t]:>3}  {p:5.0%} {r:5.0%}")
    print(f"\n[{tokens} input tokens total, {ms / n:.0f} ms avg, 1 call/msg]")

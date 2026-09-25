import json
import httpx

# ── SERVER ────────────────────────────────────────────────────────────────
# Requires load_laya.py running:  .venv/bin/python load_laya.py
SERVER = "http://127.0.0.1:8000"
MODEL = "english"
THRESHOLD = 0.5
MAX_CALLS = 5

# ── TOOLS the llm could call ───────────────────────────────────────────────
TOOLS = {
    "get_weather": "current weather or forecast for a location",
    "web_search": "look up facts, news, or information on the internet",
    "calculator": "evaluate a math expression",
    "send_email": "compose and send an email",
}

# ── CASES ─────────────────────────────────────────────────────────────────
# state: the user message. expect: the SET of tools needed (empty = none).
CASES = [
    {"state": "Hey, what's the weather like in Tokyo right now?", "expect": {"get_weather"}},
    {"state": "What is 128 * 46 + 17?", "expect": {"calculator"}},
    {"state": "Check the weather in Tokyo and email it to john@acme.com.",
     "expect": {"get_weather", "send_email"}},
    {"state": "Search for the cheapest flights to Lisbon and email the result to ana@acme.com.",
     "expect": {"web_search", "send_email"}},
    {"state": "Find the latest Fed rate news, check if it's sunny in Miami, and email both to me.",
     "expect": {"web_search", "get_weather", "send_email"}},
    {"state": "What is the capital of France?", "expect": set()},
    {"state": "Hi! How are you today?", "expect": set()},
]

# ── MOCK TOOL EXECUTION (no lfm yet) ──────────────────────────────────────
def run_tool(name, state):
    fake = {
        "get_weather": "22C, clear skies",
        "web_search": "flights from $430, top headline: Fed holds rates",
        "calculator": "5905",
        "send_email": "email sent",
    }
    return fake[name]


# ── THE LOOP: noul gate -> choice -> execute -> repeat ────────────────────
def decide(user_msg):
    state = user_msg  # plain string; tool results appended as "[tool -> result]" lines
    calls = []
    while len(calls) < MAX_CALLS:
        remaining = {t: d for t, d in TOOLS.items() if t not in calls}
        if not remaining:
            break
        gate = ("Should the assistant call a tool (weather, search, calculator, email) "
                "to answer this request?" if not calls else
                "Is there any remaining part of this request that still needs a tool call?")
        questions = {
            "more": {"type": "noul", "instructions": gate},
            "which": {"type": "choice",
                      "instructions": "Which tool should be called to answer this request?",
                      "criteria": remaining},
        }
        r = httpx.post(f"{SERVER}/evaluate",
                       json={"state": state, "questions": questions, "model": MODEL},
                       timeout=120).json()
        answers = r["answers"]
        p_more = answers["more"]["noul"]
        if p_more <= THRESHOLD:
            print(f"    gate: more={p_more:.2f} -> stop")
            break
        tool = answers["which"]["choice"]
        p_tool = answers["which"]["probabilities"][tool]
        print(f"    gate: more={p_more:.2f} -> call {tool} ({p_tool:.2f})")
        calls.append(tool)
        state += f"\n[{tool} -> {run_tool(tool, state)}]"
    return calls


# ── RUN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hits = totals = 0
    for i, case in enumerate(CASES):
        print(f"\n{'=' * 70}\n[{i}] {case['state'][:66]}\n  expect: {sorted(case['expect']) or '[]'}")
        calls = decide(case["state"])
        ok = set(calls) == case["expect"]
        hits += ok
        totals += 1
        print(f"  plan:   {calls or '[]'}  {'ok' if ok else 'MISS'}")

    print(f"\n{'=' * 70}\nexact-set match: {hits}/{totals}  {hits / totals:.0%}")

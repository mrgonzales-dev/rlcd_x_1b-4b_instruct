import json
import httpx

from people_data import DATA

# ── SERVER ────────────────────────────────────────────────────────────────
# Requires load_laya.py running:  .venv/bin/python load_laya.py
SERVER = "http://127.0.0.1:8000"
MODEL = "english"
THRESHOLD = 0.5

# ── TOOLS: real (toy) search over people_data ─────────────────────────────
def find_person(name):
    """Look up one person's bio by name (fuzzy)."""
    q = name.lower().replace(" ", "_")
    for key, person in DATA.items():
        if q in key or name.lower() in person["full_name"].lower():
            return person
    return None

def search_people(query):
    """Substring search across every field; returns matching full names."""
    return [p["full_name"] for p in DATA.values()
            if all(w in json.dumps(p).lower() for w in query.lower().split())]

TOOLS = {
    "find_person": {"desc": "look up one specific person's bio by name", "fn": find_person},
    "search_people": {"desc": "find people matching an attribute like occupation or birthplace",
                      "fn": search_people},
}

# ── LAYA QUESTIONS ────────────────────────────────────────────────────────
# tier 1: casual vs needs-data gate (the boundary laya CAN draw reliably)
# tier 2: which search tool, asked only when the gate says "data"
QUESTIONS = {
    "casual": {"type": "noul",
               "instructions": "Is this message casual conversation or small talk rather than a request for information?"},
    "which": {"type": "choice",
              "instructions": "Which tool should be called to answer this request?",
              "criteria": {"none": "casual chat or general knowledge, no lookup needed",
                           **{t: d["desc"] for t, d in TOOLS.items()}}},
}

# ── CASES ─────────────────────────────────────────────────────────────────
CASES = [
    # data requests - expect needs_data=True; tool = ideal sub-route
    {"state": "When was Manny Pacquiao born?", "needs_data": True, "tool": "find_person"},
    {"state": "Who is Lea Salonga's spouse?", "needs_data": True, "tool": "find_person"},
    {"state": "Where was Coco Martin born?", "needs_data": True, "tool": "find_person"},
    {"state": "Which celebrities in the dataset are YouTubers?", "needs_data": True, "tool": "search_people"},
    {"state": "List the actors who were born in Quezon City.", "needs_data": True, "tool": "search_people"},
    {"state": "Which singers are in the dataset, and when was Sarah Geronimo born?",
     "needs_data": True, "tool": "search_people+find_person"},
    {"state": "Look up Vice Ganda's bio, and which celebrities are comedians.",
     "needs_data": True, "tool": "search_people+find_person"},
    # casual chat - no tools
    {"state": "Hi! How are you today?", "needs_data": False, "tool": None},
    {"state": "Tell me a joke.", "needs_data": False, "tool": None},
    {"state": "What's your favorite movie?", "needs_data": False, "tool": None},
    {"state": "How do I cook rice?", "needs_data": False, "tool": None},
    {"state": "Thanks, that really helped!", "needs_data": False, "tool": None},
]

# ── RUN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gate_ok = gate_n = 0
    route_ok = route_n = 0
    tokens, ms = 0, 0

    for i, case in enumerate(CASES):
        r = httpx.post(f"{SERVER}/evaluate",
                       json={"state": case["state"], "questions": QUESTIONS, "model": MODEL},
                       timeout=120)
        result = r.json()
        if r.status_code != 200:
            raise SystemExit(f"{SERVER} returned {r.status_code}: {result.get('error')}")
        a = result["answers"]
        tokens += result["usage"]["input_tokens"]
        ms += result["elapsed_ms"]

        p_casual = a["casual"]["noul"]
        needs_data = p_casual <= THRESHOLD
        gok = needs_data == case["needs_data"]
        gate_ok += gok
        gate_n += 1

        line = (f"[{i}] {case['state'][:58]:<60} casual={p_casual:.2f} "
                f"-> {'DATA' if needs_data else 'chat':<5} {'ok' if gok else 'MISS'}")
        if needs_data:
            pick = a["which"]["choice"]
            rok = pick in (case["tool"] or "").split("+") if case["tool"] else False
            route_ok += rok
            route_n += 1
            line += f"  | route={pick} want~{case['tool']} {'ok' if rok else 'MISS'}"
        print(line)

    n = len(CASES)
    print(f"\n{'=' * 60}")
    print(f"tier 1 - casual-vs-data gate: {gate_ok}/{gate_n}  {gate_ok / gate_n:.0%}")
    print(f"tier 2 - tool sub-routing (data msgs only): {route_ok}/{route_n}  {route_ok / route_n:.0%}")
    print(f"\n[{tokens} input tokens total, {ms / n:.0f} ms avg, 1 call/msg]")

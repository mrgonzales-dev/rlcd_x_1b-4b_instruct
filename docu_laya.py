import json
import re
from html.parser import HTMLParser
from pathlib import Path

import httpx

# ── SERVERS ───────────────────────────────────────────────────────────────
# laya is PREPARED below but not used yet - this file only loads/chunks docs.
LAYA = "http://127.0.0.1:8000"
LAYA_MODEL = "english"  # 512-token ctx; each chunk must stay small

DOCS_DIR = Path(__file__).parent / "docs"
HEADING_TAGS = {"h1", "h2", "h3", "h4"}
SKIP_TAGS = {"script", "style", "nav", "header", "footer"}

# ── HTML -> TEXT ──────────────────────────────────────────────────────────
class TextExtractor(HTMLParser):
    """Strips tags, keeps heading boundaries as \n@@ markers for chunking.
    Boilerplate regions are regex-removed before parsing (tags can be unbalanced)."""

    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in HEADING_TAGS:
            self.parts.append("\n@@")  # heading boundary marker
        elif tag in ("p", "li", "pre", "tr", "br"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in HEADING_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)

    def text(self):
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", "".join(self.parts))).strip()


def load_doc(path):
    """.html -> stripped text; .md -> headings rewritten as @@ markers; .txt as-is."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    if str(path).endswith((".html", ".htm")):
        for tag in SKIP_TAGS:
            raw = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", raw, flags=re.S | re.I)
            raw = re.sub(rf"<{tag}[^>]*/>", " ", raw, flags=re.I)
        p = TextExtractor()
        p.feed(raw)
        return p.text()
    if str(path).endswith(".md"):
        return re.sub(r"(?m)^(#{1,4})\s+", lambda m: f"\n@@{len(m.group(1))}", raw)
    return raw

# ── CHUNKING ──────────────────────────────────────────────────────────────
# laya's head fits only ~48 tokens per option label and ~300 tokens of state,
# so chunks are kept small; a short title is what laya would scan later.
MAX_CHUNK_CHARS = 1200  # ~300 tokens at ~4 chars/token

def chunk_doc(text, source):
    """Split on @@N heading markers -> [{id, title, text, source}].
    title is the heading path ('Types > Boolean types'); empty parent
    headings contribute to the path but emit no chunk of their own."""
    chunks, n, stack = [], 0, {}
    for section in text.split("\n@@"):
        section = section.strip()
        if not section:
            continue
        m = re.match(r"(\d)", section)
        level = int(m.group(1)) if m else 2
        title, _, body = section.lstrip("0123456789").partition("\n")
        title = title.strip()
        stack[level] = title
        for lv in [lv for lv in stack if lv > level]:
            del stack[lv]
        path = " > ".join(stack[k] for k in sorted(stack))[:90]
        body = body.strip()
        if not body:
            continue
        while len(body) > MAX_CHUNK_CHARS:  # split on paragraph boundaries
            cut = body[:MAX_CHUNK_CHARS]
            para = cut.rfind("\n\n")
            if para > MAX_CHUNK_CHARS // 3:
                cut = body[:para + 2]
            chunks.append({"id": f"{source}#{n}", "title": path,
                           "text": cut.strip(), "source": source})
            n += 1
            body = body[len(cut):].lstrip()
        if body:
            chunks.append({"id": f"{source}#{n}", "title": path,
                           "text": body, "source": source})
            n += 1
    return chunks

def load_all(docs_dir=DOCS_DIR):
    chunks = []
    for f in sorted(Path(docs_dir).iterdir()):
        chunks += chunk_doc(load_doc(f), f.stem)
    return chunks

# ══ SCRAPER CONFIG — play with these ══════════════════════════════════════
QTYPE = "noul"            # "noul" (P(yes)) or "score" (expected relevance level)
INSTRUCTION = "Is this section relevant to the question?"
SCORE_CRITERIA = ["irrelevant", "somewhat related", "directly answers the question"]
STATE_FMT = "Question: {q}\nSection: {t}\n{x}"   # {q}=query {t}=title {x}=text
DECISION = "topn"         # "thr" (over threshold) | "topn" | "gap" (biggest drop)
THRESHOLD = 0.5           # noul p for "thr"
TOP_N = 8                 # chunks kept by "topn"
BM25_TOP = 30             # recall shortlist size fed to laya (phase 1 -> 2)
# sweep findings (goroutine query, 156 chunks, 12 needles):
#   title in state is critical: avg needle rank 13 w/ title vs 36 without
#   topn=8 is the robust rule (7/8 needles under every titled phrasing)
#   "answer"+gap = surgical 5/5 needles; score head can't discriminate

# ── LAYA SCRAPER ──────────────────────────────────────────────────────────
# Controlled-state scan: laya reads each chunk, one question per chunk.
# Plain-string state (dicts serialize to JSON and degrade answers).
_client = httpx.Client(timeout=120)

def probe(query, chunk, qtype=QTYPE, instruction=INSTRUCTION, state_fmt=STATE_FMT):
    """Score one chunk. Returns noul P(yes) or score level (0..k-1)."""
    if qtype == "score":
        q = {"hit": {"type": "score", "instructions": instruction, "criteria": SCORE_CRITERIA}}
    else:
        q = {"hit": {"type": "noul", "instructions": instruction}}
    r = _client.post(f"{LAYA}/evaluate", json={
        "state": state_fmt.format(q=query, t=chunk["title"], x=chunk["text"]),
        "questions": q, "model": LAYA_MODEL})
    result = r.json()
    if r.status_code != 200:
        raise SystemExit(f"laya returned {r.status_code}: {result.get('error')}")
    a = result["answers"]["hit"]
    return a["noul"] if qtype == "noul" else a["score"]

def scan(query, chunks, qtype=QTYPE, instruction=INSTRUCTION, state_fmt=STATE_FMT,
         progress=25):
    """Score every chunk -> [(val, chunk)] sorted best first."""
    import time
    scored, t0 = [], time.time()
    for i, c in enumerate(chunks):
        if progress and i % progress == 0:
            print(f"\r  scanning {i}/{len(chunks)} ({time.time() - t0:.0f}s)", end="", flush=True)
        scored.append((probe(query, c, qtype, instruction, state_fmt), c))
    print(f"\r  scanned {len(chunks)} chunks in {time.time() - t0:.0f}s" + " " * 30)
    return sorted(scored, key=lambda x: -x[0])

def decide(scored, rule=DECISION, threshold=THRESHOLD, top_n=TOP_N):
    """Apply a decision rule to ranked (val, chunk) -> the hit set."""
    if rule == "topn":
        return scored[:top_n]
    if rule == "gap":
        vals = [v for v, _ in scored]
        if len(vals) < 2:
            return scored
        i = max(range(len(vals) - 1), key=lambda j: vals[j] - vals[j + 1])
        return scored[:i + 1]
    return [s for s in scored if s[0] > threshold]  # "thr"

# ── BM25 RECALL STAGE ─────────────────────────────────────────────────────
# Phase 1 of the funnel: pure-math keyword ranking, ~10ms for all chunks.
# score = Σ query_terms: idf(term) * saturated_tf(term, chunk)
#   idf    = ln((N - df + .5) / (df + .5) + 1)   -- rare terms count more
#   sat_tf = tf*(k1+1) / (tf + k1*(1-b+b*len/avglen))  -- diminishing returns
STOPWORDS = set("a an and are as at be by do does for from how i in is it of "
                "on or the to what when where which who why with".split())
K1, B = 1.5, 0.75

def _toks(s):
    return [t.rstrip("s") for t in re.findall(r"[a-z0-9]+", s.lower())
            if t not in STOPWORDS and len(t) > 1]

def bm25_rank(query, chunks):
    """Instant keyword ranking -> [(score, chunk)] best first."""
    import math
    docs = [_toks(c["title"] + " " + c["title"] + " " + c["text"])  # title x2 boost
            for c in chunks]
    q = _toks(query)
    N = len(docs)
    df = {t: sum(1 for d in docs if t in d) for t in q}
    avg = sum(map(len, docs)) / N
    out = []
    for d, c in zip(docs, chunks):
        s = 0.0
        for t in q:
            if t not in d:
                continue
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
            tf = d.count(t)
            s += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len(d) / avg))
        out.append((s, c))
    return sorted(out, key=lambda x: -x[0])

# ── THE PIPELINE: bm25 recall -> laya verify -> hits ──────────────────────
def scrape(query, chunks, top=BM25_TOP, progress=0):
    """BM25 shortlists candidates; laya noul-verifies each; decide() cuts."""
    cand = [c for s, c in bm25_rank(query, chunks)[:top] if s > 0]
    print(f"bm25: {len(cand)} candidates (top {top}, zero-score dropped)")
    if not cand:
        return []
    scored = scan(query, cand, progress=progress)
    return decide(scored)

# ── RUN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    chunks = load_all()
    print(f"loaded {len(chunks)} chunks from {len(list(DOCS_DIR.iterdir()))} docs")

    if len(sys.argv) > 1:  # docu_laya.py "your question" -> bm25+laya funnel
        query = " ".join(sys.argv[1:])
        print(f"\nscraping for: {query}")
        hits = scrape(query, chunks)
        for p, c in hits:
            print(f"\n  {p:.2f}  {c['id']}  {c['title']}")
            print(f"       {c['text'][:160]}")
        if not hits:
            print("  no hits")
    else:  # stats only
        sizes = [len(c["text"]) for c in chunks]
        print(f"chunk chars: min={min(sizes)} max={max(sizes)} "
              f"avg={sum(sizes) // len(sizes)} total={sum(sizes) // 1024}KB")
        for c in chunks[:20]:
            print(f"  {c['id']:<28} {len(c['text']):>5} ch  {c['title'][:60]}")
        print(f"  ... +{len(chunks) - 20} more")

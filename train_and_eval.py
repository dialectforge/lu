"""
Lu 4D training + evaluation harness.

Downloads HuggingFace datasets, extracts S-R-O triples, trains the Lu 4D brain
via its MCP server (JSON-RPC over stdio), runs a 40-question eval, and
demonstrates temporal session features.

No npm / axios. Pure stdlib + requests + datasets.
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LU4D_BIN = os.path.join(os.path.dirname(__file__), "target", "release", "lu4d.exe")
BRAIN_FILE = os.path.join(os.path.dirname(__file__), "lu4d_trained.lbr")
SCIQ_LIMIT = 2500  # how many SciQ rows to mine for triples

STOPWORDS = set("""
a an the of to in on at by for with from into onto off and or but not no nor
is are was were be been being am has have had having do does did doing
this that these those it its as if then than so such also very most more less
can could may might should would will shall must ought some any all both each
other another one two three four five six seven eight nine ten
i you he she we they me him her us them my your his their our its mine yours hers
what which who whom whose when where why how there here
about above below over under between among during before after since until while
because though although however therefore thus hence due
does do did done doing being been
only just also even still yet already
""".split())


# ---------------------------------------------------------------------------
# MCP client: JSON-RPC over stdio to lu4d.exe --mcp
# ---------------------------------------------------------------------------

class MCPClient:
    def __init__(self, brain_path):
        self.brain_path = brain_path
        self.proc = None
        self.next_id = 1

    def start(self):
        # --no-save disables the per-tool autosave that would otherwise
        # rewrite the full brain file after every teach call.
        cmd = [LU4D_BIN, "--mcp", "--brain", self.brain_path, "--no-save"]
        print(f"[mcp] spawning: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.request("initialize", {})
        self.notify("notifications/initialized", {})

    def stop(self):
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()

    def _send(self, msg):
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed stdout")
        return json.loads(line)

    def request(self, method, params):
        rid = self.next_id
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        resp = self._read()
        while resp.get("id") != rid:
            resp = self._read()
        if "error" in resp:
            raise RuntimeError(f"MCP error {resp['error']}")
        return resp["result"]

    def notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def call_tool(self, name, args):
        return self.request("tools/call", {"name": name, "arguments": args})


# ---------------------------------------------------------------------------
# Triple extraction
# ---------------------------------------------------------------------------

def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def content_tokens(text, min_len=3, max_len=25):
    """Extract reasonably meaningful tokens from a chunk of text."""
    norm = normalize(text)
    out = []
    for tok in norm.split():
        if len(tok) < min_len or len(tok) > max_len:
            continue
        if tok in STOPWORDS:
            continue
        if tok.isdigit():
            continue
        out.append(tok)
    return out

def top_noun_phrase(text):
    """Pick a reasonable 'topic' phrase from a sentence/question."""
    toks = content_tokens(text)
    # Prefer the longest token as a rough proxy for a salient noun
    if not toks:
        return None
    # Use last content token, which is usually the focal noun in an English question
    return toks[-1]

def triples_from_sciq_row(row):
    """Emit S-R-O triples for one SciQ row.

    All nodes are single content tokens to avoid creating long multi-word
    phrase-nodes that get indexed on stopwords (which then hijack
    keyword_search on any query containing "is", "the", etc.).
    """
    question = row.get("question", "") or ""
    answer = row.get("correct_answer", "") or ""
    support = row.get("support", "") or ""
    out = []

    ans_toks = content_tokens(answer)
    if not ans_toks:
        return out

    # Question content tokens (last one is usually the focal noun)
    q_toks = content_tokens(question)
    if not q_toks:
        return out
    q_focus = q_toks[-1]

    # Primary triples: focal question token <-> RelatedTo <-> each answer token
    # (emit both directions so spread activation can reach either side)
    for a in ans_toks[:3]:
        if a != q_focus:
            out.append((q_focus, "RelatedTo", a))
            out.append((a, "RelatedTo", q_focus))

    # Bridge via the top answer token (the "canonical" answer single word)
    ans_head = ans_toks[0]

    # Extract unique support tokens, keep first-occurrence order
    seen = set()
    sup_uniq = []
    for t in content_tokens(support):
        if t not in seen:
            seen.add(t)
            sup_uniq.append(t)
        if len(sup_uniq) >= 6:
            break

    # Anchor: answer head -> RelatedTo -> each salient support token
    for t in sup_uniq[:4]:
        if t != ans_head:
            out.append((ans_head, "RelatedTo", t))

    return out


# ---------------------------------------------------------------------------
# Curated multi-domain triples to ensure breadth
# ---------------------------------------------------------------------------

CURATED = [
    # --- Geography (capitals) ---
    ("paris", "IsA", "capital"), ("france", "HasProperty", "paris"),
    ("tokyo", "IsA", "capital"), ("japan", "HasProperty", "tokyo"),
    ("london", "IsA", "capital"), ("england", "HasProperty", "london"),
    ("berlin", "IsA", "capital"), ("germany", "HasProperty", "berlin"),
    ("madrid", "IsA", "capital"), ("spain", "HasProperty", "madrid"),
    ("rome", "IsA", "capital"), ("italy", "HasProperty", "rome"),
    ("moscow", "IsA", "capital"), ("russia", "HasProperty", "moscow"),
    ("beijing", "IsA", "capital"), ("china", "HasProperty", "beijing"),
    ("cairo", "IsA", "capital"), ("egypt", "HasProperty", "cairo"),
    ("ottawa", "IsA", "capital"), ("canada", "HasProperty", "ottawa"),
    ("canberra", "IsA", "capital"), ("australia", "HasProperty", "canberra"),
    ("brasilia", "IsA", "capital"), ("brazil", "HasProperty", "brasilia"),
    ("washington", "IsA", "capital"), ("usa", "HasProperty", "washington"),
    ("new delhi", "IsA", "capital"), ("india", "HasProperty", "new delhi"),
    ("seoul", "IsA", "capital"), ("korea", "HasProperty", "seoul"),
    ("mexico city", "IsA", "capital"), ("mexico", "HasProperty", "mexico city"),
    # continents
    ("france", "PartOf", "europe"), ("germany", "PartOf", "europe"),
    ("italy", "PartOf", "europe"), ("spain", "PartOf", "europe"),
    ("japan", "PartOf", "asia"), ("china", "PartOf", "asia"),
    ("india", "PartOf", "asia"), ("korea", "PartOf", "asia"),
    ("egypt", "PartOf", "africa"), ("brazil", "PartOf", "south america"),
    ("usa", "PartOf", "north america"), ("canada", "PartOf", "north america"),
    ("australia", "PartOf", "oceania"),
    # rivers and mountains
    ("nile", "IsA", "river"), ("nile", "PartOf", "egypt"),
    ("amazon", "IsA", "river"), ("amazon", "PartOf", "brazil"),
    ("mississippi", "IsA", "river"), ("mississippi", "PartOf", "usa"),
    ("yangtze", "IsA", "river"), ("yangtze", "PartOf", "china"),
    ("thames", "IsA", "river"), ("thames", "PartOf", "england"),
    ("everest", "IsA", "mountain"), ("everest", "PartOf", "himalayas"),
    ("kilimanjaro", "IsA", "mountain"), ("kilimanjaro", "PartOf", "africa"),
    ("sahara", "IsA", "desert"), ("sahara", "PartOf", "africa"),
    ("gobi", "IsA", "desert"), ("gobi", "PartOf", "asia"),

    # --- History ---
    ("world war", "IsA", "war"),
    ("ww1", "Causes", "treaty of versailles"),
    ("ww2", "Causes", "united nations"),
    ("napoleon", "IsA", "emperor"), ("napoleon", "PartOf", "france"),
    ("julius caesar", "IsA", "emperor"), ("julius caesar", "PartOf", "rome"),
    ("lincoln", "IsA", "president"), ("lincoln", "PartOf", "usa"),
    ("washington", "IsA", "president"), ("kennedy", "IsA", "president"),
    ("einstein", "IsA", "physicist"), ("einstein", "Produces", "relativity"),
    ("newton", "IsA", "physicist"), ("newton", "Produces", "gravity"),
    ("darwin", "IsA", "biologist"), ("darwin", "Produces", "evolution"),
    ("curie", "IsA", "chemist"), ("curie", "Produces", "radium"),
    ("tesla", "IsA", "inventor"), ("tesla", "Produces", "alternating current"),
    ("edison", "IsA", "inventor"), ("edison", "Produces", "light bulb"),
    ("shakespeare", "IsA", "playwright"), ("shakespeare", "Produces", "hamlet"),
    ("shakespeare", "Produces", "macbeth"), ("shakespeare", "PartOf", "england"),
    ("mozart", "IsA", "composer"), ("mozart", "PartOf", "austria"),
    ("beethoven", "IsA", "composer"), ("beethoven", "PartOf", "germany"),
    ("renaissance", "IsA", "era"), ("renaissance", "PartOf", "europe"),
    ("industrial revolution", "Produces", "factories"),
    ("colonialism", "Causes", "slavery"), ("slavery", "Opposes", "freedom"),
    ("revolution", "Causes", "government change"),

    # --- Technology ---
    ("python", "IsA", "programming language"),
    ("rust", "IsA", "programming language"),
    ("javascript", "IsA", "programming language"),
    ("c", "IsA", "programming language"),
    ("java", "IsA", "programming language"),
    ("go", "IsA", "programming language"),
    ("linux", "IsA", "operating system"),
    ("windows", "IsA", "operating system"),
    ("macos", "IsA", "operating system"),
    ("cpu", "IsA", "hardware"), ("gpu", "IsA", "hardware"),
    ("ram", "IsA", "memory"), ("ssd", "IsA", "storage"),
    ("tcp", "IsA", "protocol"), ("http", "IsA", "protocol"),
    ("html", "UsedFor", "web pages"), ("css", "UsedFor", "styling"),
    ("sql", "UsedFor", "databases"), ("json", "UsedFor", "data exchange"),
    ("git", "UsedFor", "version control"),
    ("docker", "UsedFor", "containers"),
    ("kubernetes", "UsedFor", "orchestration"),
    ("neural network", "UsedFor", "machine learning"),
    ("transformer", "IsA", "neural network"),
    ("gpt", "IsA", "transformer"), ("claude", "IsA", "transformer"),
    ("ai", "Requires", "data"), ("machine learning", "Requires", "data"),
    ("encryption", "UsedFor", "security"),
    ("blockchain", "UsedFor", "cryptocurrency"),
    ("bitcoin", "IsA", "cryptocurrency"),
    ("internet", "Requires", "servers"),
    ("smartphone", "IsA", "device"), ("laptop", "IsA", "computer"),

    # --- Science (biology / chemistry / physics / astronomy) ---
    ("water", "IsA", "molecule"), ("water", "HasProperty", "h2o"),
    ("oxygen", "IsA", "element"), ("hydrogen", "IsA", "element"),
    ("carbon", "IsA", "element"), ("nitrogen", "IsA", "element"),
    ("photosynthesis", "Produces", "oxygen"),
    ("photosynthesis", "Requires", "sunlight"),
    ("plants", "Produces", "oxygen"),
    ("cell", "PartOf", "organism"),
    ("mitochondria", "PartOf", "cell"),
    ("mitochondria", "Produces", "atp"),
    ("dna", "PartOf", "cell"), ("dna", "UsedFor", "genetics"),
    ("heart", "PartOf", "body"), ("heart", "UsedFor", "pumping blood"),
    ("lungs", "PartOf", "body"), ("lungs", "UsedFor", "breathing"),
    ("brain", "PartOf", "body"), ("brain", "UsedFor", "thinking"),
    ("gravity", "Causes", "falling"),
    ("sun", "IsA", "star"), ("earth", "IsA", "planet"),
    ("moon", "IsA", "satellite"), ("moon", "PartOf", "earth"),
    ("mars", "IsA", "planet"), ("jupiter", "IsA", "planet"),
    ("saturn", "IsA", "planet"), ("venus", "IsA", "planet"),
    ("mercury", "IsA", "planet"), ("neptune", "IsA", "planet"),
    ("uranus", "IsA", "planet"),
    ("solar system", "PartOf", "galaxy"),
    ("galaxy", "PartOf", "universe"),
    ("light", "IsA", "electromagnetic radiation"),
    ("sound", "Requires", "medium"),
    ("electron", "PartOf", "atom"), ("proton", "PartOf", "atom"),
    ("neutron", "PartOf", "atom"),
    ("acid", "ContrastedWith", "base"),
    ("metal", "ContrastedWith", "nonmetal"),

    # --- Math ---
    ("algebra", "IsA", "math"), ("geometry", "IsA", "math"),
    ("calculus", "IsA", "math"), ("trigonometry", "IsA", "math"),
    ("pi", "RelatedTo", "circle"), ("pi", "HasProperty", "3.14159"),
    ("pythagoras", "Produces", "pythagorean theorem"),
    ("euclid", "Produces", "geometry"),

    # --- Art / Culture ---
    ("picasso", "IsA", "painter"), ("picasso", "PartOf", "spain"),
    ("davinci", "IsA", "painter"), ("davinci", "Produces", "mona lisa"),
    ("vangogh", "IsA", "painter"), ("vangogh", "PartOf", "netherlands"),
    ("monet", "IsA", "painter"), ("monet", "PartOf", "france"),

    # Extra linking concepts
    ("president", "IsA", "leader"), ("emperor", "IsA", "leader"),
    ("capital", "PartOf", "country"), ("river", "PartOf", "geography"),
    ("mountain", "PartOf", "geography"), ("desert", "PartOf", "geography"),
    ("planet", "PartOf", "solar system"),
    ("programming language", "UsedFor", "software"),
    ("software", "Requires", "computer"),
]


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------

def load_sciq(limit):
    """Load SciQ rows via the HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[load] datasets not available: {e}", flush=True)
        return []
    print(f"[load] loading sciq (limit={limit})...", flush=True)
    ds = load_dataset("sciq", split="train")
    n = min(limit, len(ds))
    rows = ds.select(range(n))
    print(f"[load] loaded {n} sciq rows", flush=True)
    return list(rows)


def train(client, sciq_rows):
    counts = Counter()
    start = time.time()

    # 1) Curated triples (high confidence). Emit each in both directions since
    #    spread activation follows outgoing bonds only -- asking "evolution"
    #    should reach "darwin" even though the canonical bond is darwin->evolution.
    print(f"[train] teaching {len(CURATED)} curated triples (+ reverse bonds)...",
          flush=True)
    for (s, r, o) in CURATED:
        s_n, o_n = normalize(s), normalize(o)
        try:
            client.call_tool("lu4d_teach", {
                "subject": s_n, "relation": r, "object": o_n,
                "weight": 0.9, "confidence": "Verified",
            })
            counts["curated"] += 1
            # Reverse edge for traversability (softer relation + weight)
            client.call_tool("lu4d_teach", {
                "subject": o_n, "relation": "RelatedTo", "object": s_n,
                "weight": 0.85, "confidence": "Verified",
            })
            counts["curated_reverse"] += 1
        except Exception as e:
            print(f"[train] curated error {s!r}->{o!r}: {e}", flush=True)

    # 2) SciQ-derived triples
    print(f"[train] mining triples from {len(sciq_rows)} sciq rows...", flush=True)
    total_triples = 0
    for i, row in enumerate(sciq_rows):
        triples = triples_from_sciq_row(row)
        for (s, r, o) in triples:
            if not s or not o or s == o:
                continue
            if len(s) > 60 or len(o) > 60:
                continue
            try:
                client.call_tool("lu4d_teach", {
                    "subject": s,
                    "relation": r,
                    "object": o,
                    "weight": 0.7,
                    "confidence": "Verified",
                })
                total_triples += 1
                counts["sciq"] += 1
            except Exception as e:
                # Don't spam the console; just tick a counter
                counts["errors"] += 1
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            print(f"[train]   row {i+1}/{len(sciq_rows)}, triples so far={total_triples}, "
                  f"{elapsed:.1f}s elapsed", flush=True)

    elapsed = time.time() - start
    print(f"[train] done in {elapsed:.1f}s -- {dict(counts)}", flush=True)
    return counts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

EVAL_QUESTIONS = [
    # --- Geography (10) ---
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("What is the capital of Germany?", ["berlin"]),
    ("What is the capital of Italy?", ["rome"]),
    ("What is the capital of Russia?", ["moscow"]),
    ("In which continent is Egypt?", ["africa"]),
    ("In which continent is Japan?", ["asia"]),
    ("The Nile river is in which country?", ["egypt"]),
    ("The Amazon river is in which country?", ["brazil"]),
    ("On which mountain range is Everest?", ["himalayas"]),

    # --- History & Figures (10) ---
    ("Who discovered the theory of relativity?", ["einstein"]),
    ("Who is famous for the theory of evolution?", ["darwin"]),
    ("Who discovered gravity?", ["newton"]),
    ("Who painted the Mona Lisa?", ["davinci"]),
    ("Who wrote Hamlet?", ["shakespeare"]),
    ("Which emperor ruled France in the 1800s?", ["napoleon"]),
    ("What did Edison invent?", ["light bulb", "bulb"]),
    ("What did Tesla invent?", ["alternating current", "current"]),
    ("Which composer was from Austria?", ["mozart"]),
    ("Which US president led during the Civil War era?", ["lincoln"]),

    # --- Technology (10) ---
    ("Which language is Rust?", ["programming language", "programming"]),
    ("What is Python used for?", ["programming", "software", "language"]),
    ("What is Docker used for?", ["containers"]),
    ("What is SQL used for?", ["databases", "database"]),
    ("What is HTML used for?", ["web", "pages"]),
    ("What is Kubernetes used for?", ["orchestration"]),
    ("What is Linux?", ["operating system", "system"]),
    ("What is a GPU?", ["hardware"]),
    ("What is Bitcoin?", ["cryptocurrency"]),
    ("What is a transformer in AI?", ["neural network", "network"]),

    # --- Science (10) ---
    ("What do plants produce during photosynthesis?", ["oxygen"]),
    ("What is the chemical formula related concept for water?", ["h2o", "molecule"]),
    ("Which organelle produces ATP in a cell?", ["mitochondria"]),
    ("What is the sun classified as?", ["star"]),
    ("What is the moon of Earth?", ["satellite", "moon"]),
    ("What are protons part of?", ["atom"]),
    ("What carries genetic information in cells?", ["dna"]),
    ("What organ pumps blood?", ["heart"]),
    ("What organ is used for breathing?", ["lungs"]),
    ("What planet is Jupiter?", ["planet"]),
]


def score_answer(activated, expected_keywords):
    """activated: list of {content, activation, ...}. Return (hit, top_hit)."""
    contents = [a.get("content", "").lower() for a in activated]
    joined = " | ".join(contents)
    for kw in expected_keywords:
        if kw.lower() in joined:
            return True, kw
    return False, None


def evaluate(client):
    print("\n" + "=" * 68, flush=True)
    print("EVALUATION: 40-question knowledge recall test", flush=True)
    print("=" * 68, flush=True)
    hits = 0
    detail = []
    for i, (q, expected) in enumerate(EVAL_QUESTIONS, start=1):
        try:
            result = client.call_tool("lu4d_ask", {"query": q, "depth": 3, "top_k": 12})
            structured = result.get("structuredContent", {})
            recall = structured.get("recall", {})
            activated = recall.get("activated", [])
        except Exception as e:
            activated = []
            print(f"[eval] error on Q{i}: {e}", flush=True)

        hit, matched = score_answer(activated, expected)
        if hit:
            hits += 1

        top3 = [a.get("content", "") for a in activated[:3]]
        mark = "PASS" if hit else "MISS"
        print(f"  Q{i:02d} [{mark}] {q}", flush=True)
        print(f"        expected={expected} | top-3={top3} | matched={matched}",
              flush=True)
        detail.append({
            "q": q, "expected": expected, "hit": hit, "matched": matched,
            "top3": top3, "activated": len(activated),
        })

    print("\n" + "-" * 68, flush=True)
    print(f"SCORE: {hits}/{len(EVAL_QUESTIONS)} questions answered "
          f"with relevant recall ({100*hits/len(EVAL_QUESTIONS):.1f}%)", flush=True)
    print("-" * 68, flush=True)
    return hits, len(EVAL_QUESTIONS), detail


# ---------------------------------------------------------------------------
# Temporal-features demo
# ---------------------------------------------------------------------------

def temporal_demo(client):
    print("\n" + "=" * 68, flush=True)
    print("TEMPORAL FEATURES DEMO", flush=True)
    print("=" * 68, flush=True)

    # Snapshot before any session-specific learning
    status = client.call_tool("lu4d_status", {})
    print("[temporal] pre-session status:\n  " +
          status.get("content", [{}])[0].get("text", "").replace("\n", "\n  "),
          flush=True)

    # Baseline query for a made-up concept
    probe_q = "What is Gluonite used for?"
    print(f"\n[temporal] BEFORE session: asking '{probe_q}'", flush=True)
    r = client.call_tool("lu4d_ask", {"query": probe_q, "depth": 3, "top_k": 5})
    activated = r.get("structuredContent", {}).get("recall", {}).get("activated", [])
    print(f"  -> activated={[a.get('content') for a in activated]}", flush=True)

    # Start a session, teach session-specific knowledge
    print("\n[temporal] starting session...", flush=True)
    s = client.call_tool("lu4d_session_start", {})
    session_id = s.get("structuredContent", {}).get("session_id")
    print(f"  -> session_id={session_id}", flush=True)

    session_triples = [
        ("gluonite", "IsA", "fictional mineral"),
        ("gluonite", "UsedFor", "powering warp drives"),
        ("gluonite", "HasProperty", "bright green glow"),
        ("warp drives", "Requires", "gluonite"),
    ]
    for (a, rel, b) in session_triples:
        client.call_tool("lu4d_teach", {
            "subject": a, "relation": rel, "object": b,
            "weight": 0.9, "confidence": "Unverified",
        })
    print(f"[temporal] taught {len(session_triples)} session-local triples",
          flush=True)

    # Query during session -> session knowledge should activate
    print(f"\n[temporal] DURING session: asking '{probe_q}'", flush=True)
    r = client.call_tool("lu4d_ask", {"query": probe_q, "depth": 3, "top_k": 5})
    activated = r.get("structuredContent", {}).get("recall", {}).get("activated", [])
    print(f"  -> activated={[a.get('content') for a in activated]}", flush=True)

    # End session
    print("\n[temporal] ending session...", flush=True)
    client.call_tool("lu4d_session_end", {})

    # Query after session (baseline recall)
    print(f"\n[temporal] AFTER session end: asking '{probe_q}'", flush=True)
    r = client.call_tool("lu4d_ask", {"query": probe_q, "depth": 3, "top_k": 5})
    activated = r.get("structuredContent", {}).get("recall", {}).get("activated", [])
    print(f"  -> activated={[a.get('content') for a in activated]}", flush=True)

    # Time travel: query as of 1 hour ago (before training) vs now
    print("\n[temporal] time travel diff: query '\"einstein\"' vs state 1 hour ago",
          flush=True)
    tt = client.call_tool("lu4d_timetravel",
                          {"query": "einstein", "ms_ago": 3600 * 1000})
    print("  -> " + tt.get("content", [{}])[0].get("text", ""), flush=True)

    # Consolidation stats
    print("\n[temporal] running consolidation...", flush=True)
    c = client.call_tool("lu4d_consolidate", {})
    print("  -> " + c.get("content", [{}])[0].get("text", ""), flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(LU4D_BIN):
        print(f"FATAL: lu4d.exe not found at {LU4D_BIN}", flush=True)
        sys.exit(1)

    # Wipe any prior brain file so this is a clean run
    if os.path.exists(BRAIN_FILE):
        os.remove(BRAIN_FILE)
        print(f"[setup] removed prior brain file", flush=True)

    sciq_rows = load_sciq(SCIQ_LIMIT)

    client = MCPClient(BRAIN_FILE)
    client.start()
    try:
        # status pre-training
        pre = client.call_tool("lu4d_status", {})
        print("[status] pre-train:\n  " +
              pre.get("content", [{}])[0].get("text", "").replace("\n", "\n  "),
              flush=True)

        counts = train(client, sciq_rows)

        post = client.call_tool("lu4d_status", {})
        print("\n[status] post-train:\n  " +
              post.get("content", [{}])[0].get("text", "").replace("\n", "\n  "),
              flush=True)

        hits, total, detail = evaluate(client)

        temporal_demo(client)

        print("\n" + "=" * 68, flush=True)
        print("FINAL SUMMARY", flush=True)
        print("=" * 68, flush=True)
        print(f"  triples taught (curated): {counts.get('curated', 0)}", flush=True)
        print(f"  triples taught (sciq):    {counts.get('sciq', 0)}", flush=True)
        print(f"  teach errors:             {counts.get('errors', 0)}", flush=True)
        print(f"  eval score:               {hits}/{total} "
              f"({100*hits/total:.1f}%)", flush=True)
    finally:
        client.stop()


if __name__ == "__main__":
    main()

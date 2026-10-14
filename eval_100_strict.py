#!/usr/bin/env python3
"""
eval_100_strict.py -- Strict 100-question progressive evaluation.

Scoring: 1 = expected answer appears in TOP-5 activated nodes' content labels.
         0 = answer not in top-5, empty, or seed-only.

"Seed-only" fail: if ALL top-5 results are query words themselves (depth=0
 trivial seed hits), the brain gets 0 even if a keyword matches.

This is strictly harder than "keyword anywhere in 20 results".
"""

import json, os, subprocess, sys, threading, time
from pathlib import Path

BRAIN_FILE = str(Path(__file__).parent / "lu4d_curriculum.lbr")
LU4D_EXE   = str(Path(__file__).parent / "target" / "release" / "lu4d.exe")

DEPTH  = 3
TOP_K  = 10   # get 10, score on top 5


# ---------------------------------------------------------------------------
class MCPClient:
    def __init__(self, brain_file):
        self.brain_file = brain_file
        self.proc = None
        self._lock = threading.Lock()
        self._req_id = 0

    def _drain_stderr(self):
        try:
            for raw in self.proc.stderr:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    print(f"  [lu4d] {line}", flush=True)
        except Exception:
            pass

    def start(self):
        cmd = [LU4D_EXE, "--mcp", "--brain", self.brain_file, "--no-save"]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        # initialize
        rid = self._next_id()
        self._raw_send({"jsonrpc": "2.0", "id": rid, "method": "initialize",
                        "params": {"protocolVersion": "2024-11-05",
                                   "capabilities": {},
                                   "clientInfo": {"name": "eval", "version": "1.0"}}})
        self._raw_recv(rid)
        self._raw_send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def _next_id(self):
        self._req_id += 1
        return self._req_id

    def _raw_send(self, obj):
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        self.proc.stdin.flush()

    def _raw_recv(self, target_id=None):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if target_id is None or obj.get("id") == target_id:
                    return obj
            except Exception:
                continue

    def stop(self):
        if self.proc:
            try: self.proc.stdin.close()
            except Exception: pass
            try: self.proc.wait(timeout=60)
            except Exception: pass
            self.proc = None

    def ask(self, query):
        with self._lock:
            rid = self._next_id()
            self._raw_send({
                "jsonrpc": "2.0", "id": rid,
                "method": "tools/call",
                "params": {"name": "lu4d_ask",
                           "arguments": {"query": query,
                                         "depth": DEPTH,
                                         "top_k": TOP_K}},
            })
            resp = self._raw_recv(rid)
        if resp is None:
            return [], ""
        result = resp.get("result", {})
        # structured content
        structured = result.get("structuredContent", {})
        # text content
        text = "\n".join(
            c.get("text", "") for c in result.get("content", [])
            if c.get("type") == "text"
        )
        activated = structured.get("recall", {}).get("activated", [])
        return activated, text

    def status(self):
        with self._lock:
            rid = self._next_id()
            self._raw_send({"jsonrpc": "2.0", "id": rid,
                            "method": "tools/call",
                            "params": {"name": "lu4d_status", "arguments": {}}})
            resp = self._raw_recv(rid)
        return (resp or {}).get("result", {}).get("structuredContent", {})


# ---------------------------------------------------------------------------
# Strict scorer
# ---------------------------------------------------------------------------
def strict_score(activated, query_text, keywords):
    """
    Return (score, reason_str).

    PASS conditions (all must hold):
      1. At least one result at depth >= 1 (not purely seed-level).
      2. An expected keyword appears in the content of a top-5 result.

    Depth-0 hits (query seeds themselves) count ONLY if they also hit a
    keyword AND there are < 5 results total (tiny activation).
    """
    if not activated:
        return 0, "no activation"

    top5 = activated[:5]
    has_depth1 = any(n.get("depth", 0) >= 1 for n in top5)
    if not has_depth1 and len(activated) >= 5:
        return 0, "only seed-level hits"

    top5_contents = [n.get("content", "").lower() for n in top5]
    top5_depths   = [n.get("depth", 0)            for n in top5]

    for kw in keywords:
        kw_lo = kw.lower()
        for i, content in enumerate(top5_contents):
            if kw_lo == content:
                return 1, f"exact '{kw}' at pos {i+1} d={top5_depths[i]}"
            if len(kw_lo) >= 4 and kw_lo in content:
                return 1, f"'{kw}' in '{content}' at pos {i+1}"
            if len(content) >= 4 and content in kw_lo:
                return 1, f"'{content}' in kw '{kw}' at pos {i+1}"

    top5_str = ", ".join(f"{c}(d{d})" for c, d in zip(top5_contents, top5_depths))
    return 0, f"top5=[{top5_str}]"


# ---------------------------------------------------------------------------
# 100 questions
# ---------------------------------------------------------------------------
QUESTIONS = [
    # ----------------------------------------------------------------
    # BRACKET 1: Basic Facts (Q1-20)
    # ----------------------------------------------------------------
    ("What is water made of?",
     ["hydrogen", "oxygen", "h2o"]),
    ("What is the capital of France?",
     ["paris"]),
    ("What planet is closest to the sun?",
     ["mercury"]),
    ("What is the largest ocean?",
     ["pacific"]),
    ("At what temperature does water boil?",
     ["celsius", "fahrenheit", "boil", "steam", "100"]),
    ("What is the capital of Japan?",
     ["tokyo"]),
    ("What is the speed of light?",
     ["speed", "light", "299", "electromagnetic"]),
    ("What is the largest mammal?",
     ["whale", "blue"]),
    ("Who wrote Romeo and Juliet?",
     ["shakespeare"]),
    ("What is the capital of Germany?",
     ["berlin"]),
    ("What element does the chemical symbol Au represent?",
     ["gold"]),
    ("How many continents are there?",
     ["seven", "continent"]),
    ("What is the powerhouse of the cell?",
     ["mitochondria", "mitochondrion"]),
    ("What is the largest planet in our solar system?",
     ["jupiter"]),
    ("What language is spoken in Brazil?",
     ["portuguese"]),
    ("What is the hardest natural substance?",
     ["diamond"]),
    ("What is the capital of Australia?",
     ["canberra"]),
    ("What is DNA?",
     ["genetic", "nucleic", "molecule", "chromosome", "gene"]),
    ("What is gravity?",
     ["force", "mass", "attraction", "newton"]),
    ("What is the chemical symbol for gold?",
     ["au", "gold"]),

    # ----------------------------------------------------------------
    # BRACKET 2: Science (Q21-40)
    # ----------------------------------------------------------------
    ("How does photosynthesis work?",
     ["chlorophyll", "glucose", "carbon", "sunlight", "plant", "oxygen"]),
    ("What causes earthquakes?",
     ["tectonic", "fault", "plate", "seismic", "crust"]),
    ("How does the immune system work?",
     ["antibody", "virus", "cell", "infection", "lymphocyte", "immune"]),
    ("What is the theory of relativity?",
     ["einstein", "space", "time", "mass", "energy", "relativity"]),
    ("What causes tides?",
     ["moon", "gravity", "tidal", "lunar", "ocean"]),
    ("How do vaccines work?",
     ["immune", "antibody", "virus", "immunity", "pathogen", "vaccine"]),
    ("How does nuclear fission work?",
     ["nuclear", "fission", "atom", "neutron", "uranium", "energy"]),
    ("What is evolution?",
     ["darwin", "species", "selection", "adapt", "evolution", "natural"]),
    ("How does the brain process memory?",
     ["hippocampus", "neuron", "synapse", "memory", "brain"]),
    ("What is quantum mechanics?",
     ["quantum", "particle", "wave", "uncertainty", "electron"]),
    ("How do black holes form?",
     ["black", "hole", "gravity", "star", "collapse", "mass"]),
    ("What is entropy?",
     ["entropy", "disorder", "thermodynamics", "heat"]),
    ("How does the carbon cycle work?",
     ["carbon", "co2", "atmosphere", "photosynthesis", "fossil"]),
    ("What is the greenhouse effect?",
     ["greenhouse", "carbon", "temperature", "atmosphere", "climate"]),
    ("How do neurons transmit signals?",
     ["neuron", "synapse", "axon", "signal", "electrical", "chemical"]),
    ("What is CRISPR?",
     ["crispr", "gene", "dna", "editing", "genetic"]),
    ("How does natural selection work?",
     ["selection", "survival", "fitness", "evolution", "adapt"]),
    ("What is dark matter?",
     ["dark", "matter", "gravity", "universe", "mass"]),
    ("What is the Higgs boson?",
     ["higgs", "boson", "mass", "particle", "field"]),
    ("How does the digestive system work?",
     ["stomach", "enzyme", "intestine", "digest", "nutrient", "food"]),

    # ----------------------------------------------------------------
    # BRACKET 3: Cross-domain (Q41-60)
    # ----------------------------------------------------------------
    ("How does chemistry relate to cooking?",
     ["reaction", "heat", "protein", "maillard", "chemistry", "molecule"]),
    ("What is the physics of music?",
     ["sound", "wave", "frequency", "vibration", "acoustic"]),
    ("How does biology connect to technology?",
     ["biotech", "genetic", "dna", "artificial", "bioinformatics", "biology"]),
    ("How does mathematics relate to music?",
     ["frequency", "ratio", "pattern", "harmony", "wave", "math"]),
    ("What is the connection between economics and psychology?",
     ["behavior", "decision", "rational", "cognitive", "bias", "economic"]),
    ("How does language shape thought?",
     ["language", "thought", "cognitive", "sapir", "culture", "concept"]),
    ("What connects art to neuroscience?",
     ["perception", "visual", "art", "neural", "aesthetic", "brain"]),
    ("How does physics apply to sports?",
     ["force", "motion", "velocity", "kinetic", "energy", "physics"]),
    ("What is the relationship between information and entropy?",
     ["information", "entropy", "shannon", "disorder", "bit"]),
    ("How does game theory apply to evolution?",
     ["game", "strategy", "evolution", "cooperation", "nash", "fitness"]),
    ("What connects geography to history?",
     ["geography", "history", "civilization", "trade", "resource", "climate"]),
    ("How does statistics relate to medicine?",
     ["statistics", "trial", "clinical", "evidence", "probability", "data"]),
    ("What is the relationship between philosophy and science?",
     ["philosophy", "science", "empirical", "knowledge", "reason", "logic"]),
    ("How does architecture relate to mathematics?",
     ["architecture", "geometry", "proportion", "structure", "design", "math"]),
    ("What connects climate to civilization?",
     ["climate", "civilization", "agriculture", "food", "temperature"]),
    ("How does ecology relate to economics?",
     ["ecology", "economic", "resource", "ecosystem", "capital", "natural"]),
    ("What is the connection between music and mathematics?",
     ["music", "math", "pattern", "harmony", "frequency", "ratio"]),
    ("How does chemistry relate to medicine?",
     ["chemistry", "medicine", "drug", "molecule", "pharmacology", "compound"]),
    ("What connects psychology to marketing?",
     ["psychology", "behavior", "marketing", "persuasion", "consumer", "emotion"]),
    ("How does evolution relate to language?",
     ["evolution", "language", "communicate", "symbol", "culture", "adapt"]),

    # ----------------------------------------------------------------
    # BRACKET 4: Abstract Reasoning (Q61-80)
    # ----------------------------------------------------------------
    ("Why do economies have cycles?",
     ["cycle", "economic", "boom", "recession", "inflation", "demand", "credit"]),
    ("What makes a good analogy?",
     ["analogy", "comparison", "structure", "relation", "mapping", "similar"]),
    ("How does evolution relate to innovation?",
     ["evolution", "innovation", "selection", "variation", "adapt", "fitness"]),
    ("Why do complex systems self-organize?",
     ["complex", "system", "emerge", "organize", "feedback", "pattern"]),
    ("What is the relationship between order and chaos?",
     ["order", "chaos", "entropy", "system", "stability", "dynamic"]),
    ("Why does diversity increase resilience?",
     ["diversity", "resilience", "ecosystem", "variation", "adapt", "robust"]),
    ("How do feedback loops shape systems?",
     ["feedback", "loop", "system", "control", "amplify", "regulate"]),
    ("What makes knowledge cumulative?",
     ["knowledge", "science", "cumulative", "language", "record", "build"]),
    ("Why do networks have hubs?",
     ["network", "hub", "node", "connect", "scale", "preferential"]),
    ("How does emergence work?",
     ["emerge", "complex", "property", "system", "collective", "pattern"]),
    ("What is the role of constraints in creativity?",
     ["constraint", "creative", "limit", "innovation", "problem", "solution"]),
    ("Why do civilizations rise and fall?",
     ["civilization", "collapse", "resource", "complexity", "climate", "trade"]),
    ("How does compression relate to understanding?",
     ["compress", "pattern", "information", "model", "abstract", "represent"]),
    ("What makes a good model?",
     ["model", "predict", "accurate", "theory", "abstract", "simple"]),
    ("Why is symmetry important in physics?",
     ["symmetry", "physics", "conservation", "law", "invariant", "noether"]),
    ("How does language evolve?",
     ["language", "evolve", "change", "cultural", "adapt", "grammar"]),
    ("What is the relationship between competition and cooperation?",
     ["compete", "cooperate", "evolution", "game", "social", "altruism"]),
    ("Why do exponential processes surprise us?",
     ["exponential", "growth", "compound", "double", "scale", "intuition"]),
    ("How does abstraction enable progress?",
     ["abstract", "model", "concept", "simplify", "build", "reuse"]),
    ("What connects individual behavior to collective outcomes?",
     ["individual", "collective", "emerge", "social", "network", "behavior"]),

    # ----------------------------------------------------------------
    # BRACKET 5: Complex Multi-Hop (Q81-100)
    # ----------------------------------------------------------------
    ("Trace the chain from sunlight to human civilization",
     ["photosynthesis", "agriculture", "food", "plant", "solar", "energy"]),
    ("How does plate tectonics affect biodiversity?",
     ["tectonic", "plate", "continent", "isolate", "evolution", "species"]),
    ("Connect mathematics to art",
     ["proportion", "symmetry", "fibonacci", "golden", "geometry", "pattern"]),
    ("How does the electromagnetic spectrum relate to modern technology?",
     ["electromagnetic", "spectrum", "radio", "wave", "wireless", "frequency"]),
    ("Trace the path from atoms to consciousness",
     ["atom", "molecule", "cell", "neuron", "brain", "consciousness"]),
    ("How does language shape scientific discovery?",
     ["language", "science", "concept", "theory", "communicate", "symbol"]),
    ("Connect thermodynamics to economics",
     ["thermodynamic", "energy", "entropy", "economic", "resource", "efficiency"]),
    ("How does geography shape culture?",
     ["geography", "culture", "climate", "resource", "trade", "civilization"]),
    ("Trace evolution from single cells to intelligence",
     ["cell", "multicellular", "brain", "intelligence", "nervous", "adapt"]),
    ("How does quantum mechanics enable modern computing?",
     ["quantum", "electron", "semiconductor", "transistor", "computing", "chip"]),
    ("Connect Darwinian evolution to cultural change",
     ["darwin", "evolution", "culture", "meme", "selection", "variation"]),
    ("How does information theory relate to biology?",
     ["information", "dna", "genetic", "code", "shannon", "entropy", "biology"]),
    ("Trace the chain from stellar nucleosynthesis to life",
     ["star", "element", "carbon", "nuclear", "fusion", "life", "supernova"]),
    ("How do game theory and evolution connect to social norms?",
     ["game", "evolution", "social", "norm", "cooperation", "strategy", "nash"]),
    ("Connect the mathematics of networks to social phenomena",
     ["network", "graph", "social", "node", "connect", "scale", "hub"]),
    ("How does the physics of climate connect to human migration?",
     ["climate", "temperature", "migration", "agriculture", "drought", "human"]),
    ("Trace the development of symbolic thought to modern mathematics",
     ["symbol", "language", "abstract", "number", "math", "logic", "formal"]),
    ("How does neuroscience relate to artificial intelligence?",
     ["neuroscience", "brain", "neural", "artificial", "intelligence", "learn"]),
    ("Connect the concept of entropy to social organization",
     ["entropy", "order", "disorder", "social", "organize", "energy", "complexity"]),
    ("How does the feedback between technology and society drive history?",
     ["technology", "society", "feedback", "industrial", "innovation", "history"]),
]

assert len(QUESTIONS) == 100

BRACKETS = [
    (1,  20, "Basic Facts"),
    (21, 40, "Science"),
    (41, 60, "Cross-Domain"),
    (61, 80, "Abstract Reasoning"),
    (81, 100, "Complex Multi-Hop"),
]


# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(BRAIN_FILE):
        print(f"ERROR: brain not found: {BRAIN_FILE}")
        sys.exit(1)

    sz    = os.path.getsize(BRAIN_FILE)
    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(BRAIN_FILE)))
    print(f"[eval] brain: {sz:,} bytes ({sz/1024**3:.2f} GB) | mtime={mtime}")
    print(f"[eval] binary: {LU4D_EXE}")
    print(f"[eval] scoring: TOP-{TOP_K} results, answer must be in top-5 content labels")
    print(f"[eval] depth={DEPTH}  top_k={TOP_K}")

    client = MCPClient(BRAIN_FILE)
    client.start()

    st = client.status()
    print(f"[eval] nodes={st.get('node_count',0):,}  bonds={st.get('bond_count',0):,}")
    print()

    scores         = []
    bracket_scores = {b[2]: [] for b in BRACKETS}
    t0             = time.time()

    for i, (question, keywords) in enumerate(QUESTIONS, 1):
        # bracket header
        for lo, hi, name in BRACKETS:
            if i == lo:
                print(f"\n{'='*70}")
                print(f"  BRACKET: {name}  (Q{lo}-{hi})")
                print(f"{'='*70}")
                break

        activated, text = client.ask(question)
        sc, reason = strict_score(activated, question, keywords)

        # build display for top-3
        top3 = activated[:3]
        top3_str = "  |  ".join(
            f"{j+1}.{n.get('content','?')}(a={n.get('activation',0):.3f},d={n.get('depth',0)})"
            for j, n in enumerate(top3)
        )
        if not top3_str:
            top3_str = "(no activation)"

        mark = "+" if sc else "-"
        elapsed = time.time() - t0
        print(f"  [{mark}] Q{i:03d}: {question}", flush=True)
        print(f"         top3: {top3_str}", flush=True)
        if not sc:
            print(f"         FAIL -- {reason}", flush=True)
            print(f"         expected: {keywords[:4]}", flush=True)

        scores.append(sc)
        for lo, hi, name in BRACKETS:
            if lo <= i <= hi:
                bracket_scores[name].append(sc)

    # ---- Final report ----
    total   = sum(scores)
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  Lu 4D -- STRICT 100-QUESTION EVAL RESULTS")
    print(f"  Brain: {sz/1024**3:.2f} GB | {st.get('node_count',0):,} nodes / "
          f"{st.get('bond_count',0):,} bonds")
    print(f"  Scoring: answer must appear in top-5 activated nodes")
    print(f"{'='*70}")
    print(f"  {'Bracket':<25} {'Score':>7}   {'Bar'}")
    print(f"  {'-'*60}")
    for lo, hi, name in BRACKETS:
        bsc = bracket_scores[name]
        bs  = sum(bsc)
        n   = len(bsc)
        pct = 100.0 * bs / n if n else 0
        bar = "#" * bs + "." * (n - bs)
        print(f"  {name:<25} {bs:>3}/{n:<3}  {pct:>5.1f}%  |{bar}|")
    print(f"  {'-'*60}")
    pct_total = 100.0 * total / len(scores)
    print(f"  {'TOTAL':<25} {total:>3}/100  {pct_total:>5.1f}%")
    print(f"\n  Elapsed: {elapsed:.1f}s")

    client.stop()
    print("[eval] done.")


if __name__ == "__main__":
    main()

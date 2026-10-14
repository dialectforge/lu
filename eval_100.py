#!/usr/bin/env python3
"""
eval_100.py -- 100-question progressive evaluation of the Lu 4D brain.
Brackets: Basic (1-20), Science (21-40), Cross-domain (41-60),
          Abstract (61-80), Multi-hop (81-100).
Scoring: 1 = meaningful/correct response, 0 = empty/irrelevant.
"""

import json, os, subprocess, sys, threading, time
from pathlib import Path

BRAIN_FILE = str(Path(__file__).parent / "lu4d_curriculum.lbr")
LU4D_EXE   = str(Path(__file__).parent / "target" / "release" / "lu4d.exe")

# ---------------------------------------------------------------------------
# MCP client (read-only, no session, no Hebbian)
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
        print(f"[mcp] {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def stop(self):
        if self.proc:
            try: self.proc.stdin.close()
            except Exception: pass
            self.proc.wait(timeout=60)
            self.proc = None

    def _send(self, method, params):
        with self._lock:
            self._req_id += 1
            rid = self._req_id
            msg = json.dumps({"jsonrpc": "2.0", "id": rid,
                              "method": method, "params": params}) + "\n"
            self.proc.stdin.write(msg.encode())
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if not line: raise RuntimeError("MCP EOF")
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                if obj.get("id") == rid:
                    return obj

    def ask(self, query, depth=4, top_k=20):
        resp = self._send("tools/call", {
            "name": "lu4d_ask",
            "arguments": {"query": query, "depth": depth, "top_k": top_k},
        })
        result = resp.get("result", {})
        content = result.get("content", [])
        text = "\n".join(c.get("text", "") for c in content
                         if c.get("type") == "text")
        structured = result.get("structuredContent", {})
        if structured:
            return {**structured, "text": text}
        return {"text": text}

    def status(self):
        resp = self._send("tools/call",
                          {"name": "lu4d_status", "arguments": {}})
        result = resp.get("result", {})
        structured = result.get("structuredContent", {})
        return structured

# ---------------------------------------------------------------------------
# 100 questions with expected keywords for automated scoring
# (keywords are OR'd: any hit => score 1)
# ---------------------------------------------------------------------------
QUESTIONS = [
    # --- BRACKET 1: Basic facts (1-20) ---
    ("What is water made of?",
     ["hydrogen", "oxygen", "h2o"]),
    ("What is the capital of France?",
     ["paris"]),
    ("What planet is closest to the sun?",
     ["mercury"]),
    ("What is the largest ocean?",
     ["pacific"]),
    ("What is the boiling point of water?",
     ["100", "celsius", "212", "fahrenheit", "boil"]),
    ("What is the capital of Japan?",
     ["tokyo"]),
    ("What is the speed of light?",
     ["light", "speed", "299", "electromagnetic"]),
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
     ["genetic", "dna", "molecule", "nucleic"]),
    ("What is gravity?",
     ["force", "mass", "gravity", "newton", "attraction"]),
    ("What is the chemical symbol for gold?",
     ["au", "gold"]),

    # --- BRACKET 2: Science (21-40) ---
    ("How does photosynthesis work?",
     ["light", "chlorophyll", "plant", "glucose", "energy", "carbon"]),
    ("What causes earthquakes?",
     ["tectonic", "fault", "plate", "seismic", "crust"]),
    ("How does the immune system work?",
     ["immune", "antibody", "virus", "cell", "infection", "white"]),
    ("What is the theory of relativity?",
     ["einstein", "relativity", "space", "time", "energy", "mass"]),
    ("What causes tides?",
     ["moon", "gravity", "tidal", "ocean", "lunar"]),
    ("How do vaccines work?",
     ["immune", "vaccine", "antibody", "virus", "immunity", "pathogen"]),
    ("How does nuclear fission work?",
     ["nuclear", "fission", "atom", "neutron", "energy", "uranium"]),
    ("What is evolution?",
     ["evolution", "natural", "selection", "darwin", "species", "adapt"]),
    ("How does the brain process memory?",
     ["brain", "memory", "neuron", "hippocampus", "synapse"]),
    ("What is quantum mechanics?",
     ["quantum", "particle", "wave", "electron", "uncertainty", "physics"]),
    ("How do black holes form?",
     ["black", "hole", "gravity", "star", "collapse", "mass"]),
    ("What is entropy?",
     ["entropy", "disorder", "thermodynamics", "energy", "heat"]),
    ("How does the carbon cycle work?",
     ["carbon", "cycle", "co2", "atmosphere", "photosynthesis", "fossil"]),
    ("What is the greenhouse effect?",
     ["greenhouse", "carbon", "temperature", "atmosphere", "heat", "climate"]),
    ("How do neurons transmit signals?",
     ["neuron", "synapse", "signal", "axon", "electrical", "chemical"]),
    ("What is CRISPR?",
     ["crispr", "gene", "dna", "editing", "genetic"]),
    ("How does natural selection work?",
     ["selection", "survival", "fitness", "evolution", "adapt", "reproduce"]),
    ("What is dark matter?",
     ["dark", "matter", "gravity", "universe", "mass", "invisible"]),
    ("What is the Higgs boson?",
     ["higgs", "boson", "mass", "particle", "field"]),
    ("How does the digestive system work?",
     ["digest", "stomach", "enzyme", "intestine", "food", "nutrient"]),

    # --- BRACKET 3: Cross-domain (41-60) ---
    ("How does chemistry relate to cooking?",
     ["chemical", "reaction", "heat", "protein", "maillard", "molecule",
      "chemistry", "cooking", "food"]),
    ("What is the physics of music?",
     ["sound", "wave", "frequency", "vibration", "acoustic", "music"]),
    ("How does biology connect to technology?",
     ["biology", "technology", "biotech", "genetic", "dna", "artificial",
      "bioinformatics"]),
    ("How does mathematics relate to music?",
     ["math", "frequency", "ratio", "pattern", "harmony", "wave", "music"]),
    ("What is the connection between economics and psychology?",
     ["behavior", "decision", "economic", "psychology", "rational",
      "cognitive", "bias"]),
    ("How does language shape thought?",
     ["language", "thought", "cognitive", "culture", "sapir", "concept"]),
    ("What connects art to neuroscience?",
     ["brain", "perception", "visual", "art", "neural", "aesthetic"]),
    ("How does physics apply to sports?",
     ["force", "motion", "velocity", "physics", "energy", "kinetic"]),
    ("What is the relationship between information and entropy?",
     ["information", "entropy", "shannon", "disorder", "bit", "uncertainty"]),
    ("How does game theory apply to evolution?",
     ["game", "strategy", "evolution", "cooperation", "nash", "fitness"]),
    ("What connects geography to history?",
     ["geography", "history", "civilization", "trade", "resource", "climate"]),
    ("How does statistics relate to medicine?",
     ["statistics", "trial", "clinical", "evidence", "probability", "data"]),
    ("What is the relationship between philosophy and science?",
     ["philosophy", "science", "empirical", "knowledge", "reason", "logic"]),
    ("How does architecture relate to mathematics?",
     ["architecture", "geometry", "proportion", "math", "structure", "design"]),
    ("What connects climate to civilization?",
     ["climate", "civilization", "agriculture", "food", "temperature",
      "collapse"]),
    ("How does ecology relate to economics?",
     ["ecology", "economic", "resource", "ecosystem", "capital", "natural"]),
    ("What is the connection between music and mathematics?",
     ["music", "math", "pattern", "harmony", "frequency", "ratio"]),
    ("How does chemistry relate to medicine?",
     ["chemistry", "medicine", "drug", "molecule", "pharmacology", "compound"]),
    ("What connects psychology to marketing?",
     ["psychology", "behavior", "marketing", "persuasion", "consumer",
      "emotion"]),
    ("How does evolution relate to language?",
     ["evolution", "language", "communicate", "symbol", "culture", "adapt"]),

    # --- BRACKET 4: Abstract reasoning (61-80) ---
    ("Why do economies have cycles?",
     ["cycle", "economic", "boom", "recession", "inflation", "demand",
      "supply", "credit"]),
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
     ["knowledge", "build", "science", "cumulative", "language", "record"]),
    ("Why do networks have hubs?",
     ["network", "hub", "node", "connect", "scale", "preferential"]),
    ("How does emergence work?",
     ["emerge", "complex", "property", "system", "collective", "pattern"]),
    ("What is the role of constraints in creativity?",
     ["constraint", "creative", "limit", "innovation", "problem", "solution"]),
    ("Why do civilizations rise and fall?",
     ["civilization", "collapse", "resource", "complexity", "climate",
      "trade", "war"]),
    ("How does compression relate to understanding?",
     ["compress", "pattern", "information", "model", "abstract", "represent"]),
    ("What makes a good model?",
     ["model", "predict", "accurate", "simple", "abstract", "theory"]),
    ("Why is symmetry important in physics?",
     ["symmetry", "physics", "conservation", "law", "invariant", "noether"]),
    ("How does language evolve?",
     ["language", "evolve", "change", "cultural", "adapt", "grammar"]),
    ("What is the relationship between competition and cooperation?",
     ["compete", "cooperate", "evolution", "game", "social", "altruism"]),
    ("Why do exponential processes surprise us?",
     ["exponential", "growth", "compound", "intuition", "double", "scale"]),
    ("How does abstraction enable progress?",
     ["abstract", "model", "concept", "simplify", "build", "reuse"]),
    ("What connects individual behavior to collective outcomes?",
     ["individual", "collective", "emerge", "social", "network", "behavior"]),

    # --- BRACKET 5: Complex multi-hop (81-100) ---
    ("Trace the chain from sunlight to human civilization",
     ["sun", "photosynthesis", "energy", "plant", "agriculture", "food",
      "civilization", "light"]),
    ("How does plate tectonics affect biodiversity?",
     ["tectonic", "plate", "continent", "isolate", "evolution", "species",
      "diversity", "geography"]),
    ("Connect mathematics to art",
     ["math", "art", "proportion", "symmetry", "pattern", "fibonacci",
      "golden", "geometry"]),
    ("How does the electromagnetic spectrum relate to modern technology?",
     ["electromagnetic", "spectrum", "radio", "light", "wave", "wireless",
      "technology", "frequency"]),
    ("Trace the path from atoms to consciousness",
     ["atom", "molecule", "cell", "neuron", "brain", "consciousness",
      "emerge", "chemistry"]),
    ("How does language shape scientific discovery?",
     ["language", "science", "concept", "model", "theory", "communicate",
      "symbol", "knowledge"]),
    ("Connect thermodynamics to economics",
     ["thermodynamic", "energy", "entropy", "economic", "resource",
      "efficiency", "work", "system"]),
    ("How does geography shape culture?",
     ["geography", "culture", "climate", "resource", "trade", "isolate",
      "adapt", "civilization"]),
    ("Trace evolution from single cells to intelligence",
     ["cell", "evolution", "multicellular", "brain", "intelligence",
      "nervous", "complex", "adapt"]),
    ("How does quantum mechanics enable modern computing?",
     ["quantum", "electron", "semiconductor", "transistor", "computing",
      "chip", "physics"]),
    ("Connect Darwinian evolution to cultural change",
     ["darwin", "evolution", "culture", "meme", "selection", "adapt",
      "variation", "change"]),
    ("How does information theory relate to biology?",
     ["information", "dna", "genetic", "code", "shannon", "entropy",
      "biology", "bit"]),
    ("Trace the chain from stellar nucleosynthesis to life",
     ["star", "element", "carbon", "nuclear", "fusion", "life", "atom",
      "supernova"]),
    ("How do game theory and evolution connect to social norms?",
     ["game", "evolution", "social", "norm", "cooperation", "strategy",
      "behavior", "nash"]),
    ("Connect the mathematics of networks to social phenomena",
     ["network", "graph", "social", "node", "connect", "scale", "hub",
      "community"]),
    ("How does the physics of climate connect to human migration?",
     ["climate", "temperature", "migration", "agriculture", "drought",
      "sea", "human", "resource"]),
    ("Trace the development of symbolic thought to modern mathematics",
     ["symbol", "language", "abstract", "number", "math", "logic",
      "formal", "proof"]),
    ("How does neuroscience relate to artificial intelligence?",
     ["neuroscience", "brain", "neural", "artificial", "intelligence",
      "network", "learn", "neuron"]),
    ("Connect the concept of entropy to social organization",
     ["entropy", "order", "disorder", "social", "organize", "energy",
      "complexity", "system"]),
    ("How does the feedback between technology and society drive history?",
     ["technology", "society", "feedback", "change", "industrial",
      "culture", "innovation", "history"]),
]

assert len(QUESTIONS) == 100, f"Expected 100 questions, got {len(QUESTIONS)}"

BRACKET_NAMES = [
    (1,  20, "Basic Facts"),
    (21, 40, "Science"),
    (41, 60, "Cross-Domain"),
    (61, 80, "Abstract Reasoning"),
    (81, 100, "Complex Multi-Hop"),
]

# ---------------------------------------------------------------------------
def score_response(text, keywords):
    """1 if any keyword found in response AND response is non-trivial."""
    tl = text.lower()
    # non-trivial: must have some activation (not completely empty)
    if "activated: 0" in tl or ("seed hits: 0" in tl and "activated: 0" in tl):
        return 0
    return 1 if any(kw.lower() in tl for kw in keywords) else 0

def run_eval(client):
    print("\n" + "=" * 72, flush=True)
    print("Lu 4D PROGRESSIVE EVALUATION -- 100 QUESTIONS", flush=True)
    print("=" * 72, flush=True)

    scores = []
    bracket_scores = {b[2]: [] for b in BRACKET_NAMES}

    current_bracket = BRACKET_NAMES[0]

    for i, (question, keywords) in enumerate(QUESTIONS, 1):
        # print bracket header
        for lo, hi, name in BRACKET_NAMES:
            if i == lo:
                print(f"\n--- BRACKET: {name} (Q{lo}-{hi}) ---", flush=True)
                current_bracket = (lo, hi, name)
                break

        try:
            r    = client.ask(question, depth=4, top_k=20)
            text = r.get("text", "")
            sc   = score_response(text, keywords)

            # extract top result for display
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            top_result = ""
            for l in lines:
                if l.startswith("1.") or l.startswith("  1."):
                    top_result = l[:100]
                    break
            if not top_result:
                # grab activated count line
                for l in lines:
                    if "activated" in l.lower():
                        top_result = l[:80]
                        break

            mark = "+" if sc else "-"
            print(f"  [{mark}] Q{i:03d}: {question}", flush=True)
            if top_result:
                print(f"         {top_result}", flush=True)
            if not sc:
                print(f"         (expected: {keywords[:3]})", flush=True)

        except Exception as e:
            print(f"  [!] Q{i:03d}: ERROR -- {e}", flush=True)
            sc = 0

        scores.append(sc)
        for lo, hi, name in BRACKET_NAMES:
            if lo <= i <= hi:
                bracket_scores[name].append(sc)

    # final report
    print("\n" + "=" * 72, flush=True)
    print("RESULTS", flush=True)
    print("=" * 72, flush=True)
    total = sum(scores)
    print(f"\n  {'Bracket':<25} {'Score':>8}  {'Pct':>6}", flush=True)
    print(f"  {'-'*45}", flush=True)
    for lo, hi, name in BRACKET_NAMES:
        bsc = bracket_scores[name]
        bs  = sum(bsc)
        pct = 100.0 * bs / len(bsc) if bsc else 0
        bar = "#" * bs + "." * (len(bsc) - bs)
        print(f"  {name:<25} {bs:>3}/{len(bsc):<3}  {pct:>5.1f}%  |{bar}|",
              flush=True)
    print(f"  {'-'*45}", flush=True)
    pct_total = 100.0 * total / len(scores)
    print(f"  {'TOTAL':<25} {total:>3}/100  {pct_total:>5.1f}%", flush=True)

    return scores, bracket_scores

def main():
    if not os.path.exists(BRAIN_FILE):
        print(f"ERROR: brain file not found: {BRAIN_FILE}", flush=True)
        sys.exit(1)
    sz = os.path.getsize(BRAIN_FILE)
    print(f"[eval] brain file: {sz:,} bytes ({sz/1024**3:.2f} GB)", flush=True)

    client = MCPClient(BRAIN_FILE)
    client.start()

    st = client.status()
    print(f"[eval] nodes={st.get('node_count',0):,}  bonds={st.get('bond_count',0):,}",
          flush=True)

    run_eval(client)
    client.stop()
    print("\n[eval] done.", flush=True)

if __name__ == "__main__":
    main()

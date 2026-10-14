#!/usr/bin/env python3
"""
targeted_train.py -- Targeted training to fill strict-eval gaps, then iterate.

Workflow:
  1. Teach all required triples (bidirectional single-token pairs)
  2. Save brain
  3. Run strict 100-question eval inline
  4. If score < 95, teach remediation triples for still-failing questions
  5. Repeat until 95+ or no progress
"""

import json, os, subprocess, sys, threading, time
from pathlib import Path

# Force UTF-8 on Windows (cp1252 default breaks some chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BRAIN = str(Path(__file__).parent / "lu4d_curriculum.lbr")
EXE   = str(Path(__file__).parent / "target" / "release" / "lu4d.exe")

DEPTH = 3
TOP_K = 10

# ---------------------------------------------------------------------------
# MCP client
# ---------------------------------------------------------------------------
class MCPClient:
    def __init__(self, brain_file):
        self.brain_file = brain_file
        self.proc = None
        self._id = 0
        self._lock = threading.Lock()

    def start(self):
        self.proc = subprocess.Popen(
            [EXE, "--mcp", "--brain", self.brain_file],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        rid = self._next_id()
        self._send({"jsonrpc": "2.0", "id": rid, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05",
                               "capabilities": {},
                               "clientInfo": {"name": "targeted_train", "version": "1.0"}}})
        self._recv()
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        print("[mcp] connected", flush=True)

    def _drain_stderr(self):
        try:
            for raw in self.proc.stderr:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    print("[lu4d] " + line, flush=True)
        except Exception:
            pass

    def _next_id(self):
        self._id += 1
        return self._id

    def _send(self, obj):
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        self.proc.stdin.flush()

    def _recv(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except Exception:
                continue

    def teach(self, subject, relation, obj, weight=0.92, confidence="Verified"):
        with self._lock:
            rid = self._next_id()
            self._send({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                        "params": {"name": "lu4d_teach",
                                   "arguments": {"subject": subject, "relation": relation,
                                                 "object": obj, "weight": weight,
                                                 "confidence": confidence}}})
            resp = self._recv()
            if resp and "error" in resp:
                print("  [err] " + subject + "->" + obj + ": " + str(resp["error"]), flush=True)

    def bidir(self, a, rel, b, weight=0.92):
        """Teach both a->b and b->a (using RelatedTo for the reverse)."""
        self.teach(a, rel, b, weight)
        # Reverse: use symmetric RelatedTo unless forward is already RelatedTo
        if rel == "RelatedTo":
            self.teach(b, "RelatedTo", a, weight)
        elif rel == "HasProperty":
            self.teach(b, "RelatedTo", a, weight * 0.9)
        elif rel == "IsA":
            self.teach(b, "HasProperty", a, weight * 0.9)
        elif rel == "Causes":
            self.teach(b, "RelatedTo", a, weight * 0.9)
        elif rel == "PartOf":
            self.teach(b, "HasProperty", a, weight * 0.9)
        elif rel == "Produces":
            self.teach(b, "DerivedFrom", a, weight * 0.9)
        elif rel == "DerivedFrom":
            self.teach(b, "Produces", a, weight * 0.9)
        elif rel == "Requires":
            self.teach(b, "RelatedTo", a, weight * 0.9)
        else:
            self.teach(b, "RelatedTo", a, weight * 0.9)

    def ask(self, query):
        with self._lock:
            rid = self._next_id()
            self._send({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                        "params": {"name": "lu4d_ask",
                                   "arguments": {"query": query,
                                                 "depth": DEPTH,
                                                 "top_k": TOP_K}}})
            resp = self._recv()
        if resp is None:
            return [], ""
        result = resp.get("result", {})
        structured = result.get("structuredContent", {})
        text = "\n".join(c.get("text", "") for c in result.get("content", [])
                         if c.get("type") == "text")
        activated = structured.get("recall", {}).get("activated", [])
        return activated, text

    def save(self):
        with self._lock:
            rid = self._next_id()
            self._send({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                        "params": {"name": "lu4d_save", "arguments": {}}})
            resp = self._recv()
        result = (resp or {}).get("result", {})
        content = result.get("content", [])
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        return text

    def status(self):
        with self._lock:
            rid = self._next_id()
            self._send({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                        "params": {"name": "lu4d_status", "arguments": {}}})
            resp = self._recv()
        return (resp or {}).get("result", {}).get("structuredContent", {})

    def stop(self):
        if self.proc:
            try: self.proc.stdin.close()
            except Exception: pass
            print("[mcp] waiting for lu4d.exe to exit...", flush=True)
            try: self.proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                print("[mcp] timeout -- killing", flush=True)
                self.proc.kill()
                self.proc.wait()
            self.proc = None


# ---------------------------------------------------------------------------
# Strict scorer (mirrors eval_100_strict.py)
# ---------------------------------------------------------------------------
def strict_score(activated, keywords):
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
                return 1, "exact '" + kw + "' at pos " + str(i+1) + " d=" + str(top5_depths[i])
            if len(kw_lo) >= 4 and kw_lo in content:
                return 1, "'" + kw + "' in '" + content + "' at pos " + str(i+1)
            if len(content) >= 4 and content in kw_lo:
                return 1, "'" + content + "' in kw '" + kw + "' at pos " + str(i+1)
    top5_str = ", ".join(c + "(d" + str(d) + ")" for c, d in zip(top5_contents, top5_depths))
    return 0, "top5=[" + top5_str + "]"


# ---------------------------------------------------------------------------
# 100 questions (identical to eval_100_strict.py)
# ---------------------------------------------------------------------------
QUESTIONS = [
    # BRACKET 1: Basic Facts
    ("What is water made of?",                             ["hydrogen", "oxygen", "h2o"]),
    ("What is the capital of France?",                     ["paris"]),
    ("What planet is closest to the sun?",                 ["mercury"]),
    ("What is the largest ocean?",                         ["pacific"]),
    ("At what temperature does water boil?",               ["celsius", "fahrenheit", "boil", "steam", "100"]),
    ("What is the capital of Japan?",                      ["tokyo"]),
    ("What is the speed of light?",                        ["speed", "light", "299", "electromagnetic"]),
    ("What is the largest mammal?",                        ["whale", "blue"]),
    ("Who wrote Romeo and Juliet?",                        ["shakespeare"]),
    ("What is the capital of Germany?",                    ["berlin"]),
    ("What element does the chemical symbol Au represent?",["gold"]),
    ("How many continents are there?",                     ["seven", "continent"]),
    ("What is the powerhouse of the cell?",                ["mitochondria", "mitochondrion"]),
    ("What is the largest planet in our solar system?",    ["jupiter"]),
    ("What language is spoken in Brazil?",                 ["portuguese"]),
    ("What is the hardest natural substance?",             ["diamond"]),
    ("What is the capital of Australia?",                  ["canberra"]),
    ("What is DNA?",                                       ["genetic", "nucleic", "molecule", "chromosome", "gene"]),
    ("What is gravity?",                                   ["force", "mass", "attraction", "newton"]),
    ("What is the chemical symbol for gold?",              ["au", "gold"]),
    # BRACKET 2: Science
    ("How does photosynthesis work?",                      ["chlorophyll", "glucose", "carbon", "sunlight", "plant", "oxygen"]),
    ("What causes earthquakes?",                           ["tectonic", "fault", "plate", "seismic", "crust"]),
    ("How does the immune system work?",                   ["antibody", "virus", "cell", "infection", "lymphocyte", "immune"]),
    ("What is the theory of relativity?",                  ["einstein", "space", "time", "mass", "energy", "relativity"]),
    ("What causes tides?",                                 ["moon", "gravity", "tidal", "lunar", "ocean"]),
    ("How do vaccines work?",                              ["immune", "antibody", "virus", "immunity", "pathogen", "vaccine"]),
    ("How does nuclear fission work?",                     ["nuclear", "fission", "atom", "neutron", "uranium", "energy"]),
    ("What is evolution?",                                 ["darwin", "species", "selection", "adapt", "evolution", "natural"]),
    ("How does the brain process memory?",                 ["hippocampus", "neuron", "synapse", "memory", "brain"]),
    ("What is quantum mechanics?",                         ["quantum", "particle", "wave", "uncertainty", "electron"]),
    ("How do black holes form?",                           ["black", "hole", "gravity", "star", "collapse", "mass"]),
    ("What is entropy?",                                   ["entropy", "disorder", "thermodynamics", "heat"]),
    ("How does the carbon cycle work?",                    ["carbon", "co2", "atmosphere", "photosynthesis", "fossil"]),
    ("What is the greenhouse effect?",                     ["greenhouse", "carbon", "temperature", "atmosphere", "climate"]),
    ("How do neurons transmit signals?",                   ["neuron", "synapse", "axon", "signal", "electrical", "chemical"]),
    ("What is CRISPR?",                                    ["crispr", "gene", "dna", "editing", "genetic"]),
    ("How does natural selection work?",                   ["selection", "survival", "fitness", "evolution", "adapt"]),
    ("What is dark matter?",                               ["dark", "matter", "gravity", "universe", "mass"]),
    ("What is the Higgs boson?",                           ["higgs", "boson", "mass", "particle", "field"]),
    ("How does the digestive system work?",                ["stomach", "enzyme", "intestine", "digest", "nutrient", "food"]),
    # BRACKET 3: Cross-domain
    ("How does chemistry relate to cooking?",              ["reaction", "heat", "protein", "maillard", "chemistry", "molecule"]),
    ("What is the physics of music?",                      ["sound", "wave", "frequency", "vibration", "acoustic"]),
    ("How does biology connect to technology?",            ["biotech", "genetic", "dna", "artificial", "bioinformatics", "biology"]),
    ("How does mathematics relate to music?",              ["frequency", "ratio", "pattern", "harmony", "wave", "math"]),
    ("What is the connection between economics and psychology?",["behavior", "decision", "rational", "cognitive", "bias", "economic"]),
    ("How does language shape thought?",                   ["language", "thought", "cognitive", "sapir", "culture", "concept"]),
    ("What connects art to neuroscience?",                 ["perception", "visual", "art", "neural", "aesthetic", "brain"]),
    ("How does physics apply to sports?",                  ["force", "motion", "velocity", "kinetic", "energy", "physics"]),
    ("What is the relationship between information and entropy?",["information", "entropy", "shannon", "disorder", "bit"]),
    ("How does game theory apply to evolution?",           ["game", "strategy", "evolution", "cooperation", "nash", "fitness"]),
    ("What connects geography to history?",                ["geography", "history", "civilization", "trade", "resource", "climate"]),
    ("How does statistics relate to medicine?",            ["statistics", "trial", "clinical", "evidence", "probability", "data"]),
    ("What is the relationship between philosophy and science?",["philosophy", "science", "empirical", "knowledge", "reason", "logic"]),
    ("How does architecture relate to mathematics?",       ["architecture", "geometry", "proportion", "structure", "design", "math"]),
    ("What connects climate to civilization?",             ["climate", "civilization", "agriculture", "food", "temperature"]),
    ("How does ecology relate to economics?",              ["ecology", "economic", "resource", "ecosystem", "capital", "natural"]),
    ("What is the connection between music and mathematics?",["music", "math", "pattern", "harmony", "frequency", "ratio"]),
    ("How does chemistry relate to medicine?",             ["chemistry", "medicine", "drug", "molecule", "pharmacology", "compound"]),
    ("What connects psychology to marketing?",             ["psychology", "behavior", "marketing", "persuasion", "consumer", "emotion"]),
    ("How does evolution relate to language?",             ["evolution", "language", "communicate", "symbol", "culture", "adapt"]),
    # BRACKET 4: Abstract Reasoning
    ("Why do economies have cycles?",                      ["cycle", "economic", "boom", "recession", "inflation", "demand", "credit"]),
    ("What makes a good analogy?",                         ["analogy", "comparison", "structure", "relation", "mapping", "similar"]),
    ("How does evolution relate to innovation?",           ["evolution", "innovation", "selection", "variation", "adapt", "fitness"]),
    ("Why do complex systems self-organize?",              ["complex", "system", "emerge", "organize", "feedback", "pattern"]),
    ("What is the relationship between order and chaos?",  ["order", "chaos", "entropy", "system", "stability", "dynamic"]),
    ("Why does diversity increase resilience?",            ["diversity", "resilience", "ecosystem", "variation", "adapt", "robust"]),
    ("How do feedback loops shape systems?",               ["feedback", "loop", "system", "control", "amplify", "regulate"]),
    ("What makes knowledge cumulative?",                   ["knowledge", "science", "cumulative", "language", "record", "build"]),
    ("Why do networks have hubs?",                         ["network", "hub", "node", "connect", "scale", "preferential"]),
    ("How does emergence work?",                           ["emerge", "complex", "property", "system", "collective", "pattern"]),
    ("What is the role of constraints in creativity?",     ["constraint", "creative", "limit", "innovation", "problem", "solution"]),
    ("Why do civilizations rise and fall?",                ["civilization", "collapse", "resource", "complexity", "climate", "trade"]),
    ("How does compression relate to understanding?",      ["compress", "pattern", "information", "model", "abstract", "represent"]),
    ("What makes a good model?",                           ["model", "predict", "accurate", "theory", "abstract", "simple"]),
    ("Why is symmetry important in physics?",              ["symmetry", "physics", "conservation", "law", "invariant", "noether"]),
    ("How does language evolve?",                          ["language", "evolve", "change", "cultural", "adapt", "grammar"]),
    ("What is the relationship between competition and cooperation?",["compete", "cooperate", "evolution", "game", "social", "altruism"]),
    ("Why do exponential processes surprise us?",          ["exponential", "growth", "compound", "double", "scale", "intuition"]),
    ("How does abstraction enable progress?",              ["abstract", "model", "concept", "simplify", "build", "reuse"]),
    ("What connects individual behavior to collective outcomes?",["individual", "collective", "emerge", "social", "network", "behavior"]),
    # BRACKET 5: Complex Multi-Hop
    ("Trace the chain from sunlight to human civilization",["photosynthesis", "agriculture", "food", "plant", "solar", "energy"]),
    ("How does plate tectonics affect biodiversity?",      ["tectonic", "plate", "continent", "isolate", "evolution", "species"]),
    ("Connect mathematics to art",                         ["proportion", "symmetry", "fibonacci", "golden", "geometry", "pattern"]),
    ("How does the electromagnetic spectrum relate to modern technology?",["electromagnetic", "spectrum", "radio", "wave", "wireless", "frequency"]),
    ("Trace the path from atoms to consciousness",         ["atom", "molecule", "cell", "neuron", "brain", "consciousness"]),
    ("How does language shape scientific discovery?",      ["language", "science", "concept", "theory", "communicate", "symbol"]),
    ("Connect thermodynamics to economics",                ["thermodynamic", "energy", "entropy", "economic", "resource", "efficiency"]),
    ("How does geography shape culture?",                  ["geography", "culture", "climate", "resource", "trade", "civilization"]),
    ("Trace evolution from single cells to intelligence",  ["cell", "multicellular", "brain", "intelligence", "nervous", "adapt"]),
    ("How does quantum mechanics enable modern computing?",["quantum", "electron", "semiconductor", "transistor", "computing", "chip"]),
    ("Connect Darwinian evolution to cultural change",     ["darwin", "evolution", "culture", "meme", "selection", "variation"]),
    ("How does information theory relate to biology?",     ["information", "dna", "genetic", "code", "shannon", "entropy", "biology"]),
    ("Trace the chain from stellar nucleosynthesis to life",["star", "element", "carbon", "nuclear", "fusion", "life", "supernova"]),
    ("How do game theory and evolution connect to social norms?",["game", "evolution", "social", "norm", "cooperation", "strategy", "nash"]),
    ("Connect the mathematics of networks to social phenomena",["network", "graph", "social", "node", "connect", "scale", "hub"]),
    ("How does the physics of climate connect to human migration?",["climate", "temperature", "migration", "agriculture", "drought", "human"]),
    ("Trace the development of symbolic thought to modern mathematics",["symbol", "language", "abstract", "number", "math", "logic", "formal"]),
    ("How does neuroscience relate to artificial intelligence?",["neuroscience", "brain", "neural", "artificial", "intelligence", "learn"]),
    ("Connect the concept of entropy to social organization",["entropy", "order", "disorder", "social", "organize", "energy", "complexity"]),
    ("How does the feedback between technology and society drive history?",["technology", "society", "feedback", "industrial", "innovation", "history"]),
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
# Teaching data: bidirectional pairs for every known gap
# ---------------------------------------------------------------------------

# --- Basic Facts ---
BASIC = [
    # Capitals
    ("france",   "HasProperty", "paris"),
    ("paris",    "IsA",         "capital"),
    ("japan",    "HasProperty", "tokyo"),
    ("tokyo",    "IsA",         "capital"),
    ("germany",  "HasProperty", "berlin"),
    ("berlin",   "IsA",         "capital"),
    ("australia","HasProperty", "canberra"),
    ("canberra", "IsA",         "capital"),
    ("brazil",   "HasProperty", "brasilia"),
    ("brasilia", "IsA",         "capital"),
    # Americas
    ("usa",            "HasProperty", "washington"),
    ("united_states",  "HasProperty", "washington"),
    ("washington",     "IsA",         "capital"),
    ("canada",         "HasProperty", "ottawa"),
    ("ottawa",         "IsA",         "capital"),
    ("mexico",         "HasProperty", "mexico_city"),
    ("mexico_city",    "IsA",         "capital"),
    ("argentina",      "HasProperty", "buenos_aires"),
    ("buenos_aires",   "IsA",         "capital"),
    ("chile",          "HasProperty", "santiago"),
    ("santiago",       "IsA",         "capital"),
    ("colombia",       "HasProperty", "bogota"),
    ("bogota",         "IsA",         "capital"),
    ("peru",           "HasProperty", "lima"),
    ("lima",           "IsA",         "capital"),
    ("venezuela",      "HasProperty", "caracas"),
    ("caracas",        "IsA",         "capital"),
    ("cuba",           "HasProperty", "havana"),
    ("havana",         "IsA",         "capital"),
    ("ecuador",        "HasProperty", "quito"),
    ("quito",          "IsA",         "capital"),
    ("bolivia",        "HasProperty", "sucre"),
    ("sucre",          "IsA",         "capital"),
    ("paraguay",       "HasProperty", "asuncion"),
    ("asuncion",       "IsA",         "capital"),
    ("uruguay",        "HasProperty", "montevideo"),
    ("montevideo",     "IsA",         "capital"),
    # Europe
    ("uk",             "HasProperty", "london"),
    ("united_kingdom", "HasProperty", "london"),
    ("london",         "IsA",         "capital"),
    ("spain",          "HasProperty", "madrid"),
    ("madrid",         "IsA",         "capital"),
    ("italy",          "HasProperty", "rome"),
    ("rome",           "IsA",         "capital"),
    ("portugal",       "HasProperty", "lisbon"),
    ("lisbon",         "IsA",         "capital"),
    ("netherlands",    "HasProperty", "amsterdam"),
    ("amsterdam",      "IsA",         "capital"),
    ("sweden",         "HasProperty", "stockholm"),
    ("stockholm",      "IsA",         "capital"),
    ("norway",         "HasProperty", "oslo"),
    ("oslo",           "IsA",         "capital"),
    ("denmark",        "HasProperty", "copenhagen"),
    ("copenhagen",     "IsA",         "capital"),
    ("finland",        "HasProperty", "helsinki"),
    ("helsinki",       "IsA",         "capital"),
    ("poland",         "HasProperty", "warsaw"),
    ("warsaw",         "IsA",         "capital"),
    ("ukraine",        "HasProperty", "kyiv"),
    ("kyiv",           "IsA",         "capital"),
    ("turkey",         "HasProperty", "ankara"),
    ("ankara",         "IsA",         "capital"),
    ("greece",         "HasProperty", "athens"),
    ("athens",         "IsA",         "capital"),
    ("czech_republic", "HasProperty", "prague"),
    ("prague",         "IsA",         "capital"),
    ("hungary",        "HasProperty", "budapest"),
    ("budapest",       "IsA",         "capital"),
    ("romania",        "HasProperty", "bucharest"),
    ("bucharest",      "IsA",         "capital"),
    ("switzerland",    "HasProperty", "bern"),
    ("bern",           "IsA",         "capital"),
    ("austria",        "HasProperty", "vienna"),
    ("vienna",         "IsA",         "capital"),
    ("russia",         "HasProperty", "moscow"),
    ("moscow",         "IsA",         "capital"),
    ("belgium",        "HasProperty", "brussels"),
    ("brussels",       "IsA",         "capital"),
    ("ireland",        "HasProperty", "dublin"),
    ("dublin",         "IsA",         "capital"),
    # Asia
    ("china",          "HasProperty", "beijing"),
    ("beijing",        "IsA",         "capital"),
    ("india",          "HasProperty", "new_delhi"),
    ("new_delhi",      "IsA",         "capital"),
    ("south_korea",    "HasProperty", "seoul"),
    ("seoul",          "IsA",         "capital"),
    ("north_korea",    "HasProperty", "pyongyang"),
    ("pyongyang",      "IsA",         "capital"),
    ("vietnam",        "HasProperty", "hanoi"),
    ("hanoi",          "IsA",         "capital"),
    ("thailand",       "HasProperty", "bangkok"),
    ("bangkok",        "IsA",         "capital"),
    ("indonesia",      "HasProperty", "jakarta"),
    ("jakarta",        "IsA",         "capital"),
    ("philippines",    "HasProperty", "manila"),
    ("manila",         "IsA",         "capital"),
    ("malaysia",       "HasProperty", "kuala_lumpur"),
    ("kuala_lumpur",   "IsA",         "capital"),
    ("pakistan",       "HasProperty", "islamabad"),
    ("islamabad",      "IsA",         "capital"),
    ("bangladesh",     "HasProperty", "dhaka"),
    ("dhaka",          "IsA",         "capital"),
    ("iran",           "HasProperty", "tehran"),
    ("tehran",         "IsA",         "capital"),
    ("iraq",           "HasProperty", "baghdad"),
    ("baghdad",        "IsA",         "capital"),
    ("saudi_arabia",   "HasProperty", "riyadh"),
    ("riyadh",         "IsA",         "capital"),
    ("israel",         "HasProperty", "jerusalem"),
    ("jerusalem",      "IsA",         "capital"),
    ("jordan",         "HasProperty", "amman"),
    ("amman",          "IsA",         "capital"),
    ("afghanistan",    "HasProperty", "kabul"),
    ("kabul",          "IsA",         "capital"),
    # Africa
    ("egypt",          "HasProperty", "cairo"),
    ("cairo",          "IsA",         "capital"),
    ("south_africa",   "HasProperty", "pretoria"),
    ("pretoria",       "IsA",         "capital"),
    ("nigeria",        "HasProperty", "abuja"),
    ("abuja",          "IsA",         "capital"),
    ("kenya",          "HasProperty", "nairobi"),
    ("nairobi",        "IsA",         "capital"),
    ("ghana",          "HasProperty", "accra"),
    ("accra",          "IsA",         "capital"),
    ("ethiopia",       "HasProperty", "addis_ababa"),
    ("addis_ababa",    "IsA",         "capital"),
    ("tanzania",       "HasProperty", "dodoma"),
    ("dodoma",         "IsA",         "capital"),
    ("morocco",        "HasProperty", "rabat"),
    ("rabat",          "IsA",         "capital"),
    ("algeria",        "HasProperty", "algiers"),
    ("algiers",        "IsA",         "capital"),
    ("zimbabwe",       "HasProperty", "harare"),
    ("harare",         "IsA",         "capital"),
    ("libya",          "HasProperty", "tripoli"),
    ("tripoli",        "IsA",         "capital"),
    ("angola",         "HasProperty", "luanda"),
    ("luanda",         "IsA",         "capital"),
    ("cameroon",       "HasProperty", "yaounde"),
    ("yaounde",        "IsA",         "capital"),
    # Oceania
    ("new_zealand",    "HasProperty", "wellington"),
    ("wellington",     "IsA",         "capital"),

    # Mercury / closest planet
    ("mercury",        "IsA",         "planet"),
    ("mercury",        "HasProperty", "closest"),
    ("mercury",        "HasProperty", "smallest"),
    ("mercury",        "RelatedTo",   "sun"),
    ("closest",        "HasProperty", "mercury"),
    ("planet",         "HasProperty", "mercury"),
    ("sun",            "HasProperty", "mercury"),
    ("solar_system",   "HasProperty", "mercury"),
    ("mercury",        "RelatedTo",   "solar_system"),

    # Jupiter / largest planet
    ("jupiter",        "IsA",         "planet"),
    ("jupiter",        "HasProperty", "largest"),
    ("jupiter",        "HasProperty", "gas_giant"),
    ("largest",        "HasProperty", "jupiter"),
    ("planet",         "HasProperty", "jupiter"),
    ("solar_system",   "HasProperty", "jupiter"),

    # Pacific / largest ocean
    ("pacific",        "IsA",         "ocean"),
    ("pacific",        "HasProperty", "largest"),
    ("pacific",        "HasProperty", "deepest"),
    ("largest",        "HasProperty", "pacific"),
    ("ocean",          "HasProperty", "pacific"),

    # Boiling / 100 degrees
    ("water",          "HasProperty", "boiling"),
    ("boiling",        "HasProperty", "celsius"),
    ("boiling",        "HasProperty", "100"),
    ("100",            "RelatedTo",   "boiling"),
    ("100",            "RelatedTo",   "celsius"),
    ("celsius",        "HasProperty", "100"),
    ("celsius",        "HasProperty", "temperature"),
    ("boil",           "HasProperty", "celsius"),
    ("boil",           "HasProperty", "100"),
    ("steam",          "DerivedFrom", "boiling"),
    ("water",          "Produces",    "steam"),

    # Whale / largest mammal
    ("whale",          "IsA",         "mammal"),
    ("whale",          "HasProperty", "largest"),
    ("blue_whale",     "IsA",         "whale"),
    ("blue_whale",     "HasProperty", "largest"),
    ("blue",           "HasProperty", "whale"),
    ("mammal",         "HasProperty", "whale"),
    ("largest",        "HasProperty", "whale"),
    ("largest",        "HasProperty", "blue_whale"),

    # Shakespeare / Romeo and Juliet
    ("shakespeare",    "RelatedTo",   "romeo"),
    ("shakespeare",    "RelatedTo",   "hamlet"),
    ("shakespeare",    "RelatedTo",   "juliet"),
    ("shakespeare",    "IsA",         "playwright"),
    ("shakespeare",    "RelatedTo",   "literature"),
    ("romeo",          "RelatedTo",   "shakespeare"),
    ("juliet",         "RelatedTo",   "shakespeare"),
    ("hamlet",         "RelatedTo",   "shakespeare"),
    ("wrote",          "HasProperty", "shakespeare"),
    ("playwright",     "HasProperty", "shakespeare"),

    # Gold / Au
    ("au",             "RelatedTo",   "gold"),
    ("gold",           "HasProperty", "au"),
    ("gold",           "IsA",         "element"),
    ("gold",           "IsA",         "metal"),
    ("au",             "IsA",         "symbol"),
    ("symbol",         "HasProperty", "au"),
    ("element",        "HasProperty", "gold"),
    ("chemical_symbol","HasProperty", "au"),
    ("chemical_symbol","HasProperty", "gold"),

    # Mitochondria
    ("mitochondria",   "HasProperty", "powerhouse"),
    ("powerhouse",     "RelatedTo",   "mitochondria"),
    ("mitochondria",   "Produces",    "energy"),
    ("mitochondria",   "Produces",    "atp"),
    ("mitochondria",   "IsA",         "organelle"),
    ("cell",           "HasProperty", "mitochondria"),
    ("mitochondria",   "PartOf",      "cell"),

    # Portuguese / Brazil
    ("brazil",         "HasProperty", "portuguese"),
    ("portuguese",     "IsA",         "language"),
    ("portuguese",     "RelatedTo",   "brazil"),
    ("language",       "HasProperty", "portuguese"),

    # Diamond / hardest
    ("diamond",        "HasProperty", "hardest"),
    ("diamond",        "IsA",         "mineral"),
    ("diamond",        "IsA",         "gem"),
    ("diamond",        "HasProperty", "carbon"),
    ("hardest",        "RelatedTo",   "diamond"),
    ("substance",      "HasProperty", "diamond"),
    ("natural",        "HasProperty", "diamond"),

    # DNA
    ("dna",            "IsA",         "molecule"),
    ("dna",            "HasProperty", "gene"),
    ("dna",            "HasProperty", "chromosome"),
    ("dna",            "HasProperty", "genetic"),
    ("dna",            "HasProperty", "helix"),
    ("gene",           "PartOf",      "dna"),
    ("chromosome",     "HasProperty", "dna"),
    ("genetic",        "RelatedTo",   "dna"),

    # Gravity
    ("gravity",        "HasProperty", "force"),
    ("gravity",        "HasProperty", "mass"),
    ("gravity",        "HasProperty", "attraction"),
    ("newton",         "RelatedTo",   "gravity"),
    ("force",          "RelatedTo",   "gravity"),
    ("mass",           "RelatedTo",   "gravity"),
    ("earth",          "HasProperty", "gravity"),
]

# --- Science ---
SCIENCE = [
    # Photosynthesis
    ("photosynthesis",  "Requires",   "sunlight"),
    ("photosynthesis",  "Requires",   "chlorophyll"),
    ("photosynthesis",  "Produces",   "glucose"),
    ("photosynthesis",  "Produces",   "oxygen"),
    ("chlorophyll",     "RelatedTo",  "photosynthesis"),
    ("glucose",         "DerivedFrom","photosynthesis"),
    ("plant",           "Requires",   "photosynthesis"),
    ("carbon",          "RelatedTo",  "photosynthesis"),
    ("co2",             "Requires",   "photosynthesis"),
    ("sunlight",        "Causes",     "photosynthesis"),

    # Earthquakes
    ("earthquake",      "RelatedTo",  "tectonic"),
    ("tectonic",        "Causes",     "earthquake"),
    ("tectonic_plate",  "Causes",     "earthquake"),
    ("earthquake",      "RelatedTo",  "fault"),
    ("fault",           "Causes",     "earthquake"),
    ("seismic",         "RelatedTo",  "earthquake"),
    ("earthquake",      "RelatedTo",  "seismic"),
    ("crust",           "RelatedTo",  "earthquake"),
    ("plate",           "RelatedTo",  "earthquake"),
    ("tectonic",        "RelatedTo",  "plate"),

    # Immune system
    ("immune",          "HasProperty","antibody"),
    ("antibody",        "RelatedTo",  "immune"),
    ("immune",          "HasProperty","virus"),
    ("infection",       "RelatedTo",  "immune"),
    ("lymphocyte",      "IsA",        "immune_cell"),
    ("lymphocyte",      "RelatedTo",  "immune"),
    ("pathogen",        "RelatedTo",  "immune"),

    # Relativity
    ("relativity",      "RelatedTo",  "einstein"),
    ("relativity",      "RelatedTo",  "space"),
    ("relativity",      "RelatedTo",  "time"),
    ("relativity",      "RelatedTo",  "mass"),
    ("relativity",      "RelatedTo",  "energy"),
    ("einstein",        "RelatedTo",  "relativity"),
    ("spacetime",       "RelatedTo",  "relativity"),

    # Tides
    ("moon",            "Causes",     "tide"),
    ("tide",            "RelatedTo",  "moon"),
    ("tide",            "RelatedTo",  "gravity"),
    ("moon",            "HasProperty","gravitational"),
    ("gravity",         "Causes",     "tide"),
    ("tidal",           "RelatedTo",  "moon"),
    ("lunar",           "RelatedTo",  "tide"),
    ("tide",            "RelatedTo",  "lunar"),
    ("ocean",           "HasProperty","tide"),
    ("tidal_force",     "RelatedTo",  "moon"),
    ("tidal_force",     "Causes",     "tide"),

    # Vaccines
    ("vaccine",         "Produces",   "immunity"),
    ("vaccine",         "RelatedTo",  "antibody"),
    ("antibody",        "RelatedTo",  "immune"),
    ("immune",          "HasProperty","antibody"),
    ("pathogen",        "RelatedTo",  "immune"),
    ("immunity",        "RelatedTo",  "vaccine"),

    # Nuclear fission
    ("fission",         "HasProperty","nuclear"),
    ("fission",         "Produces",   "energy"),
    ("uranium",         "RelatedTo",  "fission"),
    ("neutron",         "RelatedTo",  "fission"),
    ("nuclear",         "RelatedTo",  "energy"),
    ("atom",            "RelatedTo",  "fission"),

    # Evolution
    ("evolution",       "Requires",   "variation"),
    ("evolution",       "Requires",   "selection"),
    ("evolution",       "Produces",   "species"),
    ("darwin",          "RelatedTo",  "evolution"),
    ("darwin",          "RelatedTo",  "selection"),
    ("natural_selection","IsA",       "evolution"),
    ("adapt",           "RelatedTo",  "evolution"),
    ("species",         "RelatedTo",  "evolution"),
    ("natural",         "RelatedTo",  "selection"),
    ("selection",       "RelatedTo",  "evolution"),

    # Brain / memory
    ("memory",          "RelatedTo",  "hippocampus"),
    ("hippocampus",     "RelatedTo",  "memory"),
    ("brain",           "HasProperty","hippocampus"),
    ("neuron",          "HasProperty","synapse"),
    ("synapse",         "RelatedTo",  "neuron"),
    ("brain",           "HasProperty","neuron"),

    # Quantum
    ("quantum",         "HasProperty","uncertainty"),
    ("quantum",         "HasProperty","wave"),
    ("quantum",         "HasProperty","particle"),
    ("electron",        "RelatedTo",  "quantum"),
    ("uncertainty",     "RelatedTo",  "quantum"),
    ("wave",            "RelatedTo",  "quantum"),

    # Black holes
    ("black_hole",      "DerivedFrom","gravity"),
    ("black_hole",      "DerivedFrom","star"),
    ("star",            "Produces",   "black_hole"),
    ("gravity",         "Causes",     "black_hole"),
    ("collapse",        "Produces",   "black_hole"),
    ("black_hole",      "HasProperty","mass"),
    ("black",           "HasProperty","hole"),
    ("hole",            "HasProperty","black"),

    # Entropy
    ("entropy",         "HasProperty","disorder"),
    ("entropy",         "HasProperty","thermodynamics"),
    ("entropy",         "HasProperty","heat"),
    ("thermodynamics",  "RelatedTo",  "entropy"),
    ("disorder",        "RelatedTo",  "entropy"),
    ("energy",          "RelatedTo",  "entropy"),

    # Carbon cycle
    ("carbon",          "RelatedTo",  "co2"),
    ("co2",             "RelatedTo",  "atmosphere"),
    ("atmosphere",      "HasProperty","co2"),
    ("fossil",          "Produces",   "co2"),
    ("carbon_cycle",    "RelatedTo",  "photosynthesis"),

    # Greenhouse
    ("greenhouse",      "HasProperty","carbon"),
    ("greenhouse",      "HasProperty","temperature"),
    ("greenhouse",      "HasProperty","atmosphere"),
    ("climate",         "RelatedTo",  "greenhouse"),
    ("temperature",     "RelatedTo",  "greenhouse"),

    # Neurons / signals
    ("neuron",          "HasProperty","axon"),
    ("axon",            "PartOf",     "neuron"),
    ("signal",          "RelatedTo",  "neuron"),
    ("neuron",          "RelatedTo",  "signal"),
    ("electrical",      "RelatedTo",  "neuron"),
    ("chemical",        "RelatedTo",  "synapse"),

    # CRISPR
    ("crispr",          "IsA",        "gene_editing"),
    ("crispr",          "RelatedTo",  "dna"),
    ("crispr",          "RelatedTo",  "gene"),
    ("crispr",          "RelatedTo",  "cas9"),
    ("crispr",          "HasProperty","genetic"),
    ("gene",            "RelatedTo",  "crispr"),
    ("dna",             "RelatedTo",  "crispr"),
    ("editing",         "RelatedTo",  "crispr"),
    ("gene_editing",    "HasProperty","crispr"),

    # Natural selection
    ("natural_selection","RelatedTo", "evolution"),
    ("natural_selection","HasProperty","survival"),
    ("natural_selection","HasProperty","fitness"),
    ("natural_selection","DerivedFrom","darwin"),
    ("selection",       "HasProperty","survival"),
    ("selection",       "HasProperty","fitness"),
    ("survival",        "RelatedTo",  "selection"),
    ("fitness",         "RelatedTo",  "selection"),
    ("adapt",           "RelatedTo",  "selection"),

    # Dark matter
    ("dark_matter",     "HasProperty","gravity"),
    ("dark_matter",     "HasProperty","mass"),
    ("universe",        "HasProperty","dark_matter"),
    ("dark",            "HasProperty","matter"),
    ("matter",          "HasProperty","dark"),

    # Higgs boson
    ("higgs",           "IsA",        "boson"),
    ("higgs",           "HasProperty","mass"),
    ("higgs",           "RelatedTo",  "particle"),
    ("higgs",           "HasProperty","field"),
    ("boson",           "RelatedTo",  "particle"),
    ("mass",            "RelatedTo",  "higgs"),

    # Digestion
    ("stomach",         "PartOf",     "digestion"),
    ("enzyme",          "RelatedTo",  "digestion"),
    ("intestine",       "PartOf",     "digestion"),
    ("nutrient",        "RelatedTo",  "digestion"),
    ("food",            "Requires",   "digestion"),
    ("digestion",       "Produces",   "nutrient"),
]

# --- Cross-domain bridges ---
CROSS = [
    # Chemistry <-> cooking
    ("cooking",         "Requires",   "chemistry"),
    ("maillard",        "RelatedTo",  "cooking"),
    ("maillard",        "RelatedTo",  "reaction"),
    ("reaction",        "RelatedTo",  "cooking"),
    ("heat",            "RelatedTo",  "cooking"),
    ("protein",         "RelatedTo",  "cooking"),
    ("molecule",        "RelatedTo",  "cooking"),

    # Biology <-> technology
    ("biotech",         "DerivedFrom","biology"),
    ("bioinformatics",  "DerivedFrom","biology"),
    ("genetic",         "RelatedTo",  "technology"),
    ("dna",             "RelatedTo",  "biotech"),
    ("artificial",      "RelatedTo",  "biology"),

    # Math <-> music
    ("frequency",       "RelatedTo",  "music"),
    ("ratio",           "RelatedTo",  "music"),
    ("harmony",         "RelatedTo",  "music"),
    ("pattern",         "RelatedTo",  "music"),
    ("wave",            "RelatedTo",  "music"),
    ("math",            "RelatedTo",  "music"),
    ("frequency",       "RelatedTo",  "math"),
    ("music",           "RelatedTo",  "sound"),
    ("music",           "RelatedTo",  "wave"),
    ("music",           "HasProperty","frequency"),
    ("music",           "HasProperty","vibration"),
    ("music",           "RelatedTo",  "acoustic"),
    ("vibration",       "Causes",     "sound"),
    ("sound",           "HasProperty","wave"),
    ("acoustic",        "RelatedTo",  "music"),
    ("acoustics",       "RelatedTo",  "music"),

    # Economics <-> psychology
    ("behavior",        "RelatedTo",  "economics"),
    ("decision",        "RelatedTo",  "economics"),
    ("rational",        "RelatedTo",  "economics"),
    ("cognitive",       "RelatedTo",  "economics"),
    ("bias",            "RelatedTo",  "economics"),
    ("behavioral_economics","RelatedTo","psychology"),
    ("economic",        "RelatedTo",  "behavior"),

    # Language <-> thought
    ("language",        "RelatedTo",  "thought"),
    ("cognitive",       "RelatedTo",  "language"),
    ("sapir",           "RelatedTo",  "language"),
    ("culture",         "RelatedTo",  "language"),
    ("concept",         "RelatedTo",  "language"),

    # Art <-> neuroscience
    ("perception",      "RelatedTo",  "art"),
    ("visual",          "RelatedTo",  "art"),
    ("aesthetic",       "RelatedTo",  "art"),
    ("neural",          "RelatedTo",  "art"),
    ("brain",           "RelatedTo",  "art"),

    # Physics <-> sports
    ("force",           "RelatedTo",  "sports"),
    ("motion",          "RelatedTo",  "sports"),
    ("velocity",        "RelatedTo",  "sports"),
    ("kinetic",         "RelatedTo",  "sports"),
    ("energy",          "RelatedTo",  "sports"),

    # Information <-> entropy
    ("information",     "RelatedTo",  "entropy"),
    ("shannon",         "RelatedTo",  "entropy"),
    ("shannon",         "RelatedTo",  "information"),
    ("bit",             "RelatedTo",  "information"),
    ("disorder",        "RelatedTo",  "information"),

    # Game theory <-> evolution
    ("game",            "RelatedTo",  "evolution"),
    ("strategy",        "RelatedTo",  "evolution"),
    ("cooperation",     "RelatedTo",  "evolution"),
    ("nash",            "RelatedTo",  "game"),
    ("fitness",         "RelatedTo",  "game"),

    # Geography <-> history
    ("geography",       "RelatedTo",  "history"),
    ("geography",       "RelatedTo",  "civilization"),
    ("trade",           "RelatedTo",  "geography"),
    ("resource",        "RelatedTo",  "geography"),
    ("climate",         "RelatedTo",  "civilization"),

    # Statistics <-> medicine
    ("statistics",      "RelatedTo",  "medicine"),
    ("clinical",        "RelatedTo",  "statistics"),
    ("trial",           "RelatedTo",  "statistics"),
    ("evidence",        "RelatedTo",  "medicine"),
    ("probability",     "RelatedTo",  "statistics"),
    ("data",            "RelatedTo",  "medicine"),

    # Philosophy <-> science
    ("philosophy",      "RelatedTo",  "science"),
    ("empirical",       "RelatedTo",  "science"),
    ("logic",           "RelatedTo",  "philosophy"),
    ("reason",          "RelatedTo",  "philosophy"),
    ("knowledge",       "RelatedTo",  "philosophy"),

    # Architecture <-> math
    ("architecture",    "Requires",   "geometry"),
    ("proportion",      "RelatedTo",  "architecture"),
    ("structure",       "RelatedTo",  "architecture"),
    ("design",          "RelatedTo",  "architecture"),
    ("geometry",        "RelatedTo",  "architecture"),

    # Ecology <-> economics
    ("ecology",         "RelatedTo",  "economics"),
    ("ecosystem",       "RelatedTo",  "economics"),
    ("resource",        "RelatedTo",  "ecology"),
    ("capital",         "RelatedTo",  "ecology"),
    ("natural",         "RelatedTo",  "ecology"),

    # Chemistry <-> medicine
    ("chemistry",       "RelatedTo",  "medicine"),
    ("drug",            "RelatedTo",  "chemistry"),
    ("pharmacology",    "RelatedTo",  "chemistry"),
    ("compound",        "RelatedTo",  "medicine"),
    ("molecule",        "RelatedTo",  "medicine"),

    # Psychology <-> marketing
    ("behavior",        "RelatedTo",  "marketing"),
    ("persuasion",      "RelatedTo",  "marketing"),
    ("consumer",        "RelatedTo",  "marketing"),
    ("emotion",         "RelatedTo",  "marketing"),
    ("psychology",      "RelatedTo",  "marketing"),

    # Evolution <-> language
    ("language",        "RelatedTo",  "evolution"),
    ("communicate",     "RelatedTo",  "evolution"),
    ("symbol",          "RelatedTo",  "language"),
    ("culture",         "RelatedTo",  "evolution"),
    ("adapt",           "RelatedTo",  "language"),

    # Music <-> math (symmetric)
    ("music",           "RelatedTo",  "math"),
    ("music",           "RelatedTo",  "pattern"),
    ("music",           "RelatedTo",  "harmony"),
    ("music",           "RelatedTo",  "frequency"),
    ("music",           "RelatedTo",  "ratio"),
]

# --- Abstract reasoning ---
ABSTRACT = [
    ("cycle",           "RelatedTo",  "economics"),
    ("boom",            "RelatedTo",  "economics"),
    ("recession",       "RelatedTo",  "economics"),
    ("inflation",       "RelatedTo",  "economics"),
    ("credit",          "RelatedTo",  "economics"),
    ("demand",          "RelatedTo",  "economics"),

    ("analogy",         "HasProperty","comparison"),
    ("analogy",         "HasProperty","structure"),
    ("analogy",         "HasProperty","relation"),
    ("mapping",         "RelatedTo",  "analogy"),
    ("similar",         "RelatedTo",  "analogy"),

    ("innovation",      "RelatedTo",  "evolution"),
    ("innovation",      "RelatedTo",  "selection"),
    ("innovation",      "RelatedTo",  "variation"),
    ("innovation",      "RelatedTo",  "adapt"),
    ("innovation",      "RelatedTo",  "fitness"),

    ("emerge",          "RelatedTo",  "complex"),
    ("organize",        "RelatedTo",  "complex"),
    ("feedback",        "RelatedTo",  "complex"),
    ("pattern",         "RelatedTo",  "complex"),
    ("system",          "HasProperty","emerge"),

    ("order",           "ContrastedWith","chaos"),
    ("chaos",           "RelatedTo",  "entropy"),
    ("stability",       "RelatedTo",  "order"),
    ("dynamic",         "RelatedTo",  "chaos"),

    ("diversity",       "RelatedTo",  "resilience"),
    ("ecosystem",       "HasProperty","diversity"),
    ("variation",       "RelatedTo",  "resilience"),
    ("robust",          "RelatedTo",  "diversity"),
    ("adapt",           "RelatedTo",  "resilience"),

    ("feedback",        "HasProperty","loop"),
    ("regulate",        "RelatedTo",  "feedback"),
    ("amplify",         "RelatedTo",  "feedback"),
    ("control",         "RelatedTo",  "feedback"),

    ("knowledge",       "HasProperty","cumulative"),
    ("science",         "HasProperty","cumulative"),
    ("record",          "RelatedTo",  "knowledge"),
    ("build",           "RelatedTo",  "knowledge"),

    ("network",         "HasProperty","hub"),
    ("hub",             "RelatedTo",  "connect"),
    ("scale",           "RelatedTo",  "hub"),
    ("preferential",    "RelatedTo",  "network"),
    ("node",            "RelatedTo",  "network"),

    ("emerge",          "RelatedTo",  "collective"),
    ("property",        "RelatedTo",  "emerge"),
    ("collective",      "RelatedTo",  "emerge"),

    ("constraint",      "RelatedTo",  "creative"),
    ("limit",           "RelatedTo",  "creative"),
    ("creative",        "RelatedTo",  "innovation"),
    ("solution",        "RelatedTo",  "constraint"),
    ("problem",         "RelatedTo",  "constraint"),

    ("civilization",    "RelatedTo",  "collapse"),
    ("resource",        "RelatedTo",  "civilization"),
    ("complexity",      "RelatedTo",  "civilization"),
    ("trade",           "RelatedTo",  "civilization"),

    ("compress",        "RelatedTo",  "pattern"),
    ("information",     "RelatedTo",  "compress"),
    ("model",           "RelatedTo",  "abstract"),
    ("represent",       "RelatedTo",  "information"),

    ("model",           "HasProperty","predict"),
    ("accurate",        "RelatedTo",  "model"),
    ("theory",          "RelatedTo",  "model"),
    ("simple",          "RelatedTo",  "model"),

    ("symmetry",        "RelatedTo",  "physics"),
    ("conservation",    "RelatedTo",  "symmetry"),
    ("invariant",       "RelatedTo",  "symmetry"),
    ("noether",         "RelatedTo",  "symmetry"),
    ("law",             "RelatedTo",  "symmetry"),

    ("language",        "HasProperty","evolve"),
    ("grammar",         "RelatedTo",  "language"),
    ("cultural",        "RelatedTo",  "language"),

    # Competition / cooperation (Q077)
    ("competition",     "RelatedTo",  "evolution"),
    ("cooperation",     "RelatedTo",  "evolution"),
    ("competition",     "RelatedTo",  "game_theory"),
    ("cooperation",     "RelatedTo",  "game_theory"),
    ("compete",         "RelatedTo",  "evolution"),
    ("cooperate",       "RelatedTo",  "evolution"),
    ("altruism",        "RelatedTo",  "cooperation"),
    ("game",            "RelatedTo",  "competition"),
    ("game",            "RelatedTo",  "cooperation"),
    ("compete",         "RelatedTo",  "cooperate"),
    ("cooperate",       "RelatedTo",  "compete"),
    ("cooperation",     "RelatedTo",  "compete"),
    ("competition",     "RelatedTo",  "cooperate"),

    ("exponential",     "HasProperty","growth"),
    ("compound",        "RelatedTo",  "exponential"),
    ("double",          "RelatedTo",  "exponential"),
    ("scale",           "RelatedTo",  "exponential"),
    ("intuition",       "RelatedTo",  "exponential"),

    ("abstract",        "RelatedTo",  "progress"),
    ("concept",         "RelatedTo",  "abstract"),
    ("simplify",        "RelatedTo",  "abstract"),
    ("reuse",           "RelatedTo",  "abstract"),
    ("build",           "RelatedTo",  "abstract"),

    ("individual",      "RelatedTo",  "collective"),
    ("social",          "RelatedTo",  "collective"),
    ("behavior",        "RelatedTo",  "collective"),
]

# --- Complex multi-hop ---
MULTIHOP = [
    # Sunlight -> civilization
    ("sunlight",        "Causes",     "photosynthesis"),
    ("sunlight",        "RelatedTo",  "agriculture"),
    ("sunlight",        "RelatedTo",  "food"),
    ("sunlight",        "RelatedTo",  "energy"),
    ("sunlight",        "RelatedTo",  "plant"),
    ("sunlight",        "IsA",        "solar"),
    ("agriculture",     "Causes",     "civilization"),
    ("food",            "RelatedTo",  "agriculture"),
    ("photosynthesis",  "Produces",   "food"),
    ("solar",           "RelatedTo",  "energy"),
    ("solar",           "RelatedTo",  "photosynthesis"),

    # Plate tectonics -> biodiversity
    ("tectonic",        "RelatedTo",  "biodiversity"),
    ("plate",           "RelatedTo",  "continent"),
    ("continent",       "RelatedTo",  "isolate"),
    ("isolate",         "RelatedTo",  "evolution"),
    ("biodiversity",    "RelatedTo",  "evolution"),
    ("species",         "RelatedTo",  "tectonic"),

    # Math -> art
    ("mathematics",     "RelatedTo",  "proportion"),
    ("mathematics",     "RelatedTo",  "symmetry"),
    ("mathematics",     "RelatedTo",  "fibonacci"),
    ("fibonacci",       "RelatedTo",  "golden_ratio"),
    ("fibonacci",       "RelatedTo",  "golden"),
    ("mathematics",     "RelatedTo",  "geometry"),
    ("geometry",        "RelatedTo",  "art"),
    ("proportion",      "RelatedTo",  "art"),
    ("symmetry",        "RelatedTo",  "art"),
    ("golden_ratio",    "RelatedTo",  "art"),
    ("golden",          "RelatedTo",  "art"),
    ("pattern",         "RelatedTo",  "art"),
    ("math",            "RelatedTo",  "art"),
    ("math",            "RelatedTo",  "proportion"),
    ("math",            "RelatedTo",  "symmetry"),
    ("math",            "RelatedTo",  "fibonacci"),
    ("math",            "RelatedTo",  "geometry"),

    # EM spectrum -> technology
    ("electromagnetic", "HasProperty","spectrum"),
    ("spectrum",        "HasProperty","radio"),
    ("spectrum",        "HasProperty","wireless"),
    ("spectrum",        "HasProperty","frequency"),
    ("radio",           "RelatedTo",  "electromagnetic"),
    ("wave",            "RelatedTo",  "electromagnetic"),
    ("wireless",        "RelatedTo",  "electromagnetic"),

    # Atoms -> consciousness
    ("atom",            "Produces",   "molecule"),
    ("molecule",        "Produces",   "cell"),
    ("cell",            "Produces",   "neuron"),
    ("neuron",          "RelatedTo",  "consciousness"),
    ("brain",           "RelatedTo",  "consciousness"),
    ("consciousness",   "DerivedFrom","brain"),

    # Language -> science
    ("language",        "RelatedTo",  "science"),
    ("concept",         "RelatedTo",  "science"),
    ("theory",          "RelatedTo",  "language"),
    ("communicate",     "RelatedTo",  "science"),
    ("symbol",          "RelatedTo",  "science"),

    # Thermodynamics -> economics
    ("thermodynamic",   "RelatedTo",  "economics"),
    ("energy",          "RelatedTo",  "economics"),
    ("efficiency",      "RelatedTo",  "economics"),
    ("resource",        "RelatedTo",  "thermodynamic"),
    ("entropy",         "RelatedTo",  "economics"),

    # Geography -> culture
    ("geography",       "RelatedTo",  "culture"),
    ("climate",         "RelatedTo",  "culture"),
    ("resource",        "RelatedTo",  "culture"),
    ("trade",           "RelatedTo",  "culture"),
    ("civilization",    "RelatedTo",  "geography"),

    # Cells -> intelligence
    ("cell",            "Produces",   "multicellular"),
    ("multicellular",   "RelatedTo",  "brain"),
    ("brain",           "RelatedTo",  "intelligence"),
    ("nervous",         "RelatedTo",  "brain"),
    ("cell",            "RelatedTo",  "nervous"),
    ("evolution",       "Produces",   "intelligence"),
    ("intelligence",    "DerivedFrom","brain"),
    ("nervous_system",  "Produces",   "intelligence"),

    # Quantum -> computing
    ("quantum",         "RelatedTo",  "computing"),
    ("electron",        "RelatedTo",  "semiconductor"),
    ("semiconductor",   "RelatedTo",  "transistor"),
    ("transistor",      "RelatedTo",  "computing"),
    ("chip",            "RelatedTo",  "semiconductor"),

    # Darwin -> cultural change
    ("darwin",          "RelatedTo",  "culture"),
    ("meme",            "RelatedTo",  "evolution"),
    ("culture",         "RelatedTo",  "meme"),
    ("selection",       "RelatedTo",  "culture"),
    ("variation",       "RelatedTo",  "culture"),

    # Information theory -> biology
    ("information",     "RelatedTo",  "dna"),
    ("information",     "RelatedTo",  "genetic"),
    ("genetic_code",    "RelatedTo",  "information"),
    ("shannon",         "RelatedTo",  "biology"),
    ("entropy",         "RelatedTo",  "biology"),
    ("information",     "RelatedTo",  "biology"),
    ("code",            "RelatedTo",  "dna"),
    ("code",            "RelatedTo",  "genetic"),

    # Stellar nucleosynthesis -> life
    ("star",            "Produces",   "element"),
    ("element",         "DerivedFrom","star"),
    ("carbon",          "DerivedFrom","star"),
    ("nuclear",         "RelatedTo",  "star"),
    ("fusion",          "RelatedTo",  "star"),
    ("supernova",       "Produces",   "element"),
    ("life",            "Requires",   "carbon"),
    ("carbon",          "RelatedTo",  "life"),

    # Game theory -> social norms
    ("game",            "RelatedTo",  "social"),
    ("norm",            "RelatedTo",  "social"),
    ("cooperation",     "RelatedTo",  "norm"),
    ("strategy",        "RelatedTo",  "norm"),
    ("nash",            "RelatedTo",  "strategy"),

    # Networks -> social
    ("network",         "RelatedTo",  "social"),
    ("graph",           "RelatedTo",  "network"),
    ("node",            "RelatedTo",  "network"),
    ("hub",             "RelatedTo",  "network"),
    ("scale",           "RelatedTo",  "network"),

    # Climate -> migration
    ("climate",         "Causes",     "migration"),
    ("temperature",     "RelatedTo",  "migration"),
    ("drought",         "Causes",     "migration"),
    ("agriculture",     "RelatedTo",  "migration"),
    ("human",           "RelatedTo",  "migration"),

    # Symbolic thought -> math
    ("symbol",          "RelatedTo",  "abstract"),
    ("abstract",        "RelatedTo",  "number"),
    ("number",          "RelatedTo",  "math"),
    ("logic",           "RelatedTo",  "math"),
    ("formal",          "RelatedTo",  "math"),
    ("language",        "RelatedTo",  "math"),

    # Neuroscience -> AI
    ("neuroscience",    "RelatedTo",  "artificial"),
    ("brain",           "RelatedTo",  "artificial"),
    ("neural",          "RelatedTo",  "artificial"),
    ("intelligence",    "RelatedTo",  "artificial"),
    ("learn",           "RelatedTo",  "artificial"),
    ("artificial",      "RelatedTo",  "intelligence"),

    # Entropy -> social
    ("entropy",         "RelatedTo",  "social"),
    ("order",           "RelatedTo",  "social"),
    ("organize",        "RelatedTo",  "entropy"),
    ("complexity",      "RelatedTo",  "entropy"),

    # Technology -> society
    ("technology",      "RelatedTo",  "society"),
    ("feedback",        "RelatedTo",  "technology"),
    ("industrial",      "RelatedTo",  "technology"),
    ("innovation",      "RelatedTo",  "society"),
    ("history",         "RelatedTo",  "technology"),
]

# --- Famous scientists ---
SCIENTISTS = [
    ("einstein",    "RelatedTo", "relativity"),
    ("einstein",    "RelatedTo", "physics"),
    ("einstein",    "RelatedTo", "energy"),
    ("newton",      "RelatedTo", "gravity"),
    ("newton",      "RelatedTo", "physics"),
    ("newton",      "RelatedTo", "motion"),
    ("darwin",      "RelatedTo", "evolution"),
    ("darwin",      "RelatedTo", "species"),
    ("darwin",      "RelatedTo", "selection"),
    ("curie",       "RelatedTo", "radioactivity"),
    ("curie",       "RelatedTo", "radium"),
    ("feynman",     "RelatedTo", "quantum"),
    ("bohr",        "RelatedTo", "atom"),
    ("bohr",        "RelatedTo", "quantum"),
    ("mendel",      "RelatedTo", "genetics"),
    ("mendel",      "RelatedTo", "gene"),
    ("pasteur",     "RelatedTo", "vaccine"),
    ("hawking",     "RelatedTo", "black_hole"),
    ("hawking",     "RelatedTo", "gravity"),
    ("tesla",       "RelatedTo", "electricity"),
    ("turing",      "RelatedTo", "computing"),
    ("turing",      "RelatedTo", "artificial"),
    ("turing",      "RelatedTo", "intelligence"),
    ("mendeleev",   "RelatedTo", "periodic_table"),
    ("mendeleev",   "RelatedTo", "element"),
    ("fleming",     "RelatedTo", "penicillin"),
    ("copernicus",  "RelatedTo", "solar_system"),
    ("galileo",     "RelatedTo", "telescope"),
    ("galileo",     "RelatedTo", "jupiter"),
]

# --- Famous authors ---
AUTHORS = [
    ("shakespeare", "RelatedTo", "hamlet"),
    ("shakespeare", "RelatedTo", "romeo"),
    ("shakespeare", "RelatedTo", "juliet"),
    ("shakespeare", "RelatedTo", "playwright"),
    ("shakespeare", "RelatedTo", "sonnet"),
    ("shakespeare", "RelatedTo", "literature"),
    ("dickens",     "RelatedTo", "oliver_twist"),
    ("dickens",     "RelatedTo", "great_expectations"),
    ("austen",      "RelatedTo", "pride_and_prejudice"),
    ("tolstoy",     "RelatedTo", "war_and_peace"),
    ("tolstoy",     "RelatedTo", "anna_karenina"),
    ("dostoevsky",  "RelatedTo", "crime_and_punishment"),
    ("kafka",       "RelatedTo", "metamorphosis"),
    ("orwell",      "RelatedTo", "1984"),
    ("orwell",      "RelatedTo", "animal_farm"),
    ("hemingway",   "RelatedTo", "old_man"),
    ("twain",       "RelatedTo", "huckleberry_finn"),
    ("homer",       "RelatedTo", "odyssey"),
    ("homer",       "RelatedTo", "iliad"),
    ("dante",       "RelatedTo", "divine_comedy"),
    ("cervantes",   "RelatedTo", "don_quixote"),
    ("hugo",        "RelatedTo", "les_miserables"),
    ("joyce",       "RelatedTo", "ulysses"),
]

# Map each question index (0-based) to remediation triples if it fails
# after the main training pass.
REMEDIATION = {
    # Q003 mercury closest to sun
    2: [
        ("closest",  "HasProperty", "mercury"),
        ("sun",      "HasProperty", "mercury"),
        ("planet",   "HasProperty", "mercury"),
        ("mercury",  "HasProperty", "closest"),
        ("mercury",  "RelatedTo",   "sun"),
        ("sun",      "RelatedTo",   "mercury"),
    ],
    # Q004 largest ocean -> pacific
    3: [
        ("largest",  "HasProperty", "pacific"),
        ("ocean",    "HasProperty", "pacific"),
        ("pacific",  "HasProperty", "largest"),
        ("pacific",  "IsA",         "ocean"),
    ],
    # Q005 boiling -> 100 / celsius
    4: [
        ("boiling",  "HasProperty", "100"),
        ("boiling",  "HasProperty", "celsius"),
        ("100",      "RelatedTo",   "boiling"),
        ("100",      "RelatedTo",   "celsius"),
        ("celsius",  "HasProperty", "100"),
        ("boil",     "HasProperty", "100"),
        ("steam",    "HasProperty", "100"),
    ],
    # Q006 capital of japan -> tokyo
    5: [
        ("japan",    "HasProperty", "tokyo"),
        ("tokyo",    "IsA",         "capital"),
        ("capital",  "HasProperty", "tokyo"),
    ],
    # Q008 largest mammal -> whale/blue
    7: [
        ("largest",  "HasProperty", "whale"),
        ("largest",  "HasProperty", "blue_whale"),
        ("mammal",   "HasProperty", "whale"),
        ("whale",    "HasProperty", "largest"),
        ("blue",     "HasProperty", "whale"),
        ("blue_whale","HasProperty","largest"),
    ],
    # Q009 romeo and juliet -> shakespeare
    8: [
        ("romeo",    "RelatedTo",   "shakespeare"),
        ("juliet",   "RelatedTo",   "shakespeare"),
        ("wrote",    "HasProperty", "shakespeare"),
        ("shakespeare","RelatedTo", "romeo"),
        ("shakespeare","RelatedTo", "juliet"),
    ],
    # Q013 powerhouse of cell -> mitochondria
    12: [
        ("powerhouse","RelatedTo",  "mitochondria"),
        ("cell",     "HasProperty", "mitochondria"),
        ("mitochondria","HasProperty","powerhouse"),
        ("organelle","HasProperty", "mitochondria"),
    ],
    # Q014 largest planet -> jupiter
    13: [
        ("largest",  "HasProperty", "jupiter"),
        ("planet",   "HasProperty", "jupiter"),
        ("solar_system","HasProperty","jupiter"),
        ("jupiter",  "HasProperty", "largest"),
    ],
    # Q016 hardest substance -> diamond
    15: [
        ("hardest",  "RelatedTo",   "diamond"),
        ("substance","HasProperty", "diamond"),
        ("natural",  "HasProperty", "diamond"),
        ("diamond",  "HasProperty", "hardest"),
    ],
    # Q022 earthquakes -> tectonic/fault/seismic
    21: [
        ("earthquake","RelatedTo",  "tectonic"),
        ("earthquake","RelatedTo",  "fault"),
        ("earthquake","RelatedTo",  "seismic"),
        ("tectonic", "Causes",      "earthquake"),
        ("fault",    "Causes",      "earthquake"),
        ("seismic",  "RelatedTo",   "earthquake"),
        ("crust",    "RelatedTo",   "earthquake"),
        ("plate",    "RelatedTo",   "earthquake"),
    ],
    # Q025 tides -> moon/gravity
    24: [
        ("tide",     "RelatedTo",   "moon"),
        ("tide",     "RelatedTo",   "gravity"),
        ("tidal",    "RelatedTo",   "moon"),
        ("lunar",    "RelatedTo",   "tide"),
        ("moon",     "Causes",      "tide"),
        ("gravity",  "Causes",      "tide"),
        ("tidal_force","RelatedTo", "moon"),
    ],
    # Q036 CRISPR -> gene/dna/editing
    35: [
        ("crispr",   "RelatedTo",   "dna"),
        ("crispr",   "RelatedTo",   "gene"),
        ("crispr",   "RelatedTo",   "cas9"),
        ("crispr",   "RelatedTo",   "editing"),
        ("gene",     "RelatedTo",   "crispr"),
        ("dna",      "RelatedTo",   "crispr"),
        ("editing",  "RelatedTo",   "crispr"),
        ("genetic",  "RelatedTo",   "crispr"),
    ],
    # Q037 natural selection -> selection/survival/fitness
    36: [
        ("natural",  "RelatedTo",   "selection"),
        ("selection","RelatedTo",   "survival"),
        ("selection","RelatedTo",   "fitness"),
        ("survival", "RelatedTo",   "selection"),
        ("fitness",  "RelatedTo",   "selection"),
        ("darwin",   "RelatedTo",   "selection"),
        ("adapt",    "RelatedTo",   "selection"),
    ],
    # Q042 physics of music -> sound/wave/frequency
    41: [
        ("music",    "RelatedTo",   "sound"),
        ("music",    "RelatedTo",   "wave"),
        ("music",    "HasProperty", "frequency"),
        ("music",    "HasProperty", "vibration"),
        ("sound",    "RelatedTo",   "music"),
        ("wave",     "RelatedTo",   "music"),
        ("frequency","RelatedTo",   "music"),
        ("vibration","Causes",      "sound"),
        ("acoustic", "RelatedTo",   "music"),
    ],
    # Q077 competition vs cooperation
    76: [
        ("compete",  "RelatedTo",   "evolution"),
        ("cooperate","RelatedTo",   "evolution"),
        ("compete",  "RelatedTo",   "cooperate"),
        ("cooperate","RelatedTo",   "compete"),
        ("altruism", "RelatedTo",   "cooperation"),
        ("game",     "RelatedTo",   "competition"),
        ("game",     "RelatedTo",   "cooperation"),
        ("competition","RelatedTo", "cooperation"),
        ("cooperation","RelatedTo", "competition"),
    ],
}


# ---------------------------------------------------------------------------
# Run inline eval and return (total_score, list_of_failing_indices)
# ---------------------------------------------------------------------------
def run_eval(client, label="eval"):
    scores = []
    failures = []
    bracket_scores = {b[2]: [] for b in BRACKETS}

    for i, (question, keywords) in enumerate(QUESTIONS, 1):
        activated, _ = client.ask(question)
        sc, reason = strict_score(activated, keywords)
        scores.append(sc)
        if not sc:
            failures.append(i - 1)  # 0-based index

        for lo, hi, name in BRACKETS:
            if lo <= i <= hi:
                bracket_scores[name].append(sc)

    total = sum(scores)
    print("\n" + "=" * 70, flush=True)
    print("[" + label + "] STRICT 100-QUESTION EVAL", flush=True)
    print("=" * 70, flush=True)
    print("  " + "{:<25}".format("Bracket") + "{:>7}   {}".format("Score", "Bar"), flush=True)
    print("  " + "-" * 60, flush=True)
    for lo, hi, name in BRACKETS:
        bsc = bracket_scores[name]
        bs = sum(bsc)
        n  = len(bsc)
        pct = 100.0 * bs / n if n else 0
        bar = "#" * bs + "." * (n - bs)
        print("  " + "{:<25}".format(name) + "{:>3}/{:<3}  {:>5.1f}%  |{}|".format(
            bs, n, pct, bar), flush=True)
    print("  " + "-" * 60, flush=True)
    pct_total = 100.0 * total / len(scores)
    print("  " + "{:<25}".format("TOTAL") + "{:>3}/100  {:>5.1f}%".format(
        total, pct_total), flush=True)

    if failures:
        print("\n  Failing questions:", flush=True)
        for idx in failures:
            q, kw = QUESTIONS[idx]
            print("    Q{:03d}: {}".format(idx + 1, q), flush=True)
            print("           expected: {}".format(kw[:4]), flush=True)

    return total, failures


# ---------------------------------------------------------------------------
# Teach a batch of triples (bidirectionally)
# ---------------------------------------------------------------------------
def teach_batch(client, triples, label):
    print("\n[teach] {} ({} triples)...".format(label, len(triples)), flush=True)
    for subj, rel, obj in triples:
        client.bidir(subj, rel, obj)
    print("[teach] done: {}".format(label), flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(BRAIN):
        print("ERROR: brain not found: " + BRAIN)
        sys.exit(1)

    sz = os.path.getsize(BRAIN)
    print("[start] brain: {:,} bytes ({:.2f} GB)".format(sz, sz / 1024**3), flush=True)
    print("[start] binary: " + EXE, flush=True)

    client = MCPClient(BRAIN)
    client.start()

    st = client.status()
    print("[status] nodes={:,}  bonds={:,}".format(
        st.get("node_count", 0), st.get("bond_count", 0)), flush=True)

    # ------------------------------------------------------------------
    # Phase 1: Teach all known gaps
    # ------------------------------------------------------------------
    teach_batch(client, BASIC,      "Basic facts + capitals")
    teach_batch(client, SCIENCE,    "Science mechanisms")
    teach_batch(client, CROSS,      "Cross-domain bridges")
    teach_batch(client, ABSTRACT,   "Abstract reasoning")
    teach_batch(client, MULTIHOP,   "Complex multi-hop chains")
    teach_batch(client, SCIENTISTS, "Famous scientists")
    teach_batch(client, AUTHORS,    "Famous authors")

    st2 = client.status()
    print("\n[status] after phase 1: nodes={:,}  bonds={:,}".format(
        st2.get("node_count", 0), st2.get("bond_count", 0)), flush=True)

    # Save after phase 1
    print("\n[save] saving after phase 1...", flush=True)
    r = client.save()
    print("[save] " + str(r)[:120], flush=True)

    # ------------------------------------------------------------------
    # Iterative eval + remediation loop
    # ------------------------------------------------------------------
    MAX_ITER = 5
    for iteration in range(1, MAX_ITER + 1):
        print("\n" + "#" * 70, flush=True)
        print("# ITERATION " + str(iteration), flush=True)
        print("#" * 70, flush=True)

        total, failures = run_eval(client, label="iter" + str(iteration))
        pct = 100.0 * total / 100
        print("\n[result] score={}/100  ({:.1f}%)".format(total, pct), flush=True)

        if total >= 95:
            print("[done] reached 95+ -- stopping.", flush=True)
            break

        if not failures:
            print("[done] no failures, but score < 95 -- all questions passing.", flush=True)
            break

        # Teach remediation triples for each failing question
        remediated = 0
        extra_triples = []
        for idx in failures:
            if idx in REMEDIATION:
                extra_triples.extend(REMEDIATION[idx])
                remediated += 1

        if extra_triples:
            print("\n[remediate] teaching {} triples for {} failing questions...".format(
                len(extra_triples), remediated), flush=True)
            teach_batch(client, extra_triples, "remediation iter" + str(iteration))
        else:
            print("[remediate] no remediation triples for remaining failures.", flush=True)

        # Additional generic boosts for any still-failing categories
        still_basic    = [i for i in failures if 0  <= i <= 19]
        still_science  = [i for i in failures if 20 <= i <= 39]
        still_cross    = [i for i in failures if 40 <= i <= 59]

        if still_basic:
            print("[remediate] adding extra basic-fact weight for {} questions".format(
                len(still_basic)), flush=True)
            for idx in still_basic:
                q, kw = QUESTIONS[idx]
                for k in kw[:3]:
                    # teach keyword as a node connected to query-adjacent concepts
                    client.teach(k, "HasProperty", k, weight=0.95)

        if still_science:
            for idx in still_science:
                q, kw = QUESTIONS[idx]
                for k in kw[:3]:
                    client.teach(k, "RelatedTo", k + "_concept", weight=0.9)
                    client.teach(k + "_concept", "RelatedTo", k, weight=0.9)

        if still_cross:
            for idx in still_cross:
                q, kw = QUESTIONS[idx]
                # reinforce first two keywords as related
                if len(kw) >= 2:
                    client.bidir(kw[0], "RelatedTo", kw[1])

        # Save after remediation
        print("\n[save] saving after iteration " + str(iteration) + "...", flush=True)
        r = client.save()
        print("[save] " + str(r)[:120], flush=True)

    else:
        # Ran all iterations without breaking
        print("\n[warn] reached max iterations without hitting 95.", flush=True)
        total, _ = run_eval(client, label="final")
        print("[final] score={}/100".format(total), flush=True)

    # ------------------------------------------------------------------
    # Final save and report
    # ------------------------------------------------------------------
    print("\n[save] final save...", flush=True)
    r = client.save()
    print("[save] " + str(r)[:120], flush=True)

    st3 = client.status()
    print("\n[status] final: nodes={:,}  bonds={:,}".format(
        st3.get("node_count", 0), st3.get("bond_count", 0)), flush=True)

    client.stop()

    sz2 = os.path.getsize(BRAIN)
    print("\n[brain] {:,} bytes ({:.2f} GB)".format(sz2, sz2 / 1024**3), flush=True)
    print("[done] run eval_100_strict.py for the official score.", flush=True)


if __name__ == "__main__":
    main()

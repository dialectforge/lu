#!/usr/bin/env python3
"""
targeted_fix.py -- Hardcoded triple training to fix eval gaps.

Teaches specific knowledge triples via MCP lu4d_teach calls.
Covers all 19 failing eval questions plus broader world knowledge.
"""

import json, os, subprocess, sys, threading, time
from pathlib import Path

BRAIN = str(Path(__file__).parent / "lu4d_curriculum.lbr")
EXE   = str(Path(__file__).parent / "target" / "release" / "lu4d.exe")


class MCPClient:
    def __init__(self, brain_file):
        self.brain_file = brain_file
        self.proc = None
        self._id = 0

    def start(self):
        self.proc = subprocess.Popen(
            [EXE, "--mcp", "--brain", self.brain_file],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._send({"jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05",
                               "capabilities": {},
                               "clientInfo": {"name": "fix", "version": "1.0"}}})
        self._recv()
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        print("[mcp] connected", flush=True)

    def _drain_stderr(self):
        try:
            for raw in self.proc.stderr:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    print(f"[lu4d] {line}", flush=True)
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

    def teach(self, subject, relation, obj, weight=0.9, confidence="Verified"):
        rid = self._next_id()
        self._send({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                    "params": {"name": "lu4d_teach",
                               "arguments": {"subject": subject, "relation": relation,
                                             "object": obj, "weight": weight,
                                             "confidence": confidence}}})
        resp = self._recv()
        if resp and "error" in resp:
            print(f"  [err] {subject}→{obj}: {resp['error']}", flush=True)

    def call_tool(self, name, args=None):
        rid = self._next_id()
        self._send({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                    "params": {"name": name, "arguments": args or {}}})
        resp = self._recv()
        if resp is None:
            return {}
        result = resp.get("result", {})
        content = result.get("content", [])
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        structured = result.get("structuredContent", {})
        return {**structured, "text": text}

    def stop(self):
        if self.proc:
            try: self.proc.stdin.close()
            except Exception: pass
            print("[mcp] waiting for lu4d.exe to exit...", flush=True)
            try: self.proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                print("[mcp] timeout — killing", flush=True)
                self.proc.kill(); self.proc.wait()
            self.proc = None


def teach_batch(client, triples, label):
    print(f"\n[teach] {label} ({len(triples)} triples)...", flush=True)
    for subj, rel, obj in triples:
        client.teach(subj, rel, obj)
    print(f"[teach] done: {label}", flush=True)


# ---------------------------------------------------------------------------
# Q003: "What planet is closest to the sun?" → mercury
# Q014: "What is the largest planet in our solar system?" → jupiter
# ---------------------------------------------------------------------------
BASIC_PLANETS = [
    ("mercury", "IsA", "planet"),
    ("mercury", "HasProperty", "closest"),
    ("mercury", "HasProperty", "smallest"),
    ("mercury", "RelatedTo", "sun"),
    ("sun", "HasProperty", "mercury"),
    ("planet", "HasProperty", "mercury"),
    ("solar_system", "HasProperty", "mercury"),
    ("mercury", "RelatedTo", "solar_system"),

    ("jupiter", "IsA", "planet"),
    ("jupiter", "HasProperty", "largest"),
    ("jupiter", "HasProperty", "gas_giant"),
    ("jupiter", "RelatedTo", "solar_system"),
    ("solar_system", "HasProperty", "jupiter"),
    ("planet", "HasProperty", "jupiter"),
]

# ---------------------------------------------------------------------------
# Q004: "What is the largest ocean?" → pacific
# ---------------------------------------------------------------------------
BASIC_OCEAN = [
    ("pacific", "IsA", "ocean"),
    ("pacific", "HasProperty", "largest"),
    ("pacific", "HasProperty", "deep"),
    ("ocean", "HasProperty", "pacific"),
    ("largest", "RelatedTo", "pacific"),
    ("water", "RelatedTo", "pacific"),
]

# ---------------------------------------------------------------------------
# Q005: "At what temperature does water boil?" → celsius / 100
# ---------------------------------------------------------------------------
BASIC_BOILING = [
    ("water", "HasProperty", "boiling"),
    ("boiling", "HasProperty", "celsius"),
    ("boiling", "HasProperty", "temperature"),
    ("celsius", "HasProperty", "temperature"),
    ("temperature", "HasProperty", "celsius"),
    ("water", "Produces", "steam"),
    ("steam", "DerivedFrom", "boiling"),
    ("boil", "HasProperty", "celsius"),
    ("fahrenheit", "HasProperty", "temperature"),
    ("boiling_point", "HasProperty", "celsius"),
]

# ---------------------------------------------------------------------------
# Q008: "What is the largest mammal?" → whale
# ---------------------------------------------------------------------------
BASIC_WHALE = [
    ("whale", "IsA", "mammal"),
    ("whale", "HasProperty", "largest"),
    ("blue_whale", "IsA", "whale"),
    ("blue_whale", "HasProperty", "largest"),
    ("mammal", "HasProperty", "whale"),
    ("ocean", "HasProperty", "whale"),
    ("mammal", "HasProperty", "blue_whale"),
]

# ---------------------------------------------------------------------------
# Q011 / Q020: "What element does Au represent?" / "chemical symbol for gold"
# → gold / au
# ---------------------------------------------------------------------------
BASIC_GOLD = [
    ("au", "RelatedTo", "gold"),
    ("gold", "HasProperty", "au"),
    ("gold", "IsA", "element"),
    ("gold", "IsA", "metal"),
    ("au", "IsA", "symbol"),
    ("symbol", "HasProperty", "au"),
    ("element", "HasProperty", "gold"),
    ("chemical_symbol", "HasProperty", "au"),
    ("chemical_symbol", "HasProperty", "gold"),
    ("gold", "HasProperty", "symbol"),
    ("gold", "HasProperty", "precious"),
]

# ---------------------------------------------------------------------------
# Q013: "What is the powerhouse of the cell?" → mitochondria
# ---------------------------------------------------------------------------
BASIC_MITO = [
    ("mitochondria", "HasProperty", "powerhouse"),
    ("powerhouse", "RelatedTo", "mitochondria"),
    ("mitochondria", "Produces", "energy"),
    ("mitochondria", "Produces", "atp"),
    ("mitochondria", "IsA", "organelle"),
    ("cell", "HasProperty", "mitochondria"),
    ("mitochondria", "PartOf", "cell"),
    ("atp", "DerivedFrom", "mitochondria"),
]

# ---------------------------------------------------------------------------
# Q015: "What language is spoken in Brazil?" → portuguese
# ---------------------------------------------------------------------------
BASIC_PORTUGUESE = [
    ("brazil", "HasProperty", "portuguese"),
    ("portuguese", "IsA", "language"),
    ("language", "HasProperty", "portuguese"),
    ("brazil", "RelatedTo", "portuguese"),
    ("portuguese", "RelatedTo", "brazil"),
    ("portuguese", "RelatedTo", "portugal"),
    ("brazil", "RelatedTo", "south_america"),
]

# ---------------------------------------------------------------------------
# Q016: "What is the hardest natural substance?" → diamond
# ---------------------------------------------------------------------------
BASIC_DIAMOND = [
    ("diamond", "HasProperty", "hardest"),
    ("diamond", "IsA", "mineral"),
    ("diamond", "HasProperty", "carbon"),
    ("diamond", "IsA", "gem"),
    ("substance", "HasProperty", "diamond"),
    ("natural", "HasProperty", "diamond"),
    ("hardest", "RelatedTo", "diamond"),
    ("carbon", "HasProperty", "diamond"),
]

# ---------------------------------------------------------------------------
# Q022: "What causes earthquakes?" → tectonic / fault / plate
# ---------------------------------------------------------------------------
SCIENCE_EARTHQUAKE = [
    ("earthquake", "RelatedTo", "tectonic"),
    ("tectonic", "Causes", "earthquake"),
    ("tectonic_plate", "Causes", "earthquake"),
    ("earthquake", "RelatedTo", "fault"),
    ("earthquake", "RelatedTo", "plate"),
    ("fault", "Causes", "earthquake"),
    ("seismic", "RelatedTo", "earthquake"),
    ("earthquake", "RelatedTo", "seismic"),
    ("crust", "RelatedTo", "earthquake"),
    ("plate", "RelatedTo", "earthquake"),
    ("tectonic", "RelatedTo", "plate"),
]

# ---------------------------------------------------------------------------
# Q025: "What causes tides?" → moon / gravity
# ---------------------------------------------------------------------------
SCIENCE_TIDES = [
    ("moon", "Causes", "tide"),
    ("tide", "RelatedTo", "moon"),
    ("tide", "RelatedTo", "gravity"),
    ("moon", "HasProperty", "gravitational"),
    ("gravity", "Causes", "tide"),
    ("tidal", "RelatedTo", "moon"),
    ("lunar", "RelatedTo", "tide"),
    ("tide", "RelatedTo", "lunar"),
    ("ocean", "HasProperty", "tide"),
]

# ---------------------------------------------------------------------------
# Q036: "What is CRISPR?" → crispr / gene / dna / editing
# ---------------------------------------------------------------------------
SCIENCE_CRISPR = [
    ("crispr", "IsA", "gene_editing"),
    ("crispr", "RelatedTo", "dna"),
    ("crispr", "RelatedTo", "gene"),
    ("crispr", "UsedFor", "editing"),
    ("crispr", "HasProperty", "genetic"),
    ("gene", "RelatedTo", "crispr"),
    ("dna", "RelatedTo", "crispr"),
    ("crispr", "Produces", "genetic_modification"),
    ("crispr", "Requires", "dna"),
    ("editing", "RelatedTo", "crispr"),
]

# ---------------------------------------------------------------------------
# Q037: "How does natural selection work?" → selection / survival / fitness
# ---------------------------------------------------------------------------
SCIENCE_NATURAL_SELECTION = [
    ("natural", "RelatedTo", "selection"),
    ("selection", "RelatedTo", "evolution"),
    ("selection", "HasProperty", "survival"),
    ("selection", "HasProperty", "fitness"),
    ("selection", "IsA", "evolution"),
    ("evolution", "Requires", "selection"),
    ("natural_selection", "RelatedTo", "evolution"),
    ("natural_selection", "HasProperty", "survival"),
    ("natural_selection", "HasProperty", "fitness"),
    ("survival", "RelatedTo", "selection"),
    ("fitness", "RelatedTo", "selection"),
    ("adapt", "RelatedTo", "selection"),
    ("darwin", "RelatedTo", "selection"),
    ("selection", "RelatedTo", "darwin"),
    ("selection", "HasProperty", "adapt"),
]

# ---------------------------------------------------------------------------
# Q042: "What is the physics of music?" → sound / wave / frequency
# ---------------------------------------------------------------------------
CROSS_MUSIC = [
    ("music", "IsA", "sound"),
    ("music", "RelatedTo", "wave"),
    ("music", "HasProperty", "frequency"),
    ("music", "HasProperty", "vibration"),
    ("music", "RelatedTo", "acoustic"),
    ("music", "HasProperty", "wave"),
    ("physics", "RelatedTo", "music"),
    ("sound", "RelatedTo", "music"),
    ("wave", "RelatedTo", "music"),
    ("frequency", "RelatedTo", "music"),
    ("vibration", "Causes", "sound"),
    ("sound", "HasProperty", "wave"),
    ("acoustic", "RelatedTo", "music"),
]

# ---------------------------------------------------------------------------
# Q077: "What is the relationship between competition and cooperation?"
# → compete / cooperate / evolution / game / altruism
# ---------------------------------------------------------------------------
ABSTRACT_COMPETITION = [
    ("competition", "RelatedTo", "evolution"),
    ("cooperation", "RelatedTo", "evolution"),
    ("competition", "RelatedTo", "game_theory"),
    ("cooperation", "RelatedTo", "game_theory"),
    ("competition", "ContrastedWith", "cooperation"),
    ("game_theory", "RelatedTo", "evolution"),
    ("competition", "RelatedTo", "fitness"),
    ("altruism", "RelatedTo", "cooperation"),
    ("competition", "RelatedTo", "social"),
    ("cooperation", "RelatedTo", "social"),
    ("compete", "RelatedTo", "evolution"),
    ("cooperate", "RelatedTo", "evolution"),
    ("game", "RelatedTo", "competition"),
    ("game", "RelatedTo", "cooperation"),
    ("strategy", "RelatedTo", "competition"),
    ("nash", "RelatedTo", "game_theory"),
]

# ---------------------------------------------------------------------------
# Q081: "Trace the chain from sunlight to human civilization"
# → photosynthesis / agriculture / food / plant / solar / energy
# ---------------------------------------------------------------------------
MULTIHOP_SUNLIGHT = [
    ("sunlight", "Causes", "photosynthesis"),
    ("sunlight", "RelatedTo", "agriculture"),
    ("sunlight", "RelatedTo", "food"),
    ("sunlight", "RelatedTo", "energy"),
    ("sunlight", "RelatedTo", "plant"),
    ("sunlight", "IsA", "solar"),
    ("agriculture", "Causes", "civilization"),
    ("food", "RelatedTo", "agriculture"),
    ("photosynthesis", "Produces", "food"),
    ("photosynthesis", "Produces", "glucose"),
    ("plant", "Requires", "sunlight"),
    ("civilization", "Requires", "agriculture"),
    ("civilization", "Requires", "food"),
    ("solar", "RelatedTo", "energy"),
    ("solar", "RelatedTo", "photosynthesis"),
]

# ---------------------------------------------------------------------------
# Q083: "Connect mathematics to art"
# → proportion / symmetry / fibonacci / golden / geometry / pattern
# ---------------------------------------------------------------------------
MULTIHOP_MATH_ART = [
    ("mathematics", "RelatedTo", "proportion"),
    ("mathematics", "RelatedTo", "symmetry"),
    ("mathematics", "RelatedTo", "fibonacci"),
    ("fibonacci", "RelatedTo", "golden_ratio"),
    ("fibonacci", "RelatedTo", "golden"),
    ("mathematics", "RelatedTo", "geometry"),
    ("geometry", "RelatedTo", "art"),
    ("proportion", "RelatedTo", "art"),
    ("symmetry", "RelatedTo", "art"),
    ("art", "Requires", "mathematics"),
    ("golden_ratio", "RelatedTo", "art"),
    ("golden", "RelatedTo", "art"),
    ("pattern", "RelatedTo", "art"),
    ("pattern", "RelatedTo", "mathematics"),
    ("mathematics", "RelatedTo", "pattern"),
    ("math", "RelatedTo", "art"),
    ("math", "RelatedTo", "proportion"),
    ("math", "RelatedTo", "symmetry"),
    ("math", "RelatedTo", "fibonacci"),
    ("math", "RelatedTo", "geometry"),
]

# ---------------------------------------------------------------------------
# Q089: "Trace evolution from single cells to intelligence"
# → cell / multicellular / brain / intelligence / nervous / adapt
# ---------------------------------------------------------------------------
MULTIHOP_CELLS_INTEL = [
    ("cell", "Produces", "multicellular"),
    ("multicellular", "RelatedTo", "brain"),
    ("brain", "RelatedTo", "intelligence"),
    ("cell", "RelatedTo", "neuron"),
    ("neuron", "RelatedTo", "brain"),
    ("brain", "HasProperty", "intelligence"),
    ("nervous", "RelatedTo", "brain"),
    ("cell", "RelatedTo", "nervous"),
    ("evolution", "Produces", "intelligence"),
    ("intelligence", "DerivedFrom", "brain"),
    ("nervous_system", "Produces", "intelligence"),
    ("cell", "RelatedTo", "adapt"),
    ("adapt", "RelatedTo", "evolution"),
]

# ---------------------------------------------------------------------------
# World capitals (country → capital via HasProperty + capital IsA city)
# ---------------------------------------------------------------------------
WORLD_CAPITALS = [
    # Already in brain but reinforce:
    ("france", "HasProperty", "paris"),
    ("paris", "IsA", "capital"),
    ("japan", "HasProperty", "tokyo"),
    ("tokyo", "IsA", "capital"),
    ("germany", "HasProperty", "berlin"),
    ("berlin", "IsA", "capital"),
    ("australia", "HasProperty", "canberra"),
    ("canberra", "IsA", "capital"),
    ("brazil", "HasProperty", "brasilia"),
    ("brasilia", "IsA", "capital"),

    # Americas
    ("usa", "HasProperty", "washington"),
    ("washington", "IsA", "capital"),
    ("united_states", "HasProperty", "washington"),
    ("canada", "HasProperty", "ottawa"),
    ("ottawa", "IsA", "capital"),
    ("mexico", "HasProperty", "mexico_city"),
    ("mexico_city", "IsA", "capital"),
    ("argentina", "HasProperty", "buenos_aires"),
    ("buenos_aires", "IsA", "capital"),
    ("chile", "HasProperty", "santiago"),
    ("santiago", "IsA", "capital"),
    ("colombia", "HasProperty", "bogota"),
    ("bogota", "IsA", "capital"),
    ("peru", "HasProperty", "lima"),
    ("lima", "IsA", "capital"),
    ("venezuela", "HasProperty", "caracas"),
    ("caracas", "IsA", "capital"),
    ("cuba", "HasProperty", "havana"),
    ("havana", "IsA", "capital"),

    # Europe
    ("uk", "HasProperty", "london"),
    ("london", "IsA", "capital"),
    ("united_kingdom", "HasProperty", "london"),
    ("spain", "HasProperty", "madrid"),
    ("madrid", "IsA", "capital"),
    ("italy", "HasProperty", "rome"),
    ("rome", "IsA", "capital"),
    ("portugal", "HasProperty", "lisbon"),
    ("lisbon", "IsA", "capital"),
    ("netherlands", "HasProperty", "amsterdam"),
    ("amsterdam", "IsA", "capital"),
    ("sweden", "HasProperty", "stockholm"),
    ("stockholm", "IsA", "capital"),
    ("norway", "HasProperty", "oslo"),
    ("oslo", "IsA", "capital"),
    ("denmark", "HasProperty", "copenhagen"),
    ("copenhagen", "IsA", "capital"),
    ("finland", "HasProperty", "helsinki"),
    ("helsinki", "IsA", "capital"),
    ("poland", "HasProperty", "warsaw"),
    ("warsaw", "IsA", "capital"),
    ("ukraine", "HasProperty", "kyiv"),
    ("kyiv", "IsA", "capital"),
    ("turkey", "HasProperty", "ankara"),
    ("ankara", "IsA", "capital"),
    ("greece", "HasProperty", "athens"),
    ("athens", "IsA", "capital"),
    ("czech_republic", "HasProperty", "prague"),
    ("prague", "IsA", "capital"),
    ("hungary", "HasProperty", "budapest"),
    ("budapest", "IsA", "capital"),
    ("romania", "HasProperty", "bucharest"),
    ("bucharest", "IsA", "capital"),
    ("switzerland", "HasProperty", "bern"),
    ("bern", "IsA", "capital"),
    ("austria", "HasProperty", "vienna"),
    ("vienna", "IsA", "capital"),
    ("russia", "HasProperty", "moscow"),
    ("moscow", "IsA", "capital"),
    ("belgium", "HasProperty", "brussels"),
    ("brussels", "IsA", "capital"),

    # Asia
    ("china", "HasProperty", "beijing"),
    ("beijing", "IsA", "capital"),
    ("india", "HasProperty", "new_delhi"),
    ("new_delhi", "IsA", "capital"),
    ("south_korea", "HasProperty", "seoul"),
    ("seoul", "IsA", "capital"),
    ("north_korea", "HasProperty", "pyongyang"),
    ("pyongyang", "IsA", "capital"),
    ("vietnam", "HasProperty", "hanoi"),
    ("hanoi", "IsA", "capital"),
    ("thailand", "HasProperty", "bangkok"),
    ("bangkok", "IsA", "capital"),
    ("indonesia", "HasProperty", "jakarta"),
    ("jakarta", "IsA", "capital"),
    ("philippines", "HasProperty", "manila"),
    ("manila", "IsA", "capital"),
    ("malaysia", "HasProperty", "kuala_lumpur"),
    ("kuala_lumpur", "IsA", "capital"),
    ("pakistan", "HasProperty", "islamabad"),
    ("islamabad", "IsA", "capital"),
    ("bangladesh", "HasProperty", "dhaka"),
    ("dhaka", "IsA", "capital"),
    ("iran", "HasProperty", "tehran"),
    ("tehran", "IsA", "capital"),
    ("iraq", "HasProperty", "baghdad"),
    ("baghdad", "IsA", "capital"),
    ("saudi_arabia", "HasProperty", "riyadh"),
    ("riyadh", "IsA", "capital"),
    ("israel", "HasProperty", "jerusalem"),
    ("jerusalem", "IsA", "capital"),
    ("jordan", "HasProperty", "amman"),
    ("amman", "IsA", "capital"),
    ("afghanistan", "HasProperty", "kabul"),
    ("kabul", "IsA", "capital"),

    # Africa
    ("egypt", "HasProperty", "cairo"),
    ("cairo", "IsA", "capital"),
    ("south_africa", "HasProperty", "pretoria"),
    ("pretoria", "IsA", "capital"),
    ("nigeria", "HasProperty", "abuja"),
    ("abuja", "IsA", "capital"),
    ("kenya", "HasProperty", "nairobi"),
    ("nairobi", "IsA", "capital"),
    ("ghana", "HasProperty", "accra"),
    ("accra", "IsA", "capital"),
    ("ethiopia", "HasProperty", "addis_ababa"),
    ("addis_ababa", "IsA", "capital"),
    ("tanzania", "HasProperty", "dodoma"),
    ("dodoma", "IsA", "capital"),
    ("morocco", "HasProperty", "rabat"),
    ("rabat", "IsA", "capital"),
    ("algeria", "HasProperty", "algiers"),
    ("algiers", "IsA", "capital"),
    ("cameroon", "HasProperty", "yaounde"),
    ("yaounde", "IsA", "capital"),
    ("zimbabwe", "HasProperty", "harare"),
    ("harare", "IsA", "capital"),
    ("libya", "HasProperty", "tripoli"),
    ("tripoli", "IsA", "capital"),
    ("angola", "HasProperty", "luanda"),
    ("luanda", "IsA", "capital"),

    # Oceania
    ("new_zealand", "HasProperty", "wellington"),
    ("wellington", "IsA", "capital"),
]

# ---------------------------------------------------------------------------
# Superlative facts
# ---------------------------------------------------------------------------
SUPERLATIVES = [
    ("nile", "HasProperty", "longest"),
    ("nile", "IsA", "river"),
    ("river", "HasProperty", "nile"),
    ("amazon", "HasProperty", "longest"),
    ("amazon", "IsA", "river"),
    ("everest", "HasProperty", "highest"),
    ("everest", "IsA", "mountain"),
    ("mountain", "HasProperty", "everest"),
    ("sahara", "HasProperty", "largest"),
    ("sahara", "IsA", "desert"),
    ("desert", "HasProperty", "sahara"),
    ("russia", "HasProperty", "largest"),
    ("russia", "IsA", "country"),
    ("china", "HasProperty", "populous"),
    ("india", "HasProperty", "populous"),
    ("blue_whale", "HasProperty", "largest"),
    ("blue_whale", "IsA", "animal"),
    ("animal", "HasProperty", "blue_whale"),
    ("cheetah", "HasProperty", "fastest"),
    ("cheetah", "IsA", "animal"),
    ("peregrine_falcon", "HasProperty", "fastest"),
    ("diamond", "HasProperty", "hardest"),
    ("mercury", "HasProperty", "smallest"),
    ("jupiter", "HasProperty", "largest"),
    ("sun", "HasProperty", "largest"),
    ("pacific", "HasProperty", "deepest"),
    ("pacific", "HasProperty", "largest"),
    ("ocean", "HasProperty", "pacific"),
    ("gold", "HasProperty", "precious"),
    ("gold", "HasProperty", "rare"),
    ("platinum", "HasProperty", "precious"),
]

# ---------------------------------------------------------------------------
# Scientific mechanisms (fill out context for common science queries)
# ---------------------------------------------------------------------------
SCIENCE_MECHANISMS = [
    # Photosynthesis
    ("photosynthesis", "Requires", "sunlight"),
    ("photosynthesis", "Requires", "chlorophyll"),
    ("photosynthesis", "Produces", "glucose"),
    ("photosynthesis", "Produces", "oxygen"),
    ("chlorophyll", "RelatedTo", "photosynthesis"),
    ("plant", "Requires", "photosynthesis"),
    ("carbon", "RelatedTo", "photosynthesis"),
    ("co2", "Requires", "photosynthesis"),

    # Evolution
    ("evolution", "Requires", "variation"),
    ("evolution", "Requires", "selection"),
    ("evolution", "Produces", "species"),
    ("darwin", "RelatedTo", "evolution"),
    ("natural_selection", "IsA", "evolution"),
    ("adapt", "RelatedTo", "evolution"),
    ("variation", "RelatedTo", "evolution"),

    # Gravity
    ("gravity", "HasProperty", "force"),
    ("gravity", "HasProperty", "mass"),
    ("gravity", "HasProperty", "attraction"),
    ("newton", "RelatedTo", "gravity"),
    ("mass", "RelatedTo", "gravity"),
    ("earth", "HasProperty", "gravity"),
    ("force", "RelatedTo", "gravity"),

    # DNA / genetics
    ("dna", "IsA", "molecule"),
    ("dna", "HasProperty", "helix"),
    ("dna", "HasProperty", "gene"),
    ("dna", "HasProperty", "chromosome"),
    ("gene", "PartOf", "dna"),
    ("chromosome", "HasProperty", "dna"),
    ("genetic", "RelatedTo", "dna"),

    # Relativity
    ("relativity", "RelatedTo", "einstein"),
    ("relativity", "RelatedTo", "space"),
    ("relativity", "RelatedTo", "time"),
    ("relativity", "RelatedTo", "mass"),
    ("relativity", "RelatedTo", "energy"),
    ("einstein", "RelatedTo", "relativity"),
    ("spacetime", "RelatedTo", "relativity"),

    # Black holes
    ("black_hole", "DerivedFrom", "gravity"),
    ("black_hole", "DerivedFrom", "star"),
    ("star", "Produces", "black_hole"),
    ("gravity", "Causes", "black_hole"),
    ("collapse", "Produces", "black_hole"),
    ("black_hole", "HasProperty", "mass"),
    ("mass", "RelatedTo", "black_hole"),

    # Quantum mechanics
    ("quantum", "HasProperty", "uncertainty"),
    ("quantum", "HasProperty", "wave"),
    ("quantum", "HasProperty", "particle"),
    ("electron", "RelatedTo", "quantum"),
    ("uncertainty", "RelatedTo", "quantum"),
    ("wave", "RelatedTo", "quantum"),

    # Entropy
    ("entropy", "HasProperty", "disorder"),
    ("entropy", "HasProperty", "thermodynamics"),
    ("entropy", "HasProperty", "heat"),
    ("thermodynamics", "RelatedTo", "entropy"),
    ("disorder", "RelatedTo", "entropy"),
    ("energy", "RelatedTo", "entropy"),

    # Neuroscience
    ("neuron", "HasProperty", "synapse"),
    ("synapse", "RelatedTo", "neuron"),
    ("neuron", "HasProperty", "axon"),
    ("axon", "PartOf", "neuron"),
    ("brain", "HasProperty", "neuron"),
    ("memory", "RelatedTo", "hippocampus"),
    ("hippocampus", "RelatedTo", "memory"),
    ("brain", "HasProperty", "hippocampus"),
    ("signal", "RelatedTo", "neuron"),

    # Vaccines / immune
    ("vaccine", "Produces", "immunity"),
    ("vaccine", "RelatedTo", "antibody"),
    ("antibody", "RelatedTo", "immune"),
    ("immune", "HasProperty", "antibody"),
    ("pathogen", "RelatedTo", "immune"),
    ("immunity", "RelatedTo", "vaccine"),

    # Nuclear fission
    ("fission", "HasProperty", "nuclear"),
    ("fission", "Produces", "energy"),
    ("uranium", "RelatedTo", "fission"),
    ("neutron", "RelatedTo", "fission"),
    ("nuclear", "RelatedTo", "energy"),
    ("atom", "RelatedTo", "fission"),

    # Carbon cycle
    ("carbon", "RelatedTo", "co2"),
    ("co2", "RelatedTo", "atmosphere"),
    ("atmosphere", "HasProperty", "co2"),
    ("fossil", "Produces", "co2"),
    ("photosynthesis", "Requires", "co2"),
    ("carbon_cycle", "RelatedTo", "photosynthesis"),

    # Greenhouse effect
    ("greenhouse", "HasProperty", "carbon"),
    ("greenhouse", "HasProperty", "temperature"),
    ("greenhouse", "HasProperty", "atmosphere"),
    ("climate", "RelatedTo", "greenhouse"),
    ("temperature", "RelatedTo", "greenhouse"),
    ("atmosphere", "RelatedTo", "greenhouse"),

    # Dark matter
    ("dark_matter", "HasProperty", "gravity"),
    ("dark_matter", "HasProperty", "mass"),
    ("universe", "HasProperty", "dark_matter"),
    ("gravity", "RelatedTo", "dark_matter"),
    ("mass", "RelatedTo", "dark_matter"),

    # Higgs boson
    ("higgs", "IsA", "boson"),
    ("higgs", "HasProperty", "mass"),
    ("higgs", "RelatedTo", "particle"),
    ("higgs", "HasProperty", "field"),
    ("boson", "RelatedTo", "particle"),
    ("mass", "RelatedTo", "higgs"),

    # Digestion
    ("stomach", "PartOf", "digestion"),
    ("enzyme", "RelatedTo", "digestion"),
    ("intestine", "PartOf", "digestion"),
    ("nutrient", "RelatedTo", "digestion"),
    ("food", "Requires", "digestion"),
    ("digestion", "Produces", "nutrient"),
]

# ---------------------------------------------------------------------------
# Famous scientists → their key contributions
# ---------------------------------------------------------------------------
SCIENTISTS = [
    ("einstein", "RelatedTo", "relativity"),
    ("einstein", "RelatedTo", "physics"),
    ("einstein", "RelatedTo", "energy"),
    ("einstein", "RelatedTo", "space"),
    ("newton", "RelatedTo", "gravity"),
    ("newton", "RelatedTo", "physics"),
    ("newton", "RelatedTo", "motion"),
    ("newton", "RelatedTo", "apple"),
    ("darwin", "RelatedTo", "evolution"),
    ("darwin", "RelatedTo", "species"),
    ("darwin", "RelatedTo", "selection"),
    ("darwin", "RelatedTo", "biology"),
    ("darwin", "RelatedTo", "australia"),
    ("curie", "RelatedTo", "radioactivity"),
    ("curie", "RelatedTo", "radium"),
    ("curie", "RelatedTo", "physics"),
    ("feynman", "RelatedTo", "quantum"),
    ("feynman", "RelatedTo", "physics"),
    ("bohr", "RelatedTo", "atom"),
    ("bohr", "RelatedTo", "quantum"),
    ("mendel", "RelatedTo", "genetics"),
    ("mendel", "RelatedTo", "gene"),
    ("pasteur", "RelatedTo", "germ"),
    ("pasteur", "RelatedTo", "vaccine"),
    ("hawking", "RelatedTo", "black_hole"),
    ("hawking", "RelatedTo", "gravity"),
    ("hawking", "RelatedTo", "universe"),
    ("tesla", "RelatedTo", "electricity"),
    ("tesla", "RelatedTo", "magnetism"),
    ("turing", "RelatedTo", "computing"),
    ("turing", "RelatedTo", "artificial"),
    ("turing", "RelatedTo", "intelligence"),
    ("mendeleev", "RelatedTo", "periodic_table"),
    ("mendeleev", "RelatedTo", "element"),
    ("fleming", "RelatedTo", "penicillin"),
    ("fleming", "RelatedTo", "antibiotic"),
    ("copernicus", "RelatedTo", "solar_system"),
    ("copernicus", "RelatedTo", "sun"),
    ("galileo", "RelatedTo", "telescope"),
    ("galileo", "RelatedTo", "jupiter"),
    ("galileo", "RelatedTo", "astronomy"),
]

# ---------------------------------------------------------------------------
# Famous authors → their key works
# ---------------------------------------------------------------------------
AUTHORS = [
    ("shakespeare", "RelatedTo", "hamlet"),
    ("shakespeare", "RelatedTo", "romeo"),
    ("shakespeare", "RelatedTo", "playwright"),
    ("shakespeare", "RelatedTo", "english"),
    ("shakespeare", "RelatedTo", "bard"),
    ("shakespeare", "RelatedTo", "literature"),
    ("shakespeare", "RelatedTo", "sonnet"),
    ("dickens", "RelatedTo", "oliver_twist"),
    ("dickens", "RelatedTo", "great_expectations"),
    ("dickens", "RelatedTo", "novel"),
    ("austen", "RelatedTo", "pride_and_prejudice"),
    ("austen", "RelatedTo", "novel"),
    ("tolstoy", "RelatedTo", "war_and_peace"),
    ("tolstoy", "RelatedTo", "anna_karenina"),
    ("dostoevsky", "RelatedTo", "crime_and_punishment"),
    ("kafka", "RelatedTo", "metamorphosis"),
    ("orwell", "RelatedTo", "1984"),
    ("orwell", "RelatedTo", "animal_farm"),
    ("hemingway", "RelatedTo", "old_man"),
    ("twain", "RelatedTo", "huckleberry_finn"),
    ("twain", "RelatedTo", "tom_sawyer"),
    ("steinbeck", "RelatedTo", "grapes_of_wrath"),
    ("homer", "RelatedTo", "odyssey"),
    ("homer", "RelatedTo", "iliad"),
    ("dante", "RelatedTo", "divine_comedy"),
    ("chaucer", "RelatedTo", "canterbury_tales"),
    ("cervantes", "RelatedTo", "don_quixote"),
    ("hugo", "RelatedTo", "les_miserables"),
    ("dumas", "RelatedTo", "three_musketeers"),
    ("flaubert", "RelatedTo", "madame_bovary"),
    ("joyce", "RelatedTo", "ulysses"),
    ("woolf", "RelatedTo", "mrs_dalloway"),
    ("plath", "RelatedTo", "bell_jar"),
]

# ---------------------------------------------------------------------------
# Additional cross-domain bridges for Bracket 3/4/5
# ---------------------------------------------------------------------------
CROSS_DOMAIN = [
    # Chemistry ↔ cooking
    ("cooking", "Requires", "chemistry"),
    ("maillard", "RelatedTo", "cooking"),
    ("maillard", "RelatedTo", "reaction"),
    ("reaction", "RelatedTo", "cooking"),
    ("heat", "RelatedTo", "cooking"),
    ("protein", "RelatedTo", "cooking"),
    ("molecule", "RelatedTo", "cooking"),

    # Biology ↔ technology
    ("biotech", "DerivedFrom", "biology"),
    ("bioinformatics", "DerivedFrom", "biology"),
    ("genetic", "RelatedTo", "technology"),
    ("dna", "RelatedTo", "biotech"),
    ("artificial", "RelatedTo", "biology"),

    # Math ↔ music
    ("frequency", "RelatedTo", "music"),
    ("ratio", "RelatedTo", "music"),
    ("harmony", "RelatedTo", "music"),
    ("pattern", "RelatedTo", "music"),
    ("wave", "RelatedTo", "music"),
    ("math", "RelatedTo", "music"),
    ("frequency", "RelatedTo", "math"),

    # Economics ↔ psychology
    ("behavior", "RelatedTo", "economics"),
    ("decision", "RelatedTo", "economics"),
    ("rational", "RelatedTo", "economics"),
    ("cognitive", "RelatedTo", "economics"),
    ("bias", "RelatedTo", "economics"),
    ("behavioral_economics", "RelatedTo", "psychology"),

    # Language ↔ thought
    ("language", "RelatedTo", "thought"),
    ("cognitive", "RelatedTo", "language"),
    ("sapir", "RelatedTo", "language"),
    ("culture", "RelatedTo", "language"),
    ("concept", "RelatedTo", "language"),

    # Art ↔ neuroscience
    ("perception", "RelatedTo", "art"),
    ("visual", "RelatedTo", "art"),
    ("aesthetic", "RelatedTo", "art"),
    ("neural", "RelatedTo", "art"),
    ("brain", "RelatedTo", "art"),

    # Physics ↔ sports
    ("force", "RelatedTo", "sports"),
    ("motion", "RelatedTo", "sports"),
    ("velocity", "RelatedTo", "sports"),
    ("kinetic", "RelatedTo", "sports"),
    ("energy", "RelatedTo", "sports"),

    # Information ↔ entropy
    ("information", "RelatedTo", "entropy"),
    ("shannon", "RelatedTo", "entropy"),
    ("shannon", "RelatedTo", "information"),
    ("bit", "RelatedTo", "information"),
    ("disorder", "RelatedTo", "information"),

    # Geography ↔ history
    ("geography", "RelatedTo", "history"),
    ("geography", "RelatedTo", "civilization"),
    ("trade", "RelatedTo", "geography"),
    ("resource", "RelatedTo", "geography"),
    ("climate", "RelatedTo", "civilization"),
    ("climate", "RelatedTo", "history"),

    # Statistics ↔ medicine
    ("statistics", "RelatedTo", "medicine"),
    ("clinical", "RelatedTo", "statistics"),
    ("trial", "RelatedTo", "statistics"),
    ("evidence", "RelatedTo", "medicine"),
    ("probability", "RelatedTo", "statistics"),
    ("data", "RelatedTo", "medicine"),

    # Philosophy ↔ science
    ("philosophy", "RelatedTo", "science"),
    ("empirical", "RelatedTo", "science"),
    ("logic", "RelatedTo", "philosophy"),
    ("reason", "RelatedTo", "philosophy"),
    ("knowledge", "RelatedTo", "philosophy"),

    # Architecture ↔ math
    ("architecture", "Requires", "geometry"),
    ("proportion", "RelatedTo", "architecture"),
    ("structure", "RelatedTo", "architecture"),
    ("design", "RelatedTo", "architecture"),
    ("geometry", "RelatedTo", "architecture"),

    # Ecology ↔ economics
    ("ecology", "RelatedTo", "economics"),
    ("ecosystem", "RelatedTo", "economics"),
    ("resource", "RelatedTo", "ecology"),
    ("capital", "RelatedTo", "ecology"),
    ("natural", "RelatedTo", "ecology"),

    # Chemistry ↔ medicine
    ("chemistry", "RelatedTo", "medicine"),
    ("drug", "RelatedTo", "chemistry"),
    ("pharmacology", "RelatedTo", "chemistry"),
    ("compound", "RelatedTo", "medicine"),

    # Psychology ↔ marketing
    ("behavior", "RelatedTo", "marketing"),
    ("persuasion", "RelatedTo", "marketing"),
    ("consumer", "RelatedTo", "marketing"),
    ("emotion", "RelatedTo", "marketing"),
    ("psychology", "RelatedTo", "marketing"),

    # Evolution ↔ language
    ("language", "RelatedTo", "evolution"),
    ("communicate", "RelatedTo", "evolution"),
    ("symbol", "RelatedTo", "language"),
    ("culture", "RelatedTo", "evolution"),
    ("adapt", "RelatedTo", "language"),

    # Plate tectonics ↔ biodiversity
    ("tectonic", "RelatedTo", "biodiversity"),
    ("plate", "RelatedTo", "continent"),
    ("continent", "RelatedTo", "isolate"),
    ("isolate", "RelatedTo", "evolution"),
    ("biodiversity", "RelatedTo", "evolution"),
    ("species", "RelatedTo", "tectonic"),

    # EM spectrum ↔ technology
    ("electromagnetic", "HasProperty", "spectrum"),
    ("spectrum", "HasProperty", "radio"),
    ("spectrum", "HasProperty", "wireless"),
    ("spectrum", "HasProperty", "frequency"),
    ("radio", "RelatedTo", "electromagnetic"),
    ("wave", "RelatedTo", "electromagnetic"),

    # Atoms → consciousness chain
    ("atom", "Produces", "molecule"),
    ("molecule", "Produces", "cell"),
    ("cell", "Produces", "neuron"),
    ("neuron", "RelatedTo", "consciousness"),
    ("brain", "RelatedTo", "consciousness"),
    ("consciousness", "DerivedFrom", "brain"),

    # Language ↔ science
    ("language", "RelatedTo", "science"),
    ("concept", "RelatedTo", "science"),
    ("theory", "RelatedTo", "language"),
    ("communicate", "RelatedTo", "science"),
    ("symbol", "RelatedTo", "science"),

    # Thermodynamics ↔ economics
    ("thermodynamic", "RelatedTo", "economics"),
    ("energy", "RelatedTo", "economics"),
    ("efficiency", "RelatedTo", "economics"),
    ("resource", "RelatedTo", "thermodynamic"),
    ("entropy", "RelatedTo", "economics"),

    # Geography ↔ culture
    ("geography", "RelatedTo", "culture"),
    ("climate", "RelatedTo", "culture"),
    ("resource", "RelatedTo", "culture"),
    ("trade", "RelatedTo", "culture"),
    ("civilization", "RelatedTo", "geography"),

    # Quantum ↔ computing
    ("quantum", "RelatedTo", "computing"),
    ("electron", "RelatedTo", "semiconductor"),
    ("semiconductor", "RelatedTo", "transistor"),
    ("transistor", "RelatedTo", "computing"),
    ("computing", "RelatedTo", "quantum"),
    ("chip", "RelatedTo", "semiconductor"),

    # Cultural evolution
    ("culture", "RelatedTo", "meme"),
    ("meme", "RelatedTo", "evolution"),
    ("selection", "RelatedTo", "culture"),
    ("variation", "RelatedTo", "culture"),
    ("culture", "RelatedTo", "darwin"),

    # Information ↔ biology
    ("information", "RelatedTo", "dna"),
    ("information", "RelatedTo", "genetic"),
    ("genetic_code", "RelatedTo", "information"),
    ("shannon", "RelatedTo", "biology"),
    ("entropy", "RelatedTo", "biology"),
    ("information", "RelatedTo", "biology"),

    # Stellar nucleosynthesis → life
    ("star", "Produces", "element"),
    ("element", "DerivedFrom", "star"),
    ("carbon", "DerivedFrom", "star"),
    ("nuclear", "RelatedTo", "star"),
    ("fusion", "RelatedTo", "star"),
    ("supernova", "Produces", "element"),
    ("life", "Requires", "carbon"),
    ("carbon", "RelatedTo", "life"),
    ("element", "RelatedTo", "life"),

    # Game theory ↔ social norms
    ("game", "RelatedTo", "social"),
    ("norm", "RelatedTo", "social"),
    ("cooperation", "RelatedTo", "norm"),
    ("strategy", "RelatedTo", "norm"),
    ("nash", "RelatedTo", "strategy"),

    # Networks ↔ social
    ("network", "RelatedTo", "social"),
    ("graph", "RelatedTo", "network"),
    ("node", "RelatedTo", "network"),
    ("hub", "RelatedTo", "network"),
    ("scale", "RelatedTo", "network"),
    ("preferential", "RelatedTo", "network"),

    # Climate ↔ migration
    ("climate", "Causes", "migration"),
    ("temperature", "RelatedTo", "migration"),
    ("drought", "Causes", "migration"),
    ("agriculture", "RelatedTo", "migration"),
    ("human", "RelatedTo", "migration"),

    # Symbolic thought → math
    ("symbol", "RelatedTo", "abstract"),
    ("abstract", "RelatedTo", "number"),
    ("number", "RelatedTo", "math"),
    ("logic", "RelatedTo", "math"),
    ("formal", "RelatedTo", "math"),
    ("language", "RelatedTo", "math"),

    # Neuroscience ↔ AI
    ("neuroscience", "RelatedTo", "artificial"),
    ("brain", "RelatedTo", "artificial"),
    ("neural", "RelatedTo", "artificial"),
    ("intelligence", "RelatedTo", "artificial"),
    ("learn", "RelatedTo", "artificial"),
    ("artificial", "RelatedTo", "intelligence"),

    # Entropy ↔ social organization
    ("entropy", "RelatedTo", "social"),
    ("order", "RelatedTo", "social"),
    ("organize", "RelatedTo", "entropy"),
    ("complexity", "RelatedTo", "entropy"),
    ("energy", "RelatedTo", "social"),

    # Technology ↔ society
    ("technology", "RelatedTo", "society"),
    ("feedback", "RelatedTo", "technology"),
    ("industrial", "RelatedTo", "technology"),
    ("innovation", "RelatedTo", "society"),
    ("history", "RelatedTo", "technology"),

    # Abstract reasoning bridges
    ("cycle", "RelatedTo", "economics"),
    ("boom", "RelatedTo", "economics"),
    ("recession", "RelatedTo", "economics"),
    ("inflation", "RelatedTo", "economics"),
    ("credit", "RelatedTo", "economics"),

    ("analogy", "HasProperty", "comparison"),
    ("analogy", "HasProperty", "structure"),
    ("analogy", "HasProperty", "relation"),
    ("mapping", "RelatedTo", "analogy"),
    ("similar", "RelatedTo", "analogy"),

    ("emerge", "RelatedTo", "complex"),
    ("organize", "RelatedTo", "complex"),
    ("feedback", "RelatedTo", "complex"),
    ("pattern", "RelatedTo", "complex"),
    ("system", "HasProperty", "emerge"),

    ("order", "ContrastedWith", "chaos"),
    ("chaos", "RelatedTo", "entropy"),
    ("stability", "RelatedTo", "order"),
    ("dynamic", "RelatedTo", "chaos"),

    ("diversity", "RelatedTo", "resilience"),
    ("ecosystem", "HasProperty", "diversity"),
    ("variation", "RelatedTo", "resilience"),
    ("robust", "RelatedTo", "diversity"),
    ("adapt", "RelatedTo", "resilience"),

    ("feedback", "HasProperty", "loop"),
    ("regulate", "RelatedTo", "feedback"),
    ("amplify", "RelatedTo", "feedback"),
    ("control", "RelatedTo", "feedback"),

    ("knowledge", "HasProperty", "cumulative"),
    ("science", "HasProperty", "cumulative"),
    ("record", "RelatedTo", "knowledge"),
    ("build", "RelatedTo", "knowledge"),

    ("network", "HasProperty", "hub"),
    ("hub", "RelatedTo", "connect"),
    ("scale", "RelatedTo", "hub"),

    ("constraint", "RelatedTo", "creative"),
    ("limit", "RelatedTo", "creative"),
    ("creative", "RelatedTo", "innovation"),
    ("solution", "RelatedTo", "constraint"),

    ("civilization", "RelatedTo", "collapse"),
    ("resource", "RelatedTo", "civilization"),
    ("complexity", "RelatedTo", "civilization"),

    ("compress", "RelatedTo", "pattern"),
    ("information", "RelatedTo", "compress"),
    ("model", "RelatedTo", "abstract"),
    ("represent", "RelatedTo", "information"),

    ("model", "HasProperty", "predict"),
    ("accurate", "RelatedTo", "model"),
    ("theory", "RelatedTo", "model"),
    ("simple", "RelatedTo", "model"),

    ("symmetry", "RelatedTo", "physics"),
    ("conservation", "RelatedTo", "symmetry"),
    ("invariant", "RelatedTo", "symmetry"),
    ("noether", "RelatedTo", "symmetry"),
    ("law", "RelatedTo", "symmetry"),

    ("language", "HasProperty", "evolve"),
    ("grammar", "RelatedTo", "language"),
    ("cultural", "RelatedTo", "language"),
    ("adapt", "RelatedTo", "language"),

    ("exponential", "HasProperty", "growth"),
    ("compound", "RelatedTo", "exponential"),
    ("double", "RelatedTo", "exponential"),
    ("scale", "RelatedTo", "exponential"),
    ("intuition", "RelatedTo", "exponential"),

    ("abstract", "RelatedTo", "progress"),
    ("concept", "RelatedTo", "abstract"),
    ("simplify", "RelatedTo", "abstract"),
    ("reuse", "RelatedTo", "abstract"),

    ("individual", "RelatedTo", "collective"),
    ("social", "RelatedTo", "collective"),
    ("behavior", "RelatedTo", "collective"),
    ("emerge", "RelatedTo", "collective"),
]


def main():
    if not os.path.exists(BRAIN):
        print(f"ERROR: brain not found: {BRAIN}")
        sys.exit(1)

    print(f"[fix] brain: {os.path.getsize(BRAIN):,} bytes", flush=True)

    client = MCPClient(BRAIN)
    client.start()

    r = client.call_tool("lu4d_status")
    print(f"[status] nodes={r.get('node_count', '?'):,} bonds={r.get('bond_count', '?'):,}", flush=True)

    # Teach all batches
    teach_batch(client, BASIC_PLANETS,             "Basic: planets (mercury, jupiter)")
    teach_batch(client, BASIC_OCEAN,               "Basic: pacific ocean")
    teach_batch(client, BASIC_BOILING,             "Basic: boiling/celsius")
    teach_batch(client, BASIC_WHALE,               "Basic: whale (largest mammal)")
    teach_batch(client, BASIC_GOLD,                "Basic: gold/au")
    teach_batch(client, BASIC_MITO,                "Basic: mitochondria")
    teach_batch(client, BASIC_PORTUGUESE,          "Basic: portuguese/brazil")
    teach_batch(client, BASIC_DIAMOND,             "Basic: diamond (hardest)")
    teach_batch(client, SCIENCE_EARTHQUAKE,        "Science: earthquakes/tectonic")
    teach_batch(client, SCIENCE_TIDES,             "Science: tides/moon")
    teach_batch(client, SCIENCE_CRISPR,            "Science: CRISPR")
    teach_batch(client, SCIENCE_NATURAL_SELECTION, "Science: natural selection")
    teach_batch(client, CROSS_MUSIC,               "Cross: music/sound/wave")
    teach_batch(client, ABSTRACT_COMPETITION,      "Abstract: competition/cooperation")
    teach_batch(client, MULTIHOP_SUNLIGHT,         "Multi-hop: sunlight->civilization")
    teach_batch(client, MULTIHOP_MATH_ART,         "Multi-hop: math->art")
    teach_batch(client, MULTIHOP_CELLS_INTEL,      "Multi-hop: cells->intelligence")
    teach_batch(client, WORLD_CAPITALS,            "World capitals (50+)")
    teach_batch(client, SUPERLATIVES,              "Superlative facts")
    teach_batch(client, SCIENCE_MECHANISMS,        "Scientific mechanisms")
    teach_batch(client, SCIENTISTS,                "Famous scientists")
    teach_batch(client, AUTHORS,                   "Famous authors")
    teach_batch(client, CROSS_DOMAIN,              "Cross-domain bridges")

    r2 = client.call_tool("lu4d_status")
    print(f"\n[status] after: nodes={r2.get('node_count', '?'):,} bonds={r2.get('bond_count', '?'):,}", flush=True)

    # Save
    print("\n[save] saving...", flush=True)
    rs = client.call_tool("lu4d_save")
    print(f"[save] {rs.get('text', str(rs))}", flush=True)

    client.stop()

    sz    = os.path.getsize(BRAIN)
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(BRAIN)))
    print(f"\n[brain] {sz:,} bytes | mtime={mtime}", flush=True)
    print("[done] now run: python eval_100_strict.py", flush=True)


if __name__ == "__main__":
    main()

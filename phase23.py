"""
Lu 4D Phase 2 + Phase 3: Science deep dive + History/Culture.

Loads the Phase 1 baseline brain (lu4d_curriculum.lbr) and layers two new
temporal sessions on top:

  Session 2 - Science: physics, chemistry, biology, medicine, earth science
              pulled from curated tables + sciq + openbookqa + ai2_arc + scitail.

  Session 3 - History/Culture: empires, wars, rulers, religions, arts,
              literature, mythology, music, sports, curated + trivia_qa +
              ConceptNet (further pass).

Each phase:
  start_session -> teach (with dedup) -> end_session -> consolidate -> sanity.

The MCP process persists the brain to disk once on shutdown (patched
behaviour - see src/main.rs + src/mcp.rs).
"""

import io
import json
import os
import re
import subprocess
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
LU4D_BIN = os.path.join(ROOT, "target", "release", "lu4d.exe")
BRAIN_FILE = os.path.join(ROOT, "lu4d_curriculum.lbr")

PROGRESS_EVERY = 10_000

# ---------------------------------------------------------------------------
# MCP client (copied from phase1_foundation.py - keep self-contained)
# ---------------------------------------------------------------------------

class MCPClient:
    def __init__(self, brain_path, no_save=True):
        self.brain_path = brain_path
        self.no_save = no_save
        self.proc = None
        self.next_id = 1

    def start(self):
        cmd = [LU4D_BIN, "--mcp", "--brain", self.brain_path]
        if self.no_save:
            cmd.append("--no-save")
        print(f"[mcp] spawning: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
        )
        self.request("initialize", {})
        self.notify("notifications/initialized", {})

    def stop(self):
        if self.proc:
            try: self.proc.stdin.close()
            except Exception: pass
            try: self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired: self.proc.terminate()

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _read(self):
        line = self.proc.stdout.readline()
        if not line: raise RuntimeError("MCP server closed stdout")
        return json.loads(line)

    def request(self, method, params):
        rid = self.next_id; self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        resp = self._read()
        while resp.get("id") != rid: resp = self._read()
        if "error" in resp: raise RuntimeError(f"MCP error {resp['error']}")
        return resp["result"]

    def notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def call_tool(self, name, args):
        return self.request("tools/call", {"name": name, "arguments": args})


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

STOPWORDS = set("""
a an the of to in on at by for with from into onto off and or but not no nor
is are was were be been being am has have had having do does did doing done
this that these those it its as if then than so such also very most more less
can could may might should would will shall must ought some any all both each
other another one two three four five six seven eight nine ten
i you he she we they me him her us them my your his their our its mine yours hers
what which who whom whose when where why how there here
about above below over under between among during before after since until while
because though although however therefore thus hence due
only just even still yet already about up down out
""".split())

_norm_re = re.compile(r"[^a-z0-9\s\-]")

def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = _norm_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_single_token(s: str) -> bool:
    return bool(s) and " " not in s and len(s) >= 2 and len(s) <= 30 and not s.isdigit()

def content_tokens(text):
    out = []
    for tok in normalize(text).split():
        if len(tok) < 3 or len(tok) > 25: continue
        if tok in STOPWORDS: continue
        if tok.isdigit(): continue
        out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Teach helper with dedup + progress
# ---------------------------------------------------------------------------

def teach_triples(client, triples_iter, source, weight=0.85, confidence="Verified",
                  taught_cache=None):
    """Push (s, rel, o) triples; emits forward + RelatedTo reverse bond each."""
    taught = 0
    errors = 0
    skipped = 0
    start = time.time()
    cache = taught_cache if taught_cache is not None else set()
    for (s, rel, o) in triples_iter:
        if not s or not o or s == o:
            errors += 1; continue
        if not (is_single_token(s) and is_single_token(o)):
            errors += 1; continue
        key = (s, rel, o)
        if key in cache:
            skipped += 1; continue
        cache.add(key)
        try:
            client.call_tool("lu4d_teach", {
                "subject": s, "relation": rel, "object": o,
                "weight": weight, "confidence": confidence,
            })
            taught += 1
            # Reverse bond for bidirectional traversal
            rkey = (o, "RelatedTo", s)
            if rkey not in cache:
                cache.add(rkey)
                client.call_tool("lu4d_teach", {
                    "subject": o, "relation": "RelatedTo", "object": s,
                    "weight": weight * 0.9, "confidence": confidence,
                })
                taught += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"[{source}] teach error {s!r}->{o!r}: {e}", flush=True)
        if taught and taught % PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            rate = taught / elapsed if elapsed > 0 else 0
            print(f"[{source}] taught={taught} skipped={skipped} "
                  f"({rate:.0f}/s, {elapsed:.1f}s)", flush=True)
    elapsed = time.time() - start
    rate = taught / elapsed if elapsed > 0 else 0
    print(f"[{source}] DONE taught={taught} skipped={skipped} errors={errors} "
          f"in {elapsed:.1f}s ({rate:.0f}/s)", flush=True)
    return taught


# ===========================================================================
# PHASE 2 — SCIENCE
# ===========================================================================

# --- Chemistry extensions ---
CHEMISTRY = [
    # Acids, bases, salts
    ("hydrochloric", "IsA", "acid"), ("sulfuric", "IsA", "acid"),
    ("nitric", "IsA", "acid"), ("acetic", "IsA", "acid"),
    ("citric", "IsA", "acid"), ("lactic", "IsA", "acid"),
    ("phosphoric", "IsA", "acid"), ("carbonic", "IsA", "acid"),
    ("sodium", "Produces", "hydroxide"), ("hydroxide", "IsA", "base"),
    ("ammonia", "IsA", "base"), ("ph", "IsA", "scale"),
    ("acid", "Opposes", "base"), ("acid", "HasProperty", "sour"),
    ("base", "HasProperty", "bitter"),
    # States of matter
    ("solid", "IsA", "state"), ("liquid", "IsA", "state"),
    ("gas", "IsA", "state"), ("plasma", "IsA", "state"),
    ("ice", "IsA", "solid"), ("water", "IsA", "liquid"),
    ("steam", "IsA", "gas"), ("lightning", "HasProperty", "plasma"),
    # Organic chemistry
    ("methane", "IsA", "alkane"), ("ethane", "IsA", "alkane"),
    ("propane", "IsA", "alkane"), ("butane", "IsA", "alkane"),
    ("benzene", "IsA", "aromatic"), ("ethanol", "IsA", "alcohol"),
    ("methanol", "IsA", "alcohol"), ("glucose", "IsA", "sugar"),
    ("fructose", "IsA", "sugar"), ("sucrose", "IsA", "sugar"),
    ("lactose", "IsA", "sugar"), ("starch", "IsA", "carbohydrate"),
    ("cellulose", "IsA", "carbohydrate"), ("protein", "IsA", "polymer"),
    ("dna", "IsA", "polymer"), ("lipid", "IsA", "molecule"),
    ("fat", "IsA", "lipid"), ("oil", "IsA", "lipid"),
    ("cholesterol", "IsA", "lipid"),
    # Chemical bonds / reactions
    ("ionic", "IsA", "bond"), ("covalent", "IsA", "bond"),
    ("hydrogen", "IsA", "bond"), ("combustion", "Requires", "oxygen"),
    ("oxidation", "Produces", "rust"), ("reduction", "Opposes", "oxidation"),
    ("electrolysis", "Requires", "electricity"),
    ("catalyst", "UsedFor", "reaction"), ("enzyme", "IsA", "catalyst"),
    ("reactant", "Produces", "product"),
    # Minerals / gems
    ("diamond", "HasProperty", "carbon"), ("graphite", "HasProperty", "carbon"),
    ("quartz", "IsA", "mineral"), ("silicon", "PartOf", "quartz"),
    ("salt", "IsA", "mineral"), ("gypsum", "IsA", "mineral"),
    ("ruby", "IsA", "gem"), ("sapphire", "IsA", "gem"),
    ("emerald", "IsA", "gem"), ("diamond", "IsA", "gem"),
    ("topaz", "IsA", "gem"), ("jade", "IsA", "gem"),
]

# --- Biology taxonomy ---
ANIMAL_TAXONOMY = [
    ("lion", "mammal", "carnivore"), ("tiger", "mammal", "carnivore"),
    ("elephant", "mammal", "herbivore"), ("horse", "mammal", "herbivore"),
    ("cow", "mammal", "herbivore"), ("pig", "mammal", "omnivore"),
    ("whale", "mammal", "carnivore"), ("dolphin", "mammal", "carnivore"),
    ("bat", "mammal", "omnivore"), ("rat", "mammal", "omnivore"),
    ("mouse", "mammal", "omnivore"), ("rabbit", "mammal", "herbivore"),
    ("bear", "mammal", "omnivore"), ("wolf", "mammal", "carnivore"),
    ("fox", "mammal", "carnivore"), ("deer", "mammal", "herbivore"),
    ("giraffe", "mammal", "herbivore"), ("zebra", "mammal", "herbivore"),
    ("monkey", "mammal", "omnivore"), ("gorilla", "mammal", "herbivore"),
    ("chimpanzee", "mammal", "omnivore"), ("kangaroo", "mammal", "herbivore"),
    ("koala", "mammal", "herbivore"), ("panda", "mammal", "herbivore"),
    ("eagle", "bird", "carnivore"), ("hawk", "bird", "carnivore"),
    ("owl", "bird", "carnivore"), ("penguin", "bird", "carnivore"),
    ("parrot", "bird", "herbivore"), ("sparrow", "bird", "omnivore"),
    ("crow", "bird", "omnivore"), ("robin", "bird", "omnivore"),
    ("chicken", "bird", "omnivore"), ("duck", "bird", "omnivore"),
    ("swan", "bird", "omnivore"), ("flamingo", "bird", "omnivore"),
    ("ostrich", "bird", "omnivore"), ("peacock", "bird", "omnivore"),
    ("snake", "reptile", "carnivore"), ("lizard", "reptile", "omnivore"),
    ("crocodile", "reptile", "carnivore"), ("alligator", "reptile", "carnivore"),
    ("turtle", "reptile", "omnivore"), ("tortoise", "reptile", "herbivore"),
    ("frog", "amphibian", "carnivore"), ("toad", "amphibian", "carnivore"),
    ("salamander", "amphibian", "carnivore"),
    ("shark", "fish", "carnivore"), ("salmon", "fish", "carnivore"),
    ("tuna", "fish", "carnivore"), ("cod", "fish", "carnivore"),
    ("trout", "fish", "carnivore"), ("goldfish", "fish", "omnivore"),
    ("swordfish", "fish", "carnivore"),
    ("octopus", "mollusc", "carnivore"), ("squid", "mollusc", "carnivore"),
    ("snail", "mollusc", "herbivore"), ("clam", "mollusc", "omnivore"),
    ("spider", "arachnid", "carnivore"), ("scorpion", "arachnid", "carnivore"),
    ("ant", "insect", "omnivore"), ("bee", "insect", "herbivore"),
    ("wasp", "insect", "omnivore"), ("butterfly", "insect", "herbivore"),
    ("moth", "insect", "herbivore"), ("beetle", "insect", "omnivore"),
    ("grasshopper", "insect", "herbivore"), ("cricket", "insect", "omnivore"),
    ("fly", "insect", "omnivore"), ("mosquito", "insect", "carnivore"),
    ("dragonfly", "insect", "carnivore"),
    ("crab", "crustacean", "omnivore"), ("lobster", "crustacean", "carnivore"),
    ("shrimp", "crustacean", "omnivore"),
    ("worm", "annelid", "omnivore"),
]

PLANT_TAXONOMY = [
    ("oak", "tree"), ("pine", "tree"), ("maple", "tree"),
    ("birch", "tree"), ("elm", "tree"), ("willow", "tree"),
    ("redwood", "tree"), ("cedar", "tree"), ("palm", "tree"),
    ("baobab", "tree"), ("sequoia", "tree"), ("bamboo", "grass"),
    ("wheat", "grass"), ("rice", "grass"), ("corn", "grass"),
    ("barley", "grass"), ("oat", "grass"),
    ("rose", "flower"), ("tulip", "flower"), ("daisy", "flower"),
    ("sunflower", "flower"), ("orchid", "flower"), ("lily", "flower"),
    ("iris", "flower"), ("lavender", "herb"), ("mint", "herb"),
    ("basil", "herb"), ("thyme", "herb"), ("rosemary", "herb"),
    ("sage", "herb"), ("oregano", "herb"), ("parsley", "herb"),
    ("apple", "fruit"), ("orange", "fruit"), ("banana", "fruit"),
    ("grape", "fruit"), ("strawberry", "fruit"), ("blueberry", "fruit"),
    ("peach", "fruit"), ("pear", "fruit"), ("pineapple", "fruit"),
    ("mango", "fruit"), ("watermelon", "fruit"), ("cherry", "fruit"),
    ("lemon", "fruit"), ("lime", "fruit"), ("coconut", "fruit"),
    ("potato", "vegetable"), ("carrot", "vegetable"), ("onion", "vegetable"),
    ("tomato", "fruit"), ("lettuce", "vegetable"), ("cabbage", "vegetable"),
    ("broccoli", "vegetable"), ("spinach", "vegetable"), ("cucumber", "vegetable"),
    ("pepper", "vegetable"), ("garlic", "vegetable"),
    ("mushroom", "fungus"), ("yeast", "fungus"), ("mold", "fungus"),
    ("algae", "protist"), ("kelp", "algae"),
]

# --- Diseases & medicine ---
DISEASES = [
    ("cancer", "disease", "cell"), ("diabetes", "disease", "pancreas"),
    ("hypertension", "disease", "heart"), ("asthma", "disease", "lungs"),
    ("pneumonia", "disease", "lungs"), ("bronchitis", "disease", "lungs"),
    ("tuberculosis", "disease", "lungs"), ("influenza", "disease", "lungs"),
    ("covid", "disease", "lungs"), ("malaria", "disease", "blood"),
    ("hepatitis", "disease", "liver"), ("cirrhosis", "disease", "liver"),
    ("gastritis", "disease", "stomach"), ("ulcer", "disease", "stomach"),
    ("arthritis", "disease", "joint"), ("osteoporosis", "disease", "bone"),
    ("alzheimer", "disease", "brain"), ("parkinson", "disease", "brain"),
    ("stroke", "disease", "brain"), ("migraine", "disease", "brain"),
    ("epilepsy", "disease", "brain"), ("depression", "disease", "brain"),
    ("schizophrenia", "disease", "brain"), ("autism", "disease", "brain"),
    ("anxiety", "disease", "brain"),
    ("leukemia", "cancer", "blood"), ("lymphoma", "cancer", "blood"),
    ("melanoma", "cancer", "skin"),
    ("hiv", "virus", "immunity"), ("ebola", "virus", "blood"),
    ("measles", "virus", "skin"), ("chickenpox", "virus", "skin"),
    ("mumps", "virus", "gland"), ("rabies", "virus", "brain"),
    ("smallpox", "virus", "skin"), ("polio", "virus", "nerve"),
    ("cholera", "bacteria", "stomach"), ("salmonella", "bacteria", "stomach"),
    ("tetanus", "bacteria", "muscle"), ("syphilis", "bacteria", "blood"),
    ("anemia", "disease", "blood"), ("hemophilia", "disease", "blood"),
    ("scurvy", "disease", "nutrition"), ("rickets", "disease", "bone"),
    ("obesity", "disease", "body"),
]

VITAMINS = [
    ("vitamin", "IsA", "nutrient"), ("mineral", "IsA", "nutrient"),
    ("carbohydrate", "IsA", "nutrient"), ("protein", "IsA", "nutrient"),
    ("fat", "IsA", "nutrient"), ("fiber", "IsA", "nutrient"),
    ("calcium", "UsedFor", "bone"), ("iron", "UsedFor", "blood"),
    ("potassium", "UsedFor", "muscle"), ("sodium", "UsedFor", "balance"),
    ("magnesium", "UsedFor", "muscle"), ("zinc", "UsedFor", "immunity"),
    ("selenium", "UsedFor", "antioxidant"),
    ("iodine", "UsedFor", "thyroid"),
    ("insulin", "IsA", "hormone"), ("insulin", "UsedFor", "glucose"),
    ("adrenaline", "IsA", "hormone"), ("cortisol", "IsA", "hormone"),
    ("testosterone", "IsA", "hormone"), ("estrogen", "IsA", "hormone"),
    ("progesterone", "IsA", "hormone"), ("thyroxine", "IsA", "hormone"),
    ("dopamine", "IsA", "neurotransmitter"), ("serotonin", "IsA", "neurotransmitter"),
    ("acetylcholine", "IsA", "neurotransmitter"),
    ("endorphin", "IsA", "neurotransmitter"),
    ("antibody", "UsedFor", "immunity"), ("antigen", "IsA", "protein"),
    ("vaccine", "UsedFor", "immunity"), ("antibiotic", "UsedFor", "bacteria"),
    ("antiviral", "UsedFor", "virus"),
    ("aspirin", "IsA", "drug"), ("ibuprofen", "IsA", "drug"),
    ("paracetamol", "IsA", "drug"), ("morphine", "IsA", "drug"),
    ("penicillin", "IsA", "antibiotic"),
]

# --- Physics: laws + units + particles ---
PHYSICS = [
    ("newton", "Produces", "law"), ("einstein", "Produces", "relativity"),
    ("thermodynamics", "IsA", "physics"), ("mechanics", "IsA", "physics"),
    ("electromagnetism", "IsA", "physics"), ("optics", "IsA", "physics"),
    ("quantum", "IsA", "physics"), ("acoustics", "IsA", "physics"),
    # Units
    ("meter", "IsA", "unit"), ("kilogram", "IsA", "unit"),
    ("second", "IsA", "unit"), ("ampere", "IsA", "unit"),
    ("kelvin", "IsA", "unit"), ("mole", "IsA", "unit"),
    ("candela", "IsA", "unit"), ("joule", "IsA", "unit"),
    ("watt", "IsA", "unit"), ("newton", "IsA", "unit"),
    ("pascal", "IsA", "unit"), ("volt", "IsA", "unit"),
    ("ohm", "IsA", "unit"), ("hertz", "IsA", "unit"),
    ("tesla", "IsA", "unit"), ("coulomb", "IsA", "unit"),
    ("farad", "IsA", "unit"), ("henry", "IsA", "unit"),
    ("lumen", "IsA", "unit"), ("lux", "IsA", "unit"),
    ("radian", "IsA", "unit"), ("steradian", "IsA", "unit"),
    # Units -> measures
    ("meter", "UsedFor", "length"), ("kilogram", "UsedFor", "mass"),
    ("second", "UsedFor", "time"), ("kelvin", "UsedFor", "temperature"),
    ("joule", "UsedFor", "energy"), ("watt", "UsedFor", "power"),
    ("pascal", "UsedFor", "pressure"), ("volt", "UsedFor", "voltage"),
    ("ohm", "UsedFor", "resistance"), ("hertz", "UsedFor", "frequency"),
    ("ampere", "UsedFor", "current"),
    # Particles
    ("lepton", "IsA", "particle"), ("boson", "IsA", "particle"),
    ("quark", "IsA", "particle"), ("fermion", "IsA", "particle"),
    ("hadron", "IsA", "particle"), ("meson", "IsA", "particle"),
    ("neutrino", "IsA", "lepton"), ("muon", "IsA", "lepton"),
    ("tau", "IsA", "lepton"), ("electron", "IsA", "lepton"),
    ("gluon", "IsA", "boson"), ("photon", "IsA", "boson"),
    ("higgs", "IsA", "boson"), ("gluon", "UsedFor", "strong"),
    ("photon", "UsedFor", "electromagnetic"),
    # Phenomena
    ("refraction", "IsA", "phenomenon"), ("reflection", "IsA", "phenomenon"),
    ("diffraction", "IsA", "phenomenon"), ("interference", "IsA", "phenomenon"),
    ("polarization", "IsA", "phenomenon"),
    ("induction", "IsA", "phenomenon"), ("conduction", "IsA", "phenomenon"),
    ("convection", "IsA", "phenomenon"), ("radiation", "IsA", "phenomenon"),
    # Forces
    ("gravitational", "IsA", "force"), ("electromagnetic", "IsA", "force"),
    ("strong", "IsA", "force"), ("weak", "IsA", "force"),
    ("nuclear", "IsA", "force"), ("centripetal", "IsA", "force"),
    ("friction", "IsA", "force"), ("tension", "IsA", "force"),
    ("normal", "IsA", "force"),
]

# --- Earth science ---
EARTH_SCIENCE = [
    ("crust", "PartOf", "earth"), ("mantle", "PartOf", "earth"),
    ("core", "PartOf", "earth"), ("atmosphere", "PartOf", "earth"),
    ("troposphere", "PartOf", "atmosphere"), ("stratosphere", "PartOf", "atmosphere"),
    ("mesosphere", "PartOf", "atmosphere"), ("thermosphere", "PartOf", "atmosphere"),
    ("exosphere", "PartOf", "atmosphere"), ("ozone", "PartOf", "stratosphere"),
    # Rock types
    ("igneous", "IsA", "rock"), ("sedimentary", "IsA", "rock"),
    ("metamorphic", "IsA", "rock"), ("granite", "IsA", "igneous"),
    ("basalt", "IsA", "igneous"), ("obsidian", "IsA", "igneous"),
    ("pumice", "IsA", "igneous"), ("sandstone", "IsA", "sedimentary"),
    ("limestone", "IsA", "sedimentary"), ("shale", "IsA", "sedimentary"),
    ("marble", "IsA", "metamorphic"), ("slate", "IsA", "metamorphic"),
    ("schist", "IsA", "metamorphic"), ("gneiss", "IsA", "metamorphic"),
    # Phenomena
    ("earthquake", "Causes", "damage"), ("tsunami", "Causes", "flooding"),
    ("volcano", "Produces", "lava"), ("volcano", "Produces", "ash"),
    ("lava", "IsA", "rock"), ("magma", "PartOf", "mantle"),
    ("hurricane", "IsA", "storm"), ("tornado", "IsA", "storm"),
    ("blizzard", "IsA", "storm"), ("thunderstorm", "IsA", "storm"),
    ("drought", "Causes", "famine"), ("flood", "Causes", "damage"),
    # Weather
    ("rain", "IsA", "precipitation"), ("snow", "IsA", "precipitation"),
    ("hail", "IsA", "precipitation"), ("sleet", "IsA", "precipitation"),
    ("fog", "IsA", "weather"), ("cloud", "IsA", "weather"),
    ("cirrus", "IsA", "cloud"), ("cumulus", "IsA", "cloud"),
    ("stratus", "IsA", "cloud"), ("nimbus", "IsA", "cloud"),
    # Geology / tectonics
    ("plate", "PartOf", "crust"), ("fault", "Causes", "earthquake"),
    ("ring", "IsA", "fault"), ("andes", "IsA", "mountain"),
    ("rockies", "IsA", "mountain"), ("alps", "IsA", "mountain"),
    ("himalayas", "IsA", "mountain"), ("caucasus", "IsA", "mountain"),
    ("urals", "IsA", "mountain"), ("appalachians", "IsA", "mountain"),
    ("pyrenees", "IsA", "mountain"), ("carpathians", "IsA", "mountain"),
    # Bodies of water
    ("lake", "IsA", "water"), ("river", "IsA", "water"),
    ("stream", "IsA", "water"), ("ocean", "IsA", "water"),
    ("sea", "IsA", "water"), ("bay", "IsA", "water"),
    ("gulf", "IsA", "water"), ("strait", "IsA", "water"),
    ("pond", "IsA", "water"), ("reservoir", "IsA", "water"),
    ("glacier", "IsA", "ice"),
    # Biomes
    ("forest", "IsA", "biome"), ("rainforest", "IsA", "forest"),
    ("savanna", "IsA", "biome"), ("tundra", "IsA", "biome"),
    ("taiga", "IsA", "biome"), ("desert", "IsA", "biome"),
    ("grassland", "IsA", "biome"), ("wetland", "IsA", "biome"),
    ("coral", "IsA", "reef"), ("reef", "IsA", "ecosystem"),
    # Space / cosmology
    ("nebula", "IsA", "cloud"), ("supernova", "IsA", "explosion"),
    ("blackhole", "IsA", "object"), ("pulsar", "IsA", "star"),
    ("quasar", "IsA", "object"), ("comet", "IsA", "object"),
    ("asteroid", "IsA", "object"), ("meteor", "IsA", "object"),
    ("meteorite", "IsA", "rock"), ("satellite", "IsA", "object"),
    ("orbit", "IsA", "path"),
]


def build_chemistry_triples():
    out = list(CHEMISTRY)
    # Hierarchy
    out += [
        ("acid", "IsA", "compound"), ("base", "IsA", "compound"),
        ("salt", "IsA", "compound"),
        ("sugar", "IsA", "carbohydrate"),
        ("alkane", "IsA", "hydrocarbon"), ("alcohol", "IsA", "compound"),
        ("hydrocarbon", "IsA", "organic"), ("organic", "IsA", "compound"),
        ("polymer", "IsA", "molecule"),
        ("bond", "PartOf", "molecule"), ("reaction", "IsA", "process"),
        ("mineral", "IsA", "solid"), ("gem", "IsA", "mineral"),
    ]
    return out


def build_biology_triples():
    out = []
    for animal, clade, diet in ANIMAL_TAXONOMY:
        out.append((animal, "IsA", clade))
        out.append((animal, "IsA", diet))
        out.append((animal, "IsA", "animal"))
    for plant, clade in PLANT_TAXONOMY:
        out.append((plant, "IsA", clade))
        out.append((plant, "IsA", "plant"))
    out += [
        ("mammal", "IsA", "animal"), ("bird", "IsA", "animal"),
        ("reptile", "IsA", "animal"), ("amphibian", "IsA", "animal"),
        ("fish", "IsA", "animal"), ("insect", "IsA", "animal"),
        ("arachnid", "IsA", "animal"), ("crustacean", "IsA", "animal"),
        ("mollusc", "IsA", "animal"), ("annelid", "IsA", "animal"),
        ("carnivore", "IsA", "diet"), ("herbivore", "IsA", "diet"),
        ("omnivore", "IsA", "diet"),
        ("tree", "IsA", "plant"), ("flower", "IsA", "plant"),
        ("grass", "IsA", "plant"), ("herb", "IsA", "plant"),
        ("fruit", "PartOf", "plant"), ("vegetable", "PartOf", "plant"),
        ("fungus", "IsA", "organism"), ("protist", "IsA", "organism"),
        ("bacteria", "IsA", "organism"), ("virus", "IsA", "pathogen"),
        ("animal", "IsA", "organism"), ("plant", "IsA", "organism"),
        # Body systems from Phase 1 - re-assert
        ("cell", "PartOf", "tissue"), ("tissue", "PartOf", "organ"),
        ("organ", "PartOf", "system"),
    ]
    return out


def build_medicine_triples():
    out = []
    for disease, kind, target in DISEASES:
        out.append((disease, "IsA", kind))
        out.append((disease, "PartOf", target))
    out += VITAMINS
    out += [
        ("hormone", "IsA", "molecule"),
        ("neurotransmitter", "IsA", "molecule"),
        ("drug", "IsA", "medicine"), ("medicine", "UsedFor", "disease"),
        ("vaccine", "IsA", "medicine"),
        ("bacteria", "Causes", "infection"),
        ("virus", "Causes", "infection"),
        ("infection", "IsA", "disease"),
        ("surgery", "IsA", "treatment"), ("therapy", "IsA", "treatment"),
    ]
    return out


def build_physics_triples():
    out = list(PHYSICS)
    out += [
        ("law", "IsA", "principle"), ("physics", "IsA", "science"),
        ("unit", "UsedFor", "measurement"),
        ("particle", "PartOf", "matter"),
        ("force", "Causes", "acceleration"),
        ("energy", "ContrastedWith", "matter"),
    ]
    return out


def build_earth_triples():
    return list(EARTH_SCIENCE) + [
        ("earth", "IsA", "planet"), ("atmosphere", "PartOf", "planet"),
        ("rock", "PartOf", "crust"),
        ("mountain", "PartOf", "earth"), ("ocean", "PartOf", "earth"),
        ("biome", "PartOf", "earth"), ("storm", "IsA", "weather"),
        ("cloud", "PartOf", "atmosphere"),
        ("precipitation", "IsA", "weather"),
    ]


# --- SciQ extractor (re-used idea from phase1) ---
def stream_sciq(limit, seen):
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[sciq] datasets unavailable: {e}", flush=True); return
    print(f"[sciq] loading (limit={limit})...", flush=True)
    try:
        ds = load_dataset("sciq", split="train", streaming=True)
    except Exception as e:
        print(f"[sciq] load failed: {e}", flush=True); return
    n = 0
    for row in ds:
        if n >= limit: return
        n += 1
        q_toks = content_tokens(row.get("question", "") or "")
        ans_toks = content_tokens(row.get("correct_answer", "") or "")
        sup_toks = content_tokens(row.get("support", "") or "")
        if not q_toks or not ans_toks: continue
        q_focus = q_toks[-1]
        if not is_single_token(q_focus): continue
        for a in ans_toks[:3]:
            if a == q_focus or not is_single_token(a): continue
            key = (q_focus, "RelatedTo", a)
            if key in seen: continue
            yield key
        # answer head links to salient support tokens
        ah = ans_toks[0]
        if not is_single_token(ah): continue
        seen2 = set()
        for t in sup_toks:
            if t in seen2 or t == ah: continue
            seen2.add(t)
            if len(seen2) > 4: break
            if not is_single_token(t): continue
            key = (ah, "RelatedTo", t)
            if key in seen: continue
            yield key


def stream_openbookqa(seen):
    try:
        from datasets import load_dataset
    except Exception: return
    try:
        ds = load_dataset("openbookqa", "additional", split="train")
    except Exception as e:
        print(f"[openbookqa] load failed: {e}", flush=True); return
    print(f"[openbookqa] {len(ds)} rows", flush=True)
    for row in ds:
        fact = row.get("fact1", "") or ""
        stem = row.get("question_stem", "") or ""
        # Parse fact like "a kangaroo is a mammal" → (kangaroo, IsA, mammal)
        f_toks = content_tokens(fact)
        if len(f_toks) < 2: continue
        # Any pairwise relations between content tokens (linking)
        s = f_toks[0]
        for o in f_toks[1:5]:
            if s == o or not is_single_token(s) or not is_single_token(o): continue
            yield (s, "RelatedTo", o)
        # stem focus ↔ answer content
        q_toks = content_tokens(stem)
        choices = row.get("choices", {}) or {}
        key_label = row.get("answerKey", "")
        if q_toks and choices:
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            ans_text = ""
            for l, t in zip(labels, texts):
                if l == key_label: ans_text = t; break
            a_toks = content_tokens(ans_text)
            if q_toks and a_toks:
                qf = q_toks[-1]; af = a_toks[-1]
                if is_single_token(qf) and is_single_token(af) and qf != af:
                    yield (qf, "RelatedTo", af)


def stream_ai2_arc(seen, split="ARC-Challenge"):
    try:
        from datasets import load_dataset
    except Exception: return
    try:
        ds = load_dataset("ai2_arc", split, split="train")
    except Exception as e:
        print(f"[arc/{split}] load failed: {e}", flush=True); return
    print(f"[arc/{split}] {len(ds)} rows", flush=True)
    for row in ds:
        q = row.get("question", "") or ""
        choices = row.get("choices", {}) or {}
        key_label = row.get("answerKey", "")
        q_toks = content_tokens(q)
        if not q_toks: continue
        qf = q_toks[-1]
        if not is_single_token(qf): continue
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        ans_text = ""
        for l, t in zip(labels, texts):
            if l == key_label: ans_text = t; break
        a_toks = content_tokens(ans_text)
        for a in a_toks[:3]:
            if a == qf or not is_single_token(a): continue
            yield (qf, "RelatedTo", a)


def stream_scitail(seen):
    try:
        from datasets import load_dataset
    except Exception: return
    try:
        ds = load_dataset("scitail", "tsv_format", split="train")
    except Exception as e:
        print(f"[scitail] load failed: {e}", flush=True); return
    print(f"[scitail] {len(ds)} rows", flush=True)
    for row in ds:
        label = row.get("label", "")
        if label != "entails":
            continue
        p_toks = content_tokens(row.get("premise", ""))
        h_toks = content_tokens(row.get("hypothesis", ""))
        if not p_toks or not h_toks: continue
        # Link last content token of premise to last of hypothesis (rough topic link)
        s = p_toks[-1]; o = h_toks[-1]
        if s == o or not is_single_token(s) or not is_single_token(o):
            continue
        yield (s, "RelatedTo", o)


# ===========================================================================
# PHASE 3 — HISTORY / CULTURE
# ===========================================================================

HISTORICAL_ERAS = [
    ("stone", "era"), ("bronze", "era"), ("iron", "era"),
    ("classical", "era"), ("medieval", "era"), ("renaissance", "era"),
    ("enlightenment", "era"), ("victorian", "era"), ("edwardian", "era"),
    ("industrial", "era"), ("modern", "era"), ("postmodern", "era"),
    ("information", "era"), ("digital", "era"),
    ("antiquity", "era"), ("prehistoric", "era"),
    ("colonial", "era"), ("cold", "era"),
]

EMPIRES = [
    ("roman", "empire", "italy"), ("byzantine", "empire", "turkey"),
    ("ottoman", "empire", "turkey"), ("persian", "empire", "iran"),
    ("mongol", "empire", "mongolia"), ("british", "empire", "england"),
    ("spanish", "empire", "spain"), ("french", "empire", "france"),
    ("portuguese", "empire", "portugal"), ("dutch", "empire", "netherlands"),
    ("russian", "empire", "russia"), ("austrian", "empire", "austria"),
    ("german", "empire", "germany"), ("japanese", "empire", "japan"),
    ("chinese", "empire", "china"), ("mughal", "empire", "india"),
    ("maurya", "empire", "india"), ("gupta", "empire", "india"),
    ("aztec", "empire", "mexico"), ("inca", "empire", "peru"),
    ("maya", "empire", "mexico"), ("egyptian", "empire", "egypt"),
    ("babylonian", "empire", "iraq"), ("assyrian", "empire", "iraq"),
    ("carthaginian", "empire", "tunisia"), ("greek", "empire", "greece"),
    ("macedonian", "empire", "greece"),
]

WARS = [
    ("napoleonic", "war"), ("crusades", "war"), ("revolutionary", "war"),
    ("civil", "war"), ("korean", "war"), ("vietnam", "war"),
    ("gulf", "war"), ("iraq", "war"), ("afghan", "war"),
    ("peloponnesian", "war"), ("punic", "war"), ("trojan", "war"),
    ("hundred", "war"), ("thirty", "war"), ("seven", "war"),
    ("boer", "war"), ("crimean", "war"), ("falkland", "war"),
    ("yugoslav", "war"), ("ukrainian", "war"),
]

# Historical figures — expanded list tying into Phase 1
HISTORICAL_FIGURES = [
    # Ancient
    ("hammurabi", "king", "babylon"), ("nebuchadnezzar", "king", "babylon"),
    ("ramses", "pharaoh", "egypt"), ("tutankhamun", "pharaoh", "egypt"),
    ("akhenaten", "pharaoh", "egypt"), ("nefertiti", "queen", "egypt"),
    ("hatshepsut", "pharaoh", "egypt"),
    ("leonidas", "king", "sparta"), ("alexander", "king", "macedonia"),
    ("pericles", "leader", "athens"), ("solon", "leader", "athens"),
    ("spartacus", "gladiator", "rome"), ("cicero", "senator", "rome"),
    ("augustus", "emperor", "rome"), ("nero", "emperor", "rome"),
    ("hadrian", "emperor", "rome"), ("trajan", "emperor", "rome"),
    ("constantine", "emperor", "rome"), ("marcus", "emperor", "rome"),
    ("cyrus", "king", "persia"), ("darius", "king", "persia"),
    ("xerxes", "king", "persia"),
    ("ashoka", "emperor", "india"), ("akbar", "emperor", "india"),
    ("shah", "emperor", "india"),
    ("qin", "emperor", "china"), ("mao", "leader", "china"),
    ("confucius", "philosopher", "china"), ("laozi", "philosopher", "china"),
    ("sun", "strategist", "china"),
    # Medieval / Renaissance
    ("charlemagne", "emperor", "france"),
    ("richard", "king", "england"), ("henry", "king", "england"),
    ("tudor", "dynasty", "england"), ("stuart", "dynasty", "england"),
    ("plantagenet", "dynasty", "england"),
    ("louis", "king", "france"), ("philip", "king", "france"),
    ("ferdinand", "king", "spain"), ("isabella", "queen", "spain"),
    ("ivan", "tsar", "russia"), ("peter", "tsar", "russia"),
    ("catherine", "tsarina", "russia"), ("nicholas", "tsar", "russia"),
    ("suleiman", "sultan", "turkey"), ("mehmed", "sultan", "turkey"),
    ("saladin", "sultan", "egypt"), ("aquinas", "philosopher", "italy"),
    ("erasmus", "philosopher", "netherlands"),
    ("luther", "reformer", "germany"), ("calvin", "reformer", "switzerland"),
    ("copernicus", "astronomer", "poland"),
    # Modern
    ("bolivar", "leader", "venezuela"),
    ("bismarck", "chancellor", "germany"), ("garibaldi", "leader", "italy"),
    ("marx", "philosopher", "germany"), ("engels", "philosopher", "germany"),
    ("lenin", "leader", "russia"), ("stalin", "leader", "russia"),
    ("trotsky", "leader", "russia"), ("mao", "leader", "china"),
    ("zedong", "leader", "china"), ("ho", "leader", "vietnam"),
    ("castro", "leader", "cuba"), ("guevara", "revolutionary", "cuba"),
    ("hitler", "dictator", "germany"), ("mussolini", "dictator", "italy"),
    ("franco", "dictator", "spain"), ("tito", "leader", "yugoslavia"),
    ("nasser", "leader", "egypt"), ("sadat", "leader", "egypt"),
    ("khomeini", "leader", "iran"), ("nehru", "leader", "india"),
    ("bhutto", "leader", "pakistan"), ("khan", "leader", "pakistan"),
    ("mandela", "leader", "africa"), ("tutu", "bishop", "africa"),
    ("reagan", "president", "usa"), ("obama", "president", "usa"),
    ("clinton", "president", "usa"), ("bush", "president", "usa"),
    ("trump", "president", "usa"), ("thatcher", "prime", "england"),
    ("blair", "prime", "england"),
    ("deng", "leader", "china"), ("xi", "leader", "china"),
    ("putin", "leader", "russia"), ("gorbachev", "leader", "russia"),
    ("yeltsin", "leader", "russia"),
]

# Art movements
ART_MOVEMENTS = [
    ("renaissance", "movement"), ("baroque", "movement"),
    ("rococo", "movement"), ("neoclassicism", "movement"),
    ("romanticism", "movement"), ("realism", "movement"),
    ("impressionism", "movement"), ("expressionism", "movement"),
    ("cubism", "movement"), ("surrealism", "movement"),
    ("dadaism", "movement"), ("futurism", "movement"),
    ("minimalism", "movement"), ("abstract", "movement"),
    ("modernism", "movement"), ("postmodernism", "movement"),
    ("popart", "movement"), ("fauvism", "movement"),
    ("symbolism", "movement"), ("mannerism", "movement"),
    ("gothic", "movement"), ("romanesque", "movement"),
]

# Painters
PAINTERS = [
    ("raphael", "renaissance", "italy"), ("titian", "renaissance", "italy"),
    ("donatello", "renaissance", "italy"), ("botticelli", "renaissance", "italy"),
    ("caravaggio", "baroque", "italy"), ("rubens", "baroque", "belgium"),
    ("vermeer", "baroque", "netherlands"), ("rembrandt", "baroque", "netherlands"),
    ("goya", "romanticism", "spain"), ("delacroix", "romanticism", "france"),
    ("turner", "romanticism", "england"), ("constable", "romanticism", "england"),
    ("renoir", "impressionism", "france"), ("degas", "impressionism", "france"),
    ("manet", "impressionism", "france"), ("cezanne", "impressionism", "france"),
    ("gauguin", "impressionism", "france"), ("seurat", "impressionism", "france"),
    ("pissarro", "impressionism", "france"), ("morisot", "impressionism", "france"),
    ("matisse", "fauvism", "france"), ("klimt", "symbolism", "austria"),
    ("munch", "expressionism", "norway"),
    ("braque", "cubism", "france"), ("leger", "cubism", "france"),
    ("magritte", "surrealism", "belgium"), ("ernst", "surrealism", "germany"),
    ("kahlo", "surrealism", "mexico"), ("rivera", "muralism", "mexico"),
    ("pollock", "abstract", "usa"), ("rothko", "abstract", "usa"),
    ("mondrian", "abstract", "netherlands"), ("kandinsky", "abstract", "russia"),
    ("lichtenstein", "popart", "usa"), ("banksy", "street", "england"),
    ("hockney", "popart", "england"), ("basquiat", "neoexpressionism", "usa"),
    ("vermeer", "baroque", "netherlands"),
]

# Music genres + composers
MUSIC_GENRES = [
    ("classical", "genre"), ("baroque", "genre"), ("romantic", "genre"),
    ("modern", "genre"), ("contemporary", "genre"),
    ("jazz", "genre"), ("blues", "genre"), ("rock", "genre"),
    ("pop", "genre"), ("hiphop", "genre"), ("country", "genre"),
    ("folk", "genre"), ("reggae", "genre"), ("ska", "genre"),
    ("punk", "genre"), ("metal", "genre"), ("electronic", "genre"),
    ("techno", "genre"), ("house", "genre"), ("ambient", "genre"),
    ("funk", "genre"), ("soul", "genre"), ("gospel", "genre"),
    ("disco", "genre"), ("rnb", "genre"), ("indie", "genre"),
    ("opera", "genre"), ("symphony", "genre"), ("concerto", "genre"),
]

COMPOSERS = [
    ("vivaldi", "baroque", "italy"), ("handel", "baroque", "germany"),
    ("haydn", "classical", "austria"), ("schubert", "romantic", "austria"),
    ("mendelssohn", "romantic", "germany"), ("brahms", "romantic", "germany"),
    ("schumann", "romantic", "germany"), ("liszt", "romantic", "hungary"),
    ("dvorak", "romantic", "czechia"), ("smetana", "romantic", "czechia"),
    ("verdi", "opera", "italy"), ("puccini", "opera", "italy"),
    ("rossini", "opera", "italy"), ("bizet", "opera", "france"),
    ("debussy", "impressionist", "france"), ("ravel", "impressionist", "france"),
    ("stravinsky", "modern", "russia"), ("prokofiev", "modern", "russia"),
    ("shostakovich", "modern", "russia"), ("rachmaninoff", "romantic", "russia"),
    ("mahler", "romantic", "austria"), ("bruckner", "romantic", "austria"),
    ("sibelius", "romantic", "finland"), ("grieg", "romantic", "norway"),
    ("strauss", "romantic", "austria"), ("wagner", "opera", "germany"),
    ("gershwin", "jazz", "usa"), ("copland", "modern", "usa"),
    ("bernstein", "modern", "usa"),
]

MUSICIANS = [
    ("beatles", "rock", "england"), ("stones", "rock", "england"),
    ("floyd", "rock", "england"), ("zeppelin", "rock", "england"),
    ("queen", "rock", "england"), ("bowie", "rock", "england"),
    ("lennon", "rock", "england"), ("mccartney", "rock", "england"),
    ("jagger", "rock", "england"),
    ("dylan", "folk", "usa"), ("springsteen", "rock", "usa"),
    ("elvis", "rock", "usa"), ("presley", "rock", "usa"),
    ("hendrix", "rock", "usa"), ("morrison", "rock", "usa"),
    ("cobain", "rock", "usa"), ("madonna", "pop", "usa"),
    ("jackson", "pop", "usa"), ("prince", "pop", "usa"),
    ("beyonce", "pop", "usa"), ("swift", "pop", "usa"),
    ("eminem", "hiphop", "usa"), ("tupac", "hiphop", "usa"),
    ("biggie", "hiphop", "usa"), ("jayz", "hiphop", "usa"),
    ("kanye", "hiphop", "usa"), ("drake", "hiphop", "canada"),
    ("marley", "reggae", "jamaica"), ("tosh", "reggae", "jamaica"),
    ("armstrong", "jazz", "usa"), ("ellington", "jazz", "usa"),
    ("parker", "jazz", "usa"), ("davis", "jazz", "usa"),
    ("coltrane", "jazz", "usa"), ("fitzgerald", "jazz", "usa"),
    ("sinatra", "jazz", "usa"), ("cash", "country", "usa"),
    ("dolly", "country", "usa"),
]

# Literature
LITERATURE = [
    ("homer", "odyssey", "greece"), ("homer", "iliad", "greece"),
    ("virgil", "aeneid", "rome"), ("ovid", "metamorphoses", "rome"),
    ("chaucer", "tales", "england"), ("dante", "inferno", "italy"),
    ("boccaccio", "decameron", "italy"), ("petrarch", "sonnet", "italy"),
    ("cervantes", "quixote", "spain"), ("milton", "paradise", "england"),
    ("austen", "pride", "england"), ("bronte", "heights", "england"),
    ("dickens", "twist", "england"), ("hardy", "tess", "england"),
    ("stevenson", "jekyll", "scotland"), ("carroll", "wonderland", "england"),
    ("kipling", "jungle", "england"), ("orwell", "1984", "england"),
    ("huxley", "brave", "england"), ("tolkien", "hobbit", "england"),
    ("rowling", "potter", "england"), ("christie", "poirot", "england"),
    ("doyle", "holmes", "scotland"), ("wilde", "dorian", "ireland"),
    ("joyce", "ulysses", "ireland"), ("beckett", "godot", "ireland"),
    ("yeats", "poem", "ireland"),
    ("hugo", "miserables", "france"), ("dumas", "mousquetaires", "france"),
    ("flaubert", "bovary", "france"), ("zola", "germinal", "france"),
    ("proust", "time", "france"), ("sartre", "existentialism", "france"),
    ("camus", "stranger", "france"), ("balzac", "comedy", "france"),
    ("goethe", "faust", "germany"), ("schiller", "ballad", "germany"),
    ("kafka", "metamorphosis", "czechia"), ("mann", "mountain", "germany"),
    ("hesse", "siddhartha", "germany"),
    ("pushkin", "onegin", "russia"), ("gogol", "dead", "russia"),
    ("chekhov", "seagull", "russia"), ("nabokov", "lolita", "russia"),
    ("solzhenitsyn", "gulag", "russia"),
    ("garciamarquez", "solitude", "colombia"), ("borges", "labyrinth", "argentina"),
    ("neruda", "ode", "chile"), ("allende", "spirits", "chile"),
    ("twain", "finn", "usa"), ("melville", "dick", "usa"),
    ("poe", "raven", "usa"), ("whitman", "leaves", "usa"),
    ("thoreau", "walden", "usa"), ("emerson", "essay", "usa"),
    ("faulkner", "sound", "usa"), ("fitzgerald", "gatsby", "usa"),
    ("steinbeck", "wrath", "usa"), ("salinger", "catcher", "usa"),
    ("morrison", "beloved", "usa"), ("angelou", "caged", "usa"),
    ("murakami", "wood", "japan"), ("mishima", "pavilion", "japan"),
    ("tagore", "gitanjali", "india"), ("rushdie", "children", "india"),
]

RELIGIONS = [
    ("christianity", "religion"), ("islam", "religion"),
    ("judaism", "religion"), ("hinduism", "religion"),
    ("buddhism", "religion"), ("taoism", "religion"),
    ("shinto", "religion"), ("sikhism", "religion"),
    ("jainism", "religion"), ("confucianism", "religion"),
    ("zoroastrianism", "religion"),
    ("catholicism", "christianity"), ("orthodox", "christianity"),
    ("protestant", "christianity"), ("lutheran", "christianity"),
    ("baptist", "christianity"), ("methodist", "christianity"),
    ("sunni", "islam"), ("shia", "islam"), ("sufi", "islam"),
    ("mahayana", "buddhism"), ("theravada", "buddhism"),
    ("vajrayana", "buddhism"), ("zen", "buddhism"),
    ("jesus", "prophet", "christianity"), ("moses", "prophet", "judaism"),
    ("muhammad", "prophet", "islam"), ("buddha", "teacher", "buddhism"),
    ("krishna", "deity", "hinduism"), ("shiva", "deity", "hinduism"),
    ("vishnu", "deity", "hinduism"), ("brahma", "deity", "hinduism"),
    ("zeus", "deity", "greek"), ("apollo", "deity", "greek"),
    ("athena", "deity", "greek"), ("hermes", "deity", "greek"),
    ("poseidon", "deity", "greek"), ("hades", "deity", "greek"),
    ("ares", "deity", "greek"), ("aphrodite", "deity", "greek"),
    ("hera", "deity", "greek"), ("artemis", "deity", "greek"),
    ("odin", "deity", "norse"), ("thor", "deity", "norse"),
    ("loki", "deity", "norse"), ("freya", "deity", "norse"),
    ("ra", "deity", "egyptian"), ("anubis", "deity", "egyptian"),
    ("osiris", "deity", "egyptian"), ("isis", "deity", "egyptian"),
    ("horus", "deity", "egyptian"), ("thoth", "deity", "egyptian"),
    ("jupiter", "deity", "roman"), ("mars", "deity", "roman"),
    ("venus", "deity", "roman"),
    # Scriptures
    ("bible", "IsA", "scripture"), ("quran", "IsA", "scripture"),
    ("torah", "IsA", "scripture"), ("vedas", "IsA", "scripture"),
    ("upanishads", "IsA", "scripture"), ("gita", "IsA", "scripture"),
    ("sutras", "IsA", "scripture"), ("tripitaka", "IsA", "scripture"),
    # Concepts
    ("monotheism", "IsA", "belief"), ("polytheism", "IsA", "belief"),
    ("atheism", "IsA", "belief"), ("agnosticism", "IsA", "belief"),
    ("pantheism", "IsA", "belief"),
    ("prayer", "IsA", "practice"), ("meditation", "IsA", "practice"),
    ("fasting", "IsA", "practice"), ("pilgrimage", "IsA", "practice"),
    ("karma", "IsA", "concept"), ("dharma", "IsA", "concept"),
    ("nirvana", "IsA", "concept"), ("reincarnation", "IsA", "concept"),
    ("soul", "IsA", "concept"), ("heaven", "IsA", "concept"),
    ("hell", "IsA", "concept"),
]

SPORTS = [
    ("football", "sport"), ("soccer", "sport"), ("basketball", "sport"),
    ("baseball", "sport"), ("cricket", "sport"), ("tennis", "sport"),
    ("golf", "sport"), ("hockey", "sport"), ("rugby", "sport"),
    ("volleyball", "sport"), ("swimming", "sport"), ("cycling", "sport"),
    ("running", "sport"), ("boxing", "sport"), ("wrestling", "sport"),
    ("judo", "sport"), ("karate", "sport"), ("taekwondo", "sport"),
    ("skiing", "sport"), ("snowboarding", "sport"), ("surfing", "sport"),
    ("skating", "sport"), ("gymnastics", "sport"), ("archery", "sport"),
    ("fencing", "sport"), ("badminton", "sport"), ("squash", "sport"),
    ("polo", "sport"), ("equestrian", "sport"), ("marathon", "sport"),
    ("triathlon", "sport"), ("decathlon", "sport"), ("curling", "sport"),
    ("rowing", "sport"), ("sailing", "sport"), ("diving", "sport"),
    ("climbing", "sport"), ("chess", "sport"), ("poker", "sport"),
    # Famous athletes
    ("ali", "boxer", "usa"), ("tyson", "boxer", "usa"),
    ("mayweather", "boxer", "usa"), ("pacquiao", "boxer", "philippines"),
    ("jordan", "basketball", "usa"), ("lebron", "basketball", "usa"),
    ("kobe", "basketball", "usa"), ("bryant", "basketball", "usa"),
    ("pele", "soccer", "brazil"), ("maradona", "soccer", "argentina"),
    ("messi", "soccer", "argentina"), ("ronaldo", "soccer", "portugal"),
    ("zidane", "soccer", "france"), ("beckham", "soccer", "england"),
    ("federer", "tennis", "switzerland"), ("nadal", "tennis", "spain"),
    ("djokovic", "tennis", "serbia"), ("williams", "tennis", "usa"),
    ("woods", "golf", "usa"), ("nicklaus", "golf", "usa"),
    ("phelps", "swimming", "usa"), ("bolt", "running", "jamaica"),
    ("owens", "running", "usa"), ("ruth", "baseball", "usa"),
    ("gretzky", "hockey", "canada"),
]

FOOD_CUISINE = [
    ("pizza", "italian", "food"), ("pasta", "italian", "food"),
    ("risotto", "italian", "food"), ("tiramisu", "italian", "food"),
    ("croissant", "french", "food"), ("baguette", "french", "food"),
    ("quiche", "french", "food"), ("souffle", "french", "food"),
    ("sushi", "japanese", "food"), ("ramen", "japanese", "food"),
    ("tempura", "japanese", "food"), ("sashimi", "japanese", "food"),
    ("kimchi", "korean", "food"), ("bibimbap", "korean", "food"),
    ("curry", "indian", "food"), ("biryani", "indian", "food"),
    ("samosa", "indian", "food"), ("naan", "indian", "food"),
    ("tandoori", "indian", "food"),
    ("paella", "spanish", "food"), ("tapas", "spanish", "food"),
    ("gazpacho", "spanish", "food"),
    ("taco", "mexican", "food"), ("burrito", "mexican", "food"),
    ("enchilada", "mexican", "food"), ("guacamole", "mexican", "food"),
    ("dimsum", "chinese", "food"), ("wonton", "chinese", "food"),
    ("chowmein", "chinese", "food"), ("peking", "chinese", "food"),
    ("hummus", "mediterranean", "food"), ("falafel", "mediterranean", "food"),
    ("shawarma", "mediterranean", "food"), ("couscous", "north african", "food"),
    ("tagine", "moroccan", "food"), ("borscht", "russian", "food"),
    ("pierogi", "polish", "food"), ("schnitzel", "austrian", "food"),
    ("bratwurst", "german", "food"), ("sauerkraut", "german", "food"),
    ("fondue", "swiss", "food"), ("waffle", "belgian", "food"),
    ("hamburger", "american", "food"), ("hotdog", "american", "food"),
    ("pancake", "american", "food"), ("donut", "american", "food"),
    ("bagel", "american", "food"), ("brownie", "american", "food"),
    ("cheesecake", "american", "food"), ("pavlova", "australian", "food"),
    ("fish", "english", "food"), ("scone", "english", "food"),
    # Drinks
    ("coffee", "IsA", "drink"), ("tea", "IsA", "drink"),
    ("wine", "IsA", "drink"), ("beer", "IsA", "drink"),
    ("whiskey", "IsA", "drink"), ("vodka", "IsA", "drink"),
    ("rum", "IsA", "drink"), ("gin", "IsA", "drink"),
    ("tequila", "IsA", "drink"), ("sake", "IsA", "drink"),
    ("champagne", "IsA", "wine"), ("merlot", "IsA", "wine"),
    ("chardonnay", "IsA", "wine"),
]


def build_history_triples():
    out = []
    for era, kind in HISTORICAL_ERAS:
        out.append((era, "IsA", kind))
        out.append((era, "PartOf", "history"))
    for emp, kind, country in EMPIRES:
        out.append((emp, "IsA", kind))
        out.append((emp, "PartOf", country))
    for war, kind in WARS:
        out.append((war, "IsA", kind))
        out.append((war, "PartOf", "history"))
    for person, role, country in HISTORICAL_FIGURES:
        out.append((person, "IsA", role))
        out.append((person, "PartOf", country))
    out += [
        ("war", "IsA", "conflict"), ("conflict", "PartOf", "history"),
        ("empire", "IsA", "state"), ("dynasty", "IsA", "state"),
        ("king", "IsA", "ruler"), ("queen", "IsA", "ruler"),
        ("tsar", "IsA", "ruler"), ("tsarina", "IsA", "ruler"),
        ("emperor", "IsA", "ruler"), ("sultan", "IsA", "ruler"),
        ("pharaoh", "IsA", "ruler"), ("dictator", "IsA", "ruler"),
        ("reformer", "IsA", "leader"), ("strategist", "IsA", "leader"),
        ("ruler", "IsA", "person"), ("senator", "IsA", "person"),
        ("gladiator", "IsA", "person"), ("revolutionary", "IsA", "person"),
        ("bishop", "IsA", "religious"),
    ]
    return out


def build_arts_triples():
    out = []
    for mov, kind in ART_MOVEMENTS:
        out.append((mov, "IsA", kind))
        out.append((mov, "PartOf", "art"))
    for painter, movement, country in PAINTERS:
        out.append((painter, "IsA", "painter"))
        out.append((painter, "Produces", movement))
        out.append((painter, "PartOf", country))
    for genre, kind in MUSIC_GENRES:
        out.append((genre, "IsA", kind))
        out.append((genre, "PartOf", "music"))
    for comp, genre, country in COMPOSERS:
        out.append((comp, "IsA", "composer"))
        out.append((comp, "Produces", genre))
        out.append((comp, "PartOf", country))
    for mus, genre, country in MUSICIANS:
        out.append((mus, "IsA", "musician"))
        out.append((mus, "Produces", genre))
        out.append((mus, "PartOf", country))
    for author, work, country in LITERATURE:
        out.append((author, "IsA", "writer"))
        out.append((author, "Produces", work))
        out.append((author, "PartOf", country))
    out += [
        ("painter", "IsA", "artist"), ("composer", "IsA", "musician"),
        ("musician", "IsA", "artist"), ("writer", "IsA", "artist"),
        ("artist", "IsA", "person"), ("movement", "PartOf", "culture"),
        ("genre", "PartOf", "music"), ("art", "PartOf", "culture"),
        ("music", "PartOf", "culture"), ("literature", "PartOf", "culture"),
    ]
    return out


def build_religion_triples():
    out = []
    for tup in RELIGIONS:
        if len(tup) == 2:
            out.append((tup[0], "IsA", tup[1]))
        elif len(tup) == 3:
            name, role, tradition = tup
            out.append((name, "IsA", role))
            out.append((name, "PartOf", tradition))
    out += [
        ("religion", "PartOf", "culture"), ("prophet", "IsA", "person"),
        ("deity", "IsA", "god"), ("god", "IsA", "being"),
        ("scripture", "IsA", "text"), ("belief", "IsA", "concept"),
        ("practice", "IsA", "behavior"),
    ]
    return out


def build_sports_triples():
    out = []
    for tup in SPORTS:
        if len(tup) == 2:
            out.append((tup[0], "IsA", tup[1]))
            out.append((tup[0], "PartOf", "recreation"))
        elif len(tup) == 3:
            name, sport, country = tup
            out.append((name, "IsA", "athlete"))
            out.append((name, "PartOf", sport))
            out.append((name, "PartOf", country))
    out += [
        ("sport", "IsA", "activity"), ("athlete", "IsA", "person"),
        ("boxer", "IsA", "athlete"),
        ("recreation", "PartOf", "culture"),
    ]
    return out


def build_cuisine_triples():
    out = []
    for tup in FOOD_CUISINE:
        if len(tup) == 2:
            out.append((tup[0], "IsA", tup[1]))
        elif len(tup) == 3:
            name, cuisine, kind = tup
            out.append((name, "IsA", kind))
            out.append((name, "IsA", cuisine))
    out += [
        ("food", "IsA", "substance"), ("drink", "IsA", "substance"),
        ("cuisine", "PartOf", "culture"), ("wine", "IsA", "drink"),
    ]
    return out


# ----- History HF: trivia_qa extractor -----

def stream_trivia_qa(limit, seen):
    try:
        from datasets import load_dataset
    except Exception: return
    try:
        ds = load_dataset("trivia_qa", "rc", split="train", streaming=True)
    except Exception as e:
        print(f"[trivia_qa] load failed: {e}", flush=True); return
    print(f"[trivia_qa] streaming (limit={limit})...", flush=True)
    n = 0
    for row in ds:
        if n >= limit: return
        n += 1
        q = row.get("question", "")
        ans = row.get("answer", {}) or {}
        ans_value = ans.get("value", "") or ""
        aliases = ans.get("aliases", []) or []
        q_toks = content_tokens(q)
        if not q_toks: continue
        qf = q_toks[-1]
        if not is_single_token(qf): continue
        ans_tokens = set()
        for s in [ans_value] + aliases[:3]:
            for t in content_tokens(s):
                if is_single_token(t):
                    ans_tokens.add(t)
        for a in list(ans_tokens)[:3]:
            if a == qf: continue
            yield (qf, "RelatedTo", a)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

PHASE2_SANITY = [
    ("What type of acid is sulfuric?", ["acid"]),
    ("What is methane?", ["alkane", "hydrocarbon"]),
    ("What state is steam?", ["gas"]),
    ("What is a diamond made of?", ["carbon", "gem"]),
    ("What is a lion?", ["mammal", "carnivore", "animal"]),
    ("What is a penguin?", ["bird"]),
    ("What is a shark?", ["fish", "carnivore"]),
    ("What is an ant?", ["insect"]),
    ("What is cancer?", ["disease"]),
    ("What organ does hepatitis affect?", ["liver"]),
    ("What is insulin?", ["hormone", "glucose"]),
    ("What is dopamine?", ["neurotransmitter"]),
    ("What is a joule?", ["unit", "energy"]),
    ("What is a quark?", ["particle"]),
    ("What is the strong force?", ["force"]),
    ("What is granite?", ["igneous", "rock"]),
    ("What is a hurricane?", ["storm"]),
    ("What is a rainforest?", ["forest", "biome"]),
    ("What type of rock is marble?", ["metamorphic", "rock"]),
    ("What is an apple?", ["fruit"]),
]

PHASE3_SANITY = [
    ("Who led the Roman empire?", ["augustus", "caesar", "nero"]),
    ("Who was Alexander?", ["king", "macedonia"]),
    ("What is the Mongol empire from?", ["mongolia"]),
    ("Who was Napoleon?", ["emperor", "france"]),
    ("Who painted in the impressionism movement?", ["renoir", "monet", "degas"]),
    ("Who composed in the baroque era?", ["bach", "vivaldi", "handel"]),
    ("Who wrote Ulysses?", ["joyce"]),
    ("Who wrote War and Peace?", ["tolstoy"]),
    ("Who wrote 1984?", ["orwell"]),
    ("Who is Muhammad?", ["prophet", "islam"]),
    ("What is Buddhism?", ["religion"]),
    ("Who is Zeus?", ["deity", "greek", "god"]),
    ("What is sushi?", ["japanese", "food"]),
    ("What is pizza?", ["italian", "food"]),
    ("Who played basketball like Jordan?", ["basketball", "athlete"]),
    ("Who was Pele?", ["soccer", "brazil"]),
    ("What is jazz?", ["genre", "music"]),
    ("Who was Beethoven?", ["composer", "germany"]),
    ("Who was Confucius?", ["philosopher", "china"]),
    ("What is Hinduism?", ["religion"]),
]


def run_sanity(client, questions, name):
    print("\n" + "=" * 68, flush=True)
    print(f"{name} SANITY ({len(questions)} questions)", flush=True)
    print("=" * 68, flush=True)
    hits = 0
    for i, (q, expected) in enumerate(questions, 1):
        try:
            r = client.call_tool("lu4d_ask", {"query": q, "depth": 3, "top_k": 15})
            activated = r.get("structuredContent", {}).get("recall", {}).get("activated", [])
        except Exception as e:
            activated = []
            print(f"[sanity] Q{i} error: {e}", flush=True)
        contents = [a.get("content", "").lower() for a in activated]
        joined = " | ".join(contents)
        hit = any(kw.lower() in joined for kw in expected)
        if hit: hits += 1
        mark = "PASS" if hit else "MISS"
        print(f"  Q{i:02d} [{mark}] {q}", flush=True)
        print(f"        expected={expected} | top-5={contents[:5]}", flush=True)
    pct = 100 * hits / len(questions)
    print(f"\n{name} SANITY SCORE: {hits}/{len(questions)} ({pct:.1f}%)", flush=True)
    return hits, len(questions)


def phase_status(client, label):
    r = client.call_tool("lu4d_status", {})
    txt = r.get("content", [{}])[0].get("text", "")
    print(f"\n[status] {label}:\n  " + txt.replace("\n", "\n  "), flush=True)
    return txt


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase2(client, taught_cache):
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 2 — SCIENCE DEEP DIVE", flush=True)
    print("#" * 70, flush=True)

    phase_status(client, "pre-Phase-2")

    s = client.call_tool("lu4d_session_start", {})
    sid = s.get("structuredContent", {}).get("session_id")
    print(f"[session] Phase 2 session_id={sid}", flush=True)

    start = time.time()
    total = 0

    # Curated
    for name, builder, w in [
        ("chemistry", build_chemistry_triples, 0.9),
        ("biology",   build_biology_triples,   0.9),
        ("medicine",  build_medicine_triples,  0.9),
        ("physics",   build_physics_triples,   0.9),
        ("earth",     build_earth_triples,     0.9),
    ]:
        triples = builder()
        print(f"\n[{name}] built {len(triples)} triples", flush=True)
        total += teach_triples(client, triples, name, weight=w,
                               confidence="Verified", taught_cache=taught_cache)

    # HF datasets
    print("\n[sciq] streaming...", flush=True)
    total += teach_triples(client, stream_sciq(11000, taught_cache), "sciq",
                           weight=0.7, confidence="Verified", taught_cache=taught_cache)

    print("\n[openbookqa] loading...", flush=True)
    total += teach_triples(client, stream_openbookqa(taught_cache), "openbookqa",
                           weight=0.75, confidence="Verified", taught_cache=taught_cache)

    print("\n[arc-challenge] loading...", flush=True)
    total += teach_triples(client, stream_ai2_arc(taught_cache, "ARC-Challenge"),
                           "arc-challenge", weight=0.7, confidence="Verified",
                           taught_cache=taught_cache)
    print("\n[arc-easy] loading...", flush=True)
    total += teach_triples(client, stream_ai2_arc(taught_cache, "ARC-Easy"),
                           "arc-easy", weight=0.7, confidence="Verified",
                           taught_cache=taught_cache)

    print("\n[scitail] loading...", flush=True)
    total += teach_triples(client, stream_scitail(taught_cache), "scitail",
                           weight=0.65, confidence="Unverified",
                           taught_cache=taught_cache)

    client.call_tool("lu4d_session_end", {})
    c = client.call_tool("lu4d_consolidate", {})
    print("[consolidate] " + c.get("content", [{}])[0].get("text", ""), flush=True)

    elapsed = time.time() - start
    status = phase_status(client, "post-Phase-2")

    hits, tot = run_sanity(client, PHASE2_SANITY, "PHASE 2")

    print("\n" + "=" * 68, flush=True)
    print("PHASE 2 SUMMARY", flush=True)
    print("=" * 68, flush=True)
    print(f"  teach calls:   {total}", flush=True)
    print(f"  sanity:        {hits}/{tot} ({100*hits/tot:.1f}%)", flush=True)
    print(f"  wall time:     {elapsed:.1f}s", flush=True)

    return status, hits, tot, total


def run_phase3(client, taught_cache):
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 3 — HISTORY + CULTURE", flush=True)
    print("#" * 70, flush=True)

    phase_status(client, "pre-Phase-3")

    s = client.call_tool("lu4d_session_start", {})
    sid = s.get("structuredContent", {}).get("session_id")
    print(f"[session] Phase 3 session_id={sid}", flush=True)

    start = time.time()
    total = 0

    for name, builder, w in [
        ("history",  build_history_triples, 0.9),
        ("arts",     build_arts_triples,    0.9),
        ("religion", build_religion_triples, 0.9),
        ("sports",   build_sports_triples,  0.9),
        ("cuisine",  build_cuisine_triples, 0.9),
    ]:
        triples = builder()
        print(f"\n[{name}] built {len(triples)} triples", flush=True)
        total += teach_triples(client, triples, name, weight=w,
                               confidence="Verified", taught_cache=taught_cache)

    print("\n[trivia_qa] streaming...", flush=True)
    total += teach_triples(client, stream_trivia_qa(30000, taught_cache),
                           "trivia_qa", weight=0.7, confidence="Unverified",
                           taught_cache=taught_cache)

    client.call_tool("lu4d_session_end", {})
    c = client.call_tool("lu4d_consolidate", {})
    print("[consolidate] " + c.get("content", [{}])[0].get("text", ""), flush=True)

    elapsed = time.time() - start
    status = phase_status(client, "post-Phase-3")
    hits, tot = run_sanity(client, PHASE3_SANITY, "PHASE 3")

    print("\n" + "=" * 68, flush=True)
    print("PHASE 3 SUMMARY", flush=True)
    print("=" * 68, flush=True)
    print(f"  teach calls:   {total}", flush=True)
    print(f"  sanity:        {hits}/{tot} ({100*hits/tot:.1f}%)", flush=True)
    print(f"  wall time:     {elapsed:.1f}s", flush=True)

    return status, hits, tot, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(LU4D_BIN):
        print(f"FATAL: lu4d.exe not found at {LU4D_BIN}", flush=True); sys.exit(1)
    if not os.path.exists(BRAIN_FILE):
        print(f"FATAL: brain file not found at {BRAIN_FILE} - run phase1 first",
              flush=True); sys.exit(1)

    # --no-save during training for speed; final save on MCP shutdown (patched)
    client = MCPClient(BRAIN_FILE, no_save=True)
    client.start()
    taught_cache = set()
    all_run_start = time.time()
    try:
        _, p2_hits, p2_tot, p2_total = run_phase2(client, taught_cache)
        _, p3_hits, p3_tot, p3_total = run_phase3(client, taught_cache)
        total_elapsed = time.time() - all_run_start
        print("\n\n" + "#" * 70, flush=True)
        print("# COMBINED PHASE 2 + 3 SUMMARY", flush=True)
        print("#" * 70, flush=True)
        print(f"  Phase 2 teaches: {p2_total}, sanity: {p2_hits}/{p2_tot}", flush=True)
        print(f"  Phase 3 teaches: {p3_total}, sanity: {p3_hits}/{p3_tot}", flush=True)
        print(f"  total wall time: {total_elapsed:.1f}s", flush=True)
    finally:
        client.stop()


if __name__ == "__main__":
    main()

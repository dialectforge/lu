"""
Lu 4D Phase 1: Foundation Knowledge.

Builds a fresh baseline brain (lu4d_curriculum.lbr) with ~200K knowledge bonds
drawn from ConceptNet5, WordNet, curated world geography, and curated science
fundamentals. All ingestion happens inside a single temporal session so later
phases can diff against this foundation.

No npm / axios. Pure stdlib + requests + datasets + nltk.
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

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

# Per-source teach targets (each triple yields 1 forward + 1 reverse bond)
TARGET_CONCEPTNET_TRIPLES = 40_000     # ~80K teaches
TARGET_WORDNET_TRIPLES    = 30_000     # ~60K teaches
TARGET_WIKIDATA_TRIPLES   = 15_000     # ~30K teaches  (web_questions + curated)
TARGET_GEOGRAPHY_TRIPLES  =  8_000     # ~16K teaches
TARGET_SCIENCE_TRIPLES    =  5_000     # ~10K teaches

PROGRESS_EVERY = 10_000

# ---------------------------------------------------------------------------
# MCP client (JSON-RPC over stdio to lu4d.exe --mcp)
# ---------------------------------------------------------------------------

class MCPClient:
    def __init__(self, brain_path):
        self.brain_path = brain_path
        self.proc = None
        self.next_id = 1

    def start(self):
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
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.terminate()

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
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
only just even still yet already
""".split())

_norm_re = re.compile(r"[^a-z0-9\s\-]")

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = _norm_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_single_token(s: str) -> bool:
    return bool(s) and " " not in s and len(s) >= 2 and len(s) <= 30 and not s.isdigit()

def acceptable_phrase(s: str, max_words: int = 3) -> bool:
    """Multi-word allowed if no stopwords and short."""
    if not s or len(s) > 60:
        return False
    toks = s.split()
    if len(toks) > max_words:
        return False
    for t in toks:
        if t in STOPWORDS or len(t) < 2:
            return False
    return True


# ---------------------------------------------------------------------------
# Curated: Geography
# ---------------------------------------------------------------------------

COUNTRY_CAPITAL = [
    # Europe
    ("france", "paris", "europe"), ("germany", "berlin", "europe"),
    ("italy", "rome", "europe"), ("spain", "madrid", "europe"),
    ("portugal", "lisbon", "europe"), ("england", "london", "europe"),
    ("scotland", "edinburgh", "europe"), ("ireland", "dublin", "europe"),
    ("netherlands", "amsterdam", "europe"), ("belgium", "brussels", "europe"),
    ("luxembourg", "luxembourg", "europe"), ("switzerland", "bern", "europe"),
    ("austria", "vienna", "europe"), ("denmark", "copenhagen", "europe"),
    ("sweden", "stockholm", "europe"), ("norway", "oslo", "europe"),
    ("finland", "helsinki", "europe"), ("iceland", "reykjavik", "europe"),
    ("poland", "warsaw", "europe"), ("czechia", "prague", "europe"),
    ("slovakia", "bratislava", "europe"), ("hungary", "budapest", "europe"),
    ("romania", "bucharest", "europe"), ("bulgaria", "sofia", "europe"),
    ("greece", "athens", "europe"), ("croatia", "zagreb", "europe"),
    ("serbia", "belgrade", "europe"), ("ukraine", "kyiv", "europe"),
    ("belarus", "minsk", "europe"), ("lithuania", "vilnius", "europe"),
    ("latvia", "riga", "europe"), ("estonia", "tallinn", "europe"),
    ("russia", "moscow", "europe"), ("turkey", "ankara", "europe"),
    ("albania", "tirana", "europe"), ("slovenia", "ljubljana", "europe"),
    ("bosnia", "sarajevo", "europe"), ("macedonia", "skopje", "europe"),
    ("montenegro", "podgorica", "europe"), ("malta", "valletta", "europe"),
    ("cyprus", "nicosia", "europe"), ("moldova", "chisinau", "europe"),
    ("andorra", "andorra", "europe"), ("monaco", "monaco", "europe"),
    # Asia
    ("japan", "tokyo", "asia"), ("china", "beijing", "asia"),
    ("india", "delhi", "asia"), ("pakistan", "islamabad", "asia"),
    ("bangladesh", "dhaka", "asia"), ("indonesia", "jakarta", "asia"),
    ("philippines", "manila", "asia"), ("vietnam", "hanoi", "asia"),
    ("thailand", "bangkok", "asia"), ("malaysia", "kuala", "asia"),
    ("singapore", "singapore", "asia"), ("myanmar", "naypyidaw", "asia"),
    ("cambodia", "phnom", "asia"), ("laos", "vientiane", "asia"),
    ("korea", "seoul", "asia"), ("mongolia", "ulaanbaatar", "asia"),
    ("taiwan", "taipei", "asia"), ("nepal", "kathmandu", "asia"),
    ("bhutan", "thimphu", "asia"), ("sri lanka", "colombo", "asia"),
    ("afghanistan", "kabul", "asia"), ("iran", "tehran", "asia"),
    ("iraq", "baghdad", "asia"), ("syria", "damascus", "asia"),
    ("israel", "jerusalem", "asia"), ("palestine", "ramallah", "asia"),
    ("jordan", "amman", "asia"), ("lebanon", "beirut", "asia"),
    ("saudi", "riyadh", "asia"), ("yemen", "sanaa", "asia"),
    ("oman", "muscat", "asia"), ("qatar", "doha", "asia"),
    ("bahrain", "manama", "asia"), ("kuwait", "kuwait", "asia"),
    ("uae", "abu dhabi", "asia"), ("kazakhstan", "astana", "asia"),
    ("uzbekistan", "tashkent", "asia"), ("turkmenistan", "ashgabat", "asia"),
    ("tajikistan", "dushanbe", "asia"), ("kyrgyzstan", "bishkek", "asia"),
    ("azerbaijan", "baku", "asia"), ("armenia", "yerevan", "asia"),
    ("georgia", "tbilisi", "asia"),
    # Africa
    ("egypt", "cairo", "africa"), ("libya", "tripoli", "africa"),
    ("tunisia", "tunis", "africa"), ("algeria", "algiers", "africa"),
    ("morocco", "rabat", "africa"), ("sudan", "khartoum", "africa"),
    ("ethiopia", "addis", "africa"), ("somalia", "mogadishu", "africa"),
    ("kenya", "nairobi", "africa"), ("tanzania", "dodoma", "africa"),
    ("uganda", "kampala", "africa"), ("rwanda", "kigali", "africa"),
    ("burundi", "gitega", "africa"), ("congo", "kinshasa", "africa"),
    ("nigeria", "abuja", "africa"), ("ghana", "accra", "africa"),
    ("senegal", "dakar", "africa"), ("mali", "bamako", "africa"),
    ("niger", "niamey", "africa"), ("chad", "ndjamena", "africa"),
    ("cameroon", "yaounde", "africa"), ("gabon", "libreville", "africa"),
    ("angola", "luanda", "africa"), ("zambia", "lusaka", "africa"),
    ("zimbabwe", "harare", "africa"), ("mozambique", "maputo", "africa"),
    ("namibia", "windhoek", "africa"), ("botswana", "gaborone", "africa"),
    ("madagascar", "antananarivo", "africa"), ("mauritius", "louis", "africa"),
    ("seychelles", "victoria", "africa"), ("malawi", "lilongwe", "africa"),
    ("lesotho", "maseru", "africa"), ("eswatini", "mbabane", "africa"),
    ("eritrea", "asmara", "africa"), ("djibouti", "djibouti", "africa"),
    ("mauritania", "nouakchott", "africa"),
    # Americas
    ("usa", "washington", "north america"), ("canada", "ottawa", "north america"),
    ("mexico", "mexico", "north america"), ("guatemala", "guatemala", "north america"),
    ("belize", "belmopan", "north america"), ("honduras", "tegucigalpa", "north america"),
    ("nicaragua", "managua", "north america"), ("panama", "panama", "north america"),
    ("cuba", "havana", "north america"), ("haiti", "prince", "north america"),
    ("jamaica", "kingston", "north america"),
    ("brazil", "brasilia", "south america"), ("argentina", "aires", "south america"),
    ("chile", "santiago", "south america"), ("peru", "lima", "south america"),
    ("colombia", "bogota", "south america"), ("venezuela", "caracas", "south america"),
    ("ecuador", "quito", "south america"), ("bolivia", "paz", "south america"),
    ("paraguay", "asuncion", "south america"), ("uruguay", "montevideo", "south america"),
    ("guyana", "georgetown", "south america"), ("suriname", "paramaribo", "south america"),
    # Oceania
    ("australia", "canberra", "oceania"), ("zealand", "wellington", "oceania"),
    ("fiji", "suva", "oceania"), ("samoa", "apia", "oceania"),
    ("tonga", "nukualofa", "oceania"), ("vanuatu", "vila", "oceania"),
    ("papua", "moresby", "oceania"),
]

# Country -> primary language
COUNTRY_LANGUAGE = {
    "france": "french", "germany": "german", "italy": "italian", "spain": "spanish",
    "portugal": "portuguese", "england": "english", "netherlands": "dutch",
    "belgium": "french", "switzerland": "german", "austria": "german",
    "denmark": "danish", "sweden": "swedish", "norway": "norwegian",
    "finland": "finnish", "iceland": "icelandic", "poland": "polish",
    "czechia": "czech", "hungary": "hungarian", "romania": "romanian",
    "bulgaria": "bulgarian", "greece": "greek", "russia": "russian",
    "ukraine": "ukrainian", "turkey": "turkish", "japan": "japanese",
    "china": "mandarin", "india": "hindi", "pakistan": "urdu",
    "bangladesh": "bengali", "indonesia": "indonesian", "thailand": "thai",
    "vietnam": "vietnamese", "korea": "korean", "mongolia": "mongolian",
    "iran": "persian", "iraq": "arabic", "syria": "arabic", "israel": "hebrew",
    "saudi": "arabic", "egypt": "arabic", "morocco": "arabic", "algeria": "arabic",
    "ethiopia": "amharic", "somalia": "somali", "kenya": "swahili",
    "nigeria": "english", "ghana": "english", "brazil": "portuguese",
    "argentina": "spanish", "chile": "spanish", "peru": "spanish",
    "colombia": "spanish", "mexico": "spanish", "usa": "english",
    "canada": "english", "australia": "english", "zealand": "english",
    "cuba": "spanish",
}

# Continent populations / basic facts
CONTINENTS = ["europe", "asia", "africa", "north america", "south america", "oceania", "antarctica"]
OCEANS = ["pacific", "atlantic", "indian", "arctic", "southern"]
RIVERS = [
    ("nile", "egypt", "africa"), ("amazon", "brazil", "south america"),
    ("yangtze", "china", "asia"), ("mississippi", "usa", "north america"),
    ("yenisei", "russia", "asia"), ("ganges", "india", "asia"),
    ("danube", "germany", "europe"), ("volga", "russia", "europe"),
    ("rhine", "germany", "europe"), ("seine", "france", "europe"),
    ("thames", "england", "europe"), ("tiber", "italy", "europe"),
    ("euphrates", "iraq", "asia"), ("tigris", "iraq", "asia"),
    ("mekong", "vietnam", "asia"), ("indus", "pakistan", "asia"),
    ("congo", "congo", "africa"), ("niger", "niger", "africa"),
    ("zambezi", "zambia", "africa"), ("colorado", "usa", "north america"),
    ("ohio", "usa", "north america"), ("hudson", "usa", "north america"),
    ("parana", "argentina", "south america"), ("orinoco", "venezuela", "south america"),
    ("po", "italy", "europe"), ("elbe", "germany", "europe"),
    ("loire", "france", "europe"),
]
MOUNTAINS = [
    ("everest", "himalayas", "asia"), ("k2", "karakoram", "asia"),
    ("kangchenjunga", "himalayas", "asia"), ("kilimanjaro", "tanzania", "africa"),
    ("denali", "alaska", "north america"), ("aconcagua", "andes", "south america"),
    ("elbrus", "caucasus", "europe"), ("vinson", "antarctica", "antarctica"),
    ("fuji", "japan", "asia"), ("matterhorn", "alps", "europe"),
    ("mont blanc", "alps", "europe"), ("olympus", "greece", "europe"),
    ("rushmore", "usa", "north america"), ("whitney", "usa", "north america"),
    ("kosciuszko", "australia", "oceania"),
]
DESERTS = [
    ("sahara", "africa"), ("gobi", "asia"), ("kalahari", "africa"),
    ("atacama", "south america"), ("mojave", "north america"),
    ("sonoran", "north america"), ("arabian", "asia"), ("thar", "asia"),
    ("karakum", "asia"), ("patagonian", "south america"),
    ("antarctic", "antarctica"),
]
SEAS = [
    ("mediterranean", "europe"), ("caribbean", "north america"),
    ("baltic", "europe"), ("adriatic", "europe"), ("aegean", "europe"),
    ("caspian", "asia"), ("black", "europe"), ("red", "africa"),
    ("arabian", "asia"), ("coral", "oceania"), ("bering", "north america"),
]


def build_geography_triples(target):
    out = []
    for country, capital, cont in COUNTRY_CAPITAL:
        if is_single_token(country) and is_single_token(capital):
            out.append((capital, "IsA", "capital"))
            out.append((country, "HasProperty", capital))
            out.append((capital, "PartOf", country))
            out.append((country, "PartOf", cont.replace(" ", "")))  # single-token continent
            out.append((country, "IsA", "country"))
    for cont in CONTINENTS:
        ct = cont.replace(" ", "")
        out.append((ct, "IsA", "continent"))
    for ocean in OCEANS:
        out.append((ocean, "IsA", "ocean"))
        out.append((ocean, "PartOf", "earth"))
    for country, lang in COUNTRY_LANGUAGE.items():
        if is_single_token(country) and is_single_token(lang):
            out.append((country, "HasProperty", lang))
            out.append((lang, "IsA", "language"))
    for river, country, cont in RIVERS:
        if is_single_token(river):
            out.append((river, "IsA", "river"))
            out.append((river, "PartOf", country))
            out.append((river, "PartOf", cont.replace(" ", "")))
    for mt, region, cont in MOUNTAINS:
        if is_single_token(mt):
            out.append((mt, "IsA", "mountain"))
            out.append((mt, "PartOf", region.replace(" ", "")))
            out.append((mt, "PartOf", cont.replace(" ", "")))
    for desert, cont in DESERTS:
        if is_single_token(desert):
            out.append((desert, "IsA", "desert"))
            out.append((desert, "PartOf", cont.replace(" ", "")))
    for sea, cont in SEAS:
        if is_single_token(sea):
            out.append((sea, "IsA", "sea"))
            out.append((sea, "PartOf", cont.replace(" ", "")))
    # Hierarchy glue
    out += [
        ("country", "PartOf", "continent"), ("continent", "PartOf", "earth"),
        ("capital", "IsA", "city"), ("city", "PartOf", "country"),
        ("river", "PartOf", "geography"), ("mountain", "PartOf", "geography"),
        ("desert", "PartOf", "geography"), ("ocean", "PartOf", "earth"),
        ("sea", "PartOf", "earth"), ("language", "UsedFor", "communication"),
    ]
    # Deduplicate and cap
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
        if len(uniq) >= target:
            break
    return uniq


# ---------------------------------------------------------------------------
# Curated: Science (periodic table + body + physics)
# ---------------------------------------------------------------------------

ELEMENTS = [
    # (symbol-lower, name, category, atomic_number)
    ("h", "hydrogen", "nonmetal"), ("he", "helium", "noble"),
    ("li", "lithium", "alkali"), ("be", "beryllium", "alkaline"),
    ("b", "boron", "metalloid"), ("c", "carbon", "nonmetal"),
    ("n", "nitrogen", "nonmetal"), ("o", "oxygen", "nonmetal"),
    ("f", "fluorine", "halogen"), ("ne", "neon", "noble"),
    ("na", "sodium", "alkali"), ("mg", "magnesium", "alkaline"),
    ("al", "aluminum", "metal"), ("si", "silicon", "metalloid"),
    ("p", "phosphorus", "nonmetal"), ("s", "sulfur", "nonmetal"),
    ("cl", "chlorine", "halogen"), ("ar", "argon", "noble"),
    ("k", "potassium", "alkali"), ("ca", "calcium", "alkaline"),
    ("sc", "scandium", "metal"), ("ti", "titanium", "metal"),
    ("v", "vanadium", "metal"), ("cr", "chromium", "metal"),
    ("mn", "manganese", "metal"), ("fe", "iron", "metal"),
    ("co", "cobalt", "metal"), ("ni", "nickel", "metal"),
    ("cu", "copper", "metal"), ("zn", "zinc", "metal"),
    ("ga", "gallium", "metal"), ("ge", "germanium", "metalloid"),
    ("as", "arsenic", "metalloid"), ("se", "selenium", "nonmetal"),
    ("br", "bromine", "halogen"), ("kr", "krypton", "noble"),
    ("rb", "rubidium", "alkali"), ("sr", "strontium", "alkaline"),
    ("y", "yttrium", "metal"), ("zr", "zirconium", "metal"),
    ("nb", "niobium", "metal"), ("mo", "molybdenum", "metal"),
    ("tc", "technetium", "metal"), ("ru", "ruthenium", "metal"),
    ("rh", "rhodium", "metal"), ("pd", "palladium", "metal"),
    ("ag", "silver", "metal"), ("cd", "cadmium", "metal"),
    ("in", "indium", "metal"), ("sn", "tin", "metal"),
    ("sb", "antimony", "metalloid"), ("te", "tellurium", "metalloid"),
    ("i", "iodine", "halogen"), ("xe", "xenon", "noble"),
    ("cs", "cesium", "alkali"), ("ba", "barium", "alkaline"),
    ("la", "lanthanum", "metal"), ("ce", "cerium", "metal"),
    ("pr", "praseodymium", "metal"), ("nd", "neodymium", "metal"),
    ("pm", "promethium", "metal"), ("sm", "samarium", "metal"),
    ("eu", "europium", "metal"), ("gd", "gadolinium", "metal"),
    ("tb", "terbium", "metal"), ("dy", "dysprosium", "metal"),
    ("ho", "holmium", "metal"), ("er", "erbium", "metal"),
    ("tm", "thulium", "metal"), ("yb", "ytterbium", "metal"),
    ("lu", "lutetium", "metal"), ("hf", "hafnium", "metal"),
    ("ta", "tantalum", "metal"), ("w", "tungsten", "metal"),
    ("re", "rhenium", "metal"), ("os", "osmium", "metal"),
    ("ir", "iridium", "metal"), ("pt", "platinum", "metal"),
    ("au", "gold", "metal"), ("hg", "mercury", "metal"),
    ("tl", "thallium", "metal"), ("pb", "lead", "metal"),
    ("bi", "bismuth", "metal"), ("po", "polonium", "metalloid"),
    ("at", "astatine", "halogen"), ("rn", "radon", "noble"),
    ("fr", "francium", "alkali"), ("ra", "radium", "alkaline"),
    ("ac", "actinium", "metal"), ("th", "thorium", "metal"),
    ("pa", "protactinium", "metal"), ("u", "uranium", "metal"),
    ("np", "neptunium", "metal"), ("pu", "plutonium", "metal"),
    ("am", "americium", "metal"), ("cm", "curium", "metal"),
]

PLANETS = [
    ("mercury", "solar", 1), ("venus", "solar", 2), ("earth", "solar", 3),
    ("mars", "solar", 4), ("jupiter", "solar", 5), ("saturn", "solar", 6),
    ("uranus", "solar", 7), ("neptune", "solar", 8), ("pluto", "solar", 9),
]

BODY_PARTS = [
    ("heart", "pumping blood", "circulatory"),
    ("brain", "thinking", "nervous"),
    ("lungs", "breathing", "respiratory"),
    ("liver", "detoxification", "digestive"),
    ("kidney", "filtering blood", "urinary"),
    ("stomach", "digestion", "digestive"),
    ("intestine", "absorption", "digestive"),
    ("pancreas", "insulin", "endocrine"),
    ("spleen", "immunity", "lymphatic"),
    ("skin", "protection", "integumentary"),
    ("bone", "structure", "skeletal"),
    ("muscle", "movement", "muscular"),
    ("eye", "vision", "sensory"),
    ("ear", "hearing", "sensory"),
    ("nose", "smelling", "sensory"),
    ("tongue", "tasting", "sensory"),
    ("blood", "transport", "circulatory"),
    ("nerve", "signal", "nervous"),
]


def build_science_triples(target):
    out = []
    for sym, name, cat in ELEMENTS:
        out.append((name, "IsA", "element"))
        out.append((name, "HasProperty", sym))
        out.append((name, "IsA", cat))
        out.append((cat, "IsA", "element"))
    out += [
        ("element", "PartOf", "periodic"), ("periodic", "IsA", "table"),
        ("metal", "ContrastedWith", "nonmetal"),
        ("noble", "IsA", "gas"), ("halogen", "IsA", "group"),
        ("alkali", "IsA", "metal"), ("alkaline", "IsA", "metal"),
        ("metalloid", "IsA", "element"),
    ]
    # Compounds
    compounds = [
        ("water", "hydrogen"), ("water", "oxygen"),
        ("salt", "sodium"), ("salt", "chlorine"),
        ("ammonia", "nitrogen"), ("ammonia", "hydrogen"),
        ("methane", "carbon"), ("methane", "hydrogen"),
        ("carbon dioxide", "carbon"), ("carbon dioxide", "oxygen"),
        ("glucose", "carbon"), ("glucose", "hydrogen"),
        ("rust", "iron"), ("rust", "oxygen"),
        ("brass", "copper"), ("brass", "zinc"),
        ("bronze", "copper"), ("bronze", "tin"),
        ("steel", "iron"), ("steel", "carbon"),
    ]
    for comp, elem in compounds:
        if is_single_token(comp):
            out.append((comp, "HasProperty", elem))
            out.append((comp, "IsA", "compound"))
    # Planets
    for planet, _sys, _order in PLANETS:
        out.append((planet, "IsA", "planet"))
        out.append((planet, "PartOf", "solar"))
    out += [
        ("solar", "IsA", "system"), ("solar", "PartOf", "galaxy"),
        ("galaxy", "PartOf", "universe"),
        ("sun", "IsA", "star"), ("moon", "IsA", "satellite"),
        ("moon", "PartOf", "earth"), ("asteroid", "PartOf", "solar"),
        ("comet", "PartOf", "solar"), ("meteor", "PartOf", "solar"),
    ]
    # Body parts
    for part, func, system in BODY_PARTS:
        if is_single_token(part):
            out.append((part, "PartOf", "body"))
            out.append((part, "PartOf", system))
            out.append((system, "IsA", "system"))
            # map function -> use single-word key
            f = func.split()[0]
            out.append((part, "UsedFor", f))
    # Cell biology
    out += [
        ("cell", "PartOf", "organism"), ("cell", "IsA", "unit"),
        ("mitochondria", "PartOf", "cell"), ("mitochondria", "Produces", "atp"),
        ("nucleus", "PartOf", "cell"), ("ribosome", "PartOf", "cell"),
        ("ribosome", "Produces", "protein"),
        ("chloroplast", "PartOf", "cell"), ("chloroplast", "Produces", "oxygen"),
        ("dna", "PartOf", "nucleus"), ("rna", "PartOf", "cell"),
        ("gene", "PartOf", "dna"), ("chromosome", "PartOf", "nucleus"),
        ("protein", "PartOf", "cell"), ("enzyme", "IsA", "protein"),
        ("membrane", "PartOf", "cell"),
        ("photosynthesis", "Produces", "oxygen"),
        ("photosynthesis", "Requires", "sunlight"),
        ("respiration", "Produces", "energy"),
        ("respiration", "Requires", "oxygen"),
    ]
    # Physics
    out += [
        ("gravity", "Causes", "falling"), ("gravity", "IsA", "force"),
        ("electromagnetism", "IsA", "force"),
        ("friction", "IsA", "force"), ("friction", "Opposes", "motion"),
        ("energy", "IsA", "quantity"), ("mass", "IsA", "quantity"),
        ("velocity", "IsA", "quantity"), ("acceleration", "IsA", "quantity"),
        ("light", "IsA", "radiation"), ("heat", "IsA", "energy"),
        ("sound", "Requires", "medium"), ("sound", "IsA", "wave"),
        ("electron", "PartOf", "atom"), ("proton", "PartOf", "atom"),
        ("neutron", "PartOf", "atom"), ("quark", "PartOf", "proton"),
        ("photon", "IsA", "particle"), ("photon", "PartOf", "light"),
        ("wave", "IsA", "phenomenon"),
        ("atom", "IsA", "particle"), ("molecule", "IsA", "particle"),
    ]
    # Cap
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
        if len(uniq) >= target:
            break
    return uniq


# ---------------------------------------------------------------------------
# ConceptNet 5 loader (streaming, English only)
# ---------------------------------------------------------------------------

# ConceptNet relation -> Lu bond type
CN_REL_MAP = {
    "IsA": "IsA",
    "HasA": "HasProperty",
    "HasProperty": "HasProperty",
    "MadeOf": "HasProperty",
    "PartOf": "PartOf",
    "AtLocation": "PartOf",
    "LocatedNear": "RelatedTo",
    "UsedFor": "UsedFor",
    "CapableOf": "UsedFor",
    "ReceivesAction": "UsedFor",
    "Causes": "Causes",
    "CausesDesire": "Causes",
    "MotivatedByGoal": "Causes",
    "ObstructedBy": "Opposes",
    "HasPrerequisite": "Requires",
    "HasFirstSubevent": "Produces",
    "HasLastSubevent": "Produces",
    "HasSubevent": "Produces",
    "CreatedBy": "DerivedFrom",
    "DerivedFrom": "DerivedFrom",
    "Antonym": "Opposes",
    "DistinctFrom": "ContrastedWith",
    "Synonym": "RelatedTo",
    "RelatedTo": "RelatedTo",
    "SimilarTo": "RelatedTo",
    "FormOf": "DerivedFrom",
}

_cn_re = re.compile(r"^/c/en/([a-z0-9_\-]+)(/[a-z](?:/\w+)?)?$")

def _cn_concept(path):
    """Return single-token lowercase concept or None."""
    m = _cn_re.match(path)
    if not m:
        return None
    term = m.group(1)
    # Skip multi-underscore compounds; we want single tokens for recall safety.
    if "_" in term:
        return None
    if len(term) < 2 or len(term) > 30:
        return None
    if term.isdigit():
        return None
    return term


def stream_conceptnet(target):
    """Yield (s, rel, o) tuples from ConceptNet5 streaming. English only, single tokens."""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[conceptnet] datasets unavailable: {e}", flush=True)
        return
    print(f"[conceptnet] streaming conceptnet5 (target={target})...", flush=True)
    try:
        ds = load_dataset("conceptnet5", "conceptnet5", split="train", streaming=True)
    except Exception as e:
        print(f"[conceptnet] load_dataset failed: {e}", flush=True)
        return

    seen = set()
    produced = 0
    examined = 0
    for row in ds:
        examined += 1
        if examined % 200_000 == 0:
            print(f"[conceptnet]   examined={examined}, produced={produced}", flush=True)
        if row.get("lang") != "en":
            continue
        rel = row.get("rel", "")
        if not rel.startswith("/r/"):
            continue
        rel_name = rel[3:]
        lu_rel = CN_REL_MAP.get(rel_name)
        if not lu_rel:
            continue
        s = _cn_concept(row.get("arg1", ""))
        o = _cn_concept(row.get("arg2", ""))
        if not s or not o or s == o:
            continue
        key = (s, lu_rel, o)
        if key in seen:
            continue
        seen.add(key)
        produced += 1
        yield key
        if produced >= target:
            return
    print(f"[conceptnet] stream ended at examined={examined}, produced={produced}", flush=True)


# ---------------------------------------------------------------------------
# WordNet hypernym loader
# ---------------------------------------------------------------------------

def stream_wordnet(target):
    try:
        import nltk
        try:
            from nltk.corpus import wordnet as wn
            _ = wn.synsets("dog")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            from nltk.corpus import wordnet as wn
    except Exception as e:
        print(f"[wordnet] unavailable: {e}", flush=True)
        return
    print(f"[wordnet] enumerating noun hypernyms (target={target})...", flush=True)
    seen = set()
    produced = 0
    for syn in wn.all_synsets("n"):
        for hyp in syn.hypernyms():
            for l in syn.lemmas():
                ln = l.name().replace("_", " ").lower()
                if " " in ln or not ln.isalpha() or len(ln) < 2:
                    continue
                for hl in hyp.lemmas():
                    hn = hl.name().replace("_", " ").lower()
                    if " " in hn or not hn.isalpha() or len(hn) < 2:
                        continue
                    if ln == hn:
                        continue
                    key = (ln, "IsA", hn)
                    if key in seen:
                        continue
                    seen.add(key)
                    produced += 1
                    yield key
                    if produced >= target:
                        return
    print(f"[wordnet] enumerated {produced} pairs", flush=True)


# ---------------------------------------------------------------------------
# Wikidata-like: web_questions (question -> answer pairs) + curated famous ppl
# ---------------------------------------------------------------------------

FAMOUS_PEOPLE = [
    ("einstein", "physicist", "relativity", "germany"),
    ("newton", "physicist", "gravity", "england"),
    ("darwin", "biologist", "evolution", "england"),
    ("curie", "chemist", "radium", "poland"),
    ("tesla", "inventor", "alternating", "serbia"),
    ("edison", "inventor", "bulb", "usa"),
    ("galileo", "astronomer", "telescope", "italy"),
    ("copernicus", "astronomer", "heliocentrism", "poland"),
    ("kepler", "astronomer", "orbits", "germany"),
    ("hawking", "physicist", "holes", "england"),
    ("feynman", "physicist", "electrodynamics", "usa"),
    ("bohr", "physicist", "atom", "denmark"),
    ("heisenberg", "physicist", "uncertainty", "germany"),
    ("schrodinger", "physicist", "cat", "austria"),
    ("dirac", "physicist", "antimatter", "england"),
    ("turing", "mathematician", "computing", "england"),
    ("godel", "mathematician", "incompleteness", "austria"),
    ("euler", "mathematician", "calculus", "switzerland"),
    ("gauss", "mathematician", "algebra", "germany"),
    ("pythagoras", "mathematician", "theorem", "greece"),
    ("archimedes", "mathematician", "lever", "greece"),
    ("euclid", "mathematician", "geometry", "greece"),
    ("aristotle", "philosopher", "logic", "greece"),
    ("plato", "philosopher", "republic", "greece"),
    ("socrates", "philosopher", "method", "greece"),
    ("kant", "philosopher", "critique", "germany"),
    ("nietzsche", "philosopher", "zarathustra", "germany"),
    ("descartes", "philosopher", "cogito", "france"),
    ("shakespeare", "playwright", "hamlet", "england"),
    ("dickens", "novelist", "copperfield", "england"),
    ("tolstoy", "novelist", "war", "russia"),
    ("dostoevsky", "novelist", "crime", "russia"),
    ("hemingway", "novelist", "bells", "usa"),
    ("homer", "poet", "odyssey", "greece"),
    ("dante", "poet", "inferno", "italy"),
    ("mozart", "composer", "symphony", "austria"),
    ("beethoven", "composer", "symphony", "germany"),
    ("bach", "composer", "fugue", "germany"),
    ("chopin", "composer", "nocturne", "poland"),
    ("tchaikovsky", "composer", "ballet", "russia"),
    ("wagner", "composer", "opera", "germany"),
    ("picasso", "painter", "guernica", "spain"),
    ("monet", "painter", "impressionism", "france"),
    ("davinci", "painter", "mona", "italy"),
    ("michelangelo", "painter", "sistine", "italy"),
    ("vangogh", "painter", "sunflowers", "netherlands"),
    ("rembrandt", "painter", "watch", "netherlands"),
    ("dali", "painter", "surrealism", "spain"),
    ("warhol", "painter", "pop", "usa"),
    ("lincoln", "president", "emancipation", "usa"),
    ("washington", "president", "revolution", "usa"),
    ("jefferson", "president", "declaration", "usa"),
    ("roosevelt", "president", "deal", "usa"),
    ("kennedy", "president", "moonshot", "usa"),
    ("churchill", "prime", "britain", "england"),
    ("napoleon", "emperor", "france", "france"),
    ("caesar", "emperor", "rome", "italy"),
    ("charlemagne", "emperor", "franks", "france"),
    ("genghis", "emperor", "mongols", "mongolia"),
    ("gandhi", "leader", "independence", "india"),
    ("mandela", "leader", "apartheid", "africa"),
    ("cleopatra", "queen", "egypt", "egypt"),
    ("elizabeth", "queen", "england", "england"),
    ("victoria", "queen", "empire", "england"),
    ("columbus", "explorer", "americas", "italy"),
    ("magellan", "explorer", "circumnavigation", "portugal"),
    ("marco polo", "explorer", "china", "italy"),
    ("polo", "explorer", "china", "italy"),
    ("cook", "explorer", "pacific", "england"),
    ("lewis", "explorer", "louisiana", "usa"),
    ("amundsen", "explorer", "pole", "norway"),
    ("armstrong", "astronaut", "moon", "usa"),
    ("gagarin", "astronaut", "space", "russia"),
    ("hippocrates", "doctor", "medicine", "greece"),
    ("pasteur", "biologist", "vaccination", "france"),
    ("fleming", "biologist", "penicillin", "scotland"),
    ("mendel", "biologist", "genetics", "austria"),
    ("watson", "biologist", "dna", "usa"),
    ("crick", "biologist", "dna", "england"),
]

def build_famous_triples():
    out = []
    for person, role, work, country in FAMOUS_PEOPLE:
        p = person.split()[0]  # single token
        if not is_single_token(p):
            continue
        out.append((p, "IsA", role))
        out.append((p, "Produces", work))
        out.append((p, "PartOf", country))
    # Add role hierarchy
    out += [
        ("physicist", "IsA", "scientist"), ("biologist", "IsA", "scientist"),
        ("chemist", "IsA", "scientist"), ("astronomer", "IsA", "scientist"),
        ("mathematician", "IsA", "scientist"), ("inventor", "IsA", "person"),
        ("philosopher", "IsA", "person"), ("playwright", "IsA", "writer"),
        ("novelist", "IsA", "writer"), ("poet", "IsA", "writer"),
        ("composer", "IsA", "musician"), ("painter", "IsA", "artist"),
        ("president", "IsA", "leader"), ("emperor", "IsA", "leader"),
        ("queen", "IsA", "leader"), ("prime", "IsA", "leader"),
        ("explorer", "IsA", "person"), ("astronaut", "IsA", "person"),
        ("doctor", "IsA", "person"), ("scientist", "IsA", "person"),
        ("writer", "IsA", "person"), ("musician", "IsA", "person"),
        ("artist", "IsA", "person"), ("leader", "IsA", "person"),
    ]
    return out


def stream_web_questions(target):
    """Yield (question_focus, RelatedTo, answer_token) triples from web_questions."""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[wq] datasets unavailable: {e}", flush=True)
        return
    try:
        ds = load_dataset("web_questions", split="train", streaming=True)
    except Exception as e:
        print(f"[wq] load failed: {e}", flush=True)
        return
    produced = 0
    seen = set()
    for row in ds:
        q = row.get("question", "") or ""
        answers = row.get("answers", [])
        if isinstance(answers, str):
            try:
                answers = eval(answers, {"__builtins__": {}})
            except Exception:
                answers = []
        q_norm = normalize(q)
        q_toks = [t for t in q_norm.split() if t not in STOPWORDS and len(t) >= 3 and not t.isdigit()]
        if len(q_toks) < 2:
            continue
        q_focus = q_toks[-1]
        for ans in (answers or [])[:2]:
            a_norm = normalize(str(ans))
            a_toks = [t for t in a_norm.split() if t not in STOPWORDS and len(t) >= 2 and not t.isdigit()]
            if not a_toks:
                continue
            a_focus = a_toks[-1]
            if q_focus == a_focus or not is_single_token(q_focus) or not is_single_token(a_focus):
                continue
            key = (q_focus, "RelatedTo", a_focus)
            if key in seen:
                continue
            seen.add(key)
            produced += 1
            yield key
            if produced >= target:
                return
    print(f"[wq] enumerated {produced} pairs from web_questions", flush=True)


# ---------------------------------------------------------------------------
# Batch teach: send a list of triples, emit forward + reverse bonds
# ---------------------------------------------------------------------------

def teach_triples(client, triples_iter, source_name, weight=0.85, confidence="Verified"):
    taught = 0
    errors = 0
    start = time.time()
    for (s, rel, o) in triples_iter:
        if not s or not o or s == o:
            errors += 1; continue
        if len(s) > 60 or len(o) > 60:
            errors += 1; continue
        try:
            client.call_tool("lu4d_teach", {
                "subject": s, "relation": rel, "object": o,
                "weight": weight, "confidence": confidence,
            })
            taught += 1
            client.call_tool("lu4d_teach", {
                "subject": o, "relation": "RelatedTo", "object": s,
                "weight": weight * 0.9, "confidence": confidence,
            })
            taught += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"[{source_name}] teach error {s!r}->{o!r}: {e}", flush=True)
        if taught % PROGRESS_EVERY == 0 and taught > 0:
            elapsed = time.time() - start
            rate = taught / elapsed if elapsed > 0 else 0
            print(f"[{source_name}] taught={taught} ({rate:.0f}/s, {elapsed:.1f}s)", flush=True)
    elapsed = time.time() - start
    rate = taught / elapsed if elapsed > 0 else 0
    print(f"[{source_name}] DONE taught={taught} errors={errors} "
          f"in {elapsed:.1f}s ({rate:.0f}/s)", flush=True)
    return taught, errors


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

SANITY_QUESTIONS = [
    # Geography
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("The Nile river is in which country?", ["egypt"]),
    ("In which continent is Brazil?", ["southamerica", "south"]),
    ("The Sahara is a what?", ["desert"]),
    # Science
    ("What produces ATP in a cell?", ["mitochondria"]),
    ("What do plants produce during photosynthesis?", ["oxygen"]),
    ("What is hydrogen?", ["element", "nonmetal"]),
    ("What is iron?", ["element", "metal"]),
    ("What planet is Jupiter?", ["planet"]),
    # Famous people
    ("Who discovered evolution?", ["darwin"]),
    ("Who discovered gravity?", ["newton"]),
    ("Who painted the Mona Lisa?", ["davinci"]),
    ("Who wrote Hamlet?", ["shakespeare"]),
    ("Who is Einstein?", ["physicist", "relativity"]),
    # WordNet / ConceptNet recall
    ("What is a dog?", ["animal", "mammal", "canine"]),
    ("What is a cat?", ["animal", "mammal", "feline"]),
    ("What is water?", ["liquid", "molecule", "compound", "hydrogen"]),
    ("What is a car?", ["vehicle", "motor"]),
    ("What is music?", ["art", "sound"]),
]


def sanity_check(client):
    print("\n" + "=" * 68, flush=True)
    print("SANITY CHECK (20 questions across domains)", flush=True)
    print("=" * 68, flush=True)
    hits = 0
    for i, (q, expected) in enumerate(SANITY_QUESTIONS, 1):
        try:
            r = client.call_tool("lu4d_ask", {"query": q, "depth": 3, "top_k": 15})
            activated = r.get("structuredContent", {}).get("recall", {}).get("activated", [])
        except Exception as e:
            activated = []
            print(f"[sanity] Q{i} error: {e}", flush=True)
        contents = [a.get("content", "").lower() for a in activated]
        joined = " | ".join(contents)
        hit = any(kw.lower() in joined for kw in expected)
        if hit:
            hits += 1
        mark = "PASS" if hit else "MISS"
        top5 = contents[:5]
        print(f"  Q{i:02d} [{mark}] {q}", flush=True)
        print(f"        expected={expected} | top-5={top5}", flush=True)
    pct = 100 * hits / len(SANITY_QUESTIONS)
    print(f"\nSANITY SCORE: {hits}/{len(SANITY_QUESTIONS)} ({pct:.1f}%)", flush=True)
    return hits, len(SANITY_QUESTIONS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(LU4D_BIN):
        print(f"FATAL: lu4d.exe not found at {LU4D_BIN}", flush=True)
        sys.exit(1)
    if os.path.exists(BRAIN_FILE):
        os.remove(BRAIN_FILE)
        print(f"[setup] removed prior brain at {BRAIN_FILE}", flush=True)

    run_start = time.time()
    client = MCPClient(BRAIN_FILE)
    client.start()

    total_taught = 0
    total_errors = 0

    try:
        # Pre-training status
        pre = client.call_tool("lu4d_status", {})
        print("[status] pre-train:\n  " +
              pre.get("content", [{}])[0].get("text", "").replace("\n", "\n  "),
              flush=True)

        # Start the baseline session (Session 1)
        print("\n[session] starting Session 1 (Foundation baseline)...", flush=True)
        s = client.call_tool("lu4d_session_start", {})
        sid = s.get("structuredContent", {}).get("session_id")
        print(f"[session] session_id={sid}", flush=True)

        # --- Phase 1a: Curated geography (fast, high confidence) ---
        print("\n--- 1a) Geography ---", flush=True)
        geo = build_geography_triples(TARGET_GEOGRAPHY_TRIPLES)
        print(f"[geography] built {len(geo)} triples", flush=True)
        t, e = teach_triples(client, geo, "geography", weight=0.9, confidence="Verified")
        total_taught += t; total_errors += e

        # --- Phase 1b: Curated science ---
        print("\n--- 1b) Science ---", flush=True)
        sci = build_science_triples(TARGET_SCIENCE_TRIPLES)
        print(f"[science] built {len(sci)} triples", flush=True)
        t, e = teach_triples(client, sci, "science", weight=0.9, confidence="Verified")
        total_taught += t; total_errors += e

        # --- Phase 1c: Famous people + web_questions (Wikidata-like) ---
        print("\n--- 1c) Famous people + web_questions ---", flush=True)
        fam = build_famous_triples()
        t, e = teach_triples(client, fam, "famous", weight=0.9, confidence="Verified")
        total_taught += t; total_errors += e

        wq_gen = stream_web_questions(max(0, TARGET_WIKIDATA_TRIPLES - len(fam)))
        t, e = teach_triples(client, wq_gen, "web_questions",
                             weight=0.7, confidence="Unverified")
        total_taught += t; total_errors += e

        # --- Phase 1d: WordNet hypernyms ---
        print("\n--- 1d) WordNet hypernyms ---", flush=True)
        wn_gen = stream_wordnet(TARGET_WORDNET_TRIPLES)
        t, e = teach_triples(client, wn_gen, "wordnet",
                             weight=0.8, confidence="Verified")
        total_taught += t; total_errors += e

        # --- Phase 1e: ConceptNet ---
        print("\n--- 1e) ConceptNet commonsense ---", flush=True)
        cn_gen = stream_conceptnet(TARGET_CONCEPTNET_TRIPLES)
        t, e = teach_triples(client, cn_gen, "conceptnet",
                             weight=0.7, confidence="Verified")
        total_taught += t; total_errors += e

        # --- End session + consolidate ---
        print("\n[session] ending Session 1...", flush=True)
        client.call_tool("lu4d_session_end", {})
        print("[consolidate] running baseline consolidation...", flush=True)
        c = client.call_tool("lu4d_consolidate", {})
        print("  -> " + c.get("content", [{}])[0].get("text", ""), flush=True)

        # Post-training status
        post = client.call_tool("lu4d_status", {})
        post_txt = post.get("content", [{}])[0].get("text", "")
        print("\n[status] post-train:\n  " + post_txt.replace("\n", "\n  "), flush=True)

        # Sanity check
        hits, total = sanity_check(client)

        elapsed = time.time() - run_start
        print("\n" + "=" * 68, flush=True)
        print("PHASE 1 FINAL SUMMARY", flush=True)
        print("=" * 68, flush=True)
        print(f"  total teach calls:   {total_taught}", flush=True)
        print(f"  teach errors:        {total_errors}", flush=True)
        print(f"  sanity score:        {hits}/{total} ({100*hits/total:.1f}%)", flush=True)
        print(f"  wall time:           {elapsed:.1f}s", flush=True)
        print(f"  brain file:          {BRAIN_FILE}", flush=True)
        print(f"  post-train status:\n    {post_txt.replace(chr(10), chr(10)+'    ')}",
              flush=True)
    finally:
        client.stop()


if __name__ == "__main__":
    main()

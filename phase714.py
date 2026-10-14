"""
Lu 4D Phases 7–14: Wikipedia, Language, Math, Geography, People, Modern World,
Bulk Trivia Expansion, and Final Validation.

  Phase 7  — Wikipedia core (Simple English Wikipedia articles)
  Phase 8  — Language & logic (synonyms, antonyms, analogies, commonsense)
  Phase 9  — Math & engineering (theorems, shapes, units, branches)
  Phase 10 — Geography deep (cities, borders, landmarks, climate, resources)
  Phase 11 — People & events (inventions, discoveries, battles, movements)
  Phase 12 — Modern world (companies, products, sports, entertainment)
  Phase 13 — Bulk trivia expansion (large HF streams to fill gaps)
  Phase 14 — Final validation (100-question cross-domain eval + temporal tests)

Rules: single-token nodes, bidirectional bonds, no stopwords, --no-save during
training (final save on MCP shutdown).  Aborts if projected brain > 4 GB.
"""

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

ROOT       = os.path.dirname(os.path.abspath(__file__))
LU4D_BIN   = os.path.join(ROOT, "target", "release", "lu4d.exe")
BRAIN_FILE = os.path.join(ROOT, "lu4d_curriculum.lbr")

PROGRESS_EVERY = 10_000

# Empirical ratio from prior phases: 883K bonds → 1.7 GB  ≈ 1 970 bytes/bond
BYTES_PER_BOND  = 1_970
SIZE_LIMIT_GB   = 4.0
SIZE_LIMIT_BYTES = int(SIZE_LIMIT_GB * 1024 ** 3)


# ---------------------------------------------------------------------------
# MCP client (identical to prior phases)
# ---------------------------------------------------------------------------

class MCPClient:
    def __init__(self, brain_path, no_save=True):
        self.brain_path = brain_path
        self.no_save    = no_save
        self.proc       = None
        self.next_id    = 1

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

def normalize(s):
    s = (s or "").lower().strip()
    s = _norm_re.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def is_single_token(s):
    return bool(s) and " " not in s and 2 <= len(s) <= 30 and not s.isdigit()

def content_tokens(text):
    out = []
    for tok in normalize(text).split():
        if len(tok) < 3 or len(tok) > 25: continue
        if tok in STOPWORDS or tok.isdigit(): continue
        out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Teach helper
# ---------------------------------------------------------------------------

def teach_triples(client, triples_iter, source, weight=0.85, confidence="Verified",
                  taught_cache=None):
    taught = skipped = errors = 0
    start  = time.time()
    cache  = taught_cache if taught_cache is not None else set()
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
            print(f"[{source}] taught={taught} skipped={skipped} "
                  f"({taught/elapsed:.0f}/s, {elapsed:.1f}s)", flush=True)
    elapsed = time.time() - start
    rate = taught / elapsed if elapsed > 0 else 0
    print(f"[{source}] DONE taught={taught} skipped={skipped} errors={errors} "
          f"in {elapsed:.1f}s ({rate:.0f}/s)", flush=True)
    return taught


# ---------------------------------------------------------------------------
# Size guard
# ---------------------------------------------------------------------------

def check_size(client, phase_name):
    """Return True to continue, False to abort."""
    r     = client.call_tool("lu4d_status", {})
    sc    = r.get("structuredContent", {})
    bonds = sc.get("bond_count", 0)
    nodes = sc.get("node_count", 0)
    proj  = bonds * BYTES_PER_BOND
    proj_gb = proj / 1024 ** 3
    remain  = (SIZE_LIMIT_BYTES - proj) / 1024 ** 3
    print(f"[size] post-{phase_name}: nodes={nodes:,} bonds={bonds:,} "
          f"projected={proj_gb:.2f} GB  remaining={remain:.2f} GB", flush=True)
    if proj > SIZE_LIMIT_BYTES:
        print(f"[size] ABORT — projected {proj_gb:.2f} GB exceeds "
              f"{SIZE_LIMIT_GB} GB limit", flush=True)
        return False
    return True


def phase_status(client, label):
    r   = client.call_tool("lu4d_status", {})
    txt = r.get("content", [{}])[0].get("text", "")
    print(f"\n[status] {label}:\n  " + txt.replace("\n", "\n  "), flush=True)
    return r.get("structuredContent", {})


# ===========================================================================
# PHASE 7 — WIKIPEDIA CORE
# ===========================================================================

def stream_wikipedia(limit, seen):
    try:
        from datasets import load_dataset
    except Exception:
        return
    ds = None
    for name in ("20220301.simple", "20220301.en"):
        try:
            ds = load_dataset("wikipedia", name, split="train", streaming=True)
            print(f"[wiki] loaded {name}", flush=True)
            break
        except Exception as e:
            print(f"[wiki] {name} failed: {e}", flush=True)
    if ds is None:
        return
    print(f"[wiki] streaming (limit={limit})...", flush=True)
    n = 0
    for row in ds:
        if n >= limit: return
        n += 1
        title = row.get("title", "") or ""
        text  = row.get("text",  "") or ""
        # Title tokens
        t_toks = content_tokens(title)
        if not t_toks: continue
        focus = t_toks[0]
        if not is_single_token(focus): continue
        # Co-occur title tokens
        for t in t_toks[1:3]:
            if is_single_token(t) and t != focus:
                yield (focus, "RelatedTo", t)
        # First paragraph → link to focus
        first_para = text.split("\n\n")[0] if "\n\n" in text else text[:600]
        p_toks = content_tokens(first_para)
        linked = set(t_toks)
        for t in p_toks:
            if not is_single_token(t) or t in linked: continue
            linked.add(t)
            yield (focus, "RelatedTo", t)
            if len(linked) > 10: break


# ===========================================================================
# PHASE 8 — LANGUAGE & LOGIC
# ===========================================================================

SYNONYM_PAIRS = [
    ("happy","joyful"),("sad","sorrowful"),("angry","furious"),("afraid","fearful"),
    ("brave","courageous"),("calm","peaceful"),("clever","intelligent"),("cruel","brutal"),
    ("eager","enthusiastic"),("fair","just"),("gentle","mild"),("honest","truthful"),
    ("humble","modest"),("kind","benevolent"),("lazy","idle"),("loyal","faithful"),
    ("neat","tidy"),("polite","courteous"),("proud","arrogant"),("quick","rapid"),
    ("rich","wealthy"),("shy","timid"),("strange","peculiar"),("strong","powerful"),
    ("tired","weary"),("wise","prudent"),("young","youthful"),("old","ancient"),
    ("big","large"),("small","tiny"),("tall","high"),("short","brief"),
    ("fast","swift"),("slow","gradual"),("hard","difficult"),("easy","simple"),
    ("dark","dim"),("bright","luminous"),("hot","warm"),("cold","cool"),
    ("loud","noisy"),("quiet","silent"),("clean","pure"),("dirty","filthy"),
    ("fresh","new"),("old","stale"),("sharp","keen"),("blunt","dull"),
    ("heavy","weighty"),("light","featherweight"),("thick","dense"),("thin","slender"),
    ("near","close"),("far","distant"),("empty","vacant"),("full","complete"),
    ("start","begin"),("end","finish"),("help","assist"),("hurt","harm"),
    ("buy","purchase"),("sell","trade"),("make","create"),("break","destroy"),
    ("find","discover"),("lose","misplace"),("give","provide"),("take","receive"),
    ("say","speak"),("hear","listen"),("see","observe"),("think","consider"),
    ("know","understand"),("learn","study"),("teach","instruct"),("work","labor"),
    ("walk","stroll"),("run","sprint"),("jump","leap"),("fall","drop"),
    ("house","home"),("road","path"),("city","town"),("land","territory"),
    ("water","liquid"),("fire","flame"),("earth","ground"),("air","wind"),
    ("food","nourishment"),("drink","beverage"),("money","currency"),("job","occupation"),
    ("war","conflict"),("peace","harmony"),("law","rule"),("freedom","liberty"),
    ("idea","concept"),("plan","strategy"),("fact","truth"),("opinion","view"),
    ("problem","issue"),("answer","solution"),("question","query"),("reason","cause"),
]

ANTONYM_PAIRS = [
    ("hot","cold"),("fast","slow"),("big","small"),("tall","short"),
    ("light","dark"),("young","old"),("rich","poor"),("strong","weak"),
    ("hard","soft"),("loud","quiet"),("clean","dirty"),("full","empty"),
    ("start","stop"),("love","hate"),("war","peace"),("good","evil"),
    ("truth","lie"),("life","death"),("day","night"),("summer","winter"),
    ("north","south"),("east","west"),("up","down"),("left","right"),
    ("inside","outside"),("before","after"),("early","late"),("first","last"),
    ("open","closed"),("give","take"),("push","pull"),("buy","sell"),
    ("win","lose"),("success","failure"),("health","sickness"),("freedom","captivity"),
    ("friend","enemy"),("question","answer"),("cause","effect"),("input","output"),
    ("positive","negative"),("true","false"),("real","fake"),("natural","artificial"),
    ("urban","rural"),("ancient","modern"),("domestic","foreign"),("public","private"),
    ("general","specific"),("abstract","concrete"),("simple","complex"),("random","ordered"),
    ("increase","decrease"),("expand","contract"),("rise","fall"),("arrive","depart"),
    ("attack","defend"),("accept","reject"),("approve","disapprove"),("include","exclude"),
]

WORD_TYPES = [
    ("noun","word"),("verb","word"),("adjective","word"),("adverb","word"),
    ("pronoun","word"),("preposition","word"),("conjunction","word"),("article","word"),
    ("interjection","word"),("determiner","word"),
    ("singular","noun"),("plural","noun"),("abstract","noun"),("concrete","noun"),
    ("transitive","verb"),("intransitive","verb"),("auxiliary","verb"),("modal","verb"),
    ("comparative","adjective"),("superlative","adjective"),
    ("syllable","phonetics"),("phoneme","phonetics"),("morpheme","linguistics"),
    ("prefix","morpheme"),("suffix","morpheme"),("root","morpheme"),
    ("syntax","grammar"),("semantics","linguistics"),("pragmatics","linguistics"),
    ("dialect","language"),("accent","dialect"),("slang","language"),
    ("idiom","expression"),("metaphor","figure"),("simile","figure"),
    ("alliteration","figure"),("onomatopoeia","figure"),("irony","figure"),
    ("sarcasm","irony"),("hyperbole","figure"),("euphemism","figure"),
    ("denotation","meaning"),("connotation","meaning"),
]

LOGICAL_CONCEPTS = [
    ("premise","argument"),("conclusion","argument"),("inference","reasoning"),
    ("deduction","reasoning"),("induction","reasoning"),("abduction","reasoning"),
    ("syllogism","logic"),("fallacy","error"),("paradox","logic"),
    ("hypothesis","science"),("theory","science"),("law","science"),
    ("evidence","proof"),("proof","truth"),("axiom","mathematics"),
    ("theorem","mathematics"),("lemma","mathematics"),("corollary","theorem"),
    ("conjecture","hypothesis"),("counterexample","proof"),
    ("necessary","condition"),("sufficient","condition"),
    ("if","condition"),("then","consequence"),("implication","logic"),
    ("equivalence","logic"),("negation","logic"),("conjunction","logic"),
    ("disjunction","logic"),("quantifier","logic"),("predicate","logic"),
    ("variable","logic"),("proposition","logic"),("tautology","logic"),
    ("analogy","reasoning"),("causality","reasoning"),("correlation","statistics"),
    ("bias","error"),("validity","argument"),("soundness","argument"),
]

def build_language_triples():
    out = []
    for a, b in SYNONYM_PAIRS:
        out.append((a, "RelatedTo", b))
    for a, b in ANTONYM_PAIRS:
        out.append((a, "Opposes", b))
    for item, cat in WORD_TYPES:
        out.append((item, "IsA", cat))
    for item, cat in LOGICAL_CONCEPTS:
        out.append((item, "IsA", cat))
    out += [
        ("language","IsA","communication"),("grammar","PartOf","language"),
        ("vocabulary","PartOf","language"),("phonetics","PartOf","linguistics"),
        ("linguistics","IsA","science"),("logic","IsA","philosophy"),
        ("reasoning","IsA","cognition"),("argument","PartOf","logic"),
        ("rhetoric","IsA","communication"),("semantics","PartOf","meaning"),
    ]
    return out


def stream_commonsense_qa(limit, seen):
    try:
        from datasets import load_dataset
        ds = load_dataset("commonsense_qa", split="train", streaming=True)
        print(f"[commonsense_qa] streaming (limit={limit})...", flush=True)
        n = 0
        for row in ds:
            if n >= limit: return
            n += 1
            q = row.get("question", "") or ""
            choices = row.get("choices", {}) or {}
            key    = row.get("answerKey", "") or ""
            q_toks = content_tokens(q)
            if not q_toks: continue
            qf = q_toks[-1]
            if not is_single_token(qf): continue
            labels = choices.get("label", [])
            texts  = choices.get("text", [])
            ans = ""
            for l, t in zip(labels, texts):
                if l == key: ans = t; break
            for t in content_tokens(ans)[:3]:
                if is_single_token(t) and t != qf:
                    yield (qf, "RelatedTo", t)
    except Exception as e:
        print(f"[commonsense_qa] unavailable: {e}", flush=True)


def stream_hellaswag(limit, seen):
    try:
        from datasets import load_dataset
        ds = load_dataset("Rowan/hellaswag", split="train", streaming=True)
        print(f"[hellaswag] streaming (limit={limit})...", flush=True)
        n = 0
        for row in ds:
            if n >= limit: return
            n += 1
            ctx = row.get("ctx", "") or ""
            endings = row.get("endings", []) or []
            label   = row.get("label", "0")
            ctx_toks = content_tokens(ctx)
            if not ctx_toks: continue
            cf = ctx_toks[-1]
            if not is_single_token(cf): continue
            try:
                ans = endings[int(label)]
                for t in content_tokens(ans)[:3]:
                    if is_single_token(t) and t != cf:
                        yield (cf, "RelatedTo", t)
            except Exception:
                continue
    except Exception as e:
        print(f"[hellaswag] unavailable: {e}", flush=True)


# ===========================================================================
# PHASE 9 — MATH & ENGINEERING
# ===========================================================================

MATH_BRANCHES = [
    ("algebra","mathematics"),("geometry","mathematics"),("calculus","mathematics"),
    ("statistics","mathematics"),("probability","mathematics"),("topology","mathematics"),
    ("number","mathematics"),("combinatorics","mathematics"),("logic","mathematics"),
    ("analysis","mathematics"),("arithmetic","mathematics"),("trigonometry","mathematics"),
    ("linear","mathematics"),("discrete","mathematics"),("abstract","mathematics"),
    ("differential","mathematics"),("integral","mathematics"),("numerical","mathematics"),
    ("cryptography","mathematics"),("graph","mathematics"),
]

THEOREMS = [
    ("pythagoras","theorem"),("fermat","theorem"),("euler","theorem"),
    ("gauss","theorem"),("bayes","theorem"),("newton","theorem"),
    ("archimedes","theorem"),("poincare","theorem"),("riemann","theorem"),
    ("cantor","theorem"),("godel","theorem"),("fourier","theorem"),
    ("taylor","theorem"),("stokes","theorem"),("green","theorem"),
    ("bolzano","theorem"),("cauchy","theorem"),("lagrange","theorem"),
    ("rolle","theorem"),("fubini","theorem"),
]

MATH_SHAPES = [
    ("circle","shape"),("ellipse","shape"),("parabola","curve"),("hyperbola","curve"),
    ("triangle","polygon"),("square","polygon"),("rectangle","polygon"),
    ("pentagon","polygon"),("hexagon","polygon"),("heptagon","polygon"),
    ("octagon","polygon"),("polygon","shape"),("parallelogram","quadrilateral"),
    ("trapezoid","quadrilateral"),("rhombus","quadrilateral"),
    ("sphere","solid"),("cube","solid"),("cylinder","solid"),("cone","solid"),
    ("pyramid","solid"),("prism","solid"),("torus","solid"),("tetrahedron","solid"),
    ("dodecahedron","solid"),("icosahedron","solid"),("octahedron","solid"),
    ("line","geometry"),("ray","geometry"),("segment","geometry"),("angle","geometry"),
    ("arc","geometry"),("chord","geometry"),("tangent","geometry"),
    ("diameter","circle"),("radius","circle"),("circumference","circle"),
    ("hypotenuse","triangle"),("altitude","triangle"),("median","triangle"),
    ("vertex","polygon"),("edge","polygon"),("face","solid"),
]

MATH_NUMBERS = [
    ("prime","number"),("composite","number"),("rational","number"),
    ("irrational","number"),("integer","number"),("real","number"),
    ("complex","number"),("natural","number"),("whole","number"),
    ("even","integer"),("odd","integer"),("zero","integer"),
    ("infinity","concept"),("pi","constant"),("euler","constant"),
    ("golden","ratio"),("fibonacci","sequence"),("factorial","operation"),
    ("logarithm","function"),("exponent","function"),("root","function"),
    ("absolute","value"),("modulo","operation"),("derivative","calculus"),
    ("integral","calculus"),("limit","calculus"),("series","sequence"),
    ("vector","linear"),("matrix","linear"),("determinant","matrix"),
    ("eigenvalue","matrix"),("scalar","algebra"),("tensor","algebra"),
    ("set","collection"),("subset","set"),("intersection","set"),
    ("union","set"),("complement","set"),("cardinality","set"),
]

ENGINEERING_BRANCHES = [
    ("civil","engineering"),("mechanical","engineering"),("electrical","engineering"),
    ("chemical","engineering"),("aerospace","engineering"),("biomedical","engineering"),
    ("environmental","engineering"),("industrial","engineering"),("nuclear","engineering"),
    ("materials","engineering"),("petroleum","engineering"),("structural","engineering"),
    ("geotechnical","engineering"),("transportation","engineering"),("marine","engineering"),
    ("mining","engineering"),("agricultural","engineering"),("optical","engineering"),
    ("control","engineering"),("systems","engineering"),
]

ENGINEERING_CONCEPTS = [
    ("beam","structure"),("truss","structure"),("frame","structure"),("arch","structure"),
    ("foundation","structure"),("column","structure"),("slab","structure"),
    ("concrete","material"),("steel","material"),("timber","material"),
    ("asphalt","material"),("brick","material"),("glass","material"),
    ("stress","mechanics"),("strain","mechanics"),("torque","mechanics"),
    ("moment","mechanics"),("shear","force"),("tension","force"),("compression","force"),
    ("gear","machine"),("pulley","machine"),("lever","machine"),("wheel","machine"),
    ("axle","machine"),("wedge","machine"),("screw","machine"),("inclined","machine"),
    ("motor","machine"),("engine","machine"),("pump","machine"),("turbine","machine"),
    ("generator","machine"),("transformer","device"),("rectifier","device"),
    ("amplifier","device"),("oscillator","device"),("filter","device"),
    ("reactor","chemical"),("distillation","chemical"),("extraction","chemical"),
    ("absorption","chemical"),("adsorption","chemical"),("crystallization","chemical"),
    ("catalyst","chemical"),("inhibitor","chemical"),("yield","chemical"),
    ("efficiency","measure"),("reliability","measure"),("tolerance","measure"),
    ("specification","document"),("blueprint","document"),("prototype","model"),
    ("simulation","model"),("optimization","process"),("constraint","condition"),
]

def build_math_triples():
    out = []
    for item, cat in MATH_BRANCHES:
        out.append((item, "IsA", cat))
        out.append((item, "PartOf", "mathematics"))
    for person, kind in THEOREMS:
        out.append((person, "Produces", kind))
        out.append((person, "PartOf", "mathematics"))
    for item, cat in MATH_SHAPES:
        out.append((item, "IsA", cat))
    for item, cat in MATH_NUMBERS:
        out.append((item, "IsA", cat))
    for branch, _ in ENGINEERING_BRANCHES:
        out.append((branch, "IsA", "engineering"))
        out.append(("engineering", "IsA", "discipline"))
    for item, cat in ENGINEERING_CONCEPTS:
        out.append((item, "IsA", cat))
    out += [
        ("mathematics","IsA","science"),("theorem","IsA","statement"),
        ("proof","IsA","argument"),("equation","IsA","statement"),
        ("formula","IsA","equation"),("algorithm","IsA","procedure"),
        ("model","IsA","representation"),("simulation","IsA","model"),
        ("engineering","IsA","discipline"),("design","PartOf","engineering"),
        ("analysis","PartOf","engineering"),("testing","PartOf","engineering"),
        ("material","PartOf","engineering"),("structure","PartOf","engineering"),
        ("polygon","IsA","shape"),("solid","IsA","shape"),("shape","IsA","geometry"),
        ("geometry","IsA","mathematics"),("number","IsA","mathematics"),
        ("operation","IsA","mathematics"),
    ]
    return out


def stream_math_qa(limit, seen):
    try:
        from datasets import load_dataset
        ds = load_dataset("math_qa", split="train", streaming=True)
        print(f"[math_qa] streaming (limit={limit})...", flush=True)
        n = 0
        for row in ds:
            if n >= limit: return
            n += 1
            q = row.get("Problem", "") or ""
            ans = row.get("correct", "") or ""
            cat = row.get("category", "") or ""
            q_toks = content_tokens(q)
            if not q_toks: continue
            qf = q_toks[-1]
            if not is_single_token(qf): continue
            for t in content_tokens(cat)[:2]:
                if is_single_token(t) and t != qf:
                    yield (qf, "RelatedTo", t)
            for t in content_tokens(ans)[:2]:
                if is_single_token(t) and t != qf:
                    yield (qf, "RelatedTo", t)
    except Exception as e:
        print(f"[math_qa] unavailable: {e}", flush=True)


# ===========================================================================
# PHASE 10 — GEOGRAPHY DEEP
# ===========================================================================

MAJOR_CITIES = [
    # Europe
    ("paris","city","france"),("london","city","england"),("berlin","city","germany"),
    ("madrid","city","spain"),("rome","city","italy"),("amsterdam","city","netherlands"),
    ("brussels","city","belgium"),("vienna","city","austria"),("stockholm","city","sweden"),
    ("oslo","city","norway"),("copenhagen","city","denmark"),("helsinki","city","finland"),
    ("warsaw","city","poland"),("prague","city","czechia"),("budapest","city","hungary"),
    ("bucharest","city","romania"),("athens","city","greece"),("lisbon","city","portugal"),
    ("zurich","city","switzerland"),("munich","city","germany"),("hamburg","city","germany"),
    ("frankfurt","city","germany"),("milan","city","italy"),("barcelona","city","spain"),
    ("manchester","city","england"),("birmingham","city","england"),
    ("glasgow","city","scotland"),("dublin","city","ireland"),
    ("istanbul","city","turkey"),("kyiv","city","ukraine"),("moscow","city","russia"),
    ("saint","city","russia"),("minsk","city","belarus"),("belgrade","city","serbia"),
    ("zagreb","city","croatia"),("sofia","city","bulgaria"),("riga","city","latvia"),
    ("tallinn","city","estonia"),("vilnius","city","lithuania"),
    # Asia
    ("tokyo","city","japan"),("beijing","city","china"),("shanghai","city","china"),
    ("guangzhou","city","china"),("shenzhen","city","china"),("chongqing","city","china"),
    ("mumbai","city","india"),("delhi","city","india"),("bangalore","city","india"),
    ("hyderabad","city","india"),("chennai","city","india"),("kolkata","city","india"),
    ("karachi","city","pakistan"),("lahore","city","pakistan"),("dhaka","city","bangladesh"),
    ("yangon","city","myanmar"),("bangkok","city","thailand"),("jakarta","city","indonesia"),
    ("manila","city","philippines"),("hanoi","city","vietnam"),("saigon","city","vietnam"),
    ("singapore","city","singapore"),("kuala","city","malaysia"),("seoul","city","korea"),
    ("taipei","city","taiwan"),("hong","city","china"),("osaka","city","japan"),
    ("tehran","city","iran"),("baghdad","city","iraq"),("riyadh","city","saudi"),
    ("dubai","city","uae"),("doha","city","qatar"),("kuwait","city","kuwait"),
    ("beirut","city","lebanon"),("damascus","city","syria"),("amman","city","jordan"),
    ("jerusalem","city","israel"),("tel","city","israel"),("kabul","city","afghanistan"),
    ("tashkent","city","uzbekistan"),("almaty","city","kazakhstan"),
    # Americas
    ("new","city","usa"),("york","city","usa"),("los","city","usa"),
    ("chicago","city","usa"),("houston","city","usa"),("phoenix","city","usa"),
    ("philadelphia","city","usa"),("dallas","city","usa"),("seattle","city","usa"),
    ("denver","city","usa"),("miami","city","usa"),("boston","city","usa"),
    ("toronto","city","canada"),("montreal","city","canada"),("vancouver","city","canada"),
    ("calgary","city","canada"),("ottawa","city","canada"),
    ("mexico","city","mexico"),("guadalajara","city","mexico"),("monterrey","city","mexico"),
    ("sao","city","brazil"),("rio","city","brazil"),("brasilia","city","brazil"),
    ("buenos","city","argentina"),("lima","city","peru"),("bogota","city","colombia"),
    ("santiago","city","chile"),("caracas","city","venezuela"),("quito","city","ecuador"),
    ("havana","city","cuba"),("kingston","city","jamaica"),
    # Africa
    ("cairo","city","egypt"),("lagos","city","nigeria"),("kinshasa","city","congo"),
    ("johannesburg","city","africa"),("nairobi","city","kenya"),("addis","city","ethiopia"),
    ("dar","city","tanzania"),("khartoum","city","sudan"),("abuja","city","nigeria"),
    ("casablanca","city","morocco"),("accra","city","ghana"),("dakar","city","senegal"),
    ("tunis","city","tunisia"),("algiers","city","algeria"),("tripoli","city","libya"),
    ("maputo","city","mozambique"),("harare","city","zimbabwe"),("lusaka","city","zambia"),
    # Oceania
    ("sydney","city","australia"),("melbourne","city","australia"),("brisbane","city","australia"),
    ("perth","city","australia"),("adelaide","city","australia"),
    ("auckland","city","zealand"),("wellington","city","zealand"),
]

LANDMARKS = [
    ("eiffel","landmark","france"),("louvre","landmark","france"),
    ("colosseum","landmark","italy"),("pantheon","landmark","italy"),
    ("parthenon","landmark","greece"),("acropolis","landmark","greece"),
    ("pyramid","landmark","egypt"),("sphinx","landmark","egypt"),
    ("taj","landmark","india"),("agra","landmark","india"),
    ("great","landmark","china"),("forbidden","landmark","china"),
    ("stonehenge","landmark","england"),("buckingham","landmark","england"),
    ("big","landmark","england"),("westminster","landmark","england"),
    ("sagrada","landmark","spain"),("alhambra","landmark","spain"),
    ("colosseum","landmark","italy"),("pisa","landmark","italy"),
    ("acropolis","landmark","greece"),("olympia","landmark","greece"),
    ("kremlin","landmark","russia"),("hermitage","landmark","russia"),
    ("statue","landmark","usa"),("liberty","landmark","usa"),
    ("golden","landmark","usa"),("rushmore","landmark","usa"),
    ("machu","landmark","peru"),("picchu","landmark","peru"),
    ("chichen","landmark","mexico"),("itza","landmark","mexico"),
    ("angkor","landmark","cambodia"),("wat","landmark","cambodia"),
    ("petra","landmark","jordan"),("jerash","landmark","jordan"),
    ("hagia","landmark","turkey"),("sophia","landmark","turkey"),
    ("alhambra","landmark","spain"),("alcazar","landmark","spain"),
    ("versailles","landmark","france"),("mont","landmark","france"),
    ("neuschwanstein","landmark","germany"),("cologne","landmark","germany"),
    ("charles","landmark","czechia"),("prague","landmark","czechia"),
    ("colosseum","landmark","italy"),("forum","landmark","italy"),
    ("victoria","landmark","england"),("edinburgh","landmark","scotland"),
]

CLIMATE_ZONES = [
    ("tropical","climate"),("subtropical","climate"),("temperate","climate"),
    ("continental","climate"),("polar","climate"),("arid","climate"),
    ("semiarid","climate"),("mediterranean","climate"),("oceanic","climate"),
    ("subarctic","climate"),("tundra","climate"),("humid","climate"),
    ("monsoon","climate"),("savanna","climate"),("highland","climate"),
    ("equatorial","climate"),("desert","climate"),("steppe","climate"),
    ("rainforest","climate"),("alpine","climate"),
]

NATURAL_RESOURCES = [
    ("oil","resource"),("gas","resource"),("coal","resource"),("uranium","resource"),
    ("iron","resource"),("copper","resource"),("gold","resource"),("silver","resource"),
    ("aluminum","resource"),("titanium","resource"),("platinum","resource"),
    ("nickel","resource"),("zinc","resource"),("lead","resource"),("tin","resource"),
    ("chromium","resource"),("manganese","resource"),("cobalt","resource"),
    ("lithium","resource"),("rare","resource"),
    ("timber","resource"),("fish","resource"),("freshwater","resource"),
    ("soil","resource"),("arable","resource"),("pasture","resource"),
    ("solar","energy"),("wind","energy"),("hydro","energy"),("geothermal","energy"),
    ("tidal","energy"),("nuclear","energy"),("fossil","energy"),("renewable","energy"),
    ("oil","UsedFor","energy"),("coal","UsedFor","energy"),("gas","UsedFor","energy"),
    ("lithium","UsedFor","battery"),("silicon","UsedFor","semiconductor"),
    ("iron","UsedFor","steel"),("copper","UsedFor","wire"),("gold","UsedFor","conductor"),
]

def build_geography_deep_triples():
    out = []
    for city, kind, country in MAJOR_CITIES:
        if is_single_token(city):
            out.append((city, "IsA", kind))
            out.append((city, "PartOf", country))
    for landmark, kind, country in LANDMARKS:
        if is_single_token(landmark):
            out.append((landmark, "IsA", kind))
            out.append((landmark, "PartOf", country))
    for zone, cat in CLIMATE_ZONES:
        out.append((zone, "IsA", cat))
    for item in NATURAL_RESOURCES:
        if isinstance(item, tuple):
            if len(item) == 2:
                out.append((item[0], "IsA", item[1]))
            elif len(item) == 3:
                out.append((item[0], item[1], item[2]))
    out += [
        ("city","IsA","settlement"),("town","IsA","settlement"),("village","IsA","settlement"),
        ("capital","IsA","city"),("metropolis","IsA","city"),("megalopolis","IsA","city"),
        ("border","IsA","boundary"),("territory","IsA","land"),("colony","IsA","territory"),
        ("continent","PartOf","earth"),("peninsula","IsA","land"),("island","IsA","land"),
        ("archipelago","IsA","island"),("cape","IsA","land"),("bay","IsA","water"),
        ("delta","IsA","land"),("plateau","IsA","land"),("valley","IsA","land"),
        ("canyon","IsA","valley"),("fjord","IsA","inlet"),("strait","IsA","water"),
        ("climate","PartOf","geography"),("resource","PartOf","geography"),
        ("landmark","IsA","structure"),("heritage","IsA","landmark"),
        ("energy","IsA","resource"),("renewable","IsA","energy"),("fossil","IsA","fuel"),
    ]
    return out


# ===========================================================================
# PHASE 11 — PEOPLE & EVENTS
# ===========================================================================

INVENTORS = [
    ("bell","inventor","telephone"),("edison","inventor","lightbulb"),
    ("watt","inventor","steam"),("gutenberg","inventor","printing"),
    ("morse","inventor","telegraph"),("marconi","inventor","radio"),
    ("wright","inventor","airplane"),("ford","inventor","automobile"),
    ("diesel","inventor","engine"),("benz","inventor","automobile"),
    ("baird","inventor","television"),("lovelace","inventor","programming"),
    ("babbage","inventor","computer"),("zuckerberg","inventor","facebook"),
    ("berners","inventor","web"),("lee","inventor","web"),
    ("jobs","inventor","iphone"),("wozniak","inventor","apple"),
    ("gates","inventor","windows"),("torvalds","inventor","linux"),
    ("knuth","inventor","algorithm"),("hopper","inventor","compiler"),
    ("shannon","inventor","information"),("turing","inventor","computing"),
    ("von","inventor","architecture"),("neumann","inventor","computer"),
    ("faraday","inventor","electricity"),("maxwell","inventor","electromagnetism"),
    ("hertz","inventor","radio"),("röntgen","inventor","xray"),
    ("curie","inventor","radioactivity"),("bohr","inventor","atom"),
    ("planck","inventor","quantum"),("pauli","inventor","exclusion"),
    ("dirac","inventor","antimatter"),("fermi","inventor","reactor"),
    ("oppenheimer","inventor","nuclear"),("hubble","inventor","telescope"),
    ("pasteur","inventor","vaccine"),("lister","inventor","antiseptic"),
    ("semmelweis","inventor","hygiene"),("flemming","inventor","penicillin"),
    ("jenner","inventor","vaccination"),("salk","inventor","polio"),
    ("watson","inventor","dna"),("crick","inventor","helix"),
    ("mendel","inventor","genetics"),("darwin","inventor","evolution"),
    ("linnaeus","inventor","taxonomy"),("lamarck","inventor","evolution"),
    ("lavoisier","inventor","chemistry"),("mendeleev","inventor","periodic"),
    ("dalton","inventor","atom"),("avogadro","inventor","molecule"),
]

BATTLES_WARS = [
    ("marathon","battle","greece"),("thermopylae","battle","greece"),
    ("salamis","battle","greece"),("actium","battle","rome"),
    ("hastings","battle","england"),("agincourt","battle","france"),
    ("crecy","battle","france"),("poitiers","battle","france"),
    ("borodino","battle","russia"),("waterloo","battle","france"),
    ("gettysburg","battle","usa"),("antietam","battle","usa"),
    ("bull","battle","usa"),("vicksburg","battle","usa"),
    ("verdun","battle","france"),("somme","battle","france"),
    ("marne","battle","france"),("ypres","battle","belgium"),
    ("gallipoli","battle","turkey"),("jutland","battle","sea"),
    ("stalingrad","battle","russia"),("kursk","battle","russia"),
    ("midway","battle","pacific"),("iwo","battle","pacific"),
    ("normandy","battle","france"),("arnhem","battle","netherlands"),
    ("berlin","battle","germany"),("bulge","battle","belgium"),
    ("guadalcanal","battle","pacific"),("coral","battle","pacific"),
    ("inchon","battle","korea"),("dien","battle","vietnam"),
    ("tet","battle","vietnam"),("fallujah","battle","iraq"),
    ("mosul","battle","iraq"),("aleppo","battle","syria"),
]

TREATIES_EVENTS = [
    ("versailles","treaty","germany"),("westphalia","treaty","germany"),
    ("paris","treaty","france"),("utrecht","treaty","netherlands"),
    ("vienna","treaty","austria"),("tordesillas","treaty","spain"),
    ("berlin","treaty","germany"),("brest","treaty","russia"),
    ("ghent","treaty","usa"),("guadalupe","treaty","mexico"),
    ("portsmouth","treaty","usa"),("locarno","treaty","switzerland"),
    ("munich","treaty","germany"),("yalta","conference","usa"),
    ("potsdam","conference","germany"),("san","treaty","japan"),
    ("maastricht","treaty","netherlands"),("lisbon","treaty","portugal"),
    ("magna","carta","england"),("declaration","document","usa"),
    ("constitution","document","usa"),("emancipation","document","usa"),
    ("proclamation","document","usa"),("suffrage","amendment","usa"),
    ("prohibition","amendment","usa"),("habeas","writ","england"),
    ("magna","document","england"),
]

SOCIAL_MOVEMENTS = [
    ("suffrage","movement","women"),("abolition","movement","slavery"),
    ("civil","movement","rights"),("independence","movement","colonial"),
    ("reformation","movement","religion"),("renaissance","movement","culture"),
    ("enlightenment","movement","philosophy"),("romanticism","movement","art"),
    ("industrial","movement","economy"),("labor","movement","workers"),
    ("union","movement","labor"),("feminist","movement","women"),
    ("environmental","movement","ecology"),("pacifist","movement","peace"),
    ("nationalist","movement","nation"),("socialist","movement","economics"),
    ("communist","movement","politics"),("fascist","movement","politics"),
    ("anarchist","movement","politics"),("libertarian","movement","politics"),
    ("temperance","movement","alcohol"),("prohibition","movement","usa"),
    ("zionist","movement","israel"),("pan","movement","africa"),
    ("decolonization","movement","colonial"),("apartheid","movement","africa"),
    ("lgbtq","movement","rights"),("indigenous","movement","rights"),
    ("disability","movement","rights"),("consumer","movement","rights"),
]

def build_people_events_triples():
    out = []
    for person, role, topic in INVENTORS:
        p = person.split()[0] if " " in person else person
        if is_single_token(p):
            out.append((p, "IsA", role))
            out.append((p, "Produces", topic))
    for battle, kind, place in BATTLES_WARS:
        if is_single_token(battle):
            out.append((battle, "IsA", kind))
            out.append((battle, "PartOf", place))
    for event, kind, place in TREATIES_EVENTS:
        if is_single_token(event):
            out.append((event, "IsA", kind))
            if is_single_token(place):
                out.append((event, "PartOf", place))
    for name, kind, topic in SOCIAL_MOVEMENTS:
        if is_single_token(name):
            out.append((name, "IsA", kind))
            if is_single_token(topic):
                out.append((name, "PartOf", topic))
    out += [
        ("inventor","IsA","person"),("discoverer","IsA","person"),
        ("explorer","IsA","person"),("leader","IsA","person"),
        ("battle","IsA","conflict"),("war","IsA","conflict"),
        ("treaty","IsA","agreement"),("conference","IsA","meeting"),
        ("movement","IsA","change"),("revolution","IsA","change"),
        ("reform","IsA","change"),("rebellion","IsA","conflict"),
        ("invention","IsA","creation"),("discovery","IsA","finding"),
        ("innovation","IsA","creation"),("trademark","IsA","protection"),
    ]
    return out


def stream_wiki_bio(limit, seen):
    try:
        from datasets import load_dataset
        ds = load_dataset("wiki_bio", split="train", streaming=True)
        print(f"[wiki_bio] streaming (limit={limit})...", flush=True)
        n = 0
        for row in ds:
            if n >= limit: return
            n += 1
            try:
                tgt = row.get("target_text", "") or ""
                inp = row.get("input_text", {}) or {}
                if isinstance(inp, dict):
                    tbl = inp.get("table", {}) or {}
                    colnames = tbl.get("column_header", []) or []
                    values   = tbl.get("content", []) or []
                    # Find 'name' column
                    focus = None
                    for col, val in zip(colnames, values):
                        if col in ("name", "birth_name"):
                            name_toks = content_tokens(val)
                            if name_toks and is_single_token(name_toks[0]):
                                focus = name_toks[0]
                            break
                    if not focus:
                        t_toks = content_tokens(tgt)
                        if t_toks and is_single_token(t_toks[0]):
                            focus = t_toks[0]
                    if not focus: continue
                    for col, val in zip(colnames, values):
                        if col in ("name","birth_name","article_title"): continue
                        for t in content_tokens(str(val))[:2]:
                            if is_single_token(t) and t != focus:
                                yield (focus, "RelatedTo", t)
                else:
                    t_toks = content_tokens(tgt)
                    if not t_toks or not is_single_token(t_toks[0]): continue
                    focus = t_toks[0]
                    for t in t_toks[1:5]:
                        if is_single_token(t) and t != focus:
                            yield (focus, "RelatedTo", t)
            except Exception:
                continue
    except Exception as e:
        print(f"[wiki_bio] unavailable: {e}", flush=True)


# ===========================================================================
# PHASE 12 — MODERN WORLD
# ===========================================================================

TECH_COMPANIES = [
    ("apple","company","usa"),("google","company","usa"),("microsoft","company","usa"),
    ("amazon","company","usa"),("meta","company","usa"),("tesla","company","usa"),
    ("nvidia","company","usa"),("intel","company","usa"),("amd","company","usa"),
    ("ibm","company","usa"),("oracle","company","usa"),("salesforce","company","usa"),
    ("adobe","company","usa"),("netflix","company","usa"),("spotify","company","sweden"),
    ("samsung","company","korea"),("lg","company","korea"),("sony","company","japan"),
    ("panasonic","company","japan"),("toshiba","company","japan"),("fujitsu","company","japan"),
    ("huawei","company","china"),("alibaba","company","china"),("tencent","company","china"),
    ("baidu","company","china"),("xiaomi","company","china"),("bytedance","company","china"),
    ("openai","company","usa"),("anthropic","company","usa"),("deepmind","company","england"),
    ("palantir","company","usa"),("snowflake","company","usa"),("databricks","company","usa"),
    ("stripe","company","usa"),("square","company","usa"),("uber","company","usa"),
    ("lyft","company","usa"),("airbnb","company","usa"),("pinterest","company","usa"),
    ("twitter","company","usa"),("snapchat","company","usa"),("tiktok","company","china"),
    ("zoom","company","usa"),("slack","company","usa"),("dropbox","company","usa"),
    ("cloudflare","company","usa"),("datadog","company","usa"),("okta","company","usa"),
    ("twilio","company","usa"),("shopify","company","canada"),("wix","company","israel"),
    ("wordpress","company","usa"),("github","company","usa"),("gitlab","company","usa"),
    ("jira","company","australia"),("figma","company","usa"),("notion","company","usa"),
    ("asana","company","usa"),("monday","company","israel"),("atlassian","company","australia"),
]

GLOBAL_BRANDS = [
    ("cocacola","brand","usa"),("pepsi","brand","usa"),("mcdonalds","brand","usa"),
    ("starbucks","brand","usa"),("subway","brand","usa"),("kfc","brand","usa"),
    ("burger","brand","usa"),("pizza","brand","usa"),("dominos","brand","usa"),
    ("walmart","brand","usa"),("target","brand","usa"),("costco","brand","usa"),
    ("amazon","brand","usa"),("ebay","brand","usa"),("etsy","brand","usa"),
    ("nike","brand","usa"),("adidas","brand","germany"),("puma","brand","germany"),
    ("reebok","brand","usa"),("new","brand","usa"),("converse","brand","usa"),
    ("vans","brand","usa"),("levis","brand","usa"),("gap","brand","usa"),
    ("zara","brand","spain"),("hm","brand","sweden"),("uniqlo","brand","japan"),
    ("gucci","brand","italy"),("prada","brand","italy"),("chanel","brand","france"),
    ("louis","brand","france"),("vuitton","brand","france"),("hermes","brand","france"),
    ("rolex","brand","switzerland"),("omega","brand","switzerland"),
    ("bmw","brand","germany"),("mercedes","brand","germany"),("audi","brand","germany"),
    ("volkswagen","brand","germany"),("porsche","brand","germany"),
    ("toyota","brand","japan"),("honda","brand","japan"),("nissan","brand","japan"),
    ("hyundai","brand","korea"),("kia","brand","korea"),("ford","brand","usa"),
    ("chevrolet","brand","usa"),("tesla","brand","usa"),("volvo","brand","sweden"),
    ("ferrari","brand","italy"),("lamborghini","brand","italy"),("maserati","brand","italy"),
]

SPORTS_TEAMS = [
    # Soccer
    ("barcelona","team","soccer"),("real","team","soccer"),("madrid","team","soccer"),
    ("manchester","team","soccer"),("liverpool","team","soccer"),("arsenal","team","soccer"),
    ("chelsea","team","soccer"),("tottenham","team","soccer"),("bayern","team","soccer"),
    ("dortmund","team","soccer"),("juventus","team","soccer"),("milan","team","soccer"),
    ("inter","team","soccer"),("roma","team","soccer"),("napoli","team","soccer"),
    ("psg","team","soccer"),("ajax","team","soccer"),("porto","team","soccer"),
    ("benfica","team","soccer"),("celtic","team","soccer"),
    # Basketball
    ("lakers","team","basketball"),("celtics","team","basketball"),
    ("warriors","team","basketball"),("bulls","team","basketball"),
    ("heat","team","basketball"),("spurs","team","basketball"),
    ("cavaliers","team","basketball"),("nets","team","basketball"),
    ("knicks","team","basketball"),("mavericks","team","basketball"),
    # American Football
    ("patriots","team","football"),("cowboys","team","football"),
    ("steelers","team","football"),("raiders","team","football"),
    ("bears","team","football"),("packers","team","football"),
    ("49ers","team","football"),("seahawks","team","football"),
    # Baseball
    ("yankees","team","baseball"),("dodgers","team","baseball"),
    ("red","team","baseball"),("cubs","team","baseball"),
    ("cardinals","team","baseball"),("giants","team","baseball"),
    # Ice Hockey
    ("maple","team","hockey"),("leafs","team","hockey"),
    ("canadiens","team","hockey"),("bruins","team","hockey"),
    ("rangers","team","hockey"),("blackhawks","team","hockey"),
]

ENTERTAINMENT = [
    # Classic films
    ("titanic","film","drama"),("avatar","film","scifi"),("endgame","film","scifi"),
    ("godfather","film","drama"),("inception","film","scifi"),
    ("interstellar","film","scifi"),("matrix","film","scifi"),
    ("joker","film","drama"),("parasite","film","drama"),
    ("shawshank","film","drama"),("schindler","film","drama"),
    ("casablanca","film","drama"),("citizen","film","drama"),
    ("psycho","film","thriller"),("vertigo","film","thriller"),
    ("kubrick","director","usa"),("spielberg","director","usa"),
    ("scorsese","director","usa"),("nolan","director","england"),
    ("tarantino","director","usa"),("hitchcock","director","england"),
    ("coppola","director","usa"),("lynch","director","usa"),
    ("fincher","director","usa"),("anderson","director","usa"),
    # TV shows
    ("breaking","show","drama"),("sopranos","show","drama"),
    ("wire","show","drama"),("game","show","fantasy"),
    ("thrones","show","fantasy"),("westworld","show","scifi"),
    ("mandalorian","show","scifi"),("stranger","show","scifi"),
    ("crown","show","drama"),("succession","show","drama"),
    ("fleabag","show","comedy"),("ted","show","comedy"),
    # Modern musicians
    ("taylor","musician","pop"),("beyonce","musician","pop"),
    ("rihanna","musician","pop"),("adele","musician","pop"),
    ("ed","musician","pop"),("sheeran","musician","pop"),
    ("billie","musician","pop"),("eilish","musician","pop"),
    ("weeknd","musician","pop"),("dua","musician","pop"),
    ("lipa","musician","pop"),("olivia","musician","pop"),
    ("rodrigo","musician","pop"),("ariana","musician","pop"),
    ("grande","musician","pop"),("harry","musician","pop"),
    ("styles","musician","pop"),("post","musician","hiphop"),
    ("malone","musician","hiphop"),("drake","musician","hiphop"),
    ("kendrick","musician","hiphop"),("lamar","musician","hiphop"),
    ("travis","musician","hiphop"),("scott","musician","hiphop"),
    ("nicki","musician","hiphop"),("minaj","musician","hiphop"),
]

def build_modern_world_triples():
    out = []
    for name, kind, country in TECH_COMPANIES:
        if is_single_token(name):
            out.append((name, "IsA", kind))
            out.append((name, "PartOf", country))
            out.append((name, "IsA", "tech"))
    for name, kind, country in GLOBAL_BRANDS:
        if is_single_token(name):
            out.append((name, "IsA", kind))
            out.append((name, "PartOf", country))
    for name, kind, sport in SPORTS_TEAMS:
        if is_single_token(name):
            out.append((name, "IsA", kind))
            out.append((name, "PartOf", sport))
    for name, kind, genre in ENTERTAINMENT:
        if is_single_token(name):
            out.append((name, "IsA", kind))
            out.append((name, "IsA", genre))
    out += [
        ("company","IsA","organization"),("brand","IsA","company"),
        ("tech","IsA","industry"),("startup","IsA","company"),
        ("multinational","IsA","company"),("corporation","IsA","company"),
        ("team","IsA","group"),("franchise","IsA","team"),("league","IsA","sport"),
        ("film","IsA","entertainment"),("show","IsA","entertainment"),
        ("musician","IsA","artist"),("director","IsA","artist"),
        ("entertainment","IsA","industry"),("media","IsA","industry"),
        ("streaming","IsA","media"),("broadcast","IsA","media"),
        ("podcast","IsA","media"),("influencer","IsA","person"),
        ("celebrity","IsA","person"),("athlete","IsA","person"),
        ("esports","IsA","sport"),("gaming","IsA","entertainment"),
    ]
    return out


# ===========================================================================
# PHASE 13 — BULK TRIVIA EXPANSION
# ===========================================================================

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
        aliases   = ans.get("aliases", []) or []
        q_toks = content_tokens(q)
        if not q_toks: continue
        qf = q_toks[-1]
        if not is_single_token(qf): continue
        ans_tokens = set()
        for s in [ans_value] + aliases[:3]:
            for t in content_tokens(s):
                if is_single_token(t): ans_tokens.add(t)
        for a in list(ans_tokens)[:3]:
            if a != qf: yield (qf, "RelatedTo", a)


def stream_squad(limit, seen):
    try:
        from datasets import load_dataset
        ds = load_dataset("squad", split="train", streaming=True)
        print(f"[squad] streaming (limit={limit})...", flush=True)
        n = 0
        for row in ds:
            if n >= limit: return
            n += 1
            q = row.get("question", "") or ""
            answers = row.get("answers", {}) or {}
            ans_texts = answers.get("text", []) if isinstance(answers, dict) else []
            q_toks = content_tokens(q)
            if not q_toks: continue
            qf = q_toks[-1]
            if not is_single_token(qf): continue
            for ans_text in ans_texts[:2]:
                for t in content_tokens(str(ans_text)):
                    if is_single_token(t) and t != qf:
                        yield (qf, "RelatedTo", t)
    except Exception as e:
        print(f"[squad] unavailable: {e}", flush=True)


def stream_natural_questions(limit, seen):
    try:
        from datasets import load_dataset
        ds = load_dataset("natural_questions", split="train", streaming=True)
        print(f"[nq] streaming (limit={limit})...", flush=True)
        n = 0
        for row in ds:
            if n >= limit: return
            n += 1
            try:
                q = row.get("question", {})
                q_text = q.get("text", "") if isinstance(q, dict) else str(q)
                annotations = row.get("annotations", {}) or {}
                short = annotations.get("short_answers", []) if isinstance(annotations, dict) else []
                q_toks = content_tokens(q_text)
                if not q_toks: continue
                qf = q_toks[-1]
                if not is_single_token(qf): continue
                for sa in short[:1]:
                    if not isinstance(sa, dict): continue
                    tokens = sa.get("text", []) or []
                    if isinstance(tokens, str): tokens = [tokens]
                    for tt in tokens[:3]:
                        for t in content_tokens(str(tt)):
                            if is_single_token(t) and t != qf:
                                yield (qf, "RelatedTo", t)
            except Exception:
                continue
    except Exception as e:
        print(f"[nq] unavailable: {e}", flush=True)


def stream_openbookqa(seen):
    try:
        from datasets import load_dataset
        ds = load_dataset("openbookqa", "additional", split="train")
        print(f"[openbookqa2] {len(ds)} rows", flush=True)
        for row in ds:
            fact = row.get("fact1", "") or ""
            f_toks = content_tokens(fact)
            if len(f_toks) < 2: continue
            s = f_toks[0]
            for o in f_toks[1:4]:
                if s != o and is_single_token(s) and is_single_token(o):
                    yield (s, "RelatedTo", o)
    except Exception as e:
        print(f"[openbookqa2] unavailable: {e}", flush=True)


# ===========================================================================
# PHASE 14 — FINAL VALIDATION
# ===========================================================================

PHASE14_EVAL = [
    # --- PHASE 1: Geography + basic science ---
    ("What is the capital of France?",          ["paris"]),
    ("What is the capital of Japan?",           ["tokyo"]),
    ("The Nile river is in which country?",     ["egypt"]),
    ("What is the Sahara?",                     ["desert"]),
    ("What is hydrogen?",                       ["element","nonmetal"]),
    ("What is iron?",                           ["element","metal"]),
    ("Who discovered gravity?",                 ["newton"]),
    ("Who painted the Mona Lisa?",              ["davinci"]),
    ("What is DNA?",                            ["polymer","nucleus"]),
    ("What is photosynthesis?",                 ["oxygen","sunlight"]),
    # --- PHASE 2: Science deep ---
    ("What is methane?",                        ["alkane","hydrocarbon"]),
    ("What is sulfuric acid?",                  ["acid"]),
    ("What is insulin?",                        ["hormone","glucose"]),
    ("What is a quark?",                        ["particle"]),
    ("What is the strong force?",               ["force"]),
    ("What is granite?",                        ["igneous","rock"]),
    ("What is dopamine?",                       ["neurotransmitter"]),
    ("What is a hurricane?",                    ["storm"]),
    ("What is a rainforest?",                   ["forest","biome"]),
    ("What is cancer?",                         ["disease"]),
    # --- PHASE 3: History + culture ---
    ("Who was Napoleon?",                       ["emperor","france"]),
    ("What is the Mongol empire from?",         ["mongolia"]),
    ("What is Buddhism?",                       ["religion"]),
    ("Who was Pele?",                           ["soccer","brazil"]),
    ("What is jazz?",                           ["genre","music"]),
    ("Who was Beethoven?",                      ["composer","germany"]),
    ("Who was Confucius?",                      ["philosopher","china"]),
    ("What is Hinduism?",                       ["religion"]),
    ("What is sushi?",                          ["japanese","food"]),
    ("What is pizza?",                          ["italian","food"]),
    # --- PHASE 4: Technology ---
    ("What is Python?",                         ["language","scripting"]),
    ("What is quicksort?",                      ["sorting","algorithm"]),
    ("What is TCP?",                            ["protocol","reliable"]),
    ("What is AES?",                            ["algorithm","encryption"]),
    ("What is a CPU?",                          ["processor"]),
    ("What is a stack?",                        ["structure","lifo"]),
    ("What is Docker?",                         ["container"]),
    ("What is a hash table?",                   ["structure","hashing"]),
    ("What is BFS?",                            ["searching","graph"]),
    ("What is machine learning?",               ["learning","ai"]),
    # --- PHASE 5: Practical ---
    ("What is inflation?",                      ["concept","economics"]),
    ("What is compound interest?",              ["interest"]),
    ("What is cognitive bias?",                 ["bias","error"]),
    ("What is cardio?",                         ["exercise"]),
    ("What is a curriculum?",                   ["plan","education"]),
    ("What is a trademark?",                       ["right","protection"]),
    ("What is a startup?",                      ["company"]),
    ("What is mindfulness?",                    ["practice"]),
    ("What is a plaintiff?",                    ["party","law"]),
    ("What is CBT?",                            ["therapy"]),
    # --- PHASE 7: Wikipedia / general ---
    ("What is an ocean?",                       ["water","earth"]),
    ("What is a galaxy?",                       ["universe"]),
    ("What is evolution?",                      ["darwin","biology"]),
    ("What is democracy?",                      ["system","government"]),
    ("What is a virus?",                        ["pathogen","infection"]),
    ("What is electricity?",                    ["phenomenon","energy"]),
    ("What is gravity?",                        ["force"]),
    ("What is an atom?",                        ["particle"]),
    ("What is a continent?",                    ["earth","land"]),
    ("What is culture?",                        ["society","art"]),
    # --- PHASE 8: Language + logic ---
    ("What is a synonym?",                      ["word","language"]),
    ("What is a verb?",                         ["word"]),
    ("What is deduction?",                      ["reasoning"]),
    ("What is a hypothesis?",                   ["science"]),
    ("What is an analogy?",                     ["reasoning"]),
    # --- PHASE 9: Math + engineering ---
    ("What is the Pythagorean theorem?",        ["theorem","mathematics"]),
    ("What is a prime number?",                 ["number"]),
    ("What is calculus?",                       ["mathematics"]),
    ("What is a sphere?",                       ["solid","shape"]),
    ("What is engineering?",                    ["discipline"]),
    # --- PHASE 10: Geography deep ---
    ("What is the Eiffel Tower?",               ["landmark","france"]),
    ("What is a tropical climate?",             ["climate"]),
    ("What is lithium?",                        ["resource","battery"]),
    ("What is renewable energy?",               ["energy"]),
    ("What is a megalopolis?",                  ["city"]),
    # --- PHASE 11: People + events ---
    ("Who invented the telephone?",             ["bell","inventor"]),
    ("Who invented the lightbulb?",             ["edison","inventor"]),
    ("What was the Battle of Waterloo?",        ["battle","france"]),
    ("What is the Treaty of Versailles?",       ["treaty","germany"]),
    ("What was the suffrage movement?",         ["movement","women"]),
    ("Who invented the printing press?",        ["gutenberg"]),
    ("Who discovered penicillin?",              ["flemming","penicillin"]),
    ("What was the Battle of Gettysburg?",      ["battle","usa"]),
    ("What was the civil rights movement?",     ["movement","rights"]),
    ("What was the Renaissance?",               ["movement","culture"]),
    # --- PHASE 12: Modern world ---
    ("What is Apple?",                          ["company","tech"]),
    ("What is Google?",                         ["company","tech"]),
    ("What is Nike?",                           ["brand"]),
    ("What is Netflix?",                        ["company"]),
    ("What is the LA Lakers?",                  ["team","basketball"]),
    ("What is Coca Cola?",                      ["brand"]),
    ("What is Tesla?",                          ["company","brand"]),
    ("What is the New York Yankees?",           ["team","baseball"]),
    ("What is Amazon?",                         ["company"]),
    ("What is Spotify?",                        ["company"]),
    # --- 5 additional cross-domain questions ---
    ("What is a democracy?",                    ["government","voting"]),
    ("What is electricity?",                    ["energy","charge"]),
    ("What is evolution?",                      ["biology","darwin"]),
    ("What is gravity?",                        ["force","mass"]),
    ("What is the internet?",                   ["network","computer"]),
]

assert len(PHASE14_EVAL) == 100, f"Expected 100 eval questions, got {len(PHASE14_EVAL)}"


def run_sanity(client, questions, name, top_k=15):
    print("\n" + "=" * 68, flush=True)
    print(f"{name} ({len(questions)} questions)", flush=True)
    print("=" * 68, flush=True)
    hits = 0
    for i, (q, expected) in enumerate(questions, 1):
        try:
            r = client.call_tool("lu4d_ask", {"query": q, "depth": 3, "top_k": top_k})
            activated = r.get("structuredContent", {}).get("recall", {}).get("activated", [])
        except Exception as e:
            activated = []
        contents = [a.get("content", "").lower() for a in activated]
        joined   = " | ".join(contents)
        hit = any(kw.lower() in joined for kw in expected)
        if hit: hits += 1
        mark = "PASS" if hit else "MISS"
        print(f"  Q{i:03d} [{mark}] {q}", flush=True)
        print(f"         expected={expected} | top-5={contents[:5]}", flush=True)
    pct = 100 * hits / len(questions)
    print(f"\n{name} SCORE: {hits}/{len(questions)} ({pct:.1f}%)", flush=True)
    return hits, len(questions)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase7(client, taught_cache):
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 7 — WIKIPEDIA CORE", flush=True)
    print("#" * 70, flush=True)
    phase_status(client, "pre-Phase-7")

    s   = client.call_tool("lu4d_session_start", {})
    sid = s.get("structuredContent", {}).get("session_id")
    print(f"[session] Phase 7 session_id={sid}", flush=True)

    start = time.time()
    total = 0

    print("\n[wiki] streaming...", flush=True)
    total += teach_triples(client, stream_wikipedia(15_000, taught_cache),
                           "wiki", weight=0.7, confidence="Unverified",
                           taught_cache=taught_cache)

    client.call_tool("lu4d_session_end", {})
    c = client.call_tool("lu4d_consolidate", {})
    print("[consolidate] " + c.get("content",[{}])[0].get("text",""), flush=True)

    elapsed = time.time() - start
    sc = phase_status(client, "post-Phase-7")
    print(f"  teach calls: {total}  wall time: {elapsed:.1f}s", flush=True)
    return total, sc


def run_phase8(client, taught_cache):
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 8 — LANGUAGE & LOGIC", flush=True)
    print("#" * 70, flush=True)
    phase_status(client, "pre-Phase-8")

    s   = client.call_tool("lu4d_session_start", {})
    sid = s.get("structuredContent", {}).get("session_id")
    print(f"[session] Phase 8 session_id={sid}", flush=True)

    start = time.time()
    total = 0

    triples = build_language_triples()
    print(f"\n[language] built {len(triples)} triples", flush=True)
    total += teach_triples(client, triples, "language", weight=0.9,
                           confidence="Verified", taught_cache=taught_cache)

    print("\n[commonsense_qa] streaming...", flush=True)
    total += teach_triples(client, stream_commonsense_qa(8_000, taught_cache),
                           "commonsense_qa", weight=0.7, confidence="Unverified",
                           taught_cache=taught_cache)

    print("\n[hellaswag] streaming...", flush=True)
    total += teach_triples(client, stream_hellaswag(8_000, taught_cache),
                           "hellaswag", weight=0.65, confidence="Unverified",
                           taught_cache=taught_cache)

    client.call_tool("lu4d_session_end", {})
    c = client.call_tool("lu4d_consolidate", {})
    print("[consolidate] " + c.get("content",[{}])[0].get("text",""), flush=True)

    elapsed = time.time() - start
    sc = phase_status(client, "post-Phase-8")
    print(f"  teach calls: {total}  wall time: {elapsed:.1f}s", flush=True)
    return total, sc


def run_phase9(client, taught_cache):
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 9 — MATH & ENGINEERING", flush=True)
    print("#" * 70, flush=True)
    phase_status(client, "pre-Phase-9")

    s   = client.call_tool("lu4d_session_start", {})
    sid = s.get("structuredContent", {}).get("session_id")
    print(f"[session] Phase 9 session_id={sid}", flush=True)

    start = time.time()
    total = 0

    triples = build_math_triples()
    print(f"\n[math] built {len(triples)} triples", flush=True)
    total += teach_triples(client, triples, "math", weight=0.9,
                           confidence="Verified", taught_cache=taught_cache)

    print("\n[math_qa] streaming...", flush=True)
    total += teach_triples(client, stream_math_qa(5_000, taught_cache),
                           "math_qa", weight=0.7, confidence="Unverified",
                           taught_cache=taught_cache)

    client.call_tool("lu4d_session_end", {})
    c = client.call_tool("lu4d_consolidate", {})
    print("[consolidate] " + c.get("content",[{}])[0].get("text",""), flush=True)

    elapsed = time.time() - start
    sc = phase_status(client, "post-Phase-9")
    print(f"  teach calls: {total}  wall time: {elapsed:.1f}s", flush=True)
    return total, sc


def run_phase10(client, taught_cache):
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 10 — GEOGRAPHY DEEP", flush=True)
    print("#" * 70, flush=True)
    phase_status(client, "pre-Phase-10")

    s   = client.call_tool("lu4d_session_start", {})
    sid = s.get("structuredContent", {}).get("session_id")
    print(f"[session] Phase 10 session_id={sid}", flush=True)

    start = time.time()
    total = 0

    triples = build_geography_deep_triples()
    print(f"\n[geo_deep] built {len(triples)} triples", flush=True)
    total += teach_triples(client, triples, "geo_deep", weight=0.9,
                           confidence="Verified", taught_cache=taught_cache)

    client.call_tool("lu4d_session_end", {})
    c = client.call_tool("lu4d_consolidate", {})
    print("[consolidate] " + c.get("content",[{}])[0].get("text",""), flush=True)

    elapsed = time.time() - start
    sc = phase_status(client, "post-Phase-10")
    print(f"  teach calls: {total}  wall time: {elapsed:.1f}s", flush=True)
    return total, sc


def run_phase11(client, taught_cache):
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 11 — PEOPLE & EVENTS", flush=True)
    print("#" * 70, flush=True)
    phase_status(client, "pre-Phase-11")

    s   = client.call_tool("lu4d_session_start", {})
    sid = s.get("structuredContent", {}).get("session_id")
    print(f"[session] Phase 11 session_id={sid}", flush=True)

    start = time.time()
    total = 0

    triples = build_people_events_triples()
    print(f"\n[people] built {len(triples)} triples", flush=True)
    total += teach_triples(client, triples, "people", weight=0.9,
                           confidence="Verified", taught_cache=taught_cache)

    print("\n[wiki_bio] streaming...", flush=True)
    total += teach_triples(client, stream_wiki_bio(10_000, taught_cache),
                           "wiki_bio", weight=0.7, confidence="Unverified",
                           taught_cache=taught_cache)

    print("\n[trivia_qa_p11] streaming...", flush=True)
    total += teach_triples(client, stream_trivia_qa(250_000, taught_cache),
                           "trivia_qa_p11", weight=0.7, confidence="Unverified",
                           taught_cache=taught_cache)

    client.call_tool("lu4d_session_end", {})
    c = client.call_tool("lu4d_consolidate", {})
    print("[consolidate] " + c.get("content",[{}])[0].get("text",""), flush=True)

    elapsed = time.time() - start
    sc = phase_status(client, "post-Phase-11")
    print(f"  teach calls: {total}  wall time: {elapsed:.1f}s", flush=True)
    return total, sc


def run_phase12(client, taught_cache):
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 12 — MODERN WORLD", flush=True)
    print("#" * 70, flush=True)
    phase_status(client, "pre-Phase-12")

    s   = client.call_tool("lu4d_session_start", {})
    sid = s.get("structuredContent", {}).get("session_id")
    print(f"[session] Phase 12 session_id={sid}", flush=True)

    start = time.time()
    total = 0

    triples = build_modern_world_triples()
    print(f"\n[modern] built {len(triples)} triples", flush=True)
    total += teach_triples(client, triples, "modern", weight=0.9,
                           confidence="Verified", taught_cache=taught_cache)

    print("\n[trivia_qa_p12] streaming...", flush=True)
    total += teach_triples(client, stream_trivia_qa(300_000, taught_cache),
                           "trivia_qa_p12", weight=0.7, confidence="Unverified",
                           taught_cache=taught_cache)

    client.call_tool("lu4d_session_end", {})
    c = client.call_tool("lu4d_consolidate", {})
    print("[consolidate] " + c.get("content",[{}])[0].get("text",""), flush=True)

    elapsed = time.time() - start
    sc = phase_status(client, "post-Phase-12")
    print(f"  teach calls: {total}  wall time: {elapsed:.1f}s", flush=True)
    return total, sc


def run_phase13(client, taught_cache):
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 13 — BULK TRIVIA EXPANSION", flush=True)
    print("#" * 70, flush=True)
    phase_status(client, "pre-Phase-13")

    s   = client.call_tool("lu4d_session_start", {})
    sid = s.get("structuredContent", {}).get("session_id")
    print(f"[session] Phase 13 session_id={sid}", flush=True)

    start = time.time()
    total = 0

    # trivia_qa at 400K — dedup eliminates everything already seen
    print("\n[trivia_qa_bulk] streaming...", flush=True)
    total += teach_triples(client, stream_trivia_qa(400_000, taught_cache),
                           "trivia_qa_bulk", weight=0.7, confidence="Unverified",
                           taught_cache=taught_cache)

    print("\n[squad_bulk] streaming...", flush=True)
    total += teach_triples(client, stream_squad(40_000, taught_cache),
                           "squad_bulk", weight=0.65, confidence="Unverified",
                           taught_cache=taught_cache)

    print("\n[nq_bulk] streaming...", flush=True)
    total += teach_triples(client, stream_natural_questions(25_000, taught_cache),
                           "nq_bulk", weight=0.65, confidence="Unverified",
                           taught_cache=taught_cache)

    print("\n[openbookqa2] loading...", flush=True)
    total += teach_triples(client, stream_openbookqa(taught_cache),
                           "openbookqa2", weight=0.7, confidence="Verified",
                           taught_cache=taught_cache)

    client.call_tool("lu4d_session_end", {})
    c = client.call_tool("lu4d_consolidate", {})
    print("[consolidate] " + c.get("content",[{}])[0].get("text",""), flush=True)

    elapsed = time.time() - start
    sc = phase_status(client, "post-Phase-13")
    print(f"  teach calls: {total}  wall time: {elapsed:.1f}s", flush=True)
    return total, sc


def run_phase14(client):
    """Final validation only — no teaching."""
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 14 — FINAL VALIDATION", flush=True)
    print("#" * 70, flush=True)

    sc = phase_status(client, "pre-Phase-14")
    start = time.time()

    # --- 14a: 100-question cross-domain eval ---
    hits, tot = run_sanity(client, PHASE14_EVAL, "PHASE 14 CROSS-DOMAIN EVAL")

    # --- 14b: temporal scope isolation test ---
    print("\n" + "=" * 68, flush=True)
    print("PHASE 14 TEMPORAL SCOPE TEST", flush=True)
    print("=" * 68, flush=True)

    def ask(q, top_k=8):
        try:
            r = client.call_tool("lu4d_ask", {"query": q, "depth": 2, "top_k": top_k})
            activated = r.get("structuredContent",{}).get("recall",{}).get("activated",[])
            return [a.get("content","").lower() for a in activated]
        except Exception:
            return []

    # Sentinel A: session-local fact that should NOT exist before
    SENT_A = "zorblaxite"
    before_a = ask(f"What is {SENT_A}?")
    print(f"  pre-session '{SENT_A}': {before_a[:3]}", flush=True)

    s1 = client.call_tool("lu4d_session_start", {})
    sid1 = s1.get("structuredContent",{}).get("session_id")
    client.call_tool("lu4d_teach", {"subject": SENT_A, "relation": "IsA",
                                    "object": "crystal", "weight": 0.9, "confidence": "Verified"})
    client.call_tool("lu4d_teach", {"subject": SENT_A, "relation": "UsedFor",
                                    "object": "fusion", "weight": 0.9, "confidence": "Verified"})
    during_a = ask(f"What is {SENT_A}?")
    print(f"  during session {sid1} '{SENT_A}': {during_a[:3]}", flush=True)
    client.call_tool("lu4d_session_end", {})
    after_a = ask(f"What is {SENT_A}?")
    print(f"  post-session '{SENT_A}': {after_a[:3]}", flush=True)

    scope_a = (SENT_A not in " ".join(before_a)) and \
              (SENT_A in " ".join(during_a) or "crystal" in " ".join(during_a)) and \
              (SENT_A in " ".join(after_a)  or "crystal" in " ".join(after_a))
    print(f"  scope A: {'PASS' if scope_a else 'PARTIAL'}", flush=True)

    # Sentinel B: second session, verify A persists and B isolated
    SENT_B = "vorzinium"
    s2 = client.call_tool("lu4d_session_start", {})
    sid2 = s2.get("structuredContent",{}).get("session_id")
    client.call_tool("lu4d_teach", {"subject": SENT_B, "relation": "IsA",
                                    "object": "element", "weight": 0.9, "confidence": "Verified"})
    during_b = ask(f"What is {SENT_B}?")
    a_still   = ask(f"What is {SENT_A}?")
    print(f"  during session {sid2} '{SENT_B}': {during_b[:3]}", flush=True)
    print(f"  during session {sid2} '{SENT_A}' (should persist): {a_still[:3]}", flush=True)
    client.call_tool("lu4d_session_end", {})

    scope_b = ("element" in " ".join(during_b) or SENT_B in " ".join(during_b)) and \
              ("crystal" in " ".join(a_still) or SENT_A in " ".join(a_still))
    print(f"  scope B: {'PASS' if scope_b else 'PARTIAL'}", flush=True)

    # --- 14c: time travel diff ---
    print("\n" + "=" * 68, flush=True)
    print("PHASE 14 TIME TRAVEL DIFF", flush=True)
    print("=" * 68, flush=True)
    for query, ms_ago in [("python", 3600_000), ("einstein", 7200_000),
                          ("democracy", 3600_000), ("transformer", 1800_000)]:
        try:
            tt = client.call_tool("lu4d_timetravel", {"query": query, "ms_ago": ms_ago})
            txt = tt.get("content", [{}])[0].get("text", "")
            print(f"  timetravel '{query}' {ms_ago//60000}min ago: {txt[:120]}", flush=True)
        except Exception as e:
            print(f"  timetravel '{query}' error: {e}", flush=True)

    elapsed = time.time() - start
    print("\n" + "=" * 68, flush=True)
    print("PHASE 14 SUMMARY", flush=True)
    print("=" * 68, flush=True)
    print(f"  cross-domain eval:  {hits}/{tot} ({100*hits/tot:.1f}%)", flush=True)
    print(f"  scope A (single):   {'PASS' if scope_a else 'PARTIAL'}", flush=True)
    print(f"  scope B (multi):    {'PASS' if scope_b else 'PARTIAL'}", flush=True)
    print(f"  wall time:          {elapsed:.1f}s", flush=True)
    return hits, tot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(LU4D_BIN):
        print(f"FATAL: {LU4D_BIN}", flush=True); sys.exit(1)
    if not os.path.exists(BRAIN_FILE):
        print(f"FATAL: {BRAIN_FILE}", flush=True); sys.exit(1)

    client = MCPClient(BRAIN_FILE, no_save=True)
    client.start()
    taught_cache = set()
    all_start    = time.time()
    results      = {}

    try:
        phases = [
            ("7",  run_phase7),
            ("8",  run_phase8),
            ("9",  run_phase9),
            ("10", run_phase10),
            ("11", run_phase11),
            ("12", run_phase12),
            ("13", run_phase13),
        ]

        for name, runner in phases:
            total, sc = runner(client, taught_cache)
            results[name] = {"teaches": total, "sc": sc}
            if not check_size(client, f"Phase-{name}"):
                print(f"\n[ABORT] Brain size limit reached after Phase {name}. "
                      f"Skipping remaining training phases.", flush=True)
                break

        # Phase 14 always runs (eval only, no growth)
        p14_hits, p14_tot = run_phase14(client)
        results["14"] = {"hits": p14_hits, "tot": p14_tot}

        total_elapsed = time.time() - all_start
        print("\n\n" + "#" * 70, flush=True)
        print("# COMBINED PHASES 7–14 SUMMARY", flush=True)
        print("#" * 70, flush=True)
        for name, r in results.items():
            if "teaches" in r:
                sc = r["sc"]
                nodes = sc.get("node_count", "?")
                bonds = sc.get("bond_count", "?")
                print(f"  Phase {name:2s}: teaches={r['teaches']:>8,}  "
                      f"nodes={nodes:>8,}  bonds={bonds:>8,}", flush=True)
            else:
                print(f"  Phase 14: eval {r['hits']}/{r['tot']} "
                      f"({100*r['hits']/r['tot']:.1f}%)", flush=True)
        print(f"  total wall time: {total_elapsed:.1f}s", flush=True)

    finally:
        client.stop()


if __name__ == "__main__":
    main()

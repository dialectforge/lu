"""
Lu 4D Phase 4 + Phase 5 + Phase 6: Technology, Practical Knowledge, Validation.

  Phase 4 - Technology: programming languages, algorithms, data structures,
            networking, operating systems, databases, AI/ML, software
            engineering, cybersecurity, hardware.

  Phase 5 - Practical Knowledge: cooking, health, law, economics, business,
            finance, psychology, education, social skills.

  Phase 6 - Validation: comprehensive cross-domain eval (science, history,
            technology, practical) + temporal-scoping test. No new teaching.

Same approach as prior phases: single-token nodes, bidirectional bonds,
no stopwords, --no-save during training (final save on MCP shutdown).
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

# ---------------------------------------------------------------------------
# MCP client
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
# Normalization  (identical to prior phases)
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
# Teach helper with dedup + progress
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


# ===========================================================================
# PHASE 4 — TECHNOLOGY
# ===========================================================================

PROGRAMMING_LANGUAGES = [
    # Languages and their paradigms/uses
    ("python",     "language", "scripting"),  ("java",       "language", "oop"),
    ("javascript", "language", "scripting"),  ("typescript", "language", "scripting"),
    ("csharp",     "language", "oop"),        ("cpp",        "language", "systems"),
    ("c",          "language", "systems"),    ("rust",       "language", "systems"),
    ("go",         "language", "systems"),    ("swift",      "language", "mobile"),
    ("kotlin",     "language", "mobile"),     ("ruby",       "language", "scripting"),
    ("php",        "language", "web"),        ("scala",      "language", "functional"),
    ("haskell",    "language", "functional"), ("erlang",     "language", "functional"),
    ("elixir",     "language", "functional"), ("clojure",    "language", "functional"),
    ("lisp",       "language", "functional"), ("scheme",     "language", "functional"),
    ("prolog",     "language", "logic"),      ("sql",        "language", "query"),
    ("r",          "language", "statistical"),("matlab",     "language", "numerical"),
    ("julia",      "language", "numerical"),  ("perl",       "language", "scripting"),
    ("lua",        "language", "scripting"),  ("dart",       "language", "mobile"),
    ("groovy",     "language", "jvm"),        ("fortran",    "language", "scientific"),
    ("cobol",      "language", "legacy"),     ("assembly",   "language", "systems"),
    ("bash",       "language", "shell"),      ("powershell", "language", "shell"),
    ("vhdl",       "language", "hardware"),   ("verilog",    "language", "hardware"),
    # Paradigms
    ("oop",        "paradigm"), ("functional", "paradigm"), ("procedural", "paradigm"),
    ("declarative","paradigm"), ("imperative", "paradigm"), ("logic",      "paradigm"),
    ("concurrent", "paradigm"), ("reactive",   "paradigm"),
    # Typing
    ("python",  "HasProperty", "dynamictyping"),
    ("ruby",    "HasProperty", "dynamictyping"),
    ("javascript","HasProperty","dynamictyping"),
    ("java",    "HasProperty", "statictyping"),
    ("cpp",     "HasProperty", "statictyping"),
    ("rust",    "HasProperty", "statictyping"),
    ("haskell", "HasProperty", "statictyping"),
    ("python",  "UsedFor", "datascience"),
    ("python",  "UsedFor", "machinelearning"),
    ("javascript","UsedFor",  "web"),
    ("java",    "UsedFor", "enterprise"),
    ("c",       "UsedFor", "kernel"),
    ("rust",    "UsedFor", "safety"),
    ("sql",     "UsedFor", "database"),
    ("r",       "UsedFor", "statistics"),
    # Frameworks / runtimes
    ("django",   "IsA", "framework"), ("django",   "UsedFor", "python"),
    ("flask",    "IsA", "framework"), ("flask",    "UsedFor", "python"),
    ("fastapi",  "IsA", "framework"), ("fastapi",  "UsedFor", "python"),
    ("spring",   "IsA", "framework"), ("spring",   "UsedFor", "java"),
    ("rails",    "IsA", "framework"), ("rails",    "UsedFor", "ruby"),
    ("laravel",  "IsA", "framework"), ("laravel",  "UsedFor", "php"),
    ("react",    "IsA", "library"),   ("react",    "UsedFor", "javascript"),
    ("angular",  "IsA", "framework"), ("angular",  "UsedFor", "typescript"),
    ("vue",      "IsA", "framework"), ("vue",      "UsedFor", "javascript"),
    ("nodejs",   "IsA", "runtime"),   ("nodejs",   "UsedFor", "javascript"),
    ("dotnet",   "IsA", "framework"), ("dotnet",   "UsedFor", "csharp"),
    ("pytorch",  "IsA", "library"),   ("pytorch",  "UsedFor", "machinelearning"),
    ("tensorflow","IsA","library"),   ("tensorflow","UsedFor","machinelearning"),
    ("numpy",    "IsA", "library"),   ("numpy",    "UsedFor", "python"),
    ("pandas",   "IsA", "library"),   ("pandas",   "UsedFor", "python"),
    ("sklearn",  "IsA", "library"),   ("sklearn",  "UsedFor", "machinelearning"),
    ("keras",    "IsA", "library"),   ("keras",    "UsedFor", "deeplearning"),
    ("spark",    "IsA", "framework"), ("spark",    "UsedFor", "bigdata"),
    ("hadoop",   "IsA", "framework"), ("hadoop",   "UsedFor", "bigdata"),
    ("kafka",    "IsA", "platform"),  ("kafka",    "UsedFor", "streaming"),
    ("jvm",      "IsA", "runtime"),   ("jvm",      "UsedFor", "java"),
]

ALGORITHMS = [
    # Sorting
    ("quicksort",     "IsA", "sorting"),  ("mergesort",     "IsA", "sorting"),
    ("heapsort",      "IsA", "sorting"),  ("bubblesort",    "IsA", "sorting"),
    ("insertionsort", "IsA", "sorting"),  ("selectionsort", "IsA", "sorting"),
    ("timsort",       "IsA", "sorting"),  ("radixsort",     "IsA", "sorting"),
    ("countingsort",  "IsA", "sorting"),  ("shellsort",     "IsA", "sorting"),
    ("quicksort",  "HasProperty", "divideandconquer"),
    ("mergesort",  "HasProperty", "divideandconquer"),
    ("heapsort",   "HasProperty", "inplace"),
    ("quicksort",  "HasProperty", "inplace"),
    # Searching
    ("binarysearch",  "IsA", "searching"), ("linearsearch", "IsA", "searching"),
    ("bfs",           "IsA", "searching"), ("dfs",          "IsA", "searching"),
    ("astar",         "IsA", "searching"), ("bfs", "PartOf", "graph"),
    ("dfs",           "PartOf", "graph"),  ("astar", "UsedFor", "pathfinding"),
    # Graph algorithms
    ("dijkstra",  "IsA", "algorithm"), ("dijkstra",  "UsedFor", "shortest"),
    ("bellman",   "IsA", "algorithm"), ("bellman",   "UsedFor", "shortest"),
    ("floyd",     "IsA", "algorithm"), ("floyd",     "UsedFor", "shortest"),
    ("prim",      "IsA", "algorithm"), ("prim",      "UsedFor", "spanning"),
    ("kruskal",   "IsA", "algorithm"), ("kruskal",   "UsedFor", "spanning"),
    ("topological","IsA","algorithm"), ("topological","UsedFor","ordering"),
    # Paradigms
    ("divideandconquer","IsA","paradigm"), ("dynamic",  "IsA", "paradigm"),
    ("greedy",    "IsA", "paradigm"),     ("backtracking","IsA","paradigm"),
    ("memoization","IsA","optimization"), ("tabulation","IsA","optimization"),
    # Complexity
    ("complexity", "IsA", "measure"),    ("bigO",      "IsA", "notation"),
    ("linear",     "HasProperty", "complexity"),
    ("quadratic",  "HasProperty", "complexity"),
    ("logarithmic","HasProperty", "complexity"),
    ("polynomial", "HasProperty", "complexity"),
    ("exponential","HasProperty", "complexity"),
    ("constant",   "HasProperty", "complexity"),
    # Classic problems
    ("knapsack",   "IsA", "problem"),    ("knapsack",  "UsedFor", "dynamic"),
    ("tsp",        "IsA", "problem"),    ("tsp",       "HasProperty", "npcomplete"),
    ("halting",    "IsA", "problem"),    ("halting",   "HasProperty", "undecidable"),
    ("sorting",    "IsA", "algorithm"),  ("searching", "IsA", "algorithm"),
]

DATA_STRUCTURES = [
    ("array",      "IsA", "structure"), ("list",       "IsA", "structure"),
    ("linkedlist", "IsA", "structure"), ("doublylinked","IsA","structure"),
    ("stack",      "IsA", "structure"), ("queue",      "IsA", "structure"),
    ("deque",      "IsA", "structure"), ("priorityqueue","IsA","structure"),
    ("tree",       "IsA", "structure"), ("binarytree", "IsA", "tree"),
    ("bst",        "IsA", "tree"),      ("avl",        "IsA", "tree"),
    ("redblack",   "IsA", "tree"),      ("btree",      "IsA", "tree"),
    ("trie",       "IsA", "tree"),      ("segment",    "IsA", "tree"),
    ("fenwick",    "IsA", "tree"),      ("heap",       "IsA", "structure"),
    ("minheap",    "IsA", "heap"),      ("maxheap",    "IsA", "heap"),
    ("hashtable",  "IsA", "structure"), ("hashmap",    "IsA", "hashtable"),
    ("hashset",    "IsA", "hashtable"), ("graph",      "IsA", "structure"),
    ("dag",        "IsA", "graph"),     ("matrix",     "IsA", "structure"),
    ("bitmap",     "IsA", "structure"), ("bloom",      "IsA", "filter"),
    ("lru",        "IsA", "cache"),     ("cache",      "IsA", "structure"),
    ("skip",       "IsA", "list"),
    # Properties
    ("stack",      "HasProperty", "lifo"),
    ("queue",      "HasProperty", "fifo"),
    ("array",      "HasProperty", "random"),
    ("linkedlist", "HasProperty", "sequential"),
    ("bst",        "HasProperty", "ordered"),
    ("hashtable",  "HasProperty", "hashing"),
    ("heap",       "HasProperty", "complete"),
    ("graph",      "HasProperty", "vertex"),
    ("graph",      "HasProperty", "edge"),
    # Use cases
    ("stack",  "UsedFor", "recursion"),
    ("queue",  "UsedFor", "scheduling"),
    ("heap",   "UsedFor", "priority"),
    ("trie",   "UsedFor", "prefix"),
    ("graph",  "UsedFor", "network"),
    ("cache",  "UsedFor", "performance"),
    ("bloom",  "UsedFor", "membership"),
    ("lru",    "UsedFor", "eviction"),
    ("structure","PartOf","computer"),
]

NETWORKING = [
    # Protocols
    ("tcp",     "IsA", "protocol"), ("udp",    "IsA", "protocol"),
    ("ip",      "IsA", "protocol"), ("http",   "IsA", "protocol"),
    ("https",   "IsA", "protocol"), ("ftp",    "IsA", "protocol"),
    ("smtp",    "IsA", "protocol"), ("imap",   "IsA", "protocol"),
    ("pop3",    "IsA", "protocol"), ("ssh",    "IsA", "protocol"),
    ("tls",     "IsA", "protocol"), ("ssl",    "IsA", "protocol"),
    ("dns",     "IsA", "protocol"), ("dhcp",   "IsA", "protocol"),
    ("arp",     "IsA", "protocol"), ("icmp",   "IsA", "protocol"),
    ("bgp",     "IsA", "protocol"), ("ospf",   "IsA", "protocol"),
    ("rip",     "IsA", "protocol"), ("snmp",   "IsA", "protocol"),
    ("websocket","IsA","protocol"), ("grpc",   "IsA", "protocol"),
    ("mqtt",    "IsA", "protocol"), ("amqp",   "IsA", "protocol"),
    # Protocol properties
    ("tcp",  "HasProperty", "reliable"),
    ("udp",  "HasProperty", "fast"),
    ("http", "HasProperty", "stateless"),
    ("https","HasProperty", "encrypted"),
    ("ssh",  "UsedFor",     "remote"),
    ("dns",  "UsedFor",     "resolution"),
    ("tls",  "UsedFor",     "encryption"),
    ("ftp",  "UsedFor",     "transfer"),
    ("smtp", "UsedFor",     "email"),
    # OSI layers
    ("physical",   "IsA", "layer"), ("datalink",  "IsA", "layer"),
    ("network",    "IsA", "layer"), ("transport", "IsA", "layer"),
    ("session",    "IsA", "layer"), ("presentation","IsA","layer"),
    ("application","IsA", "layer"),
    ("ethernet",   "PartOf", "datalink"), ("ip",  "PartOf", "network"),
    ("tcp",        "PartOf", "transport"),("http","PartOf", "application"),
    # Hardware
    ("router",   "IsA", "device"), ("switch",   "IsA", "device"),
    ("hub",      "IsA", "device"), ("modem",    "IsA", "device"),
    ("firewall", "IsA", "device"), ("gateway",  "IsA", "device"),
    ("nic",      "IsA", "device"), ("repeater", "IsA", "device"),
    ("router",   "UsedFor", "routing"),
    ("switch",   "UsedFor", "switching"),
    ("firewall", "UsedFor", "security"),
    # Concepts
    ("bandwidth",  "IsA", "measure"), ("latency",   "IsA", "measure"),
    ("throughput", "IsA", "measure"), ("packet",    "IsA", "data"),
    ("frame",      "IsA", "data"),    ("subnet",    "IsA", "network"),
    ("vlan",       "IsA", "network"), ("vpn",       "IsA", "network"),
    ("nat",        "IsA", "technique"),("cdn",      "IsA", "service"),
    ("loadbalancer","IsA","service"), ("proxy",    "IsA", "server"),
    ("ipv4",       "IsA", "address"), ("ipv6",     "IsA", "address"),
    ("mac",        "IsA", "address"), ("port",     "IsA", "number"),
    ("socket",     "IsA", "endpoint"),("wifi",     "IsA", "wireless"),
    ("bluetooth",  "IsA", "wireless"),("cellular", "IsA", "wireless"),
    ("ethernet",   "IsA", "wired"),   ("fiber",    "IsA", "wired"),
    ("protocol",   "PartOf", "network"),
]

OPERATING_SYSTEMS = [
    # OS names
    ("linux",   "IsA", "os"), ("windows", "IsA", "os"),
    ("macos",   "IsA", "os"), ("unix",    "IsA", "os"),
    ("android", "IsA", "os"), ("ios",     "IsA", "os"),
    ("freebsd", "IsA", "os"), ("openbsd", "IsA", "os"),
    ("chromeos","IsA", "os"), ("haiku",   "IsA", "os"),
    # Linux distros
    ("ubuntu",  "IsA", "linux"), ("debian",  "IsA", "linux"),
    ("fedora",  "IsA", "linux"), ("arch",    "IsA", "linux"),
    ("centos",  "IsA", "linux"), ("rhel",    "IsA", "linux"),
    ("gentoo",  "IsA", "linux"), ("mint",    "IsA", "linux"),
    ("alpine",  "IsA", "linux"), ("nixos",   "IsA", "linux"),
    # Kernel concepts
    ("kernel",     "IsA", "component"), ("kernel", "PartOf", "os"),
    ("process",    "IsA", "entity"),    ("thread",  "IsA", "entity"),
    ("coroutine",  "IsA", "entity"),    ("fiber",   "IsA", "entity"),
    ("scheduler",  "IsA", "component"), ("dispatcher","IsA","component"),
    ("interrupt",  "IsA", "signal"),    ("syscall", "IsA", "interface"),
    ("memory",     "IsA", "resource"),  ("virtual", "IsA", "memory"),
    ("page",       "IsA", "memory"),    ("segment", "IsA", "memory"),
    ("swap",       "IsA", "memory"),    ("heap",    "IsA", "memory"),
    ("stack",      "PartOf", "memory"), ("cache",   "PartOf", "memory"),
    ("tlb",        "IsA", "cache"),     ("mmu",     "IsA", "unit"),
    # Concurrency
    ("mutex",      "IsA", "primitive"), ("semaphore","IsA","primitive"),
    ("lock",       "IsA", "primitive"), ("spinlock", "IsA", "lock"),
    ("monitor",    "IsA", "primitive"), ("condition","IsA","variable"),
    ("deadlock",   "IsA", "problem"),   ("livelock", "IsA", "problem"),
    ("starvation", "IsA", "problem"),   ("race",     "IsA", "condition"),
    ("atomicity",  "IsA", "property"),  ("mutex",    "UsedFor","synchronization"),
    ("semaphore",  "UsedFor","synchronization"),
    # Filesystem
    ("filesystem", "IsA", "component"),("inode",   "IsA", "structure"),
    ("directory",  "IsA", "structure"),("symlink",  "IsA", "file"),
    ("ext4",       "IsA", "filesystem"),("ntfs",    "IsA", "filesystem"),
    ("fat32",      "IsA", "filesystem"),("apfs",    "IsA", "filesystem"),
    ("zfs",        "IsA", "filesystem"),("btrfs",   "IsA", "filesystem"),
    ("journaling", "IsA", "filesystem"),
    # Booting
    ("bios",    "IsA", "firmware"),  ("uefi",   "IsA", "firmware"),
    ("grub",    "IsA", "bootloader"),("bootloader","IsA","software"),
    ("initrd",  "IsA", "image"),     ("kernel",  "UsedFor","booting"),
    # IPC
    ("pipe",    "IsA", "ipc"),  ("socket", "IsA", "ipc"),
    ("signal",  "IsA", "ipc"),  ("sharedmemory","IsA","ipc"),
    ("message", "IsA", "queue"),
]

DATABASES = [
    # Systems
    ("mysql",      "IsA", "database"), ("postgresql","IsA","database"),
    ("sqlite",     "IsA", "database"), ("oracle",    "IsA","database"),
    ("mssql",      "IsA", "database"), ("db2",       "IsA","database"),
    ("mongodb",    "IsA", "nosql"),    ("cassandra", "IsA","nosql"),
    ("redis",      "IsA", "nosql"),    ("dynamodb",  "IsA","nosql"),
    ("elasticsearch","IsA","nosql"),   ("couchdb",   "IsA","nosql"),
    ("neo4j",      "IsA", "graph"),    ("influxdb",  "IsA","timeseries"),
    ("clickhouse", "IsA", "olap"),     ("snowflake", "IsA","warehouse"),
    ("bigquery",   "IsA", "warehouse"),("redshift",  "IsA","warehouse"),
    # SQL concepts
    ("select",  "IsA", "statement"),  ("insert",  "IsA", "statement"),
    ("update",  "IsA", "statement"),  ("delete",  "IsA", "statement"),
    ("join",    "IsA", "operation"),  ("innerjoin","IsA","join"),
    ("leftjoin","IsA", "join"),       ("rightjoin","IsA","join"),
    ("fulljoin","IsA", "join"),       ("crossjoin","IsA","join"),
    ("index",   "IsA", "structure"),  ("view",    "IsA", "object"),
    ("trigger", "IsA", "object"),     ("procedure","IsA","object"),
    ("function","IsA", "object"),     ("schema",  "IsA", "structure"),
    ("table",   "IsA", "object"),     ("column",  "IsA", "field"),
    ("row",     "IsA", "record"),     ("primary", "IsA", "key"),
    ("foreign", "IsA", "key"),        ("unique",  "IsA", "constraint"),
    ("null",    "IsA", "value"),
    # ACID
    ("acid",      "IsA", "property"),
    ("atomicity", "PartOf", "acid"),  ("consistency","PartOf","acid"),
    ("isolation", "PartOf", "acid"),  ("durability","PartOf","acid"),
    ("transaction","IsA",  "unit"),   ("commit",   "IsA","operation"),
    ("rollback",  "IsA",  "operation"),
    # NoSQL concepts
    ("nosql",    "HasProperty", "schemaless"),
    ("nosql",    "HasProperty", "horizontal"),
    ("sharding", "IsA", "technique"),("replication","IsA","technique"),
    ("partition","IsA", "technique"),("cap",       "IsA","theorem"),
    ("eventual", "HasProperty", "consistency"),
    ("document", "IsA", "model"),    ("keyvalue", "IsA","model"),
    ("columnar", "IsA", "model"),    ("graph",    "IsA","model"),
    # ORM / Query
    ("orm",      "IsA", "tool"),     ("query",    "IsA","statement"),
    ("stored",   "IsA", "procedure"),("cursor",   "IsA","object"),
    ("normalisation","IsA","process"),("denormalisation","IsA","process"),
    ("olap",     "IsA", "system"),   ("oltp",     "IsA","system"),
    ("etl",      "IsA", "process"),  ("warehouse","IsA","system"),
    ("relational","IsA","database"),
]

AI_ML = [
    # ML types
    ("supervised",    "IsA", "learning"), ("unsupervised",  "IsA", "learning"),
    ("reinforcement", "IsA", "learning"), ("semisupervised","IsA","learning"),
    ("selfsupervised","IsA","learning"),  ("transfer",      "IsA","learning"),
    ("federated",     "IsA", "learning"), ("online",        "IsA","learning"),
    # Tasks
    ("classification","IsA","task"), ("regression",  "IsA","task"),
    ("clustering",    "IsA","task"), ("generation",  "IsA","task"),
    ("detection",     "IsA","task"), ("segmentation","IsA","task"),
    ("translation",   "IsA","task"), ("summarization","IsA","task"),
    ("ranking",       "IsA","task"), ("recommendation","IsA","task"),
    # Models
    ("neuralnetwork", "IsA", "model"), ("deeplearning",  "IsA","model"),
    ("cnn",           "IsA", "model"), ("rnn",           "IsA","model"),
    ("lstm",          "IsA", "model"), ("gru",           "IsA","model"),
    ("transformer",   "IsA", "model"), ("bert",          "IsA","model"),
    ("gpt",           "IsA", "model"), ("vae",           "IsA","model"),
    ("gan",           "IsA", "model"), ("diffusion",     "IsA","model"),
    ("resnet",        "IsA", "model"), ("vit",           "IsA","model"),
    ("autoencoder",   "IsA", "model"), ("embedding",     "IsA","representation"),
    ("linearregression","IsA","model"),("logistic",      "IsA","model"),
    ("svm",           "IsA", "model"), ("knn",           "IsA","model"),
    ("dtree",         "IsA", "model"), ("randomforest",  "IsA","model"),
    ("xgboost",       "IsA", "model"), ("lightgbm",      "IsA","model"),
    ("naivebayes",    "IsA", "model"), ("kmeans",        "IsA","model"),
    ("dbscan",        "IsA", "model"), ("pca",           "IsA","model"),
    # Components
    ("neuron",        "PartOf","neuralnetwork"),
    ("layer",         "PartOf","neuralnetwork"),
    ("weight",        "PartOf","neuralnetwork"),
    ("bias",          "PartOf","neuralnetwork"),
    ("activation",    "IsA",  "function"),
    ("relu",          "IsA",  "activation"),("sigmoid", "IsA","activation"),
    ("tanh",          "IsA",  "activation"),("softmax", "IsA","activation"),
    ("dropout",       "IsA",  "regularization"),("batchnorm","IsA","normalization"),
    ("attention",     "IsA",  "mechanism"),("multihead","IsA","attention"),
    ("feedforward",   "IsA",  "layer"),   ("encoder",  "IsA","component"),
    ("decoder",       "IsA",  "component"),
    # Training
    ("gradient",      "IsA", "descent"),  ("backpropagation","IsA","algorithm"),
    ("loss",          "IsA", "function"), ("optimizer","IsA","algorithm"),
    ("adam",          "IsA", "optimizer"),("sgd",      "IsA","optimizer"),
    ("rmsprop",       "IsA", "optimizer"),("adagrad",  "IsA","optimizer"),
    ("learning",      "HasProperty","rate"),
    ("overfitting",   "IsA", "problem"),  ("underfitting","IsA","problem"),
    ("regularization","IsA", "technique"),("dropout",  "UsedFor","regularization"),
    ("crossvalidation","IsA","technique"),("hyperparameter","IsA","parameter"),
    ("epoch",         "IsA", "unit"),     ("batch",    "IsA","unit"),
    # Evaluation
    ("accuracy",   "IsA", "metric"), ("precision", "IsA","metric"),
    ("recall",     "IsA", "metric"), ("f1",        "IsA","metric"),
    ("auc",        "IsA", "metric"), ("mse",       "IsA","metric"),
    ("mae",        "IsA", "metric"), ("perplexity","IsA","metric"),
    ("bleu",       "IsA", "metric"), ("rouge",     "IsA","metric"),
    # Data concepts
    ("dataset",   "IsA", "collection"), ("training",  "IsA","split"),
    ("validation","IsA", "split"),      ("test",      "IsA","split"),
    ("feature",   "IsA", "variable"),   ("label",     "IsA","variable"),
    ("target",    "IsA", "variable"),   ("vector",    "IsA","representation"),
    ("tensor",    "IsA", "structure"),  ("tokenization","IsA","process"),
    ("normalization","IsA","process"),  ("augmentation","IsA","process"),
    # NLP concepts
    ("nlp",           "IsA","field"),   ("tokenizer","IsA","component"),
    ("vocabulary",    "IsA","set"),     ("wordembedding","IsA","embedding"),
    ("contextual",    "IsA","embedding"),("finetune","IsA","technique"),
    ("pretrain",      "IsA","technique"),("rag",     "IsA","technique"),
    ("promptengineering","IsA","technique"),("zeroshot","IsA","technique"),
    ("fewshot",       "IsA","technique"),("chain",   "IsA","technique"),
    # CV concepts
    ("convolution",   "IsA","operation"),("pooling",  "IsA","operation"),
    ("maxpooling",    "IsA","pooling"),   ("stride",   "IsA","parameter"),
    ("padding",       "IsA","parameter"), ("kernel",   "IsA","filter"),
    ("objectdetection","IsA","task"),    ("yolo",     "IsA","model"),
    ("rcnn",          "IsA","model"),    ("segmentation","IsA","task"),
]

SOFTWARE_ENGINEERING = [
    # Methodologies
    ("agile",      "IsA", "methodology"), ("scrum",   "IsA","methodology"),
    ("kanban",     "IsA", "methodology"), ("waterfall","IsA","methodology"),
    ("xp",         "IsA", "methodology"), ("lean",    "IsA","methodology"),
    ("spiral",     "IsA", "methodology"), ("rad",     "IsA","methodology"),
    ("devops",     "IsA", "practice"),    ("devsecops","IsA","practice"),
    ("sre",        "IsA", "practice"),    ("gitops",  "IsA","practice"),
    # Testing
    ("unittest",   "IsA", "testing"), ("integration","IsA","testing"),
    ("regression", "IsA", "testing"), ("acceptance", "IsA","testing"),
    ("e2e",        "IsA", "testing"), ("performance","IsA","testing"),
    ("loadtest",   "IsA", "testing"), ("fuzz",      "IsA","testing"),
    ("tdd",        "IsA", "practice"),("bdd",       "IsA","practice"),
    ("mocking",    "IsA", "technique"),("coverage", "IsA","metric"),
    # Version control
    ("git",        "IsA", "vcs"),    ("svn",      "IsA","vcs"),
    ("mercurial",  "IsA", "vcs"),    ("vcs",      "IsA","tool"),
    ("branch",     "IsA", "concept"),("merge",    "IsA","operation"),
    ("commit",     "IsA", "operation"),("rebase", "IsA","operation"),
    ("pull",       "IsA", "request"),("fork",    "IsA","operation"),
    ("clone",      "IsA", "operation"),("stash", "IsA","operation"),
    # Design patterns
    ("singleton",  "IsA", "pattern"), ("factory",   "IsA","pattern"),
    ("observer",   "IsA", "pattern"), ("decorator", "IsA","pattern"),
    ("strategy",   "IsA", "pattern"), ("command",   "IsA","pattern"),
    ("adapter",    "IsA", "pattern"), ("facade",    "IsA","pattern"),
    ("proxy",      "IsA", "pattern"), ("template",  "IsA","pattern"),
    ("iterator",   "IsA", "pattern"), ("composite", "IsA","pattern"),
    ("builder",    "IsA", "pattern"), ("prototype", "IsA","pattern"),
    ("mvc",        "IsA", "pattern"), ("mvvm",      "IsA","pattern"),
    ("pattern",    "IsA", "solution"),
    # Architecture
    ("microservices","IsA","architecture"),("monolith","IsA","architecture"),
    ("serverless", "IsA","architecture"), ("soa",    "IsA","architecture"),
    ("rest",       "IsA","style"),         ("graphql","IsA","query"),
    ("grpc",       "IsA","protocol"),      ("api",    "IsA","interface"),
    ("sdk",        "IsA","toolkit"),       ("cli",    "IsA","interface"),
    ("gui",        "IsA","interface"),     ("tui",    "IsA","interface"),
    # CI/CD
    ("cicd",       "IsA","pipeline"),  ("jenkins",  "IsA","cicd"),
    ("github",     "IsA","platform"),  ("gitlab",   "IsA","platform"),
    ("actions",    "IsA","cicd"),      ("travis",   "IsA","cicd"),
    ("circle",     "IsA","cicd"),      ("artifact", "IsA","output"),
    ("pipeline",   "IsA","process"),   ("deploy",   "IsA","operation"),
    ("build",      "IsA","operation"), ("test",     "IsA","operation"),
    ("release",    "IsA","process"),   ("rollback", "IsA","operation"),
    # Containers / infra
    ("docker",     "IsA","container"), ("kubernetes","IsA","orchestration"),
    ("container",  "IsA","runtime"),   ("image",    "IsA","artifact"),
    ("pod",        "IsA","kubernetes"),("service",  "IsA","kubernetes"),
    ("ingress",    "IsA","kubernetes"),("helm",     "IsA","package"),
    ("terraform",  "IsA","iac"),       ("ansible",  "IsA","iac"),
    ("cloudformation","IsA","iac"),    ("iac",      "IsA","practice"),
    ("aws",        "IsA","cloud"),     ("gcp",      "IsA","cloud"),
    ("azure",      "IsA","cloud"),     ("cloud",    "IsA","platform"),
    ("lambda",     "IsA","serverless"),("function", "IsA","serverless"),
    ("s3",         "IsA","storage"),   ("ec2",      "IsA","compute"),
    ("rds",        "IsA","database"),  ("vpc",      "IsA","network"),
]

CYBERSECURITY = [
    # Cryptography
    ("encryption",    "IsA", "technique"), ("decryption",   "IsA","technique"),
    ("rsa",           "IsA", "algorithm"), ("aes",          "IsA","algorithm"),
    ("des",           "IsA", "algorithm"), ("3des",         "IsA","algorithm"),
    ("ecc",           "IsA", "algorithm"), ("dsa",          "IsA","algorithm"),
    ("diffie",        "IsA", "protocol"),  ("ecdh",         "IsA","protocol"),
    ("sha256",        "IsA", "hash"),      ("sha512",       "IsA","hash"),
    ("md5",           "IsA", "hash"),      ("bcrypt",       "IsA","hash"),
    ("argon",         "IsA", "hash"),      ("pbkdf",        "IsA","hash"),
    ("symmetric",     "IsA", "encryption"),("asymmetric",   "IsA","encryption"),
    ("publickey",     "IsA", "cryptography"),("privatekey", "IsA","cryptography"),
    ("certificate",   "IsA", "document"),  ("pki",         "IsA","infrastructure"),
    ("ssl",           "UsedFor","encryption"),
    ("tls",           "UsedFor","encryption"),
    # Attacks
    ("xss",           "IsA", "attack"),    ("sqli",         "IsA","attack"),
    ("csrf",          "IsA", "attack"),    ("xxe",          "IsA","attack"),
    ("ssrf",          "IsA", "attack"),    ("idor",         "IsA","attack"),
    ("mitm",          "IsA", "attack"),    ("replay",       "IsA","attack"),
    ("bruteforce",    "IsA", "attack"),    ("dictionary",   "IsA","attack"),
    ("rainbow",       "IsA", "attack"),    ("phishing",     "IsA","attack"),
    ("spear",         "IsA", "phishing"),  ("vishing",      "IsA","attack"),
    ("smishing",      "IsA", "attack"),    ("spoofing",     "IsA","attack"),
    ("arp",           "IsA", "spoofing"),  ("dns",          "IsA","spoofing"),
    ("ddos",          "IsA", "attack"),    ("dos",          "IsA","attack"),
    ("ransomware",    "IsA", "malware"),   ("virus",        "IsA","malware"),
    ("worm",          "IsA", "malware"),   ("trojan",       "IsA","malware"),
    ("rootkit",       "IsA", "malware"),   ("spyware",      "IsA","malware"),
    ("adware",        "IsA", "malware"),   ("keylogger",    "IsA","malware"),
    ("backdoor",      "IsA", "malware"),   ("botnet",       "IsA","network"),
    ("zeroday",       "IsA", "vulnerability"),("exploit",   "IsA","technique"),
    ("payload",       "IsA", "code"),      ("shellcode",    "IsA","payload"),
    ("bufferoverflow","IsA", "vulnerability"),("heapspray", "IsA","technique"),
    ("rop",           "IsA", "technique"),
    # Defense
    ("firewall",      "IsA", "defense"),   ("ids",          "IsA","defense"),
    ("ips",           "IsA", "defense"),   ("waf",          "IsA","defense"),
    ("siem",          "IsA", "system"),    ("soc",          "IsA","team"),
    ("pentest",       "IsA", "assessment"),("redteam",      "IsA","team"),
    ("blueteam",      "IsA", "team"),      ("purpleteam",   "IsA","team"),
    ("vulnerability", "IsA", "weakness"),  ("patch",        "IsA","update"),
    ("hardening",     "IsA", "practice"),  ("sandbox",      "IsA","environment"),
    ("honeypot",      "IsA", "trap"),      ("deception",    "IsA","technique"),
    ("mfa",           "IsA", "authentication"),("2fa",      "IsA","mfa"),
    ("oauth",         "IsA", "protocol"),  ("saml",         "IsA","protocol"),
    ("jwt",           "IsA", "token"),     ("session",      "IsA","token"),
    ("rbac",          "IsA", "model"),     ("abac",         "IsA","model"),
    ("least",         "IsA", "privilege"), ("zerotrust",    "IsA","model"),
    # Standards
    ("owasp",         "IsA", "standard"),  ("nist",         "IsA","framework"),
    ("iso27001",      "IsA", "standard"),  ("gdpr",         "IsA","regulation"),
    ("hipaa",         "IsA", "regulation"),("pci",          "IsA","standard"),
    ("cvss",          "IsA", "scoring"),   ("cve",          "IsA", "database"),
]

HARDWARE = [
    # CPU
    ("cpu",        "IsA", "processor"), ("gpu",      "IsA","processor"),
    ("tpu",        "IsA", "processor"), ("npu",      "IsA","processor"),
    ("arm",        "IsA", "architecture"),("x86",    "IsA","architecture"),
    ("risc",       "IsA", "architecture"),("cisc",   "IsA","architecture"),
    ("core",       "PartOf","cpu"),     ("thread",   "PartOf","cpu"),
    ("register",   "PartOf","cpu"),     ("alu",      "PartOf","cpu"),
    ("fpu",        "PartOf","cpu"),     ("cache",    "PartOf","cpu"),
    ("l1",         "IsA", "cache"),     ("l2",       "IsA","cache"),
    ("l3",         "IsA", "cache"),     ("pipeline", "PartOf","cpu"),
    ("clockspeed", "IsA", "measure"),   ("hertz",    "IsA","unit"),
    # Memory
    ("ram",        "IsA", "memory"),    ("rom",      "IsA","memory"),
    ("dram",       "IsA", "ram"),       ("sram",     "IsA","ram"),
    ("ddr",        "IsA", "dram"),      ("lpddr",    "IsA","dram"),
    ("eeprom",     "IsA", "rom"),       ("flash",    "IsA","memory"),
    # Storage
    ("ssd",        "IsA", "storage"),   ("hdd",      "IsA","storage"),
    ("nvme",       "IsA", "ssd"),       ("sata",     "IsA","interface"),
    ("pcie",       "IsA", "interface"), ("usb",      "IsA","interface"),
    ("raid",       "IsA", "storage"),
    # Motherboard / components
    ("motherboard","IsA", "board"),     ("chipset",  "IsA","component"),
    ("bios",       "IsA", "firmware"),  ("uefi",     "IsA","firmware"),
    ("psu",        "IsA", "component"), ("heatsink", "IsA","component"),
    ("fan",        "IsA", "component"), ("thermal",  "IsA","paste"),
    ("socket",     "IsA", "connector"), ("slot",     "IsA","connector"),
    ("pcie",       "IsA", "slot"),      ("dimm",     "IsA","slot"),
    # Peripherals
    ("monitor",    "IsA", "peripheral"),("keyboard", "IsA","peripheral"),
    ("mouse",      "IsA", "peripheral"),("printer",  "IsA","peripheral"),
    ("scanner",    "IsA", "peripheral"),("webcam",   "IsA","peripheral"),
    ("microphone", "IsA", "peripheral"),("speaker",  "IsA","peripheral"),
    ("gpu",        "IsA", "peripheral"),
    # Semiconductors
    ("transistor", "IsA", "component"), ("diode",    "IsA","component"),
    ("capacitor",  "IsA", "component"), ("resistor", "IsA","component"),
    ("inductor",   "IsA", "component"), ("ic",       "IsA","circuit"),
    ("fpga",       "IsA", "chip"),      ("asic",     "IsA","chip"),
    ("mosfet",     "IsA", "transistor"),("bjt",      "IsA","transistor"),
    ("semiconductor","IsA","material"), ("silicon",  "IsA","semiconductor"),
    ("germanium",  "IsA","semiconductor"),("gaas",   "IsA","semiconductor"),
    # Buses
    ("bus",        "IsA", "channel"),   ("databus",  "IsA","bus"),
    ("addressbus", "IsA", "bus"),       ("controlbus","IsA","bus"),
    ("pcie",       "IsA", "bus"),       ("usb",      "IsA","bus"),
    ("i2c",        "IsA", "bus"),       ("spi",      "IsA","bus"),
    ("uart",       "IsA", "protocol"),
]


def build_programming_triples():
    out = []
    for t in PROGRAMMING_LANGUAGES:
        if len(t) == 3 and all(isinstance(x, str) for x in t):
            s, rel, o = t
            if rel in ("IsA", "HasProperty", "UsedFor", "PartOf"):
                out.append(t)
            else:
                out.append((s, "IsA", rel))
                out.append((s, "UsedFor", o))
        elif len(t) == 2:
            out.append((t[0], "IsA", t[1]))
    # Hierarchy glue
    out += [
        ("language", "IsA", "software"), ("framework", "IsA", "software"),
        ("library",  "IsA", "software"), ("runtime",   "IsA", "software"),
        ("paradigm", "PartOf","programming"),
        ("statictyping",  "IsA","typing"),
        ("dynamictyping", "IsA","typing"),
        ("typing",   "PartOf","language"),
        ("software", "IsA", "computer"),
    ]
    return out


def build_algorithms_triples():
    out = list(ALGORITHMS)
    out += [
        ("sorting",   "IsA", "algorithm"), ("searching",  "IsA","algorithm"),
        ("algorithm", "IsA", "procedure"), ("complexity", "PartOf","algorithm"),
        ("npcomplete","IsA", "complexity"), ("nhard",     "IsA","complexity"),
        ("polynomial","IsA", "complexity"), ("pseudocode","IsA","algorithm"),
        ("divideandconquer","PartOf","paradigm"),
        ("dynamic",  "PartOf","paradigm"),
        ("greedy",   "PartOf","paradigm"),
    ]
    return out


def build_data_structures_triples():
    out = list(DATA_STRUCTURES)
    out += [
        ("structure","IsA","computer"),
        ("abstract", "IsA","type"),
        ("lifo",     "IsA","order"),
        ("fifo",     "IsA","order"),
    ]
    return out


def build_networking_triples():
    out = list(NETWORKING)
    out += [
        ("network",  "IsA","system"),
        ("protocol", "IsA","standard"),
        ("layer",    "PartOf","osi"),
        ("osi",      "IsA","model"),
        ("packet",   "IsA","unit"),
        ("device",   "PartOf","network"),
    ]
    return out


def build_os_triples():
    out = list(OPERATING_SYSTEMS)
    out += [
        ("os",      "IsA","software"),
        ("process", "PartOf","os"),
        ("thread",  "PartOf","process"),
        ("kernel",  "IsA","software"),
        ("memory",  "PartOf","computer"),
        ("filesystem","PartOf","os"),
    ]
    return out


def build_database_triples():
    out = list(DATABASES)
    out += [
        ("database","IsA","software"),
        ("sql",     "IsA","language"),
        ("nosql",   "IsA","database"),
        ("relational","HasProperty","sql"),
        ("acid",    "PartOf","database"),
        ("table",   "PartOf","relational"),
        ("document","PartOf","nosql"),
        ("warehouse","IsA","database"),
    ]
    return out


def build_aiml_triples():
    out = list(AI_ML)
    out += [
        ("ai",          "IsA","field"),
        ("machinelearning","PartOf","ai"),
        ("deeplearning","PartOf","machinelearning"),
        ("nlp",         "PartOf","ai"),
        ("computervision","PartOf","ai"),
        ("robotics",    "PartOf","ai"),
        ("model",       "IsA","software"),
        ("training",    "IsA","process"),
        ("inference",   "IsA","process"),
        ("neuralnetwork","PartOf","deeplearning"),
        ("transformer", "PartOf","deeplearning"),
        ("llm",         "IsA","model"),
        ("llm",         "PartOf","nlp"),
    ]
    return out


def build_swe_triples():
    out = list(SOFTWARE_ENGINEERING)
    out += [
        ("software","IsA","product"),
        ("engineering","IsA","discipline"),
        ("pattern",  "PartOf","design"),
        ("architecture","IsA","design"),
        ("testing",  "PartOf","quality"),
        ("cicd",     "PartOf","devops"),
        ("container","PartOf","devops"),
        ("cloud",    "IsA","infrastructure"),
    ]
    return out


def build_cybersecurity_triples():
    out = list(CYBERSECURITY)
    out += [
        ("security",    "IsA","field"),
        ("attack",      "IsA","action"),
        ("malware",     "IsA","software"),
        ("defense",     "IsA","measure"),
        ("cryptography","IsA","field"),
        ("vulnerability","IsA","weakness"),
        ("threat",      "IsA","risk"),
        ("authentication","IsA","process"),
        ("authorization","IsA","process"),
    ]
    return out


def build_hardware_triples():
    out = list(HARDWARE)
    out += [
        ("hardware",    "IsA","component"),
        ("cpu",         "PartOf","computer"),
        ("memory",      "PartOf","computer"),
        ("storage",     "PartOf","computer"),
        ("motherboard", "PartOf","computer"),
        ("peripheral",  "PartOf","computer"),
        ("semiconductor","PartOf","hardware"),
        ("computer",    "IsA","machine"),
    ]
    return out


# ===========================================================================
# PHASE 5 — PRACTICAL KNOWLEDGE
# ===========================================================================

COOKING_METHODS = [
    ("boiling",    "IsA", "method"), ("steaming",   "IsA","method"),
    ("baking",     "IsA", "method"), ("roasting",   "IsA","method"),
    ("grilling",   "IsA", "method"), ("frying",     "IsA","method"),
    ("deepfrying", "IsA", "method"), ("stirfrying",  "IsA","method"),
    ("sauteing",   "IsA", "method"), ("braising",   "IsA","method"),
    ("poaching",   "IsA", "method"), ("smoking",    "IsA","method"),
    ("pickling",   "IsA", "method"), ("fermenting",  "IsA","method"),
    ("blanching",  "IsA", "method"), ("broiling",   "IsA","method"),
    ("simmering",  "IsA", "method"), ("stewing",    "IsA","method"),
    ("curing",     "IsA", "method"), ("caramelizing","IsA","method"),
    # Tools
    ("knife",      "IsA", "tool"),   ("pan",        "IsA","tool"),
    ("pot",        "IsA", "tool"),   ("wok",        "IsA","tool"),
    ("oven",       "IsA", "appliance"),("microwave","IsA","appliance"),
    ("blender",    "IsA", "appliance"),("mixer",    "IsA","appliance"),
    ("thermometer","IsA", "tool"),   ("scale",      "IsA","tool"),
    ("colander",   "IsA", "tool"),   ("whisk",      "IsA","tool"),
    ("spatula",    "IsA", "tool"),   ("tongs",      "IsA","tool"),
    # Ingredients categories
    ("spice",      "IsA", "ingredient"),("herb",    "IsA","ingredient"),
    ("grain",      "IsA", "ingredient"),("legume",  "IsA","ingredient"),
    ("dairy",      "IsA", "ingredient"),("meat",    "IsA","ingredient"),
    ("seafood",    "IsA", "ingredient"),
    ("pepper",     "IsA", "spice"),  ("cumin",      "IsA","spice"),
    ("turmeric",   "IsA", "spice"),  ("cinnamon",   "IsA","spice"),
    ("paprika",    "IsA", "spice"),  ("ginger",     "IsA","spice"),
    ("coriander",  "IsA", "spice"),  ("cardamom",   "IsA","spice"),
    ("saffron",    "IsA", "spice"),  ("nutmeg",     "IsA","spice"),
    ("clove",      "IsA", "spice"),  ("anise",      "IsA","spice"),
    ("vanilla",    "IsA", "spice"),  ("chilli",     "IsA","spice"),
    ("wheat",      "IsA", "grain"),  ("rice",       "IsA","grain"),
    ("oat",        "IsA", "grain"),  ("barley",     "IsA","grain"),
    ("quinoa",     "IsA", "grain"),  ("millet",     "IsA","grain"),
    ("lentil",     "IsA", "legume"), ("bean",       "IsA","legume"),
    ("chickpea",   "IsA", "legume"), ("soybean",    "IsA","legume"),
    ("peanut",     "IsA", "legume"), ("pea",        "IsA","legume"),
    ("milk",       "IsA", "dairy"),  ("cheese",     "IsA","dairy"),
    ("butter",     "IsA", "dairy"),  ("cream",      "IsA","dairy"),
    ("yoghurt",    "IsA", "dairy"),  ("egg",        "IsA","protein"),
    ("beef",       "IsA", "meat"),   ("pork",       "IsA","meat"),
    ("chicken",    "IsA", "meat"),   ("lamb",       "IsA","meat"),
    ("turkey",     "IsA", "meat"),   ("salmon",     "IsA","seafood"),
    ("tuna",       "IsA", "seafood"),("shrimp",     "IsA","seafood"),
    ("crab",       "IsA", "seafood"),("lobster",    "IsA","seafood"),
    ("tofu",       "IsA", "protein"),("tempeh",     "IsA","protein"),
    # Flavors / textures
    ("sweet",      "IsA", "flavor"), ("sour",       "IsA","flavor"),
    ("salty",      "IsA", "flavor"), ("bitter",     "IsA","flavor"),
    ("umami",      "IsA", "flavor"), ("spicy",      "IsA","flavor"),
    ("crunchy",    "IsA", "texture"),("creamy",     "IsA","texture"),
    ("tender",     "IsA", "texture"),("crispy",     "IsA","texture"),
]

HEALTH_PRACTICAL = [
    # Exercise types
    ("cardio",      "IsA", "exercise"), ("strength",   "IsA","exercise"),
    ("flexibility", "IsA", "exercise"), ("balance",    "IsA","exercise"),
    ("hiit",        "IsA", "exercise"), ("yoga",       "IsA","exercise"),
    ("pilates",     "IsA", "exercise"), ("swimming",   "IsA","exercise"),
    ("running",     "IsA", "exercise"), ("cycling",    "IsA","exercise"),
    ("walking",     "IsA", "exercise"), ("weightlifting","IsA","strength"),
    # Nutrition concepts
    ("macronutrient","IsA","nutrient"), ("micronutrient","IsA","nutrient"),
    ("calorie",     "IsA", "unit"),     ("metabolism",  "IsA","process"),
    ("digestion",   "IsA", "process"),  ("absorption",  "IsA","process"),
    ("glucose",     "IsA", "sugar"),    ("insulin",     "IsA","hormone"),
    ("cholesterol", "IsA", "lipid"),    ("omega3",      "IsA","fat"),
    ("antioxidant", "IsA", "compound"), ("probiotic",   "IsA","bacteria"),
    ("prebiotic",   "IsA", "fiber"),    ("hydration",   "IsA","process"),
    # Sleep
    ("sleep",       "IsA", "process"),  ("rem",        "IsA","sleep"),
    ("circadian",   "IsA", "rhythm"),   ("melatonin",  "IsA","hormone"),
    ("insomnia",    "IsA", "disorder"), ("apnea",      "IsA","disorder"),
    # Mental health
    ("mindfulness", "IsA", "practice"), ("meditation", "IsA","practice"),
    ("therapy",     "IsA", "treatment"),("counseling", "IsA","treatment"),
    ("cbt",         "IsA", "therapy"),  ("dbt",        "IsA","therapy"),
    ("anxiety",     "IsA", "disorder"), ("depression", "IsA","disorder"),
    ("stress",      "IsA", "response"), ("burnout",    "IsA","condition"),
    # Preventive health
    ("vaccine",     "IsA", "prevention"),("screening",  "IsA","prevention"),
    ("checkup",     "IsA", "prevention"),("hygiene",    "IsA","practice"),
    ("sanitation",  "IsA", "practice"),  ("quarantine", "IsA","measure"),
    ("immunity",    "IsA", "defense"),   ("antibody",   "IsA","protein"),
    ("blood",       "HasProperty","pressure"),
    ("bmi",         "IsA", "measure"),   ("pulse",      "IsA","measure"),
    ("temperature", "IsA", "measure"),   ("spirometry", "IsA","test"),
    # Healthcare
    ("hospital",    "IsA", "institution"),("clinic",   "IsA","institution"),
    ("pharmacy",    "IsA", "institution"),("doctor",   "IsA","profession"),
    ("nurse",       "IsA", "profession"),("surgeon",  "IsA","profession"),
    ("dentist",     "IsA", "profession"),("pharmacist","IsA","profession"),
    ("prescription","IsA", "document"),  ("diagnosis", "IsA","process"),
    ("prognosis",   "IsA", "assessment"),("triage",   "IsA","process"),
    ("firstaid",    "IsA", "treatment"), ("cpr",      "IsA","firstaid"),
]

LAW_CONCEPTS = [
    # Legal systems
    ("commonlaw",   "IsA", "system"),  ("civillaw",    "IsA","system"),
    ("sharia",      "IsA", "system"),  ("statute",     "IsA","law"),
    ("constitution","IsA", "document"),("amendment",   "IsA","change"),
    # Court types
    ("supreme",     "IsA", "court"),   ("appeals",     "IsA","court"),
    ("district",    "IsA", "court"),   ("magistrate",  "IsA","court"),
    ("tribunal",    "IsA", "court"),   ("arbitration", "IsA","process"),
    # Legal concepts
    ("plaintiff",   "IsA", "party"),   ("defendant",   "IsA","party"),
    ("prosecutor",  "IsA", "party"),   ("judge",       "IsA","role"),
    ("jury",        "IsA", "body"),    ("lawyer",      "IsA","profession"),
    ("barrister",   "IsA", "lawyer"),  ("solicitor",   "IsA","lawyer"),
    ("verdict",     "IsA", "decision"),("sentence",    "IsA","punishment"),
    ("acquittal",   "IsA", "decision"),("appeal",      "IsA","process"),
    ("warrant",     "IsA", "document"),("subpoena",    "IsA","document"),
    ("injunction",  "IsA", "order"),   ("restraining", "IsA","order"),
    ("bail",        "IsA", "release"),  ("parole",     "IsA","release"),
    # Areas of law
    ("criminal",    "IsA", "law"),     ("civil",       "IsA","law"),
    ("corporate",   "IsA", "law"),     ("contract",    "IsA","law"),
    ("tort",        "IsA", "law"),     ("property",    "IsA","law"),
    ("family",      "IsA", "law"),     ("immigration",  "IsA","law"),
    ("intellectual","IsA", "law"),     ("tax",         "IsA","law"),
    ("environmental","IsA","law"),     ("labor",       "IsA","law"),
    # Rights / concepts
    ("rights",      "IsA", "concept"), ("liberty",     "IsA","right"),
    ("privacy",     "IsA", "right"),   ("speech",      "IsA","right"),
    ("due",         "IsA", "process"), ("presumption", "IsA","principle"),
    ("burden",      "IsA", "proof"),   ("evidence",    "IsA","proof"),
    ("precedent",   "IsA", "principle"),("habeas",     "IsA","corpus"),
    ("jurisdiction","IsA", "authority"),("sovereignty","IsA","authority"),
    ("copyright",   "IsA", "right"),   ("trademark",      "IsA","right"),
    ("trademark",   "IsA", "right"),   ("trademark",   "UsedFor","protection"),
    ("copyright",   "UsedFor","protection"),
    ("trademark",      "UsedFor","protection"),
]

ECONOMICS_BUSINESS = [
    # Economics concepts
    ("supply",      "IsA", "concept"),  ("demand",     "IsA","concept"),
    ("inflation",   "IsA", "concept"),  ("deflation",  "IsA","concept"),
    ("recession",   "IsA", "cycle"),    ("depression", "IsA","cycle"),
    ("gdp",         "IsA", "measure"),  ("gnp",        "IsA","measure"),
    ("unemployment","IsA", "measure"),  ("productivity","IsA","measure"),
    ("elasticity",  "IsA", "concept"),  ("equilibrium","IsA","concept"),
    ("monopoly",    "IsA", "market"),   ("oligopoly",  "IsA","market"),
    ("competition", "IsA", "market"),   ("duopoly",    "IsA","market"),
    ("externality", "IsA", "effect"),   ("subsidy",    "IsA","policy"),
    ("tariff",      "IsA", "tax"),      ("quota",      "IsA","limit"),
    ("fiscal",      "IsA", "policy"),   ("monetary",   "IsA","policy"),
    ("austerity",   "IsA", "policy"),   ("stimulus",   "IsA","policy"),
    # Market types
    ("stock",       "IsA", "market"),   ("bond",       "IsA","market"),
    ("forex",       "IsA", "market"),   ("commodity",  "IsA","market"),
    ("futures",     "IsA", "market"),   ("options",    "IsA","market"),
    # Financial instruments
    ("equity",      "IsA", "instrument"),("debt",      "IsA","instrument"),
    ("share",       "IsA", "equity"),   ("bond",       "IsA","debt"),
    ("derivative",  "IsA", "instrument"),("option",    "IsA","derivative"),
    ("future",      "IsA", "derivative"),("swap",      "IsA","derivative"),
    ("etf",         "IsA", "fund"),     ("mutual",     "IsA","fund"),
    ("index",       "IsA", "fund"),     ("hedge",      "IsA","fund"),
    ("venture",     "IsA", "capital"),  ("private",    "IsA","equity"),
    # Business concepts
    ("startup",     "IsA", "company"),  ("enterprise", "IsA","company"),
    ("corporation", "IsA", "company"),  ("llc",        "IsA","company"),
    ("partnership", "IsA", "company"),  ("sole",       "IsA","proprietor"),
    ("merger",      "IsA", "event"),    ("acquisition","IsA","event"),
    ("ipo",         "IsA", "event"),    ("bankruptcy", "IsA","event"),
    ("revenue",     "IsA", "measure"),  ("profit",     "IsA","measure"),
    ("margin",      "IsA", "measure"),  ("ebitda",     "IsA","measure"),
    ("valuation",   "IsA", "measure"),  ("roi",        "IsA","measure"),
    ("cashflow",    "IsA", "measure"),  ("liquidity",  "IsA","property"),
    ("solvency",    "IsA", "property"), ("leverage",   "IsA","technique"),
    # Marketing
    ("marketing",   "IsA", "function"), ("branding",  "IsA","strategy"),
    ("advertising", "IsA", "strategy"), ("pr",        "IsA","function"),
    ("seo",         "IsA", "strategy"), ("sem",       "IsA","strategy"),
    ("crm",         "IsA", "system"),   ("funnel",    "IsA","model"),
    ("conversion",  "IsA", "metric"),   ("churn",     "IsA","metric"),
    ("retention",   "IsA", "metric"),   ("acquisition","IsA","metric"),
    # Management
    ("strategy",    "IsA", "planning"), ("operations","IsA","function"),
    ("hr",          "IsA", "function"), ("finance",   "IsA","function"),
    ("accounting",  "IsA", "function"), ("audit",     "IsA","process"),
    ("compliance",  "IsA", "process"),  ("governance","IsA","practice"),
    ("kpi",         "IsA", "metric"),   ("okr",       "IsA","framework"),
    ("leadership",  "IsA", "skill"),    ("management","IsA","skill"),
    ("negotiation", "IsA", "skill"),    ("delegation","IsA","skill"),
]

PSYCHOLOGY_SOCIAL = [
    # Core theories
    ("behaviorism",  "IsA", "theory"), ("cognitivism",  "IsA","theory"),
    ("psychoanalysis","IsA","theory"), ("humanism",     "IsA","theory"),
    ("gestalt",      "IsA", "theory"), ("constructivism","IsA","theory"),
    ("attachment",   "IsA", "theory"), ("behaviorism",  "PartOf","psychology"),
    # Concepts
    ("cognition",   "IsA", "process"), ("perception",  "IsA","process"),
    ("memory",      "IsA", "process"), ("attention",   "IsA","process"),
    ("emotion",     "IsA", "state"),   ("motivation",  "IsA","state"),
    ("personality", "IsA", "trait"),   ("intelligence","IsA","trait"),
    ("consciousness","IsA","state"),   ("unconscious", "IsA","state"),
    ("instinct",    "IsA", "behavior"),("habit",       "IsA","behavior"),
    ("bias",        "IsA", "error"),   ("heuristic",   "IsA","strategy"),
    ("cognitive",   "IsA", "bias"),    ("confirmation","IsA","bias"),
    ("anchoring",   "IsA", "bias"),    ("availability","IsA","bias"),
    ("sunk",        "IsA", "fallacy"), ("gambler",     "IsA","fallacy"),
    # Learning
    ("classical",   "IsA", "conditioning"),("operant", "IsA","conditioning"),
    ("reinforcement","IsA","learning"), ("punishment",  "IsA","stimulus"),
    ("reward",      "IsA", "stimulus"), ("extinction",  "IsA","process"),
    ("generalization","IsA","process"), ("discrimination","IsA","process"),
    # Development
    ("piaget",      "IsA", "theorist"), ("vygotsky",   "IsA","theorist"),
    ("erikson",     "IsA", "theorist"), ("kohlberg",   "IsA","theorist"),
    ("maslow",      "IsA", "theorist"), ("freud",      "IsA","theorist"),
    ("jung",        "IsA", "theorist"),
    ("hierarchy",   "IsA", "model"),   ("needs",       "PartOf","hierarchy"),
    # Social psychology
    ("conformity",  "IsA", "behavior"), ("obedience",  "IsA","behavior"),
    ("compliance",  "IsA", "behavior"), ("persuasion",  "IsA","technique"),
    ("influence",   "IsA", "technique"),("groupthink", "IsA","phenomenon"),
    ("diffusion",   "IsA", "phenomenon"),("bystander", "IsA","effect"),
    ("prejudice",   "IsA", "attitude"), ("stereotype",  "IsA","belief"),
    ("discrimination","IsA","behavior"),("empathy",    "IsA","skill"),
    ("altruism",    "IsA", "behavior"), ("cooperation","IsA","behavior"),
    ("communication","IsA","skill"),    ("listening",  "IsA","skill"),
    ("assertiveness","IsA","skill"),    ("conflict",   "IsA","resolution"),
    ("leadership",  "IsA", "skill"),   ("teamwork",   "IsA","skill"),
    ("resilience",  "IsA", "trait"),   ("grit",       "IsA","trait"),
    ("growthmindset","IsA","mindset"),  ("fixedmindset","IsA","mindset"),
    # Personality models
    ("extroversion","IsA", "trait"),   ("introversion","IsA","trait"),
    ("openness",    "IsA", "trait"),   ("conscientiousness","IsA","trait"),
    ("agreeableness","IsA","trait"),   ("neuroticism", "IsA","trait"),
    ("bigfive",     "IsA", "model"),   ("mbti",       "IsA","model"),
]

FINANCE_EDUCATION = [
    # Personal finance
    ("budget",      "IsA", "plan"),    ("saving",     "IsA","practice"),
    ("investing",   "IsA", "practice"),("compound",   "IsA","interest"),
    ("simple",      "IsA", "interest"),("interest",   "IsA","rate"),
    ("mortgage",    "IsA", "loan"),    ("amortization","IsA","process"),
    ("credit",      "IsA", "score"),   ("debit",      "IsA","card"),
    ("portfolio",   "IsA", "collection"),("diversification","IsA","strategy"),
    ("riskmanagement","IsA","practice"),("insurance",  "IsA","product"),
    ("annuity",     "IsA", "product"), ("pension",    "IsA","benefit"),
    ("retirement",  "IsA", "planning"),("401k",       "IsA","account"),
    ("ira",         "IsA", "account"), ("roth",       "IsA","ira"),
    ("deduction",   "IsA", "tax"),     ("depreciation","IsA","accounting"),
    ("asset",       "IsA", "property"),("liability",  "IsA","obligation"),
    ("equity",      "IsA", "asset"),   ("net",        "IsA","worth"),
    # Education concepts
    ("curriculum",  "IsA", "plan"),    ("pedagogy",   "IsA","practice"),
    ("andragogy",   "IsA", "practice"),("syllabus",   "IsA","document"),
    ("assessment",  "IsA", "process"), ("formative",  "IsA","assessment"),
    ("summative",   "IsA", "assessment"),("rubric",   "IsA","tool"),
    ("differentiation","IsA","strategy"),("scaffolding","IsA","strategy"),
    ("bloom",       "IsA", "taxonomy"),("critical",   "IsA","thinking"),
    ("creativity",  "IsA", "skill"),   ("problemsolving","IsA","skill"),
    ("collaboration","IsA","skill"),   ("digital",    "IsA","literacy"),
    ("stem",        "IsA", "field"),   ("steam",      "IsA","field"),
    ("homeschool",  "IsA", "education"),("elearning", "IsA","education"),
    ("mooc",        "IsA", "course"),  ("blended",    "IsA","learning"),
    ("lms",         "IsA", "system"),  ("scholarship","IsA","funding"),
    ("grant",       "IsA", "funding"), ("tuition",    "IsA","cost"),
    ("degree",      "IsA", "qualification"),("diploma","IsA","qualification"),
    ("certificate", "IsA", "qualification"),("phd",   "IsA","degree"),
    ("masters",     "IsA", "degree"),  ("bachelors",  "IsA","degree"),
]


def build_cooking_triples():
    out = list(COOKING_METHODS)
    out += [
        ("method",     "PartOf","cooking"),
        ("ingredient", "PartOf","cooking"),
        ("cooking",    "IsA","skill"),
        ("recipe",     "IsA","instruction"),
        ("nutrition",  "PartOf","food"),
        ("flavor",     "PartOf","food"),
        ("texture",    "PartOf","food"),
        ("appliance",  "IsA","tool"),
        ("tool",       "PartOf","kitchen"),
    ]
    return out


def build_health_triples():
    out = list(HEALTH_PRACTICAL)
    out += [
        ("health",    "IsA","state"),
        ("wellness",  "IsA","state"),
        ("exercise",  "IsA","activity"),
        ("nutrition", "IsA","science"),
        ("medicine",  "IsA","science"),
        ("mental",    "PartOf","health"),
        ("physical",  "PartOf","health"),
        ("prevention","IsA","strategy"),
        ("treatment", "IsA","process"),
    ]
    return out


def build_law_triples():
    out = list(LAW_CONCEPTS)
    out += [
        ("law",       "IsA","system"),
        ("justice",   "IsA","concept"),
        ("legal",     "IsA","system"),
        ("rights",    "PartOf","law"),
        ("court",     "IsA","institution"),
        ("government","IsA","institution"),
        ("democracy", "IsA","system"),
        ("constitution","PartOf","government"),
    ]
    return out


def build_economics_triples():
    out = list(ECONOMICS_BUSINESS)
    out += [
        ("economics", "IsA","science"),
        ("microeconomics","IsA","economics"),
        ("macroeconomics","IsA","economics"),
        ("market",    "IsA","system"),
        ("capitalism","IsA","system"),
        ("socialism", "IsA","system"),
        ("trade",     "IsA","activity"),
        ("globalization","IsA","process"),
        ("business",  "IsA","organization"),
        ("company",   "IsA","organization"),
    ]
    return out


def build_psychology_triples():
    out = list(PSYCHOLOGY_SOCIAL)
    out += [
        ("psychology","IsA","science"),
        ("sociology", "IsA","science"),
        ("anthropology","IsA","science"),
        ("behavior",  "IsA","action"),
        ("cognition", "PartOf","psychology"),
        ("social",    "PartOf","psychology"),
        ("developmental","IsA","psychology"),
        ("clinical",  "IsA","psychology"),
        ("neuroscience","IsA","science"),
    ]
    return out


def build_finance_education_triples():
    out = list(FINANCE_EDUCATION)
    out += [
        ("finance",   "IsA","field"),
        ("accounting","IsA","field"),
        ("education", "IsA","process"),
        ("learning",  "IsA","process"),
        ("school",    "IsA","institution"),
        ("university","IsA","institution"),
        ("money",     "IsA","resource"),
        ("wealth",    "IsA","resource"),
        ("investment","IsA","activity"),
    ]
    return out


# ---------------------------------------------------------------------------
# HF dataset streamers (reuse from prior phases, fresh limits)
# ---------------------------------------------------------------------------

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


def stream_natural_questions(limit, seen):
    try:
        from datasets import load_dataset
    except Exception: return
    try:
        ds = load_dataset("natural_questions", split="train", streaming=True)
    except Exception as e:
        print(f"[nq] load failed (skipping): {e}", flush=True); return
    print(f"[nq] streaming (limit={limit})...", flush=True)
    n = 0
    for row in ds:
        if n >= limit: return
        n += 1
        try:
            q = row.get("question", {})
            q_text = q.get("text", "") if isinstance(q, dict) else str(q)
            annotations = row.get("annotations", {}) or {}
            short_answers = annotations.get("short_answers", []) if isinstance(annotations, dict) else []
            if not short_answers: continue
            q_toks = content_tokens(q_text)
            if not q_toks: continue
            qf = q_toks[-1]
            if not is_single_token(qf): continue
            for sa in short_answers[:1]:
                if not isinstance(sa, dict): continue
                tokens = sa.get("text", []) or []
                if isinstance(tokens, str): tokens = [tokens]
                for tok_text in tokens[:3]:
                    for t in content_tokens(str(tok_text)):
                        if is_single_token(t) and t != qf:
                            yield (qf, "RelatedTo", t)
        except Exception:
            continue


def stream_sciq_extra(limit, seen):
    """Second pass of SciQ — the seen cache will dedup already-taught triples."""
    try:
        from datasets import load_dataset
    except Exception: return
    try:
        ds = load_dataset("sciq", split="train", streaming=True)
    except Exception as e:
        print(f"[sciq2] load failed: {e}", flush=True); return
    print(f"[sciq2] streaming extra pass (limit={limit})...", flush=True)
    n = 0
    for row in ds:
        if n >= limit: return
        n += 1
        q_toks = content_tokens(row.get("question", "") or "")
        ans_toks = content_tokens(row.get("correct_answer", "") or "")
        sup_toks = content_tokens(row.get("support", "") or "")
        if not q_toks or not ans_toks: continue
        qf = q_toks[-1]
        if not is_single_token(qf): continue
        for a in ans_toks[:3]:
            if a != qf and is_single_token(a):
                yield (qf, "RelatedTo", a)
        ah = ans_toks[0]
        for t in sup_toks[:5]:
            if t != ah and is_single_token(t):
                yield (ah, "RelatedTo", t)


def stream_squad(limit, seen):
    try:
        from datasets import load_dataset
    except Exception: return
    try:
        ds = load_dataset("squad", split="train", streaming=True)
    except Exception as e:
        print(f"[squad] load failed (skipping): {e}", flush=True); return
    print(f"[squad] streaming (limit={limit})...", flush=True)
    n = 0
    for row in ds:
        if n >= limit: return
        n += 1
        try:
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
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Sanity / eval helpers
# ---------------------------------------------------------------------------

PHASE4_SANITY = [
    # Programming
    ("What is Python?",               ["language", "scripting", "programming"]),
    ("What is JavaScript used for?",  ["web", "scripting", "language"]),
    ("What is Rust?",                 ["language", "systems"]),
    ("What is SQL?",                  ["language", "database", "query"]),
    # Algorithms
    ("What is quicksort?",            ["sorting", "algorithm"]),
    ("What does Dijkstra solve?",     ["shortest", "path", "graph"]),
    ("What is dynamic programming?",  ["paradigm", "memoization", "algorithm"]),
    ("What is BFS?",                  ["searching", "graph"]),
    # Data structures
    ("What is a stack?",              ["structure", "lifo"]),
    ("What is a hash table?",         ["structure", "hashing"]),
    ("What is a binary tree?",        ["tree", "structure"]),
    # Networking
    ("What is TCP?",                  ["protocol", "reliable"]),
    ("What is DNS?",                  ["protocol", "resolution"]),
    ("What is HTTPS?",                ["protocol", "encrypted"]),
    # Security
    ("What is AES?",                  ["algorithm", "encryption"]),
    ("What is phishing?",             ["attack"]),
    ("What is a firewall?",           ["defense", "device"]),
    # Hardware
    ("What is a CPU?",                ["processor", "computer"]),
    ("What is RAM?",                  ["memory"]),
    ("What is a transistor?",         ["component", "semiconductor"]),
]

PHASE5_SANITY = [
    ("What is sauteing?",             ["method", "cooking"]),
    ("What is a legume?",             ["ingredient"]),
    ("What is cardio?",               ["exercise"]),
    ("What is mindfulness?",          ["practice", "meditation"]),
    ("What is a plaintiff?",          ["party", "law"]),
    ("What is a trademark?",             ["right", "protection"]),
    ("What is inflation?",            ["concept", "economics"]),
    ("What is GDP?",                  ["measure"]),
    ("What is a startup?",            ["company"]),
    ("What is compound interest?",    ["interest"]),
    ("What is cognitive bias?",       ["bias", "error"]),
    ("What is conformity?",           ["behavior"]),
    ("What is a curriculum?",         ["plan", "education"]),
    ("What is scaffolding?",          ["strategy", "education"]),
    ("What is a budget?",             ["plan"]),
    ("What is diversification?",      ["strategy"]),
    ("What is the big five?",         ["model", "personality"]),
    ("What is CBT?",                  ["therapy"]),
    ("What is a MOOC?",               ["course", "elearning"]),
    ("What is critical thinking?",    ["skill"]),
]

# Phase 6: comprehensive cross-domain eval (science + history + tech + practical)
PHASE6_EVAL = [
    # Science (Phase 2 re-check)
    ("What is methane?",              ["alkane", "hydrocarbon"]),
    ("What is insulin?",              ["hormone"]),
    ("What is a quark?",              ["particle"]),
    ("What is granite?",              ["igneous", "rock"]),
    # History (Phase 3 re-check)
    ("Who was Napoleon?",             ["emperor", "france"]),
    ("What is Buddhism?",             ["religion"]),
    ("Who was Pele?",                 ["soccer", "brazil"]),
    ("What is jazz?",                 ["genre", "music"]),
    # Technology (Phase 4)
    ("What is Python?",               ["language", "scripting"]),
    ("What is quicksort?",            ["sorting", "algorithm"]),
    ("What is TCP?",                  ["protocol", "reliable"]),
    ("What is AES?",                  ["algorithm", "encryption"]),
    ("What is a CPU?",                ["processor"]),
    ("What is a stack?",              ["structure", "lifo"]),
    ("What is a neural network?",     ["model", "deeplearning"]),
    ("What is Docker?",               ["container"]),
    # Practical (Phase 5)
    ("What is inflation?",            ["concept", "economics"]),
    ("What is a trademark?",             ["right", "protection"]),
    ("What is compound interest?",    ["interest"]),
    ("What is cognitive bias?",       ["bias"]),
    ("What is cardio?",               ["exercise"]),
    ("What is a curriculum?",         ["plan", "education"]),
    # Cross-domain linkage
    ("What is machine learning?",     ["learning", "ai"]),
    ("What is a relational database?",["sql", "relational"]),
    ("What is a transformer?",        ["model", "deeplearning"]),
    ("What is encryption?",           ["technique", "security"]),
    ("What is deep learning?",        ["model", "machinelearning"]),
    ("What is a neurotransmitter?",   ["molecule", "neurotransmitter"]),
    ("What is evolution?",            ["darwin", "biology"]),
    ("What is photosynthesis?",       ["oxygen"]),
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
        hit = any(kw.lower() in " | ".join(contents) for kw in expected)
        if hit: hits += 1
        mark = "PASS" if hit else "MISS"
        print(f"  Q{i:02d} [{mark}] {q}", flush=True)
        print(f"        expected={expected} | top-5={contents[:5]}", flush=True)
    pct = 100 * hits / len(questions)
    print(f"\n{name} SANITY SCORE: {hits}/{len(questions)} ({pct:.1f}%)", flush=True)
    return hits, len(questions)


def phase_status(client, label):
    r   = client.call_tool("lu4d_status", {})
    txt = r.get("content", [{}])[0].get("text", "")
    print(f"\n[status] {label}:\n  " + txt.replace("\n", "\n  "), flush=True)
    return txt


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase4(client, taught_cache):
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 4 — TECHNOLOGY", flush=True)
    print("#" * 70, flush=True)

    phase_status(client, "pre-Phase-4")

    s   = client.call_tool("lu4d_session_start", {})
    sid = s.get("structuredContent", {}).get("session_id")
    print(f"[session] Phase 4 session_id={sid}", flush=True)

    start = time.time()
    total = 0

    for name, builder, w in [
        ("programming",   build_programming_triples,   0.9),
        ("algorithms",    build_algorithms_triples,    0.9),
        ("datastructures",build_data_structures_triples,0.9),
        ("networking",    build_networking_triples,    0.9),
        ("os",            build_os_triples,            0.9),
        ("databases",     build_database_triples,      0.9),
        ("aiml",          build_aiml_triples,          0.9),
        ("swe",           build_swe_triples,           0.9),
        ("cybersecurity", build_cybersecurity_triples, 0.9),
        ("hardware",      build_hardware_triples,      0.9),
    ]:
        triples = builder()
        print(f"\n[{name}] built {len(triples)} triples", flush=True)
        total += teach_triples(client, triples, name, weight=w,
                               confidence="Verified", taught_cache=taught_cache)

    # HF pass 1: trivia_qa (100K rows — seen cache deups P3's 30K rows)
    print("\n[trivia_qa] streaming...", flush=True)
    total += teach_triples(client, stream_trivia_qa(100_000, taught_cache),
                           "trivia_qa", weight=0.7, confidence="Unverified",
                           taught_cache=taught_cache)

    # HF pass 2: sciq second pass (dedup handles P2 overlap)
    print("\n[sciq2] streaming...", flush=True)
    total += teach_triples(client, stream_sciq_extra(12_000, taught_cache),
                           "sciq2", weight=0.65, confidence="Verified",
                           taught_cache=taught_cache)

    # HF pass 3: squad (if available)
    print("\n[squad] streaming...", flush=True)
    total += teach_triples(client, stream_squad(30_000, taught_cache),
                           "squad", weight=0.65, confidence="Unverified",
                           taught_cache=taught_cache)

    # HF pass 4: natural_questions (if available)
    print("\n[nq] streaming...", flush=True)
    total += teach_triples(client, stream_natural_questions(20_000, taught_cache),
                           "nq", weight=0.65, confidence="Unverified",
                           taught_cache=taught_cache)

    client.call_tool("lu4d_session_end", {})
    c = client.call_tool("lu4d_consolidate", {})
    print("[consolidate] " + c.get("content", [{}])[0].get("text", ""), flush=True)

    elapsed = time.time() - start
    phase_status(client, "post-Phase-4")
    hits, tot = run_sanity(client, PHASE4_SANITY, "PHASE 4")

    print("\n" + "=" * 68, flush=True)
    print("PHASE 4 SUMMARY", flush=True)
    print("=" * 68, flush=True)
    print(f"  teach calls:   {total}", flush=True)
    print(f"  sanity:        {hits}/{tot} ({100*hits/tot:.1f}%)", flush=True)
    print(f"  wall time:     {elapsed:.1f}s", flush=True)

    return hits, tot, total


def run_phase5(client, taught_cache):
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 5 — PRACTICAL KNOWLEDGE", flush=True)
    print("#" * 70, flush=True)

    phase_status(client, "pre-Phase-5")

    s   = client.call_tool("lu4d_session_start", {})
    sid = s.get("structuredContent", {}).get("session_id")
    print(f"[session] Phase 5 session_id={sid}", flush=True)

    start = time.time()
    total = 0

    for name, builder, w in [
        ("cooking",     build_cooking_triples,          0.9),
        ("health",      build_health_triples,           0.9),
        ("law",         build_law_triples,              0.9),
        ("economics",   build_economics_triples,        0.9),
        ("psychology",  build_psychology_triples,       0.9),
        ("finance_edu", build_finance_education_triples,0.9),
    ]:
        triples = builder()
        print(f"\n[{name}] built {len(triples)} triples", flush=True)
        total += teach_triples(client, triples, name, weight=w,
                               confidence="Verified", taught_cache=taught_cache)

    # More trivia_qa (200K rows total cumulative; dedup handles prior 100K)
    print("\n[trivia_qa2] streaming...", flush=True)
    total += teach_triples(client, stream_trivia_qa(200_000, taught_cache),
                           "trivia_qa2", weight=0.7, confidence="Unverified",
                           taught_cache=taught_cache)

    # More squad
    print("\n[squad2] streaming...", flush=True)
    total += teach_triples(client, stream_squad(60_000, taught_cache),
                           "squad2", weight=0.65, confidence="Unverified",
                           taught_cache=taught_cache)

    client.call_tool("lu4d_session_end", {})
    c = client.call_tool("lu4d_consolidate", {})
    print("[consolidate] " + c.get("content", [{}])[0].get("text", ""), flush=True)

    elapsed = time.time() - start
    phase_status(client, "post-Phase-5")
    hits, tot = run_sanity(client, PHASE5_SANITY, "PHASE 5")

    print("\n" + "=" * 68, flush=True)
    print("PHASE 5 SUMMARY", flush=True)
    print("=" * 68, flush=True)
    print(f"  teach calls:   {total}", flush=True)
    print(f"  sanity:        {hits}/{tot} ({100*hits/tot:.1f}%)", flush=True)
    print(f"  wall time:     {elapsed:.1f}s", flush=True)

    return hits, tot, total


def run_phase6(client):
    """Validation only — no teaching, no session. Cross-domain + temporal scope."""
    print("\n\n" + "#" * 70, flush=True)
    print("# PHASE 6 — VALIDATION", flush=True)
    print("#" * 70, flush=True)

    phase_status(client, "pre-Phase-6")

    start = time.time()

    # --- 6a: comprehensive cross-domain eval ---
    hits, tot = run_sanity(client, PHASE6_EVAL, "PHASE 6 CROSS-DOMAIN")

    # --- 6b: temporal scope test ---
    print("\n" + "=" * 68, flush=True)
    print("PHASE 6 TEMPORAL SCOPE TEST", flush=True)
    print("=" * 68, flush=True)

    SENTINEL = "glorbite"  # fictional concept not in training data

    def ask(q):
        try:
            r = client.call_tool("lu4d_ask", {"query": q, "depth": 2, "top_k": 10})
            activated = r.get("structuredContent", {}).get("recall", {}).get("activated", [])
            return [a.get("content", "").lower() for a in activated]
        except Exception:
            return []

    # Before session: glorbite unknown
    before = ask(f"What is {SENTINEL}?")
    print(f"  [temporal] pre-session ask '{SENTINEL}': {before[:3]}", flush=True)
    found_before = SENTINEL in " ".join(before)

    # Open session, teach ephemeral facts
    s   = client.call_tool("lu4d_session_start", {})
    sid = s.get("structuredContent", {}).get("session_id")
    print(f"  [temporal] started session_id={sid}", flush=True)
    client.call_tool("lu4d_teach", {
        "subject": SENTINEL, "relation": "IsA", "object": "mineral",
        "weight": 0.9, "confidence": "Verified",
    })
    client.call_tool("lu4d_teach", {
        "subject": SENTINEL, "relation": "UsedFor", "object": "quantum",
        "weight": 0.9, "confidence": "Verified",
    })

    during = ask(f"What is {SENTINEL}?")
    print(f"  [temporal] during-session ask '{SENTINEL}': {during[:3]}", flush=True)
    found_during = SENTINEL in " ".join(during) or "mineral" in " ".join(during)

    client.call_tool("lu4d_session_end", {})
    print(f"  [temporal] ended session_id={sid}", flush=True)

    after = ask(f"What is {SENTINEL}?")
    print(f"  [temporal] post-session ask '{SENTINEL}': {after[:3]}", flush=True)
    found_after = SENTINEL in " ".join(after) or "mineral" in " ".join(after)

    scope_pass = (not found_before) and found_during and found_after
    print(f"\n  scope test: before={found_before} during={found_during} "
          f"after={found_after} -> {'PASS' if scope_pass else 'PARTIAL'}", flush=True)

    elapsed = time.time() - start

    print("\n" + "=" * 68, flush=True)
    print("PHASE 6 SUMMARY", flush=True)
    print("=" * 68, flush=True)
    print(f"  cross-domain:  {hits}/{tot} ({100*hits/tot:.1f}%)", flush=True)
    print(f"  temporal scope: {'PASS' if scope_pass else 'PARTIAL'}", flush=True)
    print(f"  wall time:     {elapsed:.1f}s", flush=True)

    return hits, tot, scope_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(LU4D_BIN):
        print(f"FATAL: lu4d.exe not found at {LU4D_BIN}", flush=True)
        sys.exit(1)
    if not os.path.exists(BRAIN_FILE):
        print(f"FATAL: brain not found at {BRAIN_FILE}", flush=True)
        sys.exit(1)

    client = MCPClient(BRAIN_FILE, no_save=True)
    client.start()
    taught_cache = set()
    all_start    = time.time()

    try:
        p4_hits, p4_tot, p4_total = run_phase4(client, taught_cache)
        p5_hits, p5_tot, p5_total = run_phase5(client, taught_cache)
        p6_hits, p6_tot, p6_scope = run_phase6(client)

        total_elapsed = time.time() - all_start
        print("\n\n" + "#" * 70, flush=True)
        print("# COMBINED PHASE 4 + 5 + 6 SUMMARY", flush=True)
        print("#" * 70, flush=True)
        print(f"  Phase 4 teaches: {p4_total}, sanity: {p4_hits}/{p4_tot}", flush=True)
        print(f"  Phase 5 teaches: {p5_total}, sanity: {p5_hits}/{p5_tot}", flush=True)
        print(f"  Phase 6 cross-domain: {p6_hits}/{p6_tot}, "
              f"temporal: {'PASS' if p6_scope else 'PARTIAL'}", flush=True)
        print(f"  total wall time: {total_elapsed:.1f}s", flush=True)
    finally:
        client.stop()


if __name__ == "__main__":
    main()

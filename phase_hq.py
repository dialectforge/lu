#!/usr/bin/env python3
"""
phase_hq.py -- High-quality training on known-working HuggingFace datasets.
Target: 500K+ nodes, stay under 4GB.
Datasets: trivia_qa, sciq, squad_v2, openbookqa, commonsense_qa, ai2_arc,
          qasc, piqa, social_i_qa, winogrande, hellaswag, boolq,
          natural_questions (if available)
"""

import json, os, re, subprocess, sys, time, threading
from pathlib import Path

# -- constants --------------------------------------------------------------
BRAIN_FILE       = str(Path(__file__).parent / "lu4d_curriculum.lbr")
LU4D_EXE         = str(Path(__file__).parent / "target" / "release" / "lu4d.exe")

# Size limits: use marginal bytes per NEW bond (not total) so node overhead
# is not double-counted. Calibrated at: 994,986 bonds / 1.80 GB base.
BASE_BONDS             = 994_986        # bonds in the loaded brain at start
BASE_BYTES             = 1_934_177_083  # actual disk file size at start
MARGINAL_BYTES_PER_BOND = 500           # bytes per additional bond (empirical)
SOFT_LIMIT_BYTES       = int(3.5 * 1024 ** 3)  # stop this dataset, try next
HARD_LIMIT_BYTES       = int(3.8 * 1024 ** 3)  # abort all datasets, eval only

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","need","dare","ought","used",
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","mine","yours","hers","ours","theirs",
    "this","that","these","those","who","which","what","where","when","why","how",
    "all","both","each","every","few","more","most","other","some","such",
    "no","not","only","same","so","than","too","very","just","but","or","and",
    "if","in","of","to","for","on","at","by","with","about","from","into",
    "through","during","before","after","above","below","between","up","down",
    "out","off","over","under","again","further","then","once","here","there",
    "any","as","its","also","like","well","one","two","three","four","five",
    "many","much","even","still","now","back","long","made","make","got","get",
    "go","went","come","came","take","took","see","saw","know","knew","use",
    "said","say","think","thought","want","give","gave","their","they","them",
    "new","old","first","last","part","way","day","time","year","man","men",
    "life","world","people","place","something","someone","another","often",
    "always","never","since","while","after","before","later","early",
    "different","same","large","small","own","those","these","such","each",
    "every","both","few","more","most","other","some","per","set","lot",
}

# -- MCP client -------------------------------------------------------------
class MCPClient:
    def __init__(self, brain_file, no_save=True):
        self.brain_file = brain_file
        self.no_save    = no_save
        self.proc       = None
        self._lock      = threading.Lock()
        self._req_id    = 0

    def _drain_stderr(self):
        """Background thread: print lu4d stderr so save errors are visible."""
        try:
            for raw in self.proc.stderr:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    print(f"[lu4d] {line}", flush=True)
        except Exception:
            pass

    def start(self):
        cmd = [LU4D_EXE, "--mcp", "--brain", self.brain_file]
        if self.no_save:
            cmd.append("--no-save")
        print(f"[mcp] spawning: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def stop(self):
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                print("[mcp] WARNING: lu4d.exe did not exit in 300s, killing", flush=True)
                self.proc.kill()
                self.proc.wait()
            self.proc = None
            # verify the brain file was actually updated
            try:
                sz   = os.path.getsize(self.brain_file)
                mtime = os.path.getmtime(self.brain_file)
                print(f"[mcp] stopped. brain file: {sz:,} bytes "
                      f"(mtime={time.strftime('%H:%M:%S', time.localtime(mtime))})",
                      flush=True)
            except Exception as e:
                print(f"[mcp] stopped. could not stat brain file: {e}", flush=True)

    def _send(self, method, params):
        with self._lock:
            self._req_id += 1
            rid = self._req_id
            msg = json.dumps({
                "jsonrpc": "2.0", "id": rid,
                "method": method, "params": params,
            }) + "\n"
            self.proc.stdin.write(msg.encode())
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("MCP EOF")
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("id") == rid:
                    return obj

    def call_tool(self, tool, args):
        resp = self._send("tools/call", {"name": tool, "arguments": args})
        if "error" in resp:
            raise RuntimeError(resp["error"])
        content    = resp.get("result", {}).get("content", [])
        structured = resp.get("result", {}).get("structuredContent", {})
        text = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
        # Always merge text so both structured fields AND r.get("text") work
        if structured:
            return {**structured, "text": text}
        return {"text": text}

# -- helpers ----------------------------------------------------------------
_clean_re = re.compile(r"[^a-z0-9\-]")

def normalize(w):
    return _clean_re.sub("", w.lower().strip())

def is_single_token(w):
    if not w or " " in w:
        return False
    n = normalize(w)
    return 2 <= len(n) <= 30 and not n.isdigit() and bool(n)

def content_tokens(text, max_toks=40):
    if not text:
        return []
    tokens = []
    for raw in re.split(r"[\s\-_/,;:.!?()\[\]\"']+", str(text).lower()):
        n = normalize(raw)
        if n and n not in STOPWORDS and is_single_token(n):
            tokens.append(n)
            if len(tokens) >= max_toks:
                break
    return tokens

def teach_triples_gen(client, gen, cache, name, size_check_every=50_000):
    """Process triples lazily from a generator.

    Returns (taught, skipped, errors, stop_reason) where stop_reason is
    None, "soft" (3.5GB -- caller should skip this dataset but continue),
    or "hard" (3.8GB -- caller should abort all remaining datasets).
    """
    taught = skipped = errors = 0
    stop_reason = None
    t0 = time.time()
    for item in gen:
        subj, rel, obj, weight = item
        if not (is_single_token(subj) and is_single_token(obj)):
            continue
        key_f = (subj, rel, obj)
        if key_f in cache:
            skipped += 1
            continue
        cache.add(key_f)
        cache.add((obj, "RelatedTo", subj))
        try:
            client.call_tool("lu4d_teach", {
                "subject": subj, "relation": rel, "object": obj,
                "weight": float(weight), "bidirectional": True,
            })
            taught += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [err] {e}", flush=True)

        if taught % 10_000 == 0 and taught > 0:
            rate = taught / max(time.time() - t0, 0.001)
            print(f"[{name}] taught={taught:,} skipped={skipped:,} "
                  f"({rate:.0f}/s, {time.time()-t0:.1f}s)", flush=True)

        # periodic mid-stream size check
        if taught % size_check_every == 0 and taught > 0:
            sz_status, _, _ = check_size(client, f"{name}@{taught//1000}k")
            if sz_status == "hard":
                stop_reason = "hard"
                break
            elif sz_status == "soft":
                stop_reason = "soft"
                break

    elapsed = time.time() - t0
    rate    = taught / max(elapsed, 0.001)
    print(f"[{name}] DONE taught={taught:,} skipped={skipped:,} "
          f"errors={errors} in {elapsed:.1f}s ({rate:.0f}/s)", flush=True)
    if stop_reason:
        print(f"[{name}] stopped early: {stop_reason} size limit hit", flush=True)
    return taught, skipped, errors, stop_reason

def check_size(client, label):
    """Return ("ok"|"soft"|"hard", bonds, nodes).

    Uses marginal-cost formula: BASE_BYTES + new_bonds * MARGINAL_BYTES_PER_BOND.
    This avoids over-counting fixed node overhead on every bond check.
    """
    r         = client.call_tool("lu4d_status", {})
    bonds     = r.get("bond_count", 0)
    nodes     = r.get("node_count", 0)
    new_bonds = max(0, bonds - BASE_BONDS)
    proj      = BASE_BYTES + new_bonds * MARGINAL_BYTES_PER_BOND
    proj_gb   = proj / 1024 ** 3
    if proj >= HARD_LIMIT_BYTES:
        status = "hard"
    elif proj >= SOFT_LIMIT_BYTES:
        status = "soft"
    else:
        status = "ok"
    remain_gb = (HARD_LIMIT_BYTES - proj) / 1024 ** 3
    print(f"[size] {label}: nodes={nodes:,} bonds={bonds:,} "
          f"new={new_bonds:,} proj={proj_gb:.2f}GB "
          f"to_hard={remain_gb:.2f}GB [{status}]", flush=True)
    return status, bonds, nodes

def status_print(client, label):
    r     = client.call_tool("lu4d_status", {})
    nodes = r.get("node_count", 0)
    bonds = r.get("bond_count", 0)
    print(f"[status] {label}: nodes={nodes:,} bonds={bonds:,}", flush=True)
    return nodes, bonds

def consolidate(client):
    r        = client.call_tool("lu4d_consolidate", {})
    promoted = r.get("promoted", 0) if isinstance(r, dict) else 0
    pruned   = r.get("pruned",   0) if isinstance(r, dict) else 0
    examined = r.get("examined", 0) if isinstance(r, dict) else 0
    print(f"[consolidate] promoted={promoted} pruned={pruned} examined={examined}", flush=True)

# -- triple builder ---------------------------------------------------------
def qa_triples(question, answers, context=None, weight=1.0):
    """Yield (subj, rel, obj, weight) from a QA pair."""
    ans_toks = []
    for a in answers:
        for t in content_tokens(str(a), max_toks=8):
            if t not in ans_toks:
                ans_toks.append(t)
    ans_toks = ans_toks[:8]

    q_toks = content_tokens(str(question), max_toks=20)

    # answer tokens <-> question tokens
    for a in ans_toks:
        for q in q_toks:
            if a != q:
                yield (a, "RelatedTo", q, weight)

    # consecutive answer token pairs
    for i in range(len(ans_toks) - 1):
        yield (ans_toks[i], "RelatedTo", ans_toks[i + 1], weight)

    # consecutive question token pairs (lower weight)
    for i in range(len(q_toks) - 1):
        if q_toks[i] != q_toks[i + 1]:
            yield (q_toks[i], "RelatedTo", q_toks[i + 1], weight * 0.7)

    # context tokens connect to answer tokens
    if context:
        ctx_toks = content_tokens(str(context), max_toks=25)
        for a in ans_toks:
            for c in ctx_toks:
                if a != c:
                    yield (a, "RelatedTo", c, weight * 0.6)

# -- dataset streamers ------------------------------------------------------

def stream_trivia_qa(limit=500_000):
    try:
        from datasets import load_dataset
        ds = load_dataset("trivia_qa", "rc", split="train", streaming=True, trust_remote_code=False)
    except Exception as e:
        print(f"[trivia_qa] unavailable: {e}", flush=True)
        return
    print(f"[trivia_qa] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        q       = row.get("question", "")
        ans_val = row.get("answer", {})
        if isinstance(ans_val, dict):
            main_ans = ans_val.get("value", "")
            aliases  = ans_val.get("aliases", [])[:3]
        else:
            main_ans = str(ans_val)
            aliases  = []
        answers = [main_ans] + list(aliases)
        ctx = ""
        srcs = row.get("search_results", {})
        if isinstance(srcs, dict):
            contexts = srcs.get("search_context", [])
            if contexts:
                ctx = str(contexts[0])[:400]
        yield from qa_triples(q, answers, ctx, weight=1.0)
        count += 1

def stream_sciq(limit=20_000):
    try:
        from datasets import load_dataset
        ds = load_dataset("sciq", split="train", streaming=True, trust_remote_code=False)
    except Exception as e:
        print(f"[sciq] unavailable: {e}", flush=True)
        return
    print(f"[sciq] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        q   = row.get("question", "")
        ans = [row.get("correct_answer", "")]
        ctx = row.get("support", "")[:400]
        yield from qa_triples(q, ans, ctx, weight=1.0)
        count += 1

def stream_squad(limit=100_000):
    ds = None
    for repo in ("rajpurkar/squad_v2", "squad_v2"):
        try:
            from datasets import load_dataset
            ds = load_dataset(repo, split="train", streaming=True, trust_remote_code=False)
            break
        except Exception:
            pass
    if ds is None:
        print("[squad] unavailable", flush=True)
        return
    print(f"[squad] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        q       = row.get("question", "")
        answers = row.get("answers", {})
        texts   = answers.get("text", []) if isinstance(answers, dict) else []
        ctx     = row.get("context", "")[:400]
        if not texts:
            count += 1
            continue
        yield from qa_triples(q, texts, ctx, weight=1.0)
        count += 1

def stream_openbookqa(limit=10_000):
    ds = None
    for repo, cfg in [("allenai/openbookqa", "main"), ("openbookqa", None)]:
        try:
            from datasets import load_dataset
            kw = {"split": "train", "streaming": True, "trust_remote_code": False}
            ds = load_dataset(repo, cfg, **kw) if cfg else load_dataset(repo, **kw)
            break
        except Exception:
            pass
    if ds is None:
        print("[openbookqa] unavailable", flush=True)
        return
    print(f"[openbookqa] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        q      = row.get("question_stem", "")
        label  = row.get("answerKey", "")
        choices = row.get("choices", {})
        labels  = choices.get("label", []) if isinstance(choices, dict) else []
        texts   = choices.get("text",  []) if isinstance(choices, dict) else []
        ans = []
        if label and label in labels:
            try:
                ans = [texts[labels.index(label)]]
            except (ValueError, IndexError):
                pass
        ctx = row.get("fact1", "")
        if not ans:
            count += 1
            continue
        yield from qa_triples(q, ans, ctx, weight=1.0)
        count += 1

def stream_commonsense_qa(limit=20_000):
    try:
        from datasets import load_dataset
        ds = load_dataset("commonsense_qa", split="train", streaming=True, trust_remote_code=False)
    except Exception as e:
        print(f"[commonsense_qa] unavailable: {e}", flush=True)
        return
    print(f"[commonsense_qa] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        q      = row.get("question", "")
        label  = row.get("answerKey", "")
        choices = row.get("choices", {})
        labels  = choices.get("label", []) if isinstance(choices, dict) else []
        texts   = choices.get("text",  []) if isinstance(choices, dict) else []
        ans = []
        if label and label in labels:
            try:
                ans = [texts[labels.index(label)]]
            except (ValueError, IndexError):
                pass
        if not ans:
            count += 1
            continue
        yield from qa_triples(q, ans, weight=1.0)
        count += 1

def stream_ai2_arc(config="ARC-Challenge", limit=10_000):
    ds = None
    for repo in ("allenai/ai2_arc", "ai2_arc"):
        try:
            from datasets import load_dataset
            ds = load_dataset(repo, config, split="train", streaming=True, trust_remote_code=False)
            break
        except Exception:
            pass
    if ds is None:
        print(f"[ai2_arc/{config}] unavailable", flush=True)
        return
    print(f"[ai2_arc/{config}] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        q      = row.get("question", "")
        label  = row.get("answerKey", "")
        choices = row.get("choices", {})
        labels  = choices.get("label", []) if isinstance(choices, dict) else []
        texts   = choices.get("text",  []) if isinstance(choices, dict) else []
        ans = []
        if label and label in labels:
            try:
                ans = [texts[labels.index(label)]]
            except (ValueError, IndexError):
                pass
        if not ans:
            count += 1
            continue
        yield from qa_triples(q, ans, weight=1.0)
        count += 1

def stream_qasc(limit=10_000):
    try:
        from datasets import load_dataset
        ds = load_dataset("qasc", split="train", streaming=True, trust_remote_code=False)
    except Exception as e:
        print(f"[qasc] unavailable: {e}", flush=True)
        return
    print(f"[qasc] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        q      = row.get("question", "")
        label  = row.get("answerKey", "")
        choices = row.get("choices", {})
        labels  = choices.get("label", []) if isinstance(choices, dict) else []
        texts   = choices.get("text",  []) if isinstance(choices, dict) else []
        ans = []
        if label and label in labels:
            try:
                ans = [texts[labels.index(label)]]
            except (ValueError, IndexError):
                pass
        ctx = row.get("fact1", "") + " " + row.get("fact2", "")
        if not ans:
            count += 1
            continue
        yield from qa_triples(q, ans, ctx, weight=1.0)
        count += 1

def stream_piqa(limit=20_000):
    try:
        from datasets import load_dataset
        ds = load_dataset("piqa", split="train", streaming=True, trust_remote_code=False)
    except Exception as e:
        print(f"[piqa] unavailable: {e}", flush=True)
        return
    print(f"[piqa] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        goal  = row.get("goal", "")
        label = row.get("label", -1)
        sol1  = row.get("sol1", "")
        sol2  = row.get("sol2", "")
        ans   = [sol1 if label == 0 else sol2]
        if not ans[0]:
            count += 1
            continue
        yield from qa_triples(goal, ans, weight=1.0)
        count += 1

def stream_social_iqa(limit=30_000):
    try:
        from datasets import load_dataset
        ds = load_dataset("social_i_qa", split="train", streaming=True, trust_remote_code=False)
    except Exception as e:
        print(f"[social_i_qa] unavailable: {e}", flush=True)
        return
    print(f"[social_i_qa] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        ctx_txt = row.get("context", "")
        q       = row.get("question", "")
        label   = str(row.get("label", ""))
        ans_key = {"1": "answerA", "2": "answerB", "3": "answerC"}.get(label, "")
        ans     = [row.get(ans_key, "")] if ans_key else []
        if not ans or not ans[0]:
            count += 1
            continue
        yield from qa_triples(ctx_txt + " " + q, ans, weight=1.0)
        count += 1

def stream_winogrande(limit=40_000):
    try:
        from datasets import load_dataset
        ds = load_dataset("winogrande", "winogrande_xl", split="train",
                          streaming=True, trust_remote_code=False)
    except Exception as e:
        print(f"[winogrande] unavailable: {e}", flush=True)
        return
    print(f"[winogrande] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        sentence = row.get("sentence", "")
        answer   = str(row.get("answer", ""))
        opt1     = row.get("option1", "")
        opt2     = row.get("option2", "")
        ans      = [opt1 if answer == "1" else opt2]
        if not ans[0]:
            count += 1
            continue
        yield from qa_triples(sentence, ans, weight=0.8)
        count += 1

def stream_hellaswag(limit=40_000):
    ds = None
    for repo in ("Rowan/hellaswag", "hellaswag"):
        try:
            from datasets import load_dataset
            ds = load_dataset(repo, split="train", streaming=True, trust_remote_code=False)
            break
        except Exception:
            pass
    if ds is None:
        print("[hellaswag] unavailable", flush=True)
        return
    print(f"[hellaswag] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        ctx     = row.get("ctx", "")
        label   = row.get("label", "")
        endings = row.get("endings", [])
        if label == "" or not endings:
            count += 1
            continue
        try:
            ans = [endings[int(label)]]
        except (ValueError, IndexError):
            count += 1
            continue
        yield from qa_triples(ctx, ans, weight=0.8)
        count += 1

def stream_boolq(limit=10_000):
    ds = None
    for repo in ("google/boolq", "boolq"):
        try:
            from datasets import load_dataset
            ds = load_dataset(repo, split="train", streaming=True, trust_remote_code=False)
            break
        except Exception:
            pass
    if ds is None:
        print("[boolq] unavailable", flush=True)
        return
    print(f"[boolq] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        q   = row.get("question", "")
        ans = ["true" if row.get("answer", False) else "false"]
        ctx = str(row.get("passage", ""))[:300]
        yield from qa_triples(q, ans, ctx, weight=0.8)
        count += 1

def stream_natural_questions(limit=50_000):
    ds = None
    for repo, cfg in [("google-research-datasets/natural_questions", "default"),
                      ("natural_questions", None)]:
        try:
            from datasets import load_dataset
            kw = {"split": "train", "streaming": True, "trust_remote_code": False}
            ds = load_dataset(repo, cfg, **kw) if cfg else load_dataset(repo, **kw)
            break
        except Exception:
            pass
    if ds is None:
        print("[natural_questions] unavailable", flush=True)
        return
    print(f"[natural_questions] streaming (limit={limit:,})...", flush=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        q_field = row.get("question", {})
        q       = q_field.get("text", "") if isinstance(q_field, dict) else str(q_field)
        annotations = row.get("annotations", {})
        short_ans = []
        if isinstance(annotations, dict):
            sa = annotations.get("short_answers", [])
            for ans_list in (sa[:1] if sa else []):
                if isinstance(ans_list, list):
                    for a in ans_list[:3]:
                        txt = a.get("text", "") if isinstance(a, dict) else str(a)
                        if txt:
                            short_ans.append(txt)
                elif isinstance(ans_list, str) and ans_list:
                    short_ans.append(ans_list)
        if not short_ans:
            count += 1
            continue
        ctx = ""
        doc = row.get("document", {})
        if isinstance(doc, dict):
            tok_field = doc.get("tokens", {})
            if isinstance(tok_field, dict):
                toks = tok_field.get("token", [])
                ctx  = " ".join(str(t) for t in toks[:100])
        yield from qa_triples(q, short_ans, ctx, weight=1.0)
        count += 1

# -- 100-question eval ------------------------------------------------------
EVAL_100 = [
    # geography (10)
    ("What is the capital of France?",           ["paris"]),
    ("What is the capital of Japan?",            ["tokyo"]),
    ("What is the capital of Germany?",          ["berlin"]),
    ("What is the capital of Australia?",        ["canberra"]),
    ("What is the capital of Brazil?",           ["brasilia"]),
    ("What is the Nile?",                        ["river"]),
    ("What is the Amazon?",                      ["river", "forest"]),
    ("What is Mount Everest?",                   ["mountain"]),
    ("What is the Sahara?",                      ["desert"]),
    ("What is the Pacific?",                     ["ocean"]),
    # science (10)
    ("What is photosynthesis?",                  ["plant", "light", "energy"]),
    ("What is DNA?",                             ["genetic", "molecule"]),
    ("What is gravity?",                         ["force", "mass"]),
    ("What is evolution?",                       ["biology", "darwin"]),
    ("What is electricity?",                     ["energy", "charge"]),
    ("What is the speed of light?",              ["light", "speed"]),
    ("What is an atom?",                         ["particle", "element"]),
    ("What is a black hole?",                    ["space", "gravity"]),
    ("What is relativity?",                      ["einstein", "physics"]),
    ("What is the periodic table?",              ["chemistry", "element"]),
    # math (5)
    ("What is calculus?",                        ["math", "derivative"]),
    ("What is a prime number?",                  ["number", "divisible"]),
    ("What is the Pythagorean theorem?",         ["triangle", "math"]),
    ("What is probability?",                     ["math", "chance"]),
    ("What is statistics?",                      ["math", "data"]),
    # history (10)
    ("Who was Napoleon?",                        ["france", "emperor"]),
    ("What was World War II?",                   ["war", "germany"]),
    ("Who was Abraham Lincoln?",                 ["president", "usa"]),
    ("What was the Renaissance?",                ["movement", "culture"]),
    ("Who was Cleopatra?",                       ["egypt", "queen"]),
    ("What was the Cold War?",                   ["usa", "ussr"]),
    ("Who was Julius Caesar?",                   ["rome", "emperor"]),
    ("What was the French Revolution?",          ["france", "revolution"]),
    ("Who was Genghis Khan?",                    ["mongol", "warrior"]),
    ("What was the Industrial Revolution?",      ["industry", "machine"]),
    # technology (10)
    ("What is Python?",                          ["programming", "language"]),
    ("What is the internet?",                    ["network", "computer"]),
    ("What is machine learning?",                ["ai", "learning"]),
    ("What is a database?",                      ["data", "storage"]),
    ("What is encryption?",                      ["security", "data"]),
    ("What is artificial intelligence?",         ["ai", "machine"]),
    ("What is a computer?",                      ["machine", "computing"]),
    ("What is a transistor?",                    ["electronics", "semiconductor"]),
    ("What is an algorithm?",                    ["computing", "steps"]),
    ("What is cloud computing?",                 ["computing", "internet"]),
    # practical (5)
    ("What is cooking?",                         ["food", "heat"]),
    ("What is medicine?",                        ["health", "treatment"]),
    ("What is law?",                             ["legal", "rules"]),
    ("What is economics?",                       ["money", "market"]),
    ("What is psychology?",                      ["mind", "behavior"]),
    # people (10)
    ("Who was Albert Einstein?",                 ["physicist", "relativity"]),
    ("Who was Marie Curie?",                     ["scientist", "physics"]),
    ("Who was Isaac Newton?",                    ["physicist", "gravity"]),
    ("Who was Leonardo da Vinci?",               ["artist", "inventor"]),
    ("Who was William Shakespeare?",             ["writer", "playwright"]),
    ("Who was Charles Darwin?",                  ["biologist", "evolution"]),
    ("Who was Nikola Tesla?",                    ["inventor", "electricity"]),
    ("Who was Thomas Edison?",                   ["inventor", "electricity"]),
    ("Who was Socrates?",                        ["philosopher", "greece"]),
    ("Who was Galileo?",                         ["astronomer", "scientist"]),
    # modern world (10)
    ("What is Apple?",                           ["company", "tech"]),
    ("What is Google?",                          ["company", "tech", "search"]),
    ("What is Amazon?",                          ["company", "ecommerce"]),
    ("What is Tesla?",                           ["company", "electric"]),
    ("What is Netflix?",                         ["company", "streaming"]),
    ("What is Facebook?",                        ["company", "social"]),
    ("What is Microsoft?",                       ["company", "software"]),
    ("What is SpaceX?",                          ["company", "space"]),
    ("What is Bitcoin?",                         ["cryptocurrency", "digital"]),
    ("What is the Olympics?",                    ["sports", "competition"]),
    # language (5)
    ("What is a synonym?",                       ["word", "similar"]),
    ("What is grammar?",                         ["language", "rules"]),
    ("What is a metaphor?",                      ["language", "comparison"]),
    ("What is poetry?",                          ["writing", "art"]),
    ("What is rhetoric?",                        ["language", "persuasion"]),
    # cross-domain (10)
    ("What is democracy?",                       ["government", "voting"]),
    ("What is capitalism?",                      ["economic", "market"]),
    ("What is climate change?",                  ["environment", "temperature"]),
    ("What is biodiversity?",                    ["biology", "species"]),
    ("What is the United Nations?",              ["organization", "international"]),
    ("What is philosophy?",                      ["knowledge", "reasoning"]),
    ("What is art?",                             ["creative", "expression"]),
    ("What is music?",                           ["sound", "art"]),
    ("What is culture?",                         ["society", "tradition"]),
    ("What is religion?",                        ["belief", "spiritual"]),
    # science II (5)
    ("What is chemistry?",                       ["science", "element"]),
    ("What is physics?",                         ["science", "force"]),
    ("What is biology?",                         ["science", "life"]),
    ("What is astronomy?",                       ["science", "space"]),
    ("What is geology?",                         ["science", "earth"]),
    # QA trivia (10)
    ("Who wrote Hamlet?",                        ["shakespeare"]),
    ("What is the largest planet?",              ["jupiter"]),
    ("What is water made of?",                   ["hydrogen", "oxygen"]),
    ("What is the speed of sound?",              ["sound", "speed"]),
    ("Who invented the telephone?",              ["bell", "graham"]),
    ("What is the human genome?",                ["dna", "genetics"]),
    ("What is a vaccine?",                       ["medicine", "immune"]),
    ("What is a photon?",                        ["light", "particle"]),
    ("What is osmosis?",                         ["biology", "membrane"]),
    ("What is the Fibonacci sequence?",          ["math", "numbers"]),
]

assert len(EVAL_100) == 100, f"Expected 100 questions, got {len(EVAL_100)}"

# -- eval runners -----------------------------------------------------------
def run_sanity(client, questions, name, top_k=15):
    print(f"\n{'='*68}", flush=True)
    print(f"{name} ({len(questions)} questions)", flush=True)
    print(f"{'='*68}", flush=True)
    hits = 0
    for i, (q, expected) in enumerate(questions, 1):
        try:
            r    = client.call_tool("lu4d_ask", {"query": q, "top_k": top_k})
            text = (r.get("text", "") if isinstance(r, dict) else str(r)).lower()
            found = any(e.lower() in text for e in expected)
            if found:
                hits += 1
            if i <= 5 or not found:
                mark = "+" if found else "-"
                print(f"  [{mark}] Q{i:03d}: {q}", flush=True)
                if not found:
                    print(f"        expected: {expected}", flush=True)
                    print(f"        got: {text[:120]}", flush=True)
        except Exception as e:
            print(f"  [!] Q{i:03d}: {e}", flush=True)
    pct = 100 * hits / len(questions)
    print(f"\n[{name}] score: {hits}/{len(questions)} ({pct:.1f}%)", flush=True)
    return hits, len(questions)


def run_temporal_scope(client):
    print(f"\n{'='*68}", flush=True)
    print("TEMPORAL SCOPE TEST", flush=True)
    print(f"{'='*68}", flush=True)

    SENT_A = "zorbaxite27"
    SENT_B = "vorzinium99"
    passed = total = 0

    def ask(q):
        r = client.call_tool("lu4d_ask", {"query": q, "top_k": 5})
        return (r.get("text", "") if isinstance(r, dict) else str(r)).lower()

    def sess_start():
        r = client.call_tool("lu4d_session_start", {})
        sid = r.get("session_id", r.get("text", "?")) if isinstance(r, dict) else "?"
        print(f"  [session] start -> id={sid}", flush=True)
        return sid

    def sess_end():
        client.call_tool("lu4d_session_end", {})
        print("  [session] ended", flush=True)

    def check(cond, label):
        nonlocal passed, total
        total += 1
        if cond:
            passed += 1
            print(f"  [+] {label}", flush=True)
        else:
            print(f"  [-] FAIL: {label}", flush=True)

    # pre-session: sentinel unknown
    resp = ask(SENT_A)
    check(SENT_A not in resp, f"pre-session: {SENT_A} is unknown")

    # session 1: teach SENT_A
    sess_start()
    client.call_tool("lu4d_teach", {
        "subject": SENT_A, "relation": "IsA", "object": "mineral",
        "weight": 1.0, "bidirectional": True,
    })
    resp = ask(SENT_A)
    check(SENT_A in resp or "mineral" in resp, f"during-session1: {SENT_A} visible")
    sess_end()

    resp = ask(SENT_A)
    check(SENT_A in resp or "mineral" in resp, f"post-session1: {SENT_A} persists")

    # session 2: teach SENT_B; SENT_A should still be accessible
    sess_start()
    client.call_tool("lu4d_teach", {
        "subject": SENT_B, "relation": "IsA", "object": "element",
        "weight": 1.0, "bidirectional": True,
    })
    resp = ask(SENT_B)
    check(SENT_B in resp or "element" in resp, f"during-session2: {SENT_B} visible")

    resp = ask(SENT_A)
    check(SENT_A in resp or "mineral" in resp, f"cross-session: {SENT_A} accessible in session2")
    sess_end()

    print(f"\n[temporal] score: {passed}/{total}", flush=True)
    return passed, total


def run_timetravel(client):
    print(f"\n{'='*68}", flush=True)
    print("TIME TRAVEL DIFFS", flush=True)
    print(f"{'='*68}", flush=True)
    for query, ms_ago in [
        ("python",      3_600_000),
        ("einstein",    7_200_000),
        ("democracy",   3_600_000),
        ("transformer", 1_800_000),
    ]:
        try:
            r    = client.call_tool("lu4d_timetravel", {"query": query, "ms_ago": ms_ago})
            text = r.get("text", "") if isinstance(r, dict) else str(r)
            print(f"  [tt] {query!r} ms_ago={ms_ago:,} -> {text[:120]}", flush=True)
        except Exception as e:
            print(f"  [tt] {query!r} ERROR: {e}", flush=True)

# -- main -------------------------------------------------------------------
def main():
    print(f"\n{'#'*70}", flush=True)
    print(f"# phase_hq.py -- Known-Working Dataset Training", flush=True)
    print(f"# Target: 500K+ nodes, stay under 3.8GB", flush=True)
    print(f"{'#'*70}\n", flush=True)

    # no_save=True: lu4d.exe skips per-call autosaves (fast training).
    # main.rs always calls brain.save() once after the MCP loop exits.
    client = MCPClient(BRAIN_FILE, no_save=True)
    client.start()
    taught_cache = set()

    nodes0, bonds0 = status_print(client, "initial")

    DATASETS = [
        ("trivia_qa",           stream_trivia_qa,       {"limit": 100_000}),
        ("sciq",                stream_sciq,             {"limit": 20_000}),
        ("squad_v2",            stream_squad,            {"limit": 100_000}),
        ("openbookqa",          stream_openbookqa,       {"limit": 10_000}),
        ("commonsense_qa",      stream_commonsense_qa,   {"limit": 20_000}),
        ("ai2_arc_challenge",   stream_ai2_arc,          {"config": "ARC-Challenge", "limit": 10_000}),
        ("ai2_arc_easy",        stream_ai2_arc,          {"config": "ARC-Easy",      "limit": 10_000}),
        ("qasc",                stream_qasc,             {"limit": 10_000}),
        ("piqa",                stream_piqa,             {"limit": 20_000}),
        ("social_i_qa",         stream_social_iqa,       {"limit": 30_000}),
        ("winogrande",          stream_winogrande,       {"limit": 40_000}),
        ("hellaswag",           stream_hellaswag,        {"limit": 40_000}),
        ("boolq",               stream_boolq,            {"limit": 10_000}),
        ("natural_questions",   stream_natural_questions, {"limit": 50_000}),
    ]

    results  = {}
    hard_stop = False  # set when projected >= 3.8 GB

    for ds_name, stream_fn, kwargs in DATASETS:
        if hard_stop:
            print(f"\n[SKIP] {ds_name} -- hard size limit reached", flush=True)
            results[ds_name] = 0
            continue

        print(f"\n{'-'*68}", flush=True)
        print(f"[dataset] {ds_name}", flush=True)
        stop_reason = None
        try:
            gen = stream_fn(**kwargs)
            total, _sk, _er, stop_reason = teach_triples_gen(
                client, gen, taught_cache, ds_name)
        except Exception as e:
            print(f"[{ds_name}] ERROR: {e}", flush=True)
            total = 0
        results[ds_name] = total
        consolidate(client)
        post_status, _, _ = check_size(client, f"post-{ds_name}")

        if stop_reason == "hard" or post_status == "hard":
            hard_stop = True
            print(f"\n[HARD STOP] 3.8GB limit. Skipping remaining datasets.", flush=True)
        elif stop_reason == "soft" or post_status == "soft":
            print(f"\n[SOFT STOP] 3.5GB limit after {ds_name}. Continuing to next dataset.", flush=True)
            # soft stop: current dataset truncated, but keep running others

    # final status
    nodes_f, bonds_f = status_print(client, "final")

    # evaluation
    print(f"\n{'#'*70}", flush=True)
    print("# EVALUATION", flush=True)
    print(f"{'#'*70}", flush=True)

    eval_hits, eval_tot = run_sanity(client, EVAL_100, "100-Question Cross-Domain Eval")
    temp_pass, temp_tot = run_temporal_scope(client)
    run_timetravel(client)

    # summary
    print(f"\n{'='*68}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*68}", flush=True)
    print(f"  initial : nodes={nodes0:,}  bonds={bonds0:,}", flush=True)
    print(f"  final   : nodes={nodes_f:,}  bonds={bonds_f:,}", flush=True)
    print(f"  gained  : nodes={nodes_f - nodes0:,}  bonds={bonds_f - bonds0:,}", flush=True)
    proj_gb = bonds_f * BYTES_PER_BOND / 1024 ** 3
    print(f"  projected file size: {proj_gb:.2f}GB", flush=True)
    print(f"  eval score : {eval_hits}/{eval_tot} ({100*eval_hits/eval_tot:.1f}%)", flush=True)
    print(f"  temporal   : {temp_pass}/{temp_tot}", flush=True)
    print(f"\n  dataset breakdown:", flush=True)
    for k, v in results.items():
        status = "OK" if v > 0 else "SKIP/FAIL"
        print(f"    {k:<22} {v:>10,} teaches  [{status}]", flush=True)

    client.stop()
    print("\n[done] phase_hq.py complete.", flush=True)


if __name__ == "__main__":
    main()

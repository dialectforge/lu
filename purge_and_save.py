import subprocess, json, sys, os, time, threading

BRAIN = r".\lu4d_curriculum.lbr"
EXE   = r".\target\release\lu4d.exe"

SANITY_QUESTIONS = [
    ("mercury",       ["planet", "solar", "smallest", "sun", "orbit"]),
    ("pacific",       ["ocean", "largest", "water", "deep"]),
    ("shakespeare",   ["hamlet", "playwright", "english", "romeo", "bard", "literature"]),
    ("mitochondria",  ["energy", "atp", "cell", "powerhouse"]),
    ("diamond",       ["carbon", "hard", "mineral", "gem"]),
    ("jupiter",       ["planet", "largest", "gas", "solar", "galileo"]),
    ("berlin",        ["germany", "capital", "city", "german"]),
    ("tokyo",         ["japan", "capital", "city", "japanese"]),
    ("einstein",      ["relativity", "physics", "theory", "physicist", "energy"]),
    ("newton",        ["gravity", "physics", "motion", "apple", "law"]),
    ("gravity",       ["force", "mass", "newton", "attraction", "earth"]),
    ("evolution",     ["darwin", "species", "natural", "selection", "biology"]),
    ("dna",           ["gene", "helix", "chromosome", "genetic", "molecule"]),
    ("photosynthesis",["light", "chlorophyll", "plant", "glucose", "sun"]),
    ("python",        ["language", "programming", "code", "snake", "guido"]),
    ("entropy",       ["thermodynamics", "disorder", "energy", "heat"]),
    ("democracy",     ["vote", "election", "government", "people", "republic"]),
    ("canberra",      ["australia", "capital", "city", "australian"]),
    ("darwin",        ["evolution", "species", "natural", "selection", "biology", "australia"]),
    ("relativity",    ["einstein", "space", "time", "light", "mass", "energy"]),
]

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
        # initialize
        self._send({"jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05",
                               "capabilities": {},
                               "clientInfo": {"name": "purge", "version": "1.0"}}})
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
        data = json.dumps(obj) + "\n"
        self.proc.stdin.write(data.encode())
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

    def call_tool(self, name, args=None):
        rid = self._next_id()
        self._send({"jsonrpc": "2.0", "id": rid,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": args or {}}})
        resp = self._recv()
        if resp is None:
            return {}
        if "error" in resp:
            print(f"[error] {name}: {resp['error']}", flush=True)
            return {}
        result = resp.get("result", {})
        content = result.get("content", [])
        text = ""
        structured = {}
        for c in content:
            if c.get("type") == "text":
                text += c.get("text", "")
            if c.get("type") == "json":
                structured = c.get("json", {})
        if result.get("structuredContent"):
            structured = result["structuredContent"]
        if structured:
            return {**structured, "text": text}
        return {"text": text}

    def stop(self):
        if self.proc:
            try: self.proc.stdin.close()
            except Exception: pass
            print("[mcp] waiting for lu4d.exe to save and exit...", flush=True)
            try:
                self.proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                print("[mcp] WARNING: lu4d.exe did not exit in 300s, killing", flush=True)
                self.proc.kill(); self.proc.wait()
            self.proc = None


def print_status(label, r):
    nc = r.get("node_count", r.get("nodes", "?"))
    bc = r.get("bond_count", r.get("bonds", "?"))
    print(f"[status] {label}: nodes={nc:,} bonds={bc:,}" if isinstance(nc, int) else
          f"[status] {label}: {r.get('text','')}", flush=True)


def score_response(r, keywords):
    text = (r.get("text", "") + " " + str(r.get("activated_nodes", ""))).lower()
    if "activated: 0" in text:
        return 0
    for kw in keywords:
        if kw.lower() in text:
            return 1
    return 0


def main():
    client = MCPClient(BRAIN)
    client.start()

    # --- BEFORE status ---
    print("\n--- BEFORE PURGE ---", flush=True)
    r_before = client.call_tool("lu4d_status", {})
    print_status("before", r_before)

    # --- PURGE ---
    print("\n--- PURGING STOPWORDS ---", flush=True)
    r_purge = client.call_tool("lu4d_purge_stopwords", {})
    print(f"[purge] {r_purge.get('text', str(r_purge))}", flush=True)

    # --- AFTER status ---
    print("\n--- AFTER PURGE ---", flush=True)
    r_after = client.call_tool("lu4d_status", {})
    print_status("after", r_after)

    # --- EXPLICIT SAVE (1) ---
    print("\n--- SAVING (1) ---", flush=True)
    r_save = client.call_tool("lu4d_save", {})
    print(f"[save] {r_save.get('text', str(r_save))}", flush=True)

    # --- SANITY CHECK ---
    print("\n--- SANITY CHECK (20 questions) ---", flush=True)
    score = 0
    for i, (query, keywords) in enumerate(SANITY_QUESTIONS, 1):
        r = client.call_tool("lu4d_ask", {"query": query, "depth": 3, "top_k": 10})
        s = score_response(r, keywords)
        score += s
        top = r.get("text", "")[:120].replace("\n", " ")
        mark = "OK" if s else "MISS"
        print(f"  Q{i:02d} [{mark}] {query}: {top}", flush=True)

    print(f"\n[sanity] score={score}/{len(SANITY_QUESTIONS)}", flush=True)

    # --- EXPLICIT SAVE (2) ---
    print("\n--- SAVING (2) ---", flush=True)
    r_save2 = client.call_tool("lu4d_save", {})
    print(f"[save] {r_save2.get('text', str(r_save2))}", flush=True)

    # --- STOP (triggers save via main.rs:71) ---
    client.stop()

    # --- VERIFY FILE ---
    sz = os.path.getsize(BRAIN)
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(BRAIN)))
    print(f"\n[brain] {sz:,} bytes | mtime={mtime}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

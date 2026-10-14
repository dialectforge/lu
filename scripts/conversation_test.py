import subprocess, json, time, sys
LU = "/opt/lu/lu/v4/target/release/lu4d"
BRAIN = "/var/lib/lu/brain-pre-step2c-20260522-220722.lbr"
KEYFILE = "/var/lib/lu/master.key"
prompts = [
    "Hi Lu, how are you today?",
    "What do you know about gravity?",
    "Can you explain what photosynthesis is?",
    "What is the capital of France?",
    "Tell me about yourself, Lu.",
    "What happens when water freezes?",
    "Why is the sky blue?",
    "What is xyzqp123abc?",
    "What is two plus two?",
    "Goodbye Lu, thank you for talking.",
]
p = subprocess.Popen([LU, "--mcp", "--brain", BRAIN, "--no-save", "--keyfile", KEYFILE],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
def send(o): p.stdin.write(json.dumps(o)+chr(10)); p.stdin.flush()
def rd(t=180):
    d=time.time()+t
    while time.time()<d:
        ln=p.stdout.readline()
        if not ln: time.sleep(0.2); continue
        ln=ln.strip()
        if not ln: continue
        try: return json.loads(ln)
        except: continue
    return None
send({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"x","version":"1"}}})
r=rd(180); print("INIT:", "ok" if r and r.get("result") else r)
send({"jsonrpc":"2.0","method":"notifications/initialized"}); time.sleep(0.5)
for i, q in enumerate(prompts, 1):
    print()
    print("=== ROUND " + str(i) + " ===")
    print("ME: " + q)
    send({"jsonrpc":"2.0","id":100+i,"method":"tools/call","params":{"name":"lu4d_narrate","arguments":{"query":q, "use_multi_think":True}}})
    r=rd(180)
    if r is None:
        print("LU: <NO RESPONSE WITHIN 180s>"); continue
    if r.get("error"):
        print("LU error: " + json.dumps(r.get("error")))
        continue
    content = r.get("result",{}).get("content",[])
    if content and isinstance(content,list):
        for item in content:
            if item.get("type")=="text":
                print("LU: " + item.get("text",""))
    else:
        print("LU(raw):", json.dumps(r)[:1500])
print()
print("DONE.")
p.stdin.close()
try: p.wait(timeout=10)
except: p.terminate(); p.wait(timeout=5)

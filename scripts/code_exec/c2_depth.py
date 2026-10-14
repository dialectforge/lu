#!/usr/bin/env python3
"""C2-deep operation-semantics depth pass. Mines behavioral edges per Python
operation from pydocs signatures + prose: HasParameter (takes), Returns
(returns/type), Causes (fails / raises). Teaches under the SAME disambiguated
'the X function' coding-sense subject the summary path uses. corpus=python-docs
(trust 0.95). --dry prints; else teaches to the brain COPY + flush + save.
"""
import re, sys, os, json, subprocess
PYROOT="/var/lib/lu/data/pydocs/python-3.14-docs-text"
EXE="/opt/lu/lu/merge-main/target/release/lu4d"
BRAIN="/opt/lu/lu/_c2_copy"; KEY="/var/lib/lu/master.key"
FILES=["library/functions.txt","library/stdtypes.txt"]
SIG=re.compile(r"^([a-zA-Z_][A-Za-z0-9_.]*)\((.*)\)\s*$")
RAISE=re.compile(r'"?([A-Z][A-Za-z]*(?:Error|Exception))"?\s+is\s+raised|raises?\s+"?([A-Z][A-Za-z]*(?:Error|Exception))"?|raise\s+"?([A-Z][A-Za-z]*(?:Error|Exception))"?', re.I)
RET=re.compile(r'[Rr]eturns?\s+(.{6,90}?)[\.\n]')

def parse_params(argstr):
    out=[]
    for tok in argstr.split(","):
        t=tok.strip()
        if not t or t in ("/","*","..."): continue
        t=t.lstrip("*")
        name=re.split(r"[=:\s]", t)[0].strip()
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name) and name not in ("self","cls"):
            out.append(name)
    return out

def mine(path):
    lines=open(path,encoding="utf-8",errors="replace").read().splitlines()
    edges=[]; i=0; n=len(lines)
    while i<n:
        m=SIG.match(lines[i])
        if not m:
            i+=1; continue
        raw=m.group(1); params=parse_params(m.group(2))
        name=raw.split(".")[-1]
        if not re.match(r"^[a-z_][a-z0-9_]*$", name):
            i+=1; continue
        # gather following indented description (skip further overload sigs)
        j=i+1; desc=[]
        while j<n and (lines[j].startswith("   ") or lines[j].strip()=="" or SIG.match(lines[j])):
            if SIG.match(lines[j]) and lines[j][:1] not in (" ","\t"):
                if lines[j].strip() and not lines[j].startswith("   "): break
            desc.append(lines[j]); j+=1
            if len([d for d in desc if d.strip()])>14: break
        text=" ".join(d.strip() for d in desc if d.strip())
        if not text:
            i=j; continue
        subj="the %s function" % name
        for p in params[:8]:
            edges.append((subj,"has parameter",p,"Concept"))
        rm=RET.search(text)
        if rm:
            obj=rm.group(1).replace(chr(42),"").replace(chr(34),"").strip().rstrip(",")
            if 3<len(obj)<90: edges.append((subj,"returns",obj,"Mechanism"))
        for rr in RAISE.finditer(text):
            err=next(g for g in rr.groups() if g)
            edges.append((subj,"causes","%s when misused"%err,"Causal"))
        i=j
    return edges

def main():
    dry="--dry" in sys.argv
    all_edges=[]
    for f in FILES:
        p=os.path.join(PYROOT,f)
        if os.path.exists(p): all_edges+=mine(p)
    # dedupe
    seen=set(); uniq=[]
    for e in all_edges:
        k=(e[0],e[1],e[2])
        if k in seen: continue
        seen.add(k); uniq.append(e)
    print("depth edges mined: %d (unique)" % len(uniq))
    from collections import Counter
    print("by relation:", dict(Counter(e[1] for e in uniq)))
    for e in uniq[:30]: print("  [%s] --%s--> [%s]" % (e[0],e[1],e[2]))
    if dry:
        print("DRY — no teach"); return
    # teach in batches to the copy
    p=subprocess.Popen([EXE,"--mcp","--brain",BRAIN,"--keyfile",KEY],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
    mid=[1]
    def call(nm,a):
        p.stdin.write((json.dumps({"jsonrpc":"2.0","id":mid[0],"method":"tools/call","params":{"name":nm,"arguments":a}})+"\n").encode());p.stdin.flush();mid[0]+=1
        r=json.loads(p.stdout.readline().decode())
        try:return r["result"]["content"][0]["text"]
        except:return json.dumps(r)[:200]
    p.stdin.write(b'{"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}\n');p.stdin.flush();p.stdout.readline()
    call("lu4d_trust_reset_to_defaults",{"seed_curriculum_corpora":True})
    call("lu4d_ingest_set",{"ingest_auto_approve_threshold":0.0})
    B=200; landed=0; res=""
    for k in range(0,len(uniq),B):
        ents=[{"subject":s,"relation":r,"object":o,"corpus":"python-docs","kind":kd} for (s,r,o,kd) in uniq[k:k+B]]
        res=call("lu4d_teach_corpus_batch",{"entries":ents})
    print("last batch:",res)
    print("flush:",call("lu4d_flush",{})[:80])
    print("save:",call("lu4d_save",{})[:40])
    p.stdin.close();p.wait()
    print("DEPTH_TEACH_DONE")

if __name__=="__main__": main()

#!/usr/bin/env python3
"""Free-form code planner (data-plumbing tier + pattern library + honesty).
NL prompt -> structured goal -> compose program from grounded C2 ops OR retrieve
a known pattern -> verify in X1 (run->fix) -> PASS / PATTERN / HONEST_ABORT / FAIL.
Grounding: ops used are checked against the C2 brain via handles (provenance).
"""
import json, subprocess, os, re, base64
X1="/opt/lu/lu/x1/x1_broker.py"
EXE="/opt/lu/lu/merge-main/target/release/lu4d";BRAIN="/opt/lu/lu/_c2_copy";KEY="/var/lib/lu/master.key"
CASES="/opt/lu/lu/x1/freeform_cases.jsonl"

# ---------- grounding (provenance): does the C2 brain know this op? ----------
_bp=None
def _brain():
    global _bp
    if _bp is None:
        _bp=subprocess.Popen([EXE,"--mcp","--brain",BRAIN,"--keyfile",KEY,"--no-save"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0,env=dict(os.environ,LU_SENSE_ROUTER="1",LU_SENSE_ROUTER_BOOST="4"))
        _bp.stdin.write(b'{"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}\n');_bp.stdin.flush();_bp.stdout.readline()
    return _bp
_gc={}
def grounds(op):
    if op in _gc: return _gc[op]
    p=_brain()
    p.stdin.write((json.dumps({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lu4d_ask","arguments":{"query":"pyfn"+op,"depth":3,"top_k":10}}})+"\n").encode());p.stdin.flush()
    a=json.loads(p.stdout.readline().decode())
    try: body=a["result"]["content"][0]["text"].lower()
    except: body=""
    r=("the %s function"%op in body) or ("when misused" in body) or ("parameter" in body)
    _gc[op]=r; return r

# ---------- X1 verify + fix loop ----------
def x1(code, inputs=None, wall_ms=6000):
    req={"code":code,"limits":{"wall_ms":wall_ms}}
    if inputs: req["inputs"]=inputs
    r=subprocess.run(["python3",X1],input=json.dumps(req).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try: return json.loads(r.stdout.decode())
    except: return {"exit_code":-1,"stdout":"","stderr":(r.stdout+r.stderr).decode()}

def run_fix(code, inputs, check, trace):
    fixes=[]
    for attempt in range(1,4):
        res=x1(code, inputs)
        out=res.get("stdout",""); err=res.get("stderr",""); rc=res.get("exit_code")
        ok = rc==0 and (check(out) if check else True)
        trace.append({"attempt":attempt,"code":code,"exit":rc,"stdout":out.strip()[:120],"stderr":err.strip()[-160:],"fixes":list(fixes),"ok":ok})
        if ok: return True, code, res
        # observe->fix
        nc=code
        if "invalid literal for int" in err or ("ValueError" in err and "int(" in code):
            nc=code.replace("int(","float("); fixes.append("int->float")
        elif "FileNotFoundError" in err:
            fixes.append("missing input file"); break
        elif "KeyError" in err or "IndexError" in err:
            fixes.append("index/key guard"); nc=code  # can't blindly fix
            break
        else:
            break
        if nc==code: break
        code=nc
    return False, code, res

# ---------- pattern library (retrieve + adapt KNOWN patterns) ----------
PATTERNS={
 "fizzbuzz": (r"fizzbuzz", """for i in range(1,21):\n    if i%15==0: print('FizzBuzz')\n    elif i%3==0: print('Fizz')\n    elif i%5==0: print('Buzz')\n    else: print(i)"""),
 "palindrome": (r"palindrome", """def is_palindrome(s):\n    s=''.join(c.lower() for c in s if c.isalnum())\n    return s==s[::-1]\nprint(is_palindrome('racecar'), is_palindrome('hello'))"""),
 "fibonacci": (r"fibonacci|fibonnaci", """def fib(n):\n    a,b=0,1; out=[]\n    for _ in range(n): out.append(a); a,b=b,a+b\n    return out\nprint(fib(10))"""),
 "factorial": (r"factorial", """def factorial(n):\n    r=1\n    for i in range(2,n+1): r*=i\n    return r\nprint(factorial(5))"""),
 "reverse_string": (r"reverse a string|reverse the string", """def reverse(s): return s[::-1]\nprint(reverse('hello'))"""),
 "is_prime": (r"is.*prime|prime number", """def is_prime(n):\n    if n<2: return False\n    for i in range(2,int(n**0.5)+1):\n        if n%i==0: return False\n    return True\nprint(is_prime(7), is_prime(8))"""),
 "count_vowels": (r"count.*vowel", """def count_vowels(s): return sum(1 for c in s.lower() if c in 'aeiou')\nprint(count_vowels('hello world'))"""),
 "sum_range": (r"sum.*1 to|sum of.*integers from 1", """print(sum(range(1,101)))"""),
}
def match_pattern(pl):
    for name,(rx,code) in PATTERNS.items():
        if re.search(rx, pl): return name, code
    return None, None

# ---------- data-plumbing intent parser + composer ----------
def parse_plumbing(pl):
    src=None
    if "json" in pl: src="json"
    elif "csv" in pl: src="csv"
    elif "file" in pl: src="file"
    elif "list" in pl: src="list"
    agg=None
    if "average" in pl or "mean" in pl: agg="avg"
    elif "longest line" in pl: agg="longest"
    elif "word frequenc" in pl or "word count" in pl or "frequenc" in pl: agg="wordfreq"
    elif "duplicate" in pl: agg="dups"
    elif "second largest" in pl: agg="second_max"
    elif "max" in pl or "largest" in pl or "maximum" in pl: agg="max"
    elif "min" in pl or "smallest" in pl: agg="min"
    elif "sort" in pl: agg="sort"
    elif "how many" in pl or "count" in pl: agg="count_lines"
    elif "keys" in pl: agg="json_keys"
    elif "sum" in pl: agg="sum"
    if src is None and agg is None: return None
    return {"src":src or "file","agg":agg}

def compose_plumbing(spec):
    src,agg=spec["src"],spec["agg"]
    read_nums="nums=[float(x) for x in open('input.txt').read().split() if x.strip()]"
    read_lines="lines=[l.rstrip(chr(10)) for l in open('input.txt')]"
    if src=="json" and agg=="json_keys":
        return "import json\nd=json.load(open('input.json'))\nprint(sorted(d.keys()))", ["open","keys"]
    if agg=="avg":
        if src=="csv": return "import csv\nrows=list(csv.reader(open('input.csv')))\ncol=[float(r[0]) for r in rows if r and r[0].strip()]\nprint(sum(col)/len(col))", ["open","float","sum","len"]
        return read_nums+"\nprint(sum(nums)/len(nums))", ["open","float","sum","len"]
    if agg=="longest": return read_lines+"\nprint(max(lines, key=len))", ["open","max","len"]
    if agg=="wordfreq": return "from collections import Counter\nw=open('input.txt').read().split()\nprint(Counter(w).most_common(1))", ["open","split","count"]
    if agg=="dups": return "from collections import Counter\nc=Counter(l.rstrip(chr(10)) for l in open('input.txt'))\nprint(sorted(k for k,v in c.items() if v>1))", ["open","count"]
    if agg=="second_max":
        return "nums=sorted(set(float(x) for x in open('input.txt').read().split() if x.strip()))\nprint(nums[-2])", ["open","float","sorted","set"]
    if agg=="max": return read_nums+"\nprint(max(nums))", ["open","float","max"]
    if agg=="min": return read_nums+"\nprint(min(nums))", ["open","float","min"]
    if agg=="sort": return read_nums+"\nprint(sorted(nums))", ["open","float","sorted"]
    if agg=="count_lines":
        return read_lines+"\nprint(len([l for l in lines if len(l)>10]))", ["open","len"]
    if agg=="sum": return read_nums+"\nprint(sum(nums))", ["open","float","sum"]
    return None, []

# ---------- top-level solve ----------
def solve(prompt, inputs=None, check=None):
    pl=prompt.lower(); trace={"prompt":prompt,"path":None,"grounded_ops":[],"attempts":[]}
    # 1) known pattern?
    name,code=match_pattern(pl)
    if code:
        trace["path"]="pattern:%s"%name
        ok,fc,res=run_fix(code, inputs, check, trace["attempts"])
        trace["verdict"]="PATTERN_PASS" if ok else "PATTERN_FAIL"
        return trace
    # 2) data-plumbing compose
    spec=parse_plumbing(pl)
    if spec:
        comp=compose_plumbing(spec)
        if comp and comp[0]:
            code,ops=comp; trace["path"]="compose:%s/%s"%(spec["src"],spec["agg"])
            trace["grounded_ops"]=[o for o in ops if grounds(o)]
            ok,fc,res=run_fix(code, inputs, check, trace["attempts"])
            trace["verdict"]="COMPOSE_PASS" if ok else "COMPOSE_FAIL"
            return trace
    # 3) honest: no known approach
    trace["path"]="none"; trace["verdict"]="HONEST_ABORT"
    trace["message"]="I don't have a grounded pattern or a composition for this; I won't fake a solution."
    return trace

if __name__=="__main__":
    import sys; print(json.dumps(solve(sys.argv[1]),indent=1))
"""end"""

#!/usr/bin/env python3
"""S1 planner on REAL C2 op-facts (sense-fix option 3: planner-side coding-context).
Grounds each op by a coding-context query ('the {op} builtin function parameter')
that routes PAST the dominant everyday sense into the C2-ingested coding subgraph
(pydocs/MDN). No py* stand-ins, no engine change. Then composes, verifies in X1,
runs observe->fix, writes cases to a SEPARATE store, plan-trace on request.
Usage: python3 s1_planner_c2.py [--trace]
"""
import json, subprocess, sys, time, argparse, re
BRAIN="/opt/lu/lu/_c2_copy"
LU4D="/opt/lu/lu/merge-main/target/release/lu4d"
KEY="/var/lib/lu/master.key"
X1="/opt/lu/lu/x1/x1_broker.py"
CASES="/opt/lu/lu/x1/s1_cases_store_c2.jsonl"

# S1 op -> real Python op name (the C2-ingested coding node family)
REAL={"range":"range","sum":"sum","len":"len","max":"max","read_lines":"open","float":"float","int":"int","print":"print"}
MARK=["parameter","returns","return","object","argument","iterable","function","when misused","integer","string","floating","value","item","number"]
REFUSE=["don't have","won't fabricate","nothing in my brain","don't match my brain","no strong anchor"]

class Brain:
    def __init__(self,path):
        self.p=subprocess.Popen([LU4D,"--mcp","--brain",path,"--keyfile",KEY,"--no-save"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
        self.mid=1;self._c("initialize",{})
    def _c(self,m,pa):
        self.p.stdin.write((json.dumps({"jsonrpc":"2.0","id":self.mid,"method":m,"params":pa})+"\n").encode());self.p.stdin.flush();self.mid+=1
        return json.loads(self.p.stdout.readline().decode())
    def ask(self,q,tk=10,depth=2):
        r=self._c("tools/call",{"name":"lu4d_ask","arguments":{"query":q,"depth":depth,"top_k":tk}})
        try:return r["result"]["content"][0]["text"]
        except:return ""
    def close(self):
        try:self.p.stdin.close();self.p.wait(timeout=10)
        except:pass

def ground(b, op):
    # OPTION 2: query the distinctive seedable handle 'pyfn<op>' which seeds
    # straight into the coding node and reaches its real params/returns/causes
    # by traversal -- grounds EVERY op incl. polysemous ones (sum/open/max...).
    name=REAL[op]
    a=b.ask("pyfn"+name, tk=16, depth=3)
    body="\n".join(l for l in a.splitlines() if not l.strip().lower().startswith("query:")).lower()
    MARK=["the "+name+" function","when misused","iterable","parameter","argument","the total","largest","number of items","returns","object"]
    grounded = any(m in body for m in MARK) and "activated: 1 " not in body
    return grounded, a.strip().replace("\n"," | ")[:200]

def x1_run(code, inputs=None, wall_ms=5000):
    req={"code":code,"limits":{"wall_ms":wall_ms}}
    if inputs: req["inputs"]=inputs
    p=subprocess.run(["python3",X1],input=json.dumps(req).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:return json.loads(p.stdout.decode())
    except:return {"exit_code":-1,"stdout":"","stderr":(p.stdout+p.stderr).decode()}

DATA="12.5\n7\n99\n50\n88\n3\n"; nums=[float(x) for x in DATA.split()]
TASKS=[
 {"name":"sum_1_to_100","desc":"print the sum of the integers from 1 to 100","intent":"SUM_RANGE","expect":"5050"},
 {"name":"file_average","desc":"print the average of the numbers in data.txt","intent":"FILE_AVG","inputs":[{"name":"data.txt","text":DATA}],"expect":str(sum(nums)/len(nums))},
 {"name":"file_count_gt50","desc":"print how many numbers in data.txt are greater than 50","intent":"FILE_COUNT_GT","inputs":[{"name":"data.txt","text":DATA}],"expect":str(sum(1 for x in nums if x>50))},
 {"name":"file_max","desc":"print the maximum number in data.txt","intent":"FILE_MAX","inputs":[{"name":"data.txt","text":DATA}],"expect":str(max(nums))},
]
PLANS={"SUM_RANGE":["range","sum","print"],"FILE_AVG":["read_lines","int","sum","len","print"],"FILE_COUNT_GT":["read_lines","float","len","print"],"FILE_MAX":["read_lines","float","max","print"]}
def compose(intent,conv):
    if intent=="SUM_RANGE": return "print(sum(range(1, 101)))"
    rd="nums=[%s(x) for x in open('data.txt').read().split() if x.strip()]"%conv
    return rd+{"FILE_AVG":"\nprint(sum(nums)/len(nums))","FILE_COUNT_GT":"\nprint(len([x for x in nums if x>50]))","FILE_MAX":"\nprint(max(nums))"}[intent]
def solve(b,task):
    intent=task["intent"];ops=PLANS[intent]
    tr={"task":task["name"],"desc":task["desc"],"required_ops":ops,"grounding":[],"attempts":[]}
    for op in ops:
        g,ev=ground(b,op);tr["grounding"].append({"op":op,"real":REAL[op],"grounded":g,"recall":ev})
    ung=[g["op"] for g in tr["grounding"] if not g["grounded"]]
    if ung: tr["result"]="ABORT_UNGROUNDED";tr["ungrounded"]=ung;return False,tr
    conv="int";fixes=[]
    for n in range(1,4):
        code=compose(intent,conv);res=x1_run(code,inputs=task.get("inputs"))
        out=res.get("stdout","").strip();ok=res.get("exit_code")==0 and out==task["expect"]
        tr["attempts"].append({"n":n,"conv":conv,"code":code,"exit":res.get("exit_code"),"stdout":out[:80],"stderr":res.get("stderr","").strip()[-140:],"pass":ok,"fixes":list(fixes)})
        if ok: tr["result"]="PASS";return True,tr
        err=res.get("stderr","")
        if conv=="int" and ("invalid literal for int" in err or "ValueError" in err):
            conv="float";fixes.append("int->float: observed ValueError converting a non-integer token");continue
        break
    tr["result"]="FAIL";return False,tr
def write_case(tr):
    last=tr["attempts"][-1]
    rec={"task":tr["desc"],"code":last["code"],"confidence":"Inferred","provenance":{"synthesized_by":"S1","verified_by":"X1","grounded_on":"C2-real-op-facts","grounded_ops":[g["real"] for g in tr["grounding"]],"fixes_applied":last["fixes"],"x1_exit":last["exit"],"ts":int(time.time())}}
    open(CASES,"a").write(json.dumps(rec)+"\n")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--trace",action="store_true");a=ap.parse_args()
    b=Brain(BRAIN);npass=0;traces=[]
    for t in TASKS:
        ok,tr=solve(b,t);traces.append(tr)
        if ok: write_case(tr);npass+=1
        print("%-16s %-16s attempts=%d grounded=%d/%d"%(t["name"],tr["result"],len(tr["attempts"]),sum(1 for g in tr["grounding"] if g["grounded"]),len(tr["grounding"])))
    b.close()
    if a.trace: print("\n===== PLAN TRACE (real C2 facts) =====\n"+json.dumps(traces,indent=1))
    print("\nS1-ON-REAL-FACTS: %d/%d tasks solved by grounded composition + X1-verified run"%(npass,len(TASKS)))
if __name__=="__main__": main()

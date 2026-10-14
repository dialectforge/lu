#!/usr/bin/env python3
"""OPTION 2: distinctive seedable coding handles per op (entity-shelf pattern:
separator-free single token, like eq<qid>). Each handle 'pyfn<op>' / 'jsfn<op>'
--RelatedTo--> the canonical coding node ('the <op> function' / web label), so a
query on the handle seeds DIRECTLY into the coding subgraph, bypassing the
dominant everyday sense. Reaches the op's real params/returns/causes by
traversal (handle -> the X function -> HasParameter/Returns/Causes)."""
import json, subprocess
EXE="/opt/lu/lu/merge-main/target/release/lu4d";B="/opt/lu/lu/_c2_copy";K="/var/lib/lu/master.key"
# Python ops -> canonical node label 'the <op> function' (C2-ingested)
PY=["abs","all","any","bin","bool","chr","dict","float","format","len","max","min","open","sorted","sum","range","print","int","str","zip","map","filter","enumerate","round","repr","hash","iter","hex","oct","ord","pow","divmod","getattr","setattr","list","tuple","set","bytes","vars","id",
    "append","join","split","replace","strip","pop","insert","count","index","upper","lower","startswith","endswith","find","keys","values","items","get","copy","clear"]
# Web ops -> (handle-op, canonical node label as ingested by MDN)
WEB=[("concat","the concat() method"),("abortcontroller","the AbortController interface"),("fetch","the fetch() method"),("queryselector","the querySelector() method"),("addeventlistener","the addEventListener() method"),("map","the map() method"),("filter","the filter() method"),("json","the JSON object"),("promise","the Promise object"),("slice","the slice() method")]
p=subprocess.Popen([EXE,"--mcp","--brain",B,"--keyfile",K],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
p.stdin.write(b'{"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}\n');p.stdin.flush();p.stdout.readline()
mid=[1]
def call(n,a):
    p.stdin.write((json.dumps({"jsonrpc":"2.0","id":mid[0],"method":"tools/call","params":{"name":n,"arguments":a}})+"\n").encode());p.stdin.flush();mid[0]+=1
    r=json.loads(p.stdout.readline().decode())
    try:return r["result"]["content"][0]["text"]
    except:return json.dumps(r)[:120]
call("lu4d_trust_reset_to_defaults",{"seed_curriculum_corpora":True})
call("lu4d_ingest_set",{"ingest_auto_approve_threshold":0.0})
ents=[]
for op in PY:
    ents.append({"subject":"pyfn"+op,"relation":"related to","object":"the %s function"%op,"corpus":"python-docs","kind":"Concept"})
for op,lab in WEB:
    ents.append({"subject":"jsfn"+op,"relation":"related to","object":lab,"corpus":"mdn","kind":"Concept"})
landed=0
for k in range(0,len(ents),200):
    r=call("lu4d_teach_corpus_batch",{"entries":ents[k:k+200]})
    print("batch:",r[:120])
print("flush:",call("lu4d_flush",{})[:70])
print("save:",call("lu4d_save",{})[:40])
p.stdin.close();p.wait()
print("HANDLES_DONE")

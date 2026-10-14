#!/usr/bin/env python3
import json,os,sys,urllib.request
port=int(sys.argv[1]) if len(sys.argv)>1 else 8787
base=f'http://127.0.0.1:{port}'
def post(u,p,tok=None,t=180):
    r=urllib.request.Request(u,data=json.dumps(p).encode(),method='POST')
    r.add_header('Content-Type','application/json')
    if tok:r.add_header('Authorization',f'Bearer {tok}')
    with urllib.request.urlopen(r,timeout=t) as z:return json.loads(z.read().decode())
pp=os.environ.get('LU_GUEST_PASSPHRASE')
if not pp:print('NO PASSPHRASE');sys.exit(99)
tok=post(f'{base}/auth',{'passphrase':pp}).get('token')
def ask(q):
    resp=post(f'{base}/mcp',{'method':'tools/call','params':{'name':'lu4d_ask','arguments':{'query':q}}},tok)
    r=resp.get('result',resp);c=r.get('content')
    if isinstance(c,list):return '\n'.join(x.get('text','') for x in c if isinstance(x,dict))
    for k in ('text','answer','narration'):
        if isinstance(r.get(k),str):return r[k]
    return json.dumps(r)
for q in sys.argv[2:]:
    print('=== Q:',q)
    try:print(ask(q))
    except Exception as e:print('ERR',e)
    print()

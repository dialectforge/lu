#!/usr/bin/env python3
"""X1 gatekeeper/broker. One request -> one FRESH sandbox. Runs the candidate
program under: root-launched bwrap dropping to the unprivileged lu-sandbox user,
all namespaces unshared (incl. an empty netns => no network), minimal read-only
rootfs (NO brain / master.key / repo bind), ephemeral per-run work dir, cgroup
caps (systemd scope) + rlimits + wall-clock. Returns stdout/stderr/exit and,
optionally, one produced file (for the demo's downloadable-file path).

usage: echo '<json request>' | python3 x1_broker.py
request: {"code": "...", "want_file": "out.png"?, "limits": {"mem_mib":256,"pids":64,"cpu_cores":1,"wall_ms":5000}?}
NOTE: NOT wired to anything live. Manual/gate use only until G-X1 is signed off.
"""
import json, os, sys, subprocess, uuid, base64, time, hashlib

RUNS = "/srv/lu-sandbox/runs"
X1DIR = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(X1DIR, "launcher.py")
UID, GID = 995, 985  # lu-sandbox
OUT_CAP = 262144

def _put(path, data: bytes):
    subprocess.run(["sudo", "tee", path], input=data, stdout=subprocess.DEVNULL, check=True)

def run(req):
    code = req["code"]
    want = req.get("want_file")
    lim = req.get("limits", {})
    mem = int(lim.get("mem_mib", 256)); pids = int(lim.get("pids", 64))
    cpu = int(lim.get("cpu_cores", 1)); wall = int(lim.get("wall_ms", 5000))
    rid = uuid.uuid4().hex
    rundir = f"{RUNS}/{rid}"
    subprocess.run(["sudo", "mkdir", "-p", f"{rundir}/out"], check=True)
    _put(f"{rundir}/prog.py", code.encode())
    _put(f"{rundir}/launcher.py", open(LAUNCHER, "rb").read())
    for inp in req.get("inputs", []):
        _put(f"{rundir}/" + inp["name"], inp.get("text", "").encode())
    subprocess.run(["sudo", "chmod", "-R", "0777", rundir], check=True)
    bw = ["--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
          "--ro-bind", "/lib", "/lib", "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev"]
    if os.path.isdir("/lib64"): bw += ["--ro-bind", "/lib64", "/lib64"]
    wall_s = max(1, wall // 1000)
    cmd = ["sudo", "systemd-run", "--scope", "--quiet",
           f"-pMemoryMax={mem}M", "-pMemorySwapMax=0", f"-pTasksMax={pids}", f"-pCPUQuota={cpu*100}%",
           "--",
           "timeout", "--signal=KILL", f"{wall/1000.0}",
           "bwrap", "--unshare-all", "--uid", str(UID), "--gid", str(GID),
           "--hostname", "sandbox", "--die-with-parent", "--new-session",
           *bw, "--bind", rundir, "/work", "--chdir", "/work",
           "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/work",
           "--setenv", "X1_CPU", str(wall_s + 2),
           "--", "/usr/bin/python3", "/work/launcher.py", "/work/prog.py"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=wall/1000.0 + 20)
        rc = proc.returncode; so = proc.stdout; se = proc.stderr
    except subprocess.TimeoutExpired as e:
        rc = -9; so = e.stdout or b""; se = (e.stderr or b"") + b"\n[broker hard timeout]"
    dt = int((time.time() - t0) * 1000)
    killed = None
    if rc in (137, -9): killed = "wall"
    elif rc == 124: killed = "wall"
    elif rc == -31 or rc == 159: killed = "seccomp"  # SIGSYS
    res = {"exit_code": rc, "killed_by": killed, "timed_out": killed == "wall",
           "wall_ms": dt, "stdout": so.decode(errors="replace")[:OUT_CAP],
           "stderr": se.decode(errors="replace")[:OUT_CAP]}
    # seccomp status sidecar (diagnostic)
    sc = subprocess.run(["sudo", "cat", f"{rundir}/.x1_seccomp"], stdout=subprocess.PIPE)
    if sc.returncode == 0: res["seccomp"] = sc.stdout.decode(errors="replace").strip()
    if want:
        r = subprocess.run(["sudo", "cat", f"{rundir}/out/{want}"], stdout=subprocess.PIPE)
        if r.returncode == 0:
            data = r.stdout[: 8 * 1024 * 1024]
            res["file"] = {"name": want, "size": len(data),
                           "sha256": hashlib.sha256(data).hexdigest(),
                           "bytes_b64": base64.b64encode(data).decode()}
    subprocess.run(["sudo", "rm", "-rf", rundir])
    return res

if __name__ == "__main__":
    req = json.load(sys.stdin)
    print(json.dumps(run(req), indent=1))

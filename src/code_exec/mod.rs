//! OP-CODE-EXEC (Phase 1) -- native Rust invocation of the X1 sandbox.
//!
//! Replaces the Python `x1_broker` on the live path. One fresh sandbox per run,
//! modelled on `capability_runtime::SubprocessPackTool` (bounded stdout reader,
//! wall-clock kill, output-size cap). The bwrap/seccomp/cgroups isolation is
//! BYTE-IDENTICAL to the proven broker; only the outer gatekeeper is now Rust.
//!
//! Isolation (re-verified by G-X1): root-launched bwrap that drops to the
//! unprivileged `lu-sandbox` user, all namespaces (empty netns => no network),
//! read-only minimal rootfs (NO brain / master.key / repo bind), ephemeral
//! per-run tmpfs work dir, cgroup caps (systemd scope) + rlimits + seccomp
//! (applied in-sandbox by the launcher) + wall-clock. The in-sandbox
//! `launcher.py` (embedded below) applies rlimits + the seccomp denylist.
//!
//! PRIVILEGE NOTE (sudo-vs-setuid decision): bwrap needs root for namespace
//! setup. This phase uses `sudo` (the serving user has scoped passwordless
//! sudo for exactly: mkdir/chmod/rm/cat on /srv/lu-sandbox, systemd-run,
//! bwrap). A dedicated setuid X1 helper is the recommended hardening follow-up.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
use std::os::unix::process::ExitStatusExt;

const LAUNCHER_PY: &str = include_str!("../../scripts/code_exec/launcher.py");
const RUNS_DIR: &str = "/srv/lu-sandbox/runs";
const SANDBOX_UID: &str = "995";
const SANDBOX_GID: &str = "985";
const OUT_CAP: usize = 256 * 1024;
const FILE_CAP: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct X1Limits {
    pub mem_mib: u32,
    pub pids: u32,
    pub cpu_cores: u32,
    pub wall_ms: u64,
}
impl Default for X1Limits {
    fn default() -> Self { X1Limits { mem_mib: 256, pids: 64, cpu_cores: 1, wall_ms: 5000 } }
}

#[derive(Debug, Clone)]
pub struct X1Input { pub name: String, pub text: String }

#[derive(Debug, Clone)]
pub struct X1Request {
    pub code: String,
    pub inputs: Vec<X1Input>,
    pub want_file: Option<String>,
    pub limits: X1Limits,
}

#[derive(Debug, Clone, Default)]
pub struct X1File { pub name: String, pub size: usize, pub sha256: String, pub bytes: Vec<u8> }

#[derive(Debug, Clone, Default)]
pub struct X1Result {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub killed_by: Option<String>,
    pub timed_out: bool,
    pub wall_ms: u64,
    pub seccomp: Option<String>,
    pub file: Option<X1File>,
}

fn sudo(args: &[&str]) -> std::io::Result<std::process::Output> {
    Command::new("sudo").args(args).output()
}
fn sudo_write(path: &str, data: &[u8]) -> std::io::Result<()> {
    let mut child = Command::new("sudo").arg("tee").arg(path)
        .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null()).spawn()?;
    if let Some(mut si) = child.stdin.take() { si.write_all(data)?; }
    child.wait()?; Ok(())
}

/// Hex sha256 without adding a crate: reuse the repo's crc? No -- use a tiny
/// inline sha256. (crc32fast is a dep but not a cryptographic digest; the file
/// hash is integrity-for-display only, so we use a compact sha256 impl.)
fn sha256_hex(data: &[u8]) -> String { crate::code_exec::sha256::hex(data) }

pub fn run(req: &X1Request) -> X1Result {
    let rid: String = {
        // uuid-ish: nanos + pid, hex
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos()).unwrap_or(0);
        format!("{:x}{:x}", n, std::process::id())
    };
    let rundir = format!("{}/{}", RUNS_DIR, rid);
    let mut res = X1Result::default();

    if sudo(&["mkdir", "-p", &format!("{}/out", rundir)]).map(|o| o.status.success()).unwrap_or(false) == false {
        res.stderr = "x1: failed to create run dir".into(); res.exit_code = -1; return res;
    }
    let _ = sudo_write(&format!("{}/prog.py", rundir), req.code.as_bytes());
    let _ = sudo_write(&format!("{}/launcher.py", rundir), LAUNCHER_PY.as_bytes());
    for inp in &req.inputs {
        let safe = inp.name.replace(['/', '.'], "_").replace("__", "_");
        let _ = safe; // keep original name (broker parity) but guard traversal:
        if inp.name.contains("..") || inp.name.starts_with('/') { continue; }
        let _ = sudo_write(&format!("{}/{}", rundir, inp.name), inp.text.as_bytes());
    }
    let _ = sudo(&["chmod", "-R", "0777", &rundir]);

    let wall_s = std::cmp::max(1, req.limits.wall_ms / 1000);
    let mem = format!("-pMemoryMax={}M", req.limits.mem_mib);
    let pids = format!("-pTasksMax={}", req.limits.pids);
    let cpu = format!("-pCPUQuota={}%", req.limits.cpu_cores * 100);
    let wall_f = format!("{}", req.limits.wall_ms as f64 / 1000.0);
    let bind = format!("{}", rundir);
    let x1cpu = format!("{}", wall_s + 2);
    let mut args: Vec<String> = vec![
        "systemd-run".into(), "--scope".into(), "--quiet".into(),
        mem, "-pMemorySwapMax=0".into(), pids, cpu, "--".into(),
        "timeout".into(), "--signal=KILL".into(), wall_f,
        "bwrap".into(), "--unshare-all".into(), "--uid".into(), SANDBOX_UID.into(),
        "--gid".into(), SANDBOX_GID.into(), "--hostname".into(), "sandbox".into(),
        "--die-with-parent".into(), "--new-session".into(),
        "--ro-bind".into(), "/usr".into(), "/usr".into(),
        "--ro-bind".into(), "/bin".into(), "/bin".into(),
        "--ro-bind".into(), "/lib".into(), "/lib".into(),
        "--tmpfs".into(), "/tmp".into(), "--proc".into(), "/proc".into(),
        "--dev".into(), "/dev".into(),
    ];
    if std::path::Path::new("/lib64").is_dir() {
        args.push("--ro-bind".into()); args.push("/lib64".into()); args.push("/lib64".into());
    }
    args.extend([
        "--bind".into(), bind, "/work".into(), "--chdir".into(), "/work".into(),
        "--setenv".into(), "PATH".into(), "/usr/bin:/bin".into(),
        "--setenv".into(), "HOME".into(), "/work".into(),
        "--setenv".into(), "X1_CPU".into(), x1cpu,
        "--".into(), "/usr/bin/python3".into(), "/work/launcher.py".into(), "/work/prog.py".into(),
    ]);

    let t0 = Instant::now();
    let out = Command::new("sudo").args(&args)
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).output();
    res.wall_ms = t0.elapsed().as_millis() as u64;
    match out {
        Ok(o) => {
            res.exit_code = o.status.code().unwrap_or_else(|| o.status.signal().map(|sig| 128 + sig).unwrap_or(-1));
            let mut so = String::from_utf8_lossy(&o.stdout).into_owned();
            let mut se = String::from_utf8_lossy(&o.stderr).into_owned();
            so.truncate(OUT_CAP); se.truncate(OUT_CAP);
            res.stdout = so; res.stderr = se;
        }
        Err(e) => { res.exit_code = -1; res.stderr = format!("x1 spawn error: {}", e); }
    }
    // 137 = SIGKILL (wall via timeout), 124 = timeout exit, 159 = 128+31 SIGSYS (seccomp)
    res.killed_by = match res.exit_code {
        137 | 124 => Some("wall".into()),
        159 => Some("seccomp".into()),
        _ => None,
    };
    if res.exit_code != 0 && res.killed_by.is_none()
        && res.wall_ms + 300 >= req.limits.wall_ms {
        res.killed_by = Some("wall".into());
    }
    res.timed_out = matches!(res.killed_by.as_deref(), Some("wall"));
    if let Ok(o) = sudo(&["cat", &format!("{}/.x1_seccomp", rundir)]) {
        if o.status.success() { res.seccomp = Some(String::from_utf8_lossy(&o.stdout).trim().to_string()); }
    }
    if let Some(want) = &req.want_file {
        if !want.contains("..") && !want.starts_with('/') {
            if let Ok(o) = sudo(&["cat", &format!("{}/out/{}", rundir, want)]) {
                if o.status.success() {
                    let mut data = o.stdout; data.truncate(FILE_CAP);
                    let sha = sha256_hex(&data);
                    res.file = Some(X1File { name: want.clone(), size: data.len(), sha256: sha, bytes: data });
                }
            }
        }
    }
    let _ = sudo(&["rm", "-rf", &rundir]);
    res
}

pub mod sha256 {
    //! Minimal SHA-256 (integrity-for-display of the produced file). No crate.
    const K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    pub fn hex(data: &[u8]) -> String {
        let mut h: [u32; 8] = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
        let ml = (data.len() as u64) * 8;
        let mut msg = data.to_vec();
        msg.push(0x80);
        while msg.len() % 64 != 56 { msg.push(0); }
        msg.extend_from_slice(&ml.to_be_bytes());
        for chunk in msg.chunks(64) {
            let mut w = [0u32; 64];
            for i in 0..16 { w[i] = u32::from_be_bytes([chunk[i*4],chunk[i*4+1],chunk[i*4+2],chunk[i*4+3]]); }
            for i in 16..64 {
                let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
                let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
                w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
            }
            let (mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh) = (h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let t2 = s0.wrapping_add(maj);
                hh=g; g=f; f=e; e=d.wrapping_add(t1); d=c; c=b; b=a; a=t1.wrapping_add(t2);
            }
            h[0]=h[0].wrapping_add(a); h[1]=h[1].wrapping_add(b); h[2]=h[2].wrapping_add(c); h[3]=h[3].wrapping_add(d);
            h[4]=h[4].wrapping_add(e); h[5]=h[5].wrapping_add(f); h[6]=h[6].wrapping_add(g); h[7]=h[7].wrapping_add(hh);
        }
        let mut s = String::with_capacity(64);
        for x in h { s.push_str(&format!("{:08x}", x)); }
        s
    }
    #[test]
    fn sha256_known_answer() {
        assert_eq!(hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }
}
pub mod planner;

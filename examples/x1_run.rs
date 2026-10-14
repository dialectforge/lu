//! X1 native runner CLI (dev/verification harness). Reads one JSON request on
//! stdin, runs it through `lu4d::code_exec::run`, prints the JSON result. Same
//! wire shape as the Python broker so the G-X1 battery runs unchanged.
use std::io::Read;
use lu4d::code_exec::{run, X1Request, X1Input, X1Limits};

fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
        s.push(T[((n >> 18) & 63) as usize] as char);
        s.push(T[((n >> 12) & 63) as usize] as char);
        s.push(if c.len() > 1 { T[((n >> 6) & 63) as usize] as char } else { '=' });
        s.push(if c.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    s
}

fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).ok();
    let v: serde_json::Value = serde_json::from_str(&buf).unwrap_or(serde_json::Value::Null);
    let lim = v.get("limits").cloned().unwrap_or(serde_json::Value::Null);
    let g = |k: &str, d: u64| lim.get(k).and_then(|x| x.as_u64()).unwrap_or(d);
    let req = X1Request {
        code: v.get("code").and_then(|x| x.as_str()).unwrap_or("").to_string(),
        inputs: v.get("inputs").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|i| {
            Some(X1Input {
                name: i.get("name")?.as_str()?.to_string(),
                text: i.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string(),
            })
        }).collect()).unwrap_or_default(),
        want_file: v.get("want_file").and_then(|x| x.as_str()).map(|s| s.to_string()),
        limits: X1Limits {
            mem_mib: g("mem_mib", 256) as u32,
            pids: g("pids", 64) as u32,
            cpu_cores: g("cpu_cores", 1) as u32,
            wall_ms: g("wall_ms", 5000),
        },
    };
    let r = run(&req);
    let mut out = serde_json::json!({
        "exit_code": r.exit_code,
        "stdout": r.stdout,
        "stderr": r.stderr,
        "killed_by": r.killed_by,
        "timed_out": r.timed_out,
        "wall_ms": r.wall_ms,
        "seccomp": r.seccomp,
    });
    if let Some(f) = r.file {
        out["file"] = serde_json::json!({
            "name": f.name, "size": f.size, "sha256": f.sha256, "bytes_b64": b64(&f.bytes),
        });
    }
    println!("{}", serde_json::to_string(&out).unwrap());
}

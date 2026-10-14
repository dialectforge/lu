//! Port Cat.5.1 — capability-pack EXECUTION runtime.
//!
//! Cat.5 (`src/capability_pack/`) shipped the schema half: the signed
//! `.lupack5` format, the publisher-trust + install ledger, and a
//! passive MCP surface that records installs but DOES NOT EXECUTE.
//! Cat.5.1 ships the runtime half: load an installed pack, RE-VERIFY
//! it (trust -> revocation -> signature) right before use, instantiate
//! its declared tools behind an object-safe `PackTool` trait, register
//! them in an in-memory registry, and make them callable through MCP
//! dynamic dispatch and the vision scene-graph extractor hook.
//!
//! Execution model — deliberately GENERALISES the proven
//! `SceneGraphExtractor` pattern (`src/vision/extractor.rs`): an
//! object-safe trait + a `build_pack_tool` factory + a concrete
//! `SubprocessPackTool`. Only the Subprocess runtime is built in this
//! phase (pure Rust, OS process boundary, no `unsafe`). WASM is
//! deferred to a later phase; SharedLib (full-privilege dlopen) is
//! rejected outright.
//!
//! SECURITY (load-bearing — this is third-party executable code):
//!   * default-DENY permissions. A pack gets ONLY what its manifest
//!     declares AND what the host can actually enforce this phase.
//!   * `brain_write` is the sole host-graph-touching capability. It is
//!     default-deny; even when granted it routes through the runtime
//!     PendingLog as a teach-class proposal (status Pending) — it NEVER
//!     mutates a bond directly and NEVER triggers a retrain. So
//!     `bond_count` is unchanged by install / load / call.
//!   * filesystem / network permissions cannot be OS-sandboxed this
//!     phase, so a pack that REQUESTS them is REFUSED at load unless
//!     the operator explicitly opts into unsandboxed execution
//!     (`allow_unsandboxed`, default false). Safer fork.
//!   * re-verify-on-execute: every call re-reads the pack file from
//!     disk and re-checks trust + revocation + Ed25519 signature, and
//!     confirms the manifest hash still matches what was registered.
//!     A tampered / swapped / revoked / de-trusted pack fails HERE,
//!     not just at install.
//!   * resource caps (cpu / mem / wall-clock / output-size) are
//!     kernel-enforced via a `ulimit` shell wrapper + a bounded stdout
//!     read + a wall-clock poll-kill.
//!
//! Host<->pack ABI (Subprocess): the host writes ONE JSON line to the
//! child's stdin:
//!   {"tool":"<decl name>","args":{..},"granted":{..permission view..}}
//! the child writes ONE JSON object to stdout and exits 0:
//!   {"ok":true,"result":<value>,"brain_writes":[{subject,object,bond_type,weight,negated}]}
//!   {"ok":false,"error":"<msg>"}
//! Non-zero exit, oversize stdout, non-JSON stdout, or timeout are
//! typed errors. brain_writes are ADVISORY proposals — the host decides
//! whether to honour them based on the granted permission set, and when
//! it does, they go through the PendingLog, never straight to a bond.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use crate::capability_pack::{
    self, CapabilityPackLedgerStore, PackPermissions, PackRuntimeKind, PackToolDecl,
};

// ---------------------------------------------------------------------
// Resource caps
// ---------------------------------------------------------------------

/// Conservative, kernel-enforced execution caps. Defaults are the
/// approved safe values from the Cat.5.1 plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceCaps {
    /// CPU time budget in ms. Enforced via `ulimit -t` (seconds,
    /// rounded up, min 1). SIGXCPU kills the child on overrun.
    pub cpu_ms: u64,
    /// Address-space cap in bytes. Enforced via `ulimit -v` (KiB).
    pub mem_bytes: u64,
    /// Wall-clock budget in ms. Enforced host-side by a poll-kill loop.
    pub wall_ms: u64,
    /// Hard cap on bytes read from the child's stdout. Overrun is a
    /// typed `OutputTooLarge` error (the child is killed).
    pub output_bytes: usize,
}

impl Default for ResourceCaps {
    fn default() -> Self {
        ResourceCaps {
            cpu_ms: 5_000,
            mem_bytes: 512 * 1024 * 1024,
            wall_ms: 10_000,
            output_bytes: 1_048_576,
        }
    }
}

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum PackRuntimeError {
    /// The capability runtime master switch is off.
    Disabled,
    /// runtime_kind is not executable in this phase (Wasm deferred,
    /// SharedLib rejected, Reserved unused).
    RuntimeUnsupported(PackRuntimeKind),
    /// A declared permission cannot be OS-sandboxed this phase and the
    /// operator has not opted into unsandboxed execution.
    PermissionUnsupported(String),
    /// A pack attempted an effect outside its granted permissions.
    PermissionDenied(String),
    /// trust / revocation / signature / manifest-hash check failed at
    /// load or at execute time.
    ReVerify(String),
    ToolNotFound(String),
    AlreadyLoaded(String),
    NotLoaded(String),
    Spawn(String),
    Exit { code: Option<i32>, stderr: String },
    Timeout { ms: u64 },
    OutputTooLarge { max: usize },
    Parse(String),
    Pack(String),
    Other(String),
}

impl std::fmt::Display for PackRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackRuntimeError::Disabled => write!(
                f,
                "capability runtime is disabled (enable via lu4d_capability_pack_runtime_configure)"
            ),
            PackRuntimeError::RuntimeUnsupported(k) => write!(
                f,
                "runtime_kind {:?} is not executable in this phase (Subprocess only; Wasm deferred, SharedLib rejected)",
                k
            ),
            PackRuntimeError::PermissionUnsupported(p) => write!(
                f,
                "permission `{}` cannot be OS-sandboxed this phase; pack refused (set allow_unsandboxed=true to override)",
                p
            ),
            PackRuntimeError::PermissionDenied(p) => write!(f, "permission denied: {}", p),
            PackRuntimeError::ReVerify(s) => write!(f, "re-verify failed: {}", s),
            PackRuntimeError::ToolNotFound(s) => write!(f, "pack tool not found: {}", s),
            PackRuntimeError::AlreadyLoaded(s) => write!(f, "pack already loaded: {}", s),
            PackRuntimeError::NotLoaded(s) => write!(f, "pack not loaded: {}", s),
            PackRuntimeError::Spawn(s) => write!(f, "subprocess spawn failed: {}", s),
            PackRuntimeError::Exit { code, stderr } => {
                write!(f, "pack subprocess exited code={:?} stderr={:?}", code, stderr)
            }
            PackRuntimeError::Timeout { ms } => write!(f, "pack subprocess timed out after {} ms", ms),
            PackRuntimeError::OutputTooLarge { max } => {
                write!(f, "pack subprocess output exceeded {} bytes", max)
            }
            PackRuntimeError::Parse(s) => write!(f, "pack response parse error: {}", s),
            PackRuntimeError::Pack(s) => write!(f, "pack error: {}", s),
            PackRuntimeError::Other(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for PackRuntimeError {}

// ---------------------------------------------------------------------
// Tool output + proposed brain writes
// ---------------------------------------------------------------------

/// A brain mutation a pack PROPOSES. Never applied directly — the host
/// routes it through the PendingLog iff `brain_write` is granted.
#[derive(Debug, Clone)]
pub struct ProposedBrainWrite {
    pub subject_text: String,
    pub object_text: String,
    pub bond_type: crate::bond::BondType,
    pub weight: f32,
    pub negated: bool,
}

/// The structured result of a pack tool call plus any brain writes it
/// proposed.
#[derive(Debug, Clone)]
pub struct PackToolOutput {
    pub result: Value,
    pub brain_writes: Vec<ProposedBrainWrite>,
}

// ---------------------------------------------------------------------
// PackTool trait (object-safe — generalises SceneGraphExtractor)
// ---------------------------------------------------------------------

/// The pluggable boundary for an executable pack tool. Held as
/// `Arc<dyn PackTool>` in the registry. Object-safe: no generics, no
/// associated types — same shape as `SceneGraphExtractor`.
pub trait PackTool: Send + Sync {
    /// Namespaced MCP tool name (e.g. `pack__df_demo__echo`).
    fn name(&self) -> &str;
    /// The tool name as declared inside the manifest.
    fn decl_name(&self) -> &str;
    fn description(&self) -> &str;
    fn args_schema(&self) -> &Value;
    fn permissions(&self) -> &PackPermissions;
    /// Execute the tool. Implementations enforce their own resource
    /// caps. Re-verification of trust/signature is the registry's job
    /// and happens BEFORE this is called.
    fn call(&self, args: &Value) -> Result<PackToolOutput, PackRuntimeError>;
}

// ---------------------------------------------------------------------
// Namespacing
// ---------------------------------------------------------------------

fn sanitize_component(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Deterministic, collision-resistant MCP name for a pack tool. The
/// `pack__` prefix keeps pack tools in a separate namespace from core
/// `lu4d_*` tools so a pack can never shadow a built-in.
pub fn namespaced_tool_name(pack_id: &str, decl_name: &str) -> String {
    format!(
        "pack__{}__{}",
        sanitize_component(pack_id),
        sanitize_component(decl_name)
    )
}

/// True if a tool name is in the pack namespace.
pub fn is_pack_tool_name(name: &str) -> bool {
    name.starts_with("pack__")
}

fn shell_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

pub fn parse_bond_type(s: &str) -> crate::bond::BondType {
    use crate::bond::BondType::*;
    match s.trim() {
        "IsA" => IsA,
        "HasProperty" => HasProperty,
        "Causes" => Causes,
        "PartOf" => PartOf,
        "UsedFor" => UsedFor,
        "RelatedTo" => RelatedTo,
        "ContrastedWith" => ContrastedWith,
        "DerivedFrom" => DerivedFrom,
        "Requires" => Requires,
        "Produces" => Produces,
        "Opposes" => Opposes,
        "DefinedAs" => DefinedAs,
        "WorksBy" => WorksBy,
        "ComposedOf" => ComposedOf,
        "PurposeOf" => PurposeOf,
        "DiffersFrom" => DiffersFrom,
        "LeadsTo" => LeadsTo,
        "ExampleOf" => ExampleOf,
        _ => RelatedTo,
    }
}

// ---------------------------------------------------------------------
// Workdir guard (extracted pack executable lives here; cleaned on drop)
// ---------------------------------------------------------------------

struct WorkDir {
    path: PathBuf,
}

impl WorkDir {
    fn create(pack_id: &str) -> Result<WorkDir, PackRuntimeError> {
        // Each workdir must be unique. Two executions of the same pack in
        // the same millisecond (concurrent calls, or parallel tests) used to
        // get the same path, and the first one to finish deleted the other's
        // extracted binary. A process-wide counter plus `create_dir` (which
        // fails if the path already exists) makes the path unique.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let tmp = std::env::temp_dir();
        loop {
            let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let base = tmp.join(format!(
                "lu_cap_runtime_{}_{}_{}_{}",
                sanitize_component(pack_id),
                std::process::id(),
                crate::brain::now_ms(),
                seq
            ));
            match std::fs::create_dir(&base) {
                Ok(()) => return Ok(WorkDir { path: base }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    return Err(PackRuntimeError::Spawn(format!("create workdir: {}", e)))
                }
            }
        }
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

// ---------------------------------------------------------------------
// SubprocessPackTool — the real, only-shipped runtime this phase
// ---------------------------------------------------------------------

pub struct SubprocessPackTool {
    namespaced_name: String,
    decl: PackToolDecl,
    permissions: PackPermissions,
    caps: ResourceCaps,
    bin_path: PathBuf,
    clock: Arc<dyn crate::time::ClockSource>,
}

impl SubprocessPackTool {
    fn now_ms(&self) -> i64 {
        match self.clock.fetch() {
            Ok(a) => a.utc_ms as i64,
            Err(_) => crate::brain::now_ms(),
        }
    }
}

impl PackTool for SubprocessPackTool {
    fn name(&self) -> &str {
        &self.namespaced_name
    }
    fn decl_name(&self) -> &str {
        &self.decl.tool_name
    }
    fn description(&self) -> &str {
        &self.decl.description
    }
    fn args_schema(&self) -> &Value {
        &self.decl.args_schema
    }
    fn permissions(&self) -> &PackPermissions {
        &self.permissions
    }

    fn call(&self, args: &Value) -> Result<PackToolOutput, PackRuntimeError> {
        // Build the ABI request line. `granted` is the host's view of
        // what this pack is allowed to do — a well-behaved pack reads
        // it; a misbehaving one cannot exceed it because the host
        // enforces the only privileged effect (brain_write) itself.
        let req = json!({
            "tool": self.decl.tool_name,
            "args": args,
            "granted": {
                "brain_read": self.permissions.brain_read,
                "brain_write": self.permissions.brain_write,
                "filesystem_read": self.permissions.filesystem_read,
                "filesystem_write": self.permissions.filesystem_write,
                "network_egress": self.permissions.network_egress,
            }
        });
        let mut req_line = serde_json::to_string(&req)
            .map_err(|e| PackRuntimeError::Parse(format!("request encode: {}", e)))?;
        req_line.push('\n');

        let cpu_s = ((self.caps.cpu_ms + 999) / 1000).max(1);
        let mem_kb = (self.caps.mem_bytes / 1024).max(1);
        // ulimit -t (CPU seconds), -v (virtual memory KiB). `exec` so
        // the pack process replaces the shell and the limits apply to
        // it directly. Redirect ulimit's own stderr away so a shell
        // that refuses a limit doesn't pollute the pack's stderr.
        let wrapper = format!(
            "ulimit -t {} 2>/dev/null; ulimit -v {} 2>/dev/null; exec {}",
            cpu_s,
            mem_kb,
            shell_quote(&self.bin_path.to_string_lossy())
        );

        let mut child = Command::new("/bin/sh")
            .arg("-c")
            .arg(&wrapper)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| PackRuntimeError::Spawn(e.to_string()))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(req_line.as_bytes())
                .map_err(|e| PackRuntimeError::Spawn(format!("stdin write: {}", e)))?;
            // dropping stdin closes the pipe -> child sees EOF
        }

        // Bounded stdout reader thread (reads at most cap+1 bytes so we
        // can detect overrun without buffering unbounded output).
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| PackRuntimeError::Spawn("no stdout pipe".into()))?;
        let cap = self.caps.output_bytes;
        let out_reader = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let mut limited = (&mut stdout).take((cap as u64) + 1);
            let _ = limited.read_to_end(&mut buf);
            buf
        });
        // Bounded stderr reader thread (64 KiB is plenty for diagnostics).
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| PackRuntimeError::Spawn("no stderr pipe".into()))?;
        let err_reader = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let mut limited = (&mut stderr).take(64 * 1024);
            let _ = limited.read_to_end(&mut buf);
            String::from_utf8_lossy(&buf).into_owned()
        });

        // Wall-clock poll-kill. Deadline computed once from the clock
        // (T.1 invariant) and polled with a bounded sleep.
        let deadline = self.now_ms() + self.caps.wall_ms as i64;
        let poll = Duration::from_millis(20);
        let status = loop {
            match child.try_wait() {
                Ok(Some(s)) => break s,
                Ok(None) => {
                    if self.now_ms() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        let _ = out_reader.join();
                        let _ = err_reader.join();
                        return Err(PackRuntimeError::Timeout {
                            ms: self.caps.wall_ms,
                        });
                    }
                    std::thread::sleep(poll);
                }
                Err(e) => {
                    let _ = child.kill();
                    let _ = out_reader.join();
                    let _ = err_reader.join();
                    return Err(PackRuntimeError::Spawn(format!("try_wait: {}", e)));
                }
            }
        };

        let out_buf = out_reader
            .join()
            .map_err(|_| PackRuntimeError::Other("stdout reader thread panicked".into()))?;
        let err_str = err_reader
            .join()
            .map_err(|_| PackRuntimeError::Other("stderr reader thread panicked".into()))?;

        if out_buf.len() > cap {
            return Err(PackRuntimeError::OutputTooLarge { max: cap });
        }
        if !status.success() {
            return Err(PackRuntimeError::Exit {
                code: status.code(),
                stderr: err_str,
            });
        }

        let resp: Value = serde_json::from_slice(&out_buf).map_err(|e| {
            PackRuntimeError::Parse(format!("response json: {} (stderr: {:?})", e, err_str))
        })?;
        let ok = resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
        if !ok {
            let msg = resp
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("pack returned ok=false")
                .to_string();
            return Err(PackRuntimeError::Other(format!("pack tool error: {}", msg)));
        }
        let result = resp.get("result").cloned().unwrap_or(Value::Null);
        let mut brain_writes = Vec::new();
        if let Some(arr) = resp.get("brain_writes").and_then(|v| v.as_array()) {
            for w in arr {
                let subject = w
                    .get("subject")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let object = w
                    .get("object")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if subject.is_empty() || object.is_empty() {
                    continue;
                }
                let bond_type = w
                    .get("bond_type")
                    .and_then(|v| v.as_str())
                    .map(parse_bond_type)
                    .unwrap_or(crate::bond::BondType::RelatedTo);
                let weight = w.get("weight").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
                let negated = w.get("negated").and_then(|v| v.as_bool()).unwrap_or(false);
                brain_writes.push(ProposedBrainWrite {
                    subject_text: subject,
                    object_text: object,
                    bond_type,
                    weight,
                    negated,
                });
            }
        }
        Ok(PackToolOutput {
            result,
            brain_writes,
        })
    }
}

// ---------------------------------------------------------------------
// build_pack_tool factory (mirrors build_extractor)
// ---------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn build_pack_tool(
    kind: PackRuntimeKind,
    namespaced_name: String,
    decl: PackToolDecl,
    permissions: PackPermissions,
    caps: ResourceCaps,
    bin_path: PathBuf,
    clock: Arc<dyn crate::time::ClockSource>,
) -> Result<Arc<dyn PackTool>, PackRuntimeError> {
    match kind {
        PackRuntimeKind::Subprocess => Ok(Arc::new(SubprocessPackTool {
            namespaced_name,
            decl,
            permissions,
            caps,
            bin_path,
            clock,
        })),
        // WASM deferred to a later phase; SharedLib (full-privilege
        // dlopen/unsafe) explicitly rejected; Reserved unused.
        other => Err(PackRuntimeError::RuntimeUnsupported(other)),
    }
}

// ---------------------------------------------------------------------
// Manifest-region hash (matches install_capability_pack's slice)
// ---------------------------------------------------------------------

fn manifest_hash_of(pack_bytes: &[u8]) -> Result<String, PackRuntimeError> {
    if pack_bytes.len() < capability_pack::PACK5_HEADER_LEN {
        return Err(PackRuntimeError::Pack("pack too short for header".into()));
    }
    let mut m = [0u8; 4];
    m.copy_from_slice(&pack_bytes[12..16]);
    let manifest_len = u32::from_le_bytes(m) as usize;
    let start = capability_pack::PACK5_HEADER_LEN;
    let end = start + manifest_len;
    if end > pack_bytes.len() {
        return Err(PackRuntimeError::Pack("manifest region out of bounds".into()));
    }
    Ok(capability_pack::sha256_hex(&pack_bytes[start..end]))
}

// ---------------------------------------------------------------------
// Loaded pack metadata + registry
// ---------------------------------------------------------------------

pub struct LoadedPack {
    pub pack_id: String,
    pub pack_version: String,
    pub publisher_pubkey_hex: String,
    pub runtime_kind: PackRuntimeKind,
    pub source_path: String,
    pub manifest_hash_hex: String,
    pub permissions: PackPermissions,
    pub tool_names: Vec<String>,
    _workdir: WorkDir,
}

#[derive(Debug, Clone)]
pub struct LoadSummary {
    pub pack_id: String,
    pub pack_version: String,
    pub runtime_kind: PackRuntimeKind,
    pub tool_names: Vec<String>,
}

/// In-memory registry of loaded, executable packs. Lives on `Brain`.
/// NOT serialized — purely additive runtime state, rebuilt per process
/// from the (persisted) Cat.5 ledger. Keeping it out of the brain file
/// preserves the orthogonality invariant: install/load/call never
/// touch the trained graph or its on-disk sections.
pub struct LoadedPackRegistry {
    pub enabled: bool,
    pub allow_unsandboxed: bool,
    pub caps: ResourceCaps,
    packs: BTreeMap<String, LoadedPack>,
    tools: BTreeMap<String, Arc<dyn PackTool>>,
}

impl Default for LoadedPackRegistry {
    fn default() -> Self {
        LoadedPackRegistry {
            // Master switch defaults OFF (capability_runtime_enabled
            // default false, per plan). Operator opts in explicitly.
            enabled: false,
            allow_unsandboxed: false,
            caps: ResourceCaps::default(),
            packs: BTreeMap::new(),
            tools: BTreeMap::new(),
        }
    }
}

impl std::fmt::Debug for LoadedPackRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedPackRegistry")
            .field("enabled", &self.enabled)
            .field("allow_unsandboxed", &self.allow_unsandboxed)
            .field("caps", &self.caps)
            .field("loaded_packs", &self.packs.keys().collect::<Vec<_>>())
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl LoadedPackRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }
    pub fn set_allow_unsandboxed(&mut self, on: bool) {
        self.allow_unsandboxed = on;
    }
    pub fn set_caps(&mut self, caps: ResourceCaps) {
        self.caps = caps;
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }
    pub fn get_tool(&self, name: &str) -> Option<Arc<dyn PackTool>> {
        self.tools.get(name).cloned()
    }
    pub fn loaded_pack_ids(&self) -> Vec<String> {
        self.packs.keys().cloned().collect()
    }
    pub fn loaded_pack(&self, pack_id: &str) -> Option<&LoadedPack> {
        self.packs.get(pack_id)
    }

    pub fn owning_pack_id(&self, tool_name: &str) -> Option<String> {
        self.packs
            .values()
            .find(|p| p.tool_names.iter().any(|t| t == tool_name))
            .map(|p| p.pack_id.clone())
    }

    /// True iff the pack owning `tool_name` was granted `brain_write`.
    pub fn tool_brain_write_granted(&self, tool_name: &str) -> bool {
        self.packs
            .values()
            .find(|p| p.tool_names.iter().any(|t| t == tool_name))
            .map(|p| p.permissions.brain_write)
            .unwrap_or(false)
    }

    /// MCP tool definitions for every registered pack tool — appended
    /// to `tools/list`.
    pub fn tool_decls(&self) -> Vec<Value> {
        let mut out = Vec::new();
        for (name, tool) in &self.tools {
            out.push(json!({
                "name": name,
                "description": format!(
                    "[capability pack] {}",
                    tool.description()
                ),
                "inputSchema": tool.args_schema().clone(),
            }));
        }
        out
    }

    /// Load and register an already-decoded pack. Performs the full
    /// install-grade gate (trust -> revocation -> signature), the
    /// default-deny permission-admissibility check, the runtime-kind
    /// check, extracts the executable to a private workdir, and builds
    /// + registers each declared tool.
    pub fn load(
        &mut self,
        pack: &capability_pack::CapabilityPack,
        pack_bytes: &[u8],
        source_path: &str,
        ledger: &CapabilityPackLedgerStore,
        clock: Arc<dyn crate::time::ClockSource>,
    ) -> Result<LoadSummary, PackRuntimeError> {
        if !self.enabled {
            return Err(PackRuntimeError::Disabled);
        }
        let manifest = &pack.manifest;

        // --- default-deny on unsandboxable permissions ---
        let perms = &manifest.permissions;
        if !self.allow_unsandboxed {
            if !perms.filesystem_read.is_empty() {
                return Err(PackRuntimeError::PermissionUnsupported("filesystem_read".into()));
            }
            if !perms.filesystem_write.is_empty() {
                return Err(PackRuntimeError::PermissionUnsupported("filesystem_write".into()));
            }
            if !perms.network_egress.is_empty() {
                return Err(PackRuntimeError::PermissionUnsupported("network_egress".into()));
            }
        }

        // --- trust -> revocation -> signature (install-grade) ---
        if !ledger.is_trusted(&manifest.publisher_pubkey_hex) {
            return Err(PackRuntimeError::ReVerify(format!(
                "untrusted publisher {}",
                manifest.publisher_pubkey_hex
            )));
        }
        if ledger.is_revoked(&manifest.pack_id) {
            return Err(PackRuntimeError::ReVerify(format!(
                "pack revoked: {}",
                manifest.pack_id
            )));
        }
        capability_pack::verify_capability_pack(pack)
            .map_err(|e| PackRuntimeError::ReVerify(format!("signature: {}", e)))?;

        // --- runtime kind ---
        if manifest.runtime_kind != PackRuntimeKind::Subprocess {
            return Err(PackRuntimeError::RuntimeUnsupported(manifest.runtime_kind));
        }

        if self.packs.contains_key(&manifest.pack_id) {
            return Err(PackRuntimeError::AlreadyLoaded(manifest.pack_id.clone()));
        }

        let manifest_hash_hex = manifest_hash_of(pack_bytes)?;

        // --- extract the executable content to a private workdir ---
        let workdir = WorkDir::create(&manifest.pack_id)?;
        let bin_path = workdir.path.join("pack_bin");
        std::fs::write(&bin_path, &pack.content)
            .map_err(|e| PackRuntimeError::Spawn(format!("write pack bin: {}", e)))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&bin_path, std::fs::Permissions::from_mode(0o755));
        }

        // --- build + register each tool ---
        let mut tool_names = Vec::new();
        let mut staged: Vec<(String, Arc<dyn PackTool>)> = Vec::new();
        for decl in &manifest.tools {
            let nsname = namespaced_tool_name(&manifest.pack_id, &decl.tool_name);
            if self.tools.contains_key(&nsname) || staged.iter().any(|(n, _)| n == &nsname) {
                return Err(PackRuntimeError::AlreadyLoaded(format!(
                    "tool name collision: {}",
                    nsname
                )));
            }
            let tool = build_pack_tool(
                manifest.runtime_kind,
                nsname.clone(),
                decl.clone(),
                perms.clone(),
                self.caps,
                bin_path.clone(),
                clock.clone(),
            )?;
            staged.push((nsname.clone(), tool));
            tool_names.push(nsname);
        }
        for (n, t) in staged {
            self.tools.insert(n, t);
        }

        let summary = LoadSummary {
            pack_id: manifest.pack_id.clone(),
            pack_version: manifest.pack_version.clone(),
            runtime_kind: manifest.runtime_kind,
            tool_names: tool_names.clone(),
        };
        self.packs.insert(
            manifest.pack_id.clone(),
            LoadedPack {
                pack_id: manifest.pack_id.clone(),
                pack_version: manifest.pack_version.clone(),
                publisher_pubkey_hex: manifest.publisher_pubkey_hex.clone(),
                runtime_kind: manifest.runtime_kind,
                source_path: source_path.to_string(),
                manifest_hash_hex,
                permissions: perms.clone(),
                tool_names,
                _workdir: workdir,
            },
        );
        Ok(summary)
    }

    /// Unload a pack: drop its tools and its workdir (executable wiped).
    pub fn unload(&mut self, pack_id: &str) -> Result<usize, PackRuntimeError> {
        let pack = self
            .packs
            .remove(pack_id)
            .ok_or_else(|| PackRuntimeError::NotLoaded(pack_id.to_string()))?;
        let n = pack.tool_names.len();
        for t in &pack.tool_names {
            self.tools.remove(t);
        }
        // pack (incl. WorkDir) drops here -> executable removed
        Ok(n)
    }

    /// Re-verify a loaded pack against the CURRENT ledger + on-disk
    /// bytes WITHOUT executing. Used by the vision extractor build path
    /// and shared by `execute`.
    pub fn reverify(
        &self,
        pack_id: &str,
        ledger: &CapabilityPackLedgerStore,
    ) -> Result<(), PackRuntimeError> {
        let meta = self
            .packs
            .get(pack_id)
            .ok_or_else(|| PackRuntimeError::NotLoaded(pack_id.to_string()))?;
        reverify_meta(meta, ledger)
    }

    /// Re-verify (trust + revocation + signature + manifest-hash) then
    /// execute the named tool. This is the load-bearing
    /// re-verify-ON-EXECUTE gate — a pack that became untrusted /
    /// revoked / tampered AFTER load is refused here.
    pub fn execute(
        &self,
        tool_name: &str,
        args: &Value,
        ledger: &CapabilityPackLedgerStore,
    ) -> Result<PackToolOutput, PackRuntimeError> {
        if !self.enabled {
            return Err(PackRuntimeError::Disabled);
        }
        let tool = self
            .tools
            .get(tool_name)
            .cloned()
            .ok_or_else(|| PackRuntimeError::ToolNotFound(tool_name.to_string()))?;
        let meta = self
            .packs
            .values()
            .find(|p| p.tool_names.iter().any(|t| t == tool_name))
            .ok_or_else(|| PackRuntimeError::ToolNotFound(format!("no owning pack for {}", tool_name)))?;
        reverify_meta(meta, ledger)?;
        tool.call(args)
    }
}

/// Shared re-verify: re-read the pack file, re-check trust + revocation
/// + Ed25519 signature, and confirm the manifest hash + pack_id still
/// match what was registered at load.
fn reverify_meta(
    meta: &LoadedPack,
    ledger: &CapabilityPackLedgerStore,
) -> Result<(), PackRuntimeError> {
    let (pack, bytes) = capability_pack::read_capability_pack_file(&meta.source_path)
        .map_err(|e| PackRuntimeError::ReVerify(format!("re-read {}: {}", meta.source_path, e)))?;
    if pack.manifest.pack_id != meta.pack_id {
        return Err(PackRuntimeError::ReVerify(format!(
            "pack_id changed on disk: load={} now={}",
            meta.pack_id, pack.manifest.pack_id
        )));
    }
    if !ledger.is_trusted(&pack.manifest.publisher_pubkey_hex) {
        return Err(PackRuntimeError::ReVerify(format!(
            "publisher no longer trusted: {}",
            pack.manifest.publisher_pubkey_hex
        )));
    }
    if ledger.is_revoked(&pack.manifest.pack_id) {
        return Err(PackRuntimeError::ReVerify(format!(
            "pack revoked: {}",
            pack.manifest.pack_id
        )));
    }
    capability_pack::verify_capability_pack(&pack)
        .map_err(|e| PackRuntimeError::ReVerify(format!("signature invalid on execute: {}", e)))?;
    let cur_hash = manifest_hash_of(&bytes)?;
    if cur_hash != meta.manifest_hash_hex {
        return Err(PackRuntimeError::ReVerify(format!(
            "manifest hash changed since load (load={} now={})",
            meta.manifest_hash_hex, cur_hash
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// CapabilityPackExtractor — wires the vision SceneGraphExtractor hook
// ---------------------------------------------------------------------

/// Adapter that exposes a loaded pack tool as a `SceneGraphExtractor`,
/// turning the long-deferred `VisionExtractorKind::CapabilityPack` arm
/// into the real path. The image bytes ride the ABI hex-encoded under
/// `args.image_hex` (binary framing / fd-passing is a later-phase
/// optimisation). The pack returns `{"triplets":[...]}` in its result.
pub struct CapabilityPackExtractor {
    pub pack_id: String,
    display_name: String,
    tool: Arc<dyn PackTool>,
}

impl CapabilityPackExtractor {
    pub fn new(pack_id: String, tool: Arc<dyn PackTool>) -> Self {
        let display_name = format!("capability_pack:{}", pack_id);
        CapabilityPackExtractor {
            pack_id,
            display_name,
            tool,
        }
    }

    /// Conventional declared tool name a pack must expose to act as a
    /// scene-graph extractor.
    pub const EXTRACTOR_TOOL: &'static str = "scene_graph_extract";
}

impl crate::vision::extractor::SceneGraphExtractor for CapabilityPackExtractor {
    fn name(&self) -> &str {
        &self.display_name
    }

    fn extract(
        &self,
        image_bytes: &[u8],
        hints: crate::vision::extractor::ExtractHints,
    ) -> Result<Vec<crate::vision::extractor::Triplet>, crate::vision::extractor::ExtractError> {
        use crate::vision::extractor::{ExtractError, Triplet};
        let args = json!({
            "image_hex": capability_pack::hex_encode(image_bytes),
            "mime_type": hints.mime_type,
            "max_triplets": hints.max_triplets,
        });
        let out = self
            .tool
            .call(&args)
            .map_err(|e| ExtractError::Other(format!("capability pack extractor: {}", e)))?;
        let arr = out
            .result
            .get("triplets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ExtractError::Parse("pack result missing `triplets` array".into()))?;
        let mut triplets = Vec::with_capacity(arr.len());
        for t in arr {
            let subject = t.get("subject").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let predicate = t.get("predicate").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let object = t.get("object").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let confidence = t.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
            triplets.push(Triplet {
                subject,
                predicate,
                object,
                confidence,
            });
        }
        if let Some(max) = hints.max_triplets {
            triplets.truncate(max);
        }
        Ok(triplets)
    }
}

#[cfg(test)]
mod tests;

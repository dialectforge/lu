//! Port R input — scene-graph extractor trait + bundled implementations.
//!
//! See `vision/mod.rs` for the port-level overview. This file defines:
//!
//!   * `Triplet`                   - (subject, predicate, object, conf)
//!   * `ExtractHints`              - mime_type + max_triplets advisories
//!   * `ExtractError`              - typed failure modes (timeout,
//!                                   subprocess-spawn-failure, parse,
//!                                   unsupported-mime, oversize)
//!   * `SceneGraphExtractor` trait - the pluggable boundary
//!   * `StubExtractor`             - placeholder, dev-only
//!   * `SubprocessExtractor`       - external CLI integration
//!
//! The configured extractor is held on `Brain` as a trait object and
//! invoked by `mcp_vision::tool_image_ingest`. The trait is fully
//! object-safe (no generics, no associated types).
//!
//! Future direction (Cat.5.1):
//!
//! ```ignore
//! /// Long-term target: load an extractor implementation from a signed
//! /// `.lupack5` capability pack. Once Cat.5.1 lands the executable-pack
//! /// flavour, a `CapabilityPackExtractor { pack_id: String }` variant
//! /// will dlopen the pack's exported `lu_scene_graph_extract` symbol
//! /// and forward through the trait. NOT IMPLEMENTED yet — Port R input
//! /// only ships Stub + Subprocess. The runtime_config enum already
//! /// has the `CapabilityPack { pack_id }` arm so the config surface
//! /// is forward-compatible.
//! pub struct CapabilityPackExtractor { pub pack_id: String }
//! ```

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One scene-graph edge. Subject and object are free-form strings
/// (the extractor's choice of node label); predicate is the relation
/// string ("has", "is_behind", "on", ...). `confidence` is in [0, 1].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Triplet {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}

/// Caller-supplied advisories. Both fields are best-effort — an
/// extractor may sniff its own mime type and may emit fewer triplets
/// than `max_triplets` requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractHints {
    pub mime_type: Option<String>,
    pub max_triplets: Option<usize>,
}

/// Typed extractor failure modes. `Other` is the catch-all so we don't
/// have to grow the enum for every new wire-protocol error.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtractError {
    UnsupportedMime(String),
    OversizeImage { len: usize, max: usize },
    Timeout { ms: u64 },
    SubprocessSpawn(String),
    SubprocessExit { code: Option<i32>, stderr: String },
    Parse(String),
    Other(String),
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractError::UnsupportedMime(m) => write!(f, "unsupported mime type: {}", m),
            ExtractError::OversizeImage { len, max } => write!(
                f,
                "image size {} exceeds image_max_size_bytes ({})",
                len, max
            ),
            ExtractError::Timeout { ms } => write!(f, "extractor timed out after {} ms", ms),
            ExtractError::SubprocessSpawn(s) => write!(f, "subprocess spawn failed: {}", s),
            ExtractError::SubprocessExit { code, stderr } => write!(
                f,
                "subprocess exited code={:?} stderr={:?}",
                code, stderr
            ),
            ExtractError::Parse(s) => write!(f, "extractor output parse error: {}", s),
            ExtractError::Other(s) => write!(f, "{}", s),
        }
    }
}

/// Pluggable extractor boundary. Implementations are held on the
/// brain as `Arc<dyn SceneGraphExtractor>` so swapping at runtime is
/// cheap. Object-safe: no generics, no associated types.
pub trait SceneGraphExtractor: Send + Sync {
    /// Short identifier — "stub", "subprocess", "capability_pack:<id>".
    fn name(&self) -> &str;
    /// Extract a list of (subject, predicate, object, confidence)
    /// triplets from the supplied image bytes. Implementations are
    /// responsible for honouring `hints.max_triplets` if they can.
    fn extract(
        &self,
        image_bytes: &[u8],
        hints: ExtractHints,
    ) -> Result<Vec<Triplet>, ExtractError>;

    /// Whether this extractor genuinely perceives image *contents*. A real
    /// CV/scene extractor returns true; the dev-only `StubExtractor` returns
    /// false so honesty-sensitive callers (e.g. `tool_image_describe`) can
    /// refuse to present its placeholder output as a real description.
    /// Honesty floor: never dress a non-perceiving extractor up as a scene.
    fn perceives_contents(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------
// StubExtractor
// ---------------------------------------------------------------------

/// Development-only extractor. Returns a single placeholder triplet
/// stamped with the current RFC3339 timestamp. Always logs a warning.
/// The clock source is taken from the brain so the T.1 invariant
/// (no direct std::time calls in code paths that look at "now") is
/// preserved.
pub struct StubExtractor {
    pub clock: Arc<dyn crate::time::ClockSource>,
}

impl StubExtractor {
    pub fn new(clock: Arc<dyn crate::time::ClockSource>) -> Self {
        StubExtractor { clock }
    }
}

fn ms_to_rfc3339(ms_since_epoch: i64) -> String {
    // Minimal RFC3339 in UTC. We avoid pulling chrono (constraint C10
    // — pure Rust, but more importantly we don't want a new crate just
    // for one timestamp string). Algorithm: civil-from-days (Howard
    // Hinnant). Negative timestamps are clamped to epoch — Lu's clock
    // never goes negative in practice.
    let secs = (ms_since_epoch / 1000).max(0);
    let ms_part = (ms_since_epoch % 1000).max(0) as u32;
    let days = (secs / 86_400) as i64;
    let secs_of_day = (secs % 86_400) as u32;
    let h = secs_of_day / 3600;
    let m = (secs_of_day % 3600) / 60;
    let s = secs_of_day % 60;
    // Howard Hinnant civil_from_days, shifted so day 0 = 1970-01-01.
    let z = days + 719_468;
    let era = if z >= 0 { z / 146_097 } else { (z - 146_096) / 146_097 };
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        y, mo, d, h, m, s, ms_part
    )
}

impl SceneGraphExtractor for StubExtractor {
    fn name(&self) -> &str {
        "stub"
    }

    fn perceives_contents(&self) -> bool {
        // The stub ignores image bytes entirely -- it cannot perceive
        // contents, so callers must not treat its output as a description.
        false
    }

    fn extract(
        &self,
        _image_bytes: &[u8],
        _hints: ExtractHints,
    ) -> Result<Vec<Triplet>, ExtractError> {
        let now_ms = match self.clock.fetch() {
            Ok(a) => a.utc_ms as i64,
            Err(_) => crate::brain::now_ms(),
        };
        let stamp = ms_to_rfc3339(now_ms);
        eprintln!(
            "[lu4d][port-r-input] WARNING: StubExtractor active — no real \
            scene-graph CV is configured. Returning placeholder triplet."
        );
        Ok(vec![Triplet {
            subject: "image".to_string(),
            predicate: "uploaded_at".to_string(),
            object: stamp,
            confidence: 0.1,
        }])
    }
}

// ---------------------------------------------------------------------
// SubprocessExtractor
// ---------------------------------------------------------------------

/// Shells out to a configured external CV pipeline. The subprocess
/// receives the image bytes on stdin, returns a JSON array of
/// `Triplet` records on stdout, and exits 0 on success. A non-zero
/// exit code or non-JSON stdout is reported as `SubprocessExit` /
/// `Parse`. The timeout is enforced via wall-clock polling against
/// the supplied `ClockSource` (T.1 invariant — the stub does not
/// reach for std::time::Instant inside the polling hot path; instead
/// it computes a deadline once and uses bounded poll/sleep).
pub struct SubprocessExtractor {
    pub command_path: PathBuf,
    pub timeout_ms: u64,
    pub clock: Arc<dyn crate::time::ClockSource>,
}

impl SubprocessExtractor {
    pub fn new(
        command_path: PathBuf,
        timeout_ms: u64,
        clock: Arc<dyn crate::time::ClockSource>,
    ) -> Self {
        SubprocessExtractor {
            command_path,
            timeout_ms,
            clock,
        }
    }
}

impl SceneGraphExtractor for SubprocessExtractor {
    fn name(&self) -> &str {
        "subprocess"
    }

    fn extract(
        &self,
        image_bytes: &[u8],
        hints: ExtractHints,
    ) -> Result<Vec<Triplet>, ExtractError> {
        // Spawn with stdin piped so we can stream the bytes in.
        // stdout/stderr are also piped so we can capture the result.
        let mut child = Command::new(&self.command_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .args(hints.mime_type.iter().flat_map(|m| ["--mime".to_string(), m.clone()]))
            .args(
                hints
                    .max_triplets
                    .iter()
                    .flat_map(|n| ["--max".to_string(), n.to_string()]),
            )
            .spawn()
            .map_err(|e| ExtractError::SubprocessSpawn(e.to_string()))?;

        // Write image bytes to stdin, then close it so the subprocess
        // sees EOF.
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(image_bytes)
                .map_err(|e| ExtractError::SubprocessSpawn(format!("stdin write: {}", e)))?;
            // dropping stdin closes the pipe
        }

        // Poll wait_timeout-style on a budget. std::process doesn't
        // ship a portable timeout helper, so we use a short sleep loop
        // bounded by the configured deadline. We avoid std::time::Instant
        // for the deadline check (T.1 invariant) by using the configured
        // clock for the start point and Instant only for the elapsed
        // sleep increment. (The clock comes back as monotonic-ish UTC
        // ms; that's fine for a coarse timeout.)
        let start = match self.clock.fetch() {
            Ok(a) => a.utc_ms as i64,
            Err(_) => crate::brain::now_ms(),
        };
        let deadline = start + self.timeout_ms as i64;
        let poll = Duration::from_millis(20);
        loop {
            match child.try_wait() {
                Ok(Some(_status)) => break,
                Ok(None) => {
                    let now = match self.clock.fetch() {
                        Ok(a) => a.utc_ms as i64,
                        Err(_) => crate::brain::now_ms(),
                    };
                    if now >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(ExtractError::Timeout {
                            ms: self.timeout_ms,
                        });
                    }
                    // Use Instant only for the sleep increment, not
                    // for the deadline check itself.
                    let s = Instant::now();
                    while s.elapsed() < poll {
                        std::thread::sleep(poll - s.elapsed().min(poll));
                        break;
                    }
                }
                Err(e) => {
                    return Err(ExtractError::SubprocessSpawn(format!(
                        "try_wait: {}",
                        e
                    )))
                }
            }
        }

        let output = child
            .wait_with_output()
            .map_err(|e| ExtractError::SubprocessSpawn(format!("wait_with_output: {}", e)))?;
        if !output.status.success() {
            return Err(ExtractError::SubprocessExit {
                code: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        let triplets: Vec<Triplet> = serde_json::from_slice(&output.stdout)
            .map_err(|e| ExtractError::Parse(format!("json: {}", e)))?;
        let triplets = if let Some(max) = hints.max_triplets {
            triplets.into_iter().take(max).collect()
        } else {
            triplets
        };
        Ok(triplets)
    }
}

// ---------------------------------------------------------------------
// Helpers reused by the MCP layer
// ---------------------------------------------------------------------

/// Sniff a best-guess mime type from the leading bytes of the file.
/// Pure Rust, no `image` crate dependency for this — we only need to
/// distinguish the four formats Lu accepts by default (jpeg, png,
/// webp, gif). Returns `None` if the bytes don't match a known sig.
pub fn sniff_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("image/jpeg");
    }
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some("image/png");
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    None
}

/// Parse REAL pixel dimensions (width, height) from an image's own header
/// bytes, for the four formats Lu accepts. This READS the file -- it is not
/// a guess and not a perception of contents. Returns `None` if the header
/// cannot be parsed. Pure Rust, no `image` crate dependency.
pub fn probe_image_dimensions(bytes: &[u8], _mime: &str) -> Option<(u32, u32)> {
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        if bytes.len() >= 24 && &bytes[12..16] == b"IHDR" {
            let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
            let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
            if w > 0 && h > 0 {
                return Some((w, h));
            }
        }
        return None;
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        if bytes.len() >= 10 {
            let w = u16::from_le_bytes([bytes[6], bytes[7]]) as u32;
            let h = u16::from_le_bytes([bytes[8], bytes[9]]) as u32;
            if w > 0 && h > 0 {
                return Some((w, h));
            }
        }
        return None;
    }
    if bytes.starts_with(&[0xFF, 0xD8]) {
        let mut i = 2usize;
        while i + 9 < bytes.len() {
            if bytes[i] != 0xFF {
                i += 1;
                continue;
            }
            let marker = bytes[i + 1];
            if marker == 0xFF {
                i += 1;
                continue;
            }
            if (0xD0..=0xD9).contains(&marker) || marker == 0x01 {
                i += 2;
                continue;
            }
            let seg_len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            let is_sof = matches!(marker, 0xC0..=0xCF)
                && marker != 0xC4
                && marker != 0xC8
                && marker != 0xCC;
            if is_sof {
                if i + 9 <= bytes.len() {
                    let h = u16::from_be_bytes([bytes[i + 5], bytes[i + 6]]) as u32;
                    let w = u16::from_be_bytes([bytes[i + 7], bytes[i + 8]]) as u32;
                    if w > 0 && h > 0 {
                        return Some((w, h));
                    }
                }
                return None;
            }
            if seg_len < 2 {
                return None;
            }
            i += 2 + seg_len;
        }
        return None;
    }
    if bytes.len() >= 30 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        let fourcc = &bytes[12..16];
        if fourcc == b"VP8X" {
            let w = 1 + (bytes[24] as u32 | (bytes[25] as u32) << 8 | (bytes[26] as u32) << 16);
            let h = 1 + (bytes[27] as u32 | (bytes[28] as u32) << 8 | (bytes[29] as u32) << 16);
            return Some((w, h));
        }
        if fourcc == b"VP8 " {
            if bytes[23] == 0x9d && bytes[24] == 0x01 && bytes[25] == 0x2a {
                let w = (u16::from_le_bytes([bytes[26], bytes[27]]) & 0x3FFF) as u32;
                let h = (u16::from_le_bytes([bytes[28], bytes[29]]) & 0x3FFF) as u32;
                if w > 0 && h > 0 {
                    return Some((w, h));
                }
            }
            return None;
        }
        if fourcc == b"VP8L" {
            if bytes.len() >= 25 && bytes[20] == 0x2f {
                let b0 = bytes[21] as u32;
                let b1 = bytes[22] as u32;
                let b2 = bytes[23] as u32;
                let b3 = bytes[24] as u32;
                let w = 1 + (((b1 & 0x3F) << 8) | b0);
                let h = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6));
                if w > 0 && h > 0 {
                    return Some((w, h));
                }
            }
            return None;
        }
        return None;
    }
    None
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::SystemClockSource;

    fn clock() -> Arc<dyn crate::time::ClockSource> {
        Arc::new(SystemClockSource::new()) as Arc<dyn crate::time::ClockSource>
    }

    #[test]
    fn stub_returns_one_placeholder_triplet() {
        let s = StubExtractor::new(clock());
        let out = s.extract(b"\x89PNG\r\n\x1a\n", ExtractHints::default()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].subject, "image");
        assert_eq!(out[0].predicate, "uploaded_at");
        assert!((out[0].confidence - 0.1).abs() < 1e-6);
        assert_eq!(s.name(), "stub");
        // The object is an RFC3339 timestamp; check the shape.
        assert!(out[0].object.ends_with('Z'));
        assert!(out[0].object.contains('T'));
    }

    #[test]
    fn rfc3339_round_shape() {
        let s = ms_to_rfc3339(0);
        assert_eq!(s, "1970-01-01T00:00:00.000Z");
        let s2 = ms_to_rfc3339(1_700_000_000_000);
        // 2023-11-14T22:13:20.000Z
        assert!(s2.starts_with("2023-11-14T22:13:20."));
    }

    #[test]
    fn sniff_recognises_png_jpeg_gif_webp() {
        assert_eq!(sniff_mime(&[0xFF, 0xD8, 0xFF, 0xE0, 0, 0]), Some("image/jpeg"));
        assert_eq!(
            sniff_mime(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
            Some("image/png")
        );
        let mut webp = Vec::from(*b"RIFF\0\0\0\0WEBP");
        webp.extend_from_slice(&[0; 4]);
        assert_eq!(sniff_mime(&webp), Some("image/webp"));
        assert_eq!(sniff_mime(b"GIF89a..."), Some("image/gif"));
        assert_eq!(sniff_mime(b"random_text"), None);
    }

    #[test]
    fn subprocess_returns_parsed_json_triplets() {
        // Build a tiny shell-script extractor that ignores stdin and
        // prints a fixed JSON array.
        let dir = tempdir_for_test();
        let script = dir.join("mock_extractor.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\ncat > /dev/null\nprintf '[{\"subject\":\"dog\",\"predicate\":\"has\",\"object\":\"tail\",\"confidence\":0.85}]'\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut p = std::fs::metadata(&script).unwrap().permissions();
            p.set_mode(0o755);
            std::fs::set_permissions(&script, p).unwrap();
        }
        let s = SubprocessExtractor::new(script.clone(), 30_000, clock());
        let out = s.extract(b"PNG", ExtractHints::default()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].subject, "dog");
        assert_eq!(out[0].predicate, "has");
        assert_eq!(out[0].object, "tail");
        assert_eq!(s.name(), "subprocess");
    }

    #[test]
    fn subprocess_timeout_fires() {
        let dir = tempdir_for_test();
        let script = dir.join("slow_extractor.sh");
        std::fs::write(&script, "#!/bin/sh\nsleep 5\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut p = std::fs::metadata(&script).unwrap().permissions();
            p.set_mode(0o755);
            std::fs::set_permissions(&script, p).unwrap();
        }
        let s = SubprocessExtractor::new(script, 200, clock());
        let err = s.extract(b"PNG", ExtractHints::default()).unwrap_err();
        match err {
            ExtractError::Timeout { ms } => assert_eq!(ms, 200),
            other => panic!("expected Timeout, got {:?}", other),
        }
    }

    fn tempdir_for_test() -> PathBuf {
        let mut p = std::env::temp_dir();
        let nonce = format!(
            "lu_port_r_{}_{}",
            std::process::id(),
            crate::brain::now_ms()
        );
        p.push(nonce);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn stub_does_not_perceive_contents() {
        let s = StubExtractor::new(clock());
        assert!(!s.perceives_contents(), "stub must not claim to perceive");
    }

    #[test]
    fn subprocess_perceives_contents_by_default() {
        let s = SubprocessExtractor::new(PathBuf::from("/bin/true"), 1000, clock());
        assert!(s.perceives_contents());
    }

    #[test]
    fn probe_dims_png_reads_ihdr() {
        let mut b = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, b'I', b'H', b'D', b'R',
            0x00, 0x00, 0x01, 0x40, 0x00, 0x00, 0x00, 0x78,
        ];
        b.extend_from_slice(&[0u8; 8]);
        assert_eq!(probe_image_dimensions(&b, "image/png"), Some((320, 120)));
    }

    #[test]
    fn probe_dims_gif_reads_screen_descriptor() {
        let mut b = Vec::from(*b"GIF89a");
        b.extend_from_slice(&[0xC8, 0x00, 0x64, 0x00]);
        b.extend_from_slice(&[0u8; 4]);
        assert_eq!(probe_image_dimensions(&b, "image/gif"), Some((200, 100)));
    }

    #[test]
    fn probe_dims_unknown_returns_none() {
        assert_eq!(probe_image_dimensions(b"not an image", "image/png"), None);
    }
}

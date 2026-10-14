//! Shared MCP JSON-RPC response envelope helper.
//!
//! `wrap_content` was originally defined in `src/mcp.rs`, but that module
//! is the *admin* MCP surface and is now compiled out of the public
//! guest binary via the `admin_mcp` feature gate (OP-007 / Port U
//! privilege separation). A handful of non-admin modules (`mcp_sense`,
//! `mcp_t11`) still need to build MCP-shaped envelopes, so the helper
//! lives here in an always-compiled module with no admin coupling.

use serde_json::{json, Value};

/// Wrap `text` + `structured` into the MCP `tools/call` result shape:
/// `{ "content": [{"type":"text","text":..}], "structuredContent": .. }`.
pub(crate) fn wrap_content(text: String, structured: Value) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    })
}

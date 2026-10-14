//! Guest auth surface (OP-007 / Port U).
//!
//! Committed design internal notes security architecture): the public
//! surface authenticates John via a **passphrase** and guests via
//! single-use **invite codes**. Crucially, invite *issuance* is an ADMIN
//! function that must never live on the guest surface — so this module
//! only VERIFIES a passphrase and REDEEMS pre-seeded invite codes. Codes
//! are provisioned out-of-band (by `lu_admin_mcp`, future work) and
//! handed to this process via the `LU_GUEST_INVITES` env var.
//!
//! Scaffold scope: passphrase hash + seeded single-use invites + bearer
//! session tokens, all in-memory. Persistence, rate limiting, and
//! invite issuance are explicit follow-ups.

use axum::http::HeaderMap;
use rand::RngCore;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Mutex;

/// The authenticated principal behind a request.
#[derive(Clone, Debug)]
pub enum Principal {
    /// John, authenticated via the owner passphrase.
    Owner,
    /// A guest, authenticated via a redeemed invite code.
    Guest(String),
}

impl Principal {
    pub fn label(&self) -> String {
        match self {
            Principal::Owner => "owner".to_string(),
            Principal::Guest(code) => format!("guest:{}", code),
        }
    }
}

pub struct AuthState {
    /// SHA-256 of the owner passphrase, or None if no passphrase was
    /// configured (in which case owner auth is disabled).
    passphrase_hash: Option<[u8; 32]>,
    /// Seeded invite codes -> redeemed?. Single-use.
    invites: Mutex<HashMap<String, bool>>,
    /// Issued bearer tokens -> principal.
    tokens: Mutex<HashMap<String, Principal>>,
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&out);
    arr
}

/// Constant-time-ish equality for fixed-size hashes.
fn ct_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

fn random_token() -> String {
    let mut buf = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut buf);
    let mut s = String::with_capacity(64);
    for b in buf {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

impl AuthState {
    /// Build from env: `LU_GUEST_PASSPHRASE` (owner passphrase, hashed at
    /// startup) and `LU_GUEST_INVITES` (comma-separated seeded codes).
    pub fn from_env() -> Self {
        let passphrase_hash = std::env::var("LU_GUEST_PASSPHRASE")
            .ok()
            .filter(|p| !p.is_empty())
            .map(|p| sha256(p.as_bytes()));
        let mut invites = HashMap::new();
        if let Ok(s) = std::env::var("LU_GUEST_INVITES") {
            for code in s.split(',').map(|c| c.trim()).filter(|c| !c.is_empty()) {
                invites.insert(code.to_string(), false);
            }
        }
        AuthState {
            passphrase_hash,
            invites: Mutex::new(invites),
            tokens: Mutex::new(HashMap::new()),
        }
    }

    pub fn owner_configured(&self) -> bool {
        self.passphrase_hash.is_some()
    }

    pub fn seeded_invite_count(&self) -> usize {
        self.invites.lock().unwrap().len()
    }

    pub fn verify_passphrase(&self, passphrase: &str) -> bool {
        match &self.passphrase_hash {
            Some(h) => ct_eq(&sha256(passphrase.as_bytes()), h),
            None => false,
        }
    }

    /// Redeem a seeded single-use invite code. Returns the guest principal
    /// on success, None if the code is unknown or already redeemed.
    pub fn redeem_invite(&self, code: &str) -> Option<Principal> {
        let mut inv = self.invites.lock().unwrap();
        match inv.get_mut(code) {
            Some(redeemed) if !*redeemed => {
                *redeemed = true;
                Some(Principal::Guest(code.to_string()))
            }
            _ => None,
        }
    }

    pub fn issue_token(&self, principal: Principal) -> String {
        let token = random_token();
        self.tokens.lock().unwrap().insert(token.clone(), principal);
        token
    }

    /// Resolve the principal for an `Authorization: Bearer <token>` header.
    pub fn principal_for_bearer(&self, headers: &HeaderMap) -> Option<Principal> {
        let raw = headers.get(axum::http::header::AUTHORIZATION)?.to_str().ok()?;
        let token = raw.strip_prefix("Bearer ")?;
        self.tokens.lock().unwrap().get(token).cloned()
    }
}

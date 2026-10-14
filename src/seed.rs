//! Cryptographic seed identity for knowledge nodes.
//!
//! Each node in the Lu 4D graph is identified by a 256-bit seed,
//! generated from random bytes. Seeds are immutable once created.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A 256-bit cryptographic identity for a knowledge node.
///
/// Storage v2 Phase 2.1: also derives `rkyv::{Archive, Serialize,
/// Deserialize}` so the v3 zero-copy load path (Slice 2) can mmap a
/// brain file and read seeds without allocating. The archived form
/// keeps `Hash + PartialEq + Eq` so seeds can be used as keys in
/// `rkyv::collections::ArchivedHashSet` / `ArchivedHashMap` (see
/// `ScarLog::scarred_seeds` and `BrainFile::gap_seed_frequency`,
/// which Slice 2 will materialize via rkyv).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Hash, PartialEq, Eq))]
pub struct Seed(pub [u8; 32]);

impl Seed {
    /// Generate a new random seed using the thread-local RNG.
    pub fn generate() -> Self {
        let mut bytes = [0u8; 32];
        rand::thread_rng().fill(&mut bytes);
        Seed(bytes)
    }

    /// Create a seed from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Seed(bytes)
    }

    /// Return the raw bytes of this seed.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Create a deterministic seed from a string (for testing).
    /// Uses a simple hash-spread — NOT cryptographically secure.
    pub fn from_label(label: &str) -> Self {
        let mut bytes = [0u8; 32];
        for (i, b) in label.bytes().enumerate() {
            bytes[i % 32] ^= b;
            // Spread influence across bytes
            bytes[(i + 7) % 32] = bytes[(i + 7) % 32].wrapping_add(b.wrapping_mul(31));
        }
        Seed(bytes)
    }
}

impl fmt::Display for Seed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Seed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Seed({})", &self.to_string()[..12])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_generation_unique() {
        let a = Seed::generate();
        let b = Seed::generate();
        assert_ne!(a, b);
    }

    #[test]
    fn test_seed_display_hex() {
        let seed = Seed([0xab; 32]);
        let hex = seed.to_string();
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_seed_from_label_deterministic() {
        let a = Seed::from_label("test_node");
        let b = Seed::from_label("test_node");
        assert_eq!(a, b);
    }

    #[test]
    fn test_seed_from_label_different() {
        let a = Seed::from_label("alpha");
        let b = Seed::from_label("beta");
        assert_ne!(a, b);
    }
}

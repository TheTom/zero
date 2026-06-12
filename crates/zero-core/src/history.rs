// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! The block/projector memory model — graded decay for long contexts (loop wakes
//! and interactive sessions alike). A conversation is a `Vec<Block>`; when it
//! exceeds budget the projector **demotes** old blocks to spill pointers instead
//! of dropping them — recency tiers, first-and-last preserved, never a single
//! arbitrary cut point (PRD: *Compaction → graded decay, not a cliff*).
//!
//! Three tiers:
//! - [`Block::Pinned`] — spec, rules, the wins ledger, the original task framing.
//!   Immune to decay; shared (`Arc<str>`) across projector/ledger/display.
//! - [`Block::Verbatim`] — recent turns, full text (`Box<str>`, no spare capacity).
//! - [`Block::Stub`] — a demoted block: a one-line marker + a [`SpillPtr`] back to
//!   the lossless on-disk store. **Demotion is not deletion** — a stub re-fetches by
//!   pointer on demand, and a re-mention promotes it back ([`History::rehydrate`]).
//!
//! Budget is in **bytes** (measured, like [`crate::compress`]) — no estimated token
//! counts. Compaction is **selection, never generation**: nothing is summarized.

use std::sync::Arc;

/// A pointer into the lossless on-disk store (a transcript or spill file) — the
/// re-fetch path for a demoted block. 16 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillPtr {
    pub file: u32,
    pub off: u64,
    pub len: u32,
}

/// One block of context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Block {
    /// Never decays; shared cheaply.
    Pinned(Arc<str>),
    /// Recent, full text.
    Verbatim(Box<str>),
    /// Demoted: a short marker line + where to re-fetch the full bytes.
    Stub { line: Box<str>, src: SpillPtr },
}

impl Block {
    /// The text this block contributes to the assembled prompt (the marker line for
    /// a stub).
    pub fn render(&self) -> &str {
        match self {
            Block::Pinned(s) => s,
            Block::Verbatim(s) => s,
            Block::Stub { line, .. } => line,
        }
    }

    /// Bytes this block costs in the prompt.
    pub fn bytes(&self) -> usize {
        self.render().len()
    }

    /// Can this block be demoted? (Pinned and already-stubbed cannot.)
    fn is_demotable(&self) -> bool {
        matches!(self, Block::Verbatim(_))
    }
}

/// An ordered list of context blocks with graded-decay compaction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct History {
    blocks: Vec<Block>,
}

impl History {
    pub fn new() -> History {
        History::default()
    }

    /// Append a pinned (never-decaying) block — spec, rules, ledger, first-K framing.
    pub fn push_pinned(&mut self, text: impl Into<Arc<str>>) {
        self.blocks.push(Block::Pinned(text.into()));
    }

    /// Append a verbatim (decayable) block — a normal turn.
    pub fn push_verbatim(&mut self, text: impl Into<Box<str>>) {
        self.blocks.push(Block::Verbatim(text.into()));
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }
    pub fn len(&self) -> usize {
        self.blocks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Total prompt bytes.
    pub fn bytes(&self) -> usize {
        self.blocks.iter().map(Block::bytes).sum()
    }

    /// The assembled prompt: every block's rendered text, in order, `\n`-joined.
    pub fn render(&self) -> String {
        self.blocks
            .iter()
            .map(Block::render)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Graded decay: while over `budget` bytes, demote the **oldest** verbatim
    /// block that is not pinned and not within the most recent `keep_recent`
    /// positions. `spill(text) -> SpillPtr` persists the full bytes (lossless) and
    /// returns the re-fetch pointer; the block becomes a one-line stub naming how
    /// much was demoted. First-K-via-pinning and recent-N stay verbatim, so the
    /// middle decays first — never a single cut point. Returns `(bytes_before,
    /// bytes_after)`.
    pub fn compact(
        &mut self,
        budget: usize,
        keep_recent: usize,
        mut spill: impl FnMut(&str) -> SpillPtr,
    ) -> (usize, usize) {
        let before = self.bytes();
        let n = self.blocks.len();
        loop {
            if self.bytes() <= budget {
                break;
            }
            // Oldest demotable block outside the recent window.
            let cutoff = n.saturating_sub(keep_recent);
            let target = (0..cutoff).find(|&i| self.blocks[i].is_demotable());
            let Some(i) = target else { break }; // nothing left to demote
            let Block::Verbatim(text) = &self.blocks[i] else {
                unreachable!("is_demotable guaranteed Verbatim");
            };
            let full_len = text.len();
            let ptr = spill(text);
            let marker = stub_line(text, full_len);
            self.blocks[i] = Block::Stub {
                line: marker.into_boxed_str(),
                src: ptr,
            };
        }
        (before, self.bytes())
    }

    /// Promote a demoted block back to verbatim — the std-only relevance signal: a
    /// stub that gets re-mentioned or re-fetched is clearly still relevant, so it
    /// returns to full fidelity. No-op if the index isn't a stub.
    pub fn rehydrate(&mut self, idx: usize, text: impl Into<Box<str>>) -> bool {
        if matches!(self.blocks.get(idx), Some(Block::Stub { .. })) {
            self.blocks[idx] = Block::Verbatim(text.into());
            true
        } else {
            false
        }
    }

    /// The spill pointers of every demoted block (for a `recall` index to cover).
    pub fn stub_pointers(&self) -> Vec<SpillPtr> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                Block::Stub { src, .. } => Some(*src),
                _ => None,
            })
            .collect()
    }
}

/// The one-line marker a demoted block leaves behind: a short prefix of its first
/// line + how many bytes were demoted (recoverable via the spill pointer).
fn stub_line(text: &str, full_len: usize) -> String {
    let first = text.lines().next().unwrap_or("").trim();
    let prefix: String = first.chars().take(48).collect();
    let ell = if first.chars().count() > 48 {
        "…"
    } else {
        ""
    };
    format!("[demoted {full_len}B → spill] {prefix}{ell}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake spill sink: hands out sequential offsets, records what was spilled.
    fn spiller(store: &mut Vec<String>) -> impl FnMut(&str) -> SpillPtr + '_ {
        move |text: &str| {
            let off = store.iter().map(|s| s.len() as u64).sum();
            store.push(text.to_string());
            SpillPtr {
                file: 0,
                off,
                len: text.len() as u32,
            }
        }
    }

    #[test]
    fn under_budget_compaction_is_a_noop() {
        let mut h = History::new();
        h.push_verbatim("small");
        let mut store = Vec::new();
        let (before, after) = h.compact(1000, 2, spiller(&mut store));
        assert_eq!(before, after);
        assert!(store.is_empty());
        assert!(matches!(h.blocks()[0], Block::Verbatim(_)));
    }

    #[test]
    fn middle_decays_first_pinned_and_recent_preserved() {
        let mut h = History::new();
        h.push_pinned("PINNED task framing"); // immune
        h.push_verbatim("a".repeat(100)); // oldest verbatim → demote first
        h.push_verbatim("b".repeat(100));
        h.push_verbatim("c".repeat(100)); // recent
        h.push_verbatim("d".repeat(100)); // recent

        let mut store = Vec::new();
        // Budget chosen so one demotion (a ~71B stub replacing a 100B block ≈ -29B)
        // is enough to drop under it. Keep the 2 most recent verbatim.
        let (before, after) = h.compact(395, 2, spiller(&mut store));
        assert!(after < before && after <= 395);
        // Pinned untouched.
        assert!(matches!(h.blocks()[0], Block::Pinned(_)));
        // The two recent blocks stayed verbatim.
        assert!(matches!(h.blocks()[3], Block::Verbatim(_)));
        assert!(matches!(h.blocks()[4], Block::Verbatim(_)));
        // The OLDEST verbatim was demoted to a stub (not deleted); the next-oldest
        // is untouched — the middle decays from the front, one at a time.
        assert!(matches!(h.blocks()[1], Block::Stub { .. }));
        assert!(matches!(h.blocks()[2], Block::Verbatim(_)));
        assert!(h.blocks()[1].render().contains("demoted 100B"));
        assert_eq!(store.len(), 1); // exactly one block spilled to reach budget
    }

    #[test]
    fn demotion_stops_when_only_pinned_and_recent_remain() {
        let mut h = History::new();
        h.push_pinned("P".repeat(500));
        h.push_verbatim("x".repeat(500)); // recent (keep_recent=1)
        let mut store = Vec::new();
        // Budget is below even the pinned block — but pinned can't be demoted and
        // the one verbatim is within keep_recent, so compaction can't go lower.
        let (_before, after) = h.compact(10, 1, spiller(&mut store));
        assert!(after >= 1000, "pinned + recent are floors");
        assert!(store.is_empty());
    }

    #[test]
    fn rehydrate_promotes_a_stub_back_to_verbatim() {
        let mut h = History::new();
        h.push_verbatim("z".repeat(200));
        h.push_verbatim("recent");
        let mut store = Vec::new();
        h.compact(50, 1, spiller(&mut store));
        assert!(matches!(h.blocks()[0], Block::Stub { .. }));
        // Re-mention: promote it back.
        assert!(h.rehydrate(0, "z".repeat(200)));
        assert!(matches!(h.blocks()[0], Block::Verbatim(_)));
        // Re-hydrating a non-stub is a no-op.
        assert!(!h.rehydrate(1, "nope"));
    }

    #[test]
    fn stub_pointers_collects_demoted_sources() {
        let mut h = History::new();
        h.push_verbatim("a".repeat(100));
        h.push_verbatim("b".repeat(100));
        h.push_verbatim("recent");
        let mut store = Vec::new();
        h.compact(60, 1, spiller(&mut store));
        let ptrs = h.stub_pointers();
        assert_eq!(ptrs.len(), 2, "both old blocks demoted");
        assert_eq!(ptrs[0].len, 100);
    }

    #[test]
    fn render_joins_blocks_in_order() {
        let mut h = History::new();
        h.push_pinned("one");
        h.push_verbatim("two");
        assert_eq!(h.render(), "one\ntwo");
    }

    #[test]
    fn pinned_blocks_share_cheaply() {
        let shared: Arc<str> = Arc::from("shared rules");
        let mut h = History::new();
        h.push_pinned(Arc::clone(&shared));
        assert_eq!(Arc::strong_count(&shared), 2); // ours + the block's
        assert_eq!(h.blocks()[0].render(), "shared rules");
    }
}

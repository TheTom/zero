// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! The wins/decisions ledger — the counterweight to compaction's forgetting
//! curve. Compaction demotes old turns; the ledger holds what got *settled* so the
//! next context never relitigates it. **Extractive, not generative**: every entry
//! is a fact the harness captured or the operator pinned — nothing is summarized
//! by a model (which is where rot and dropped negative-constraints come from).
//!
//! Tiered by source precision (PRD: *Compaction → the wins/decisions ledger*):
//! 1. [`Source::Commit`] — git commits, free + authoritative, captured silently.
//! 2. [`Source::Evidence`] — commands the harness watched run and pass.
//! 3. [`Source::Pin`] — an explicit operator decision (`/pin`), *including negative
//!    results* ("we tried X, it lost") — exactly what summaries drop.
//!
//! Prose is never auto-promoted: mechanical signals (1, 2) are captured silently,
//! conceptual ones (3) are one keystroke. The ledger **persists per-project** and
//! loads into the next session, so "do not relitigate" is a cross-session property.
//! It renders to one pinned block injected at high salience every turn.

use std::fs;
use std::io;
use std::path::Path;

/// Where a win came from — its precision tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// A git commit (free, authoritative).
    Commit,
    /// A command the harness watched pass (build green / tests pass).
    Evidence,
    /// An explicit operator decision (`/pin`), including negative results.
    Pin,
}

impl Source {
    /// The on-disk / display tag.
    pub fn tag(self) -> &'static str {
        match self {
            Source::Commit => "commit",
            Source::Evidence => "evidence",
            Source::Pin => "pin",
        }
    }

    fn from_tag(s: &str) -> Option<Source> {
        match s {
            "commit" => Some(Source::Commit),
            "evidence" => Some(Source::Evidence),
            "pin" => Some(Source::Pin),
            _ => None,
        }
    }
}

/// One settled fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Win {
    pub source: Source,
    pub text: String,
}

/// The per-project ledger of wins/decisions. Append-only in spirit (dedup keeps it
/// from bloating); rendered as a pinned block and persisted across sessions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WinsLedger {
    wins: Vec<Win>,
}

impl WinsLedger {
    pub fn new() -> WinsLedger {
        WinsLedger::default()
    }

    pub fn len(&self) -> usize {
        self.wins.len()
    }
    pub fn is_empty(&self) -> bool {
        self.wins.is_empty()
    }
    pub fn wins(&self) -> &[Win] {
        &self.wins
    }

    /// Record a win. Deduplicated by normalized text (so re-capturing the same
    /// commit/decision is a no-op). Returns `true` if it was newly added. The text
    /// is sanitized to a single line so the ledger stays one-fact-per-row.
    pub fn add(&mut self, source: Source, text: &str) -> bool {
        let text = one_line(text);
        if text.is_empty() {
            return false;
        }
        let key = norm(&text);
        if self.wins.iter().any(|w| norm(&w.text) == key) {
            return false;
        }
        self.wins.push(Win { source, text });
        true
    }

    /// Render the pinned block injected every turn: `<decisions>` with one line per
    /// win, capped to `budget` bytes (newest kept; the block must stay tiny — it
    /// rides every request). Empty ledger → empty string (nothing to pin).
    pub fn render(&self, budget: usize) -> String {
        if self.wins.is_empty() {
            return String::new();
        }
        // Newest-first so the cap drops the oldest, which the scorecard/commits
        // already preserve elsewhere.
        let mut lines: Vec<String> = self
            .wins
            .iter()
            .rev()
            .map(|w| format!("- [{}] {}", w.source.tag(), w.text))
            .collect();
        let header = "<decisions> (settled — do not relitigate)\n";
        let footer = "</decisions>";
        // Greedily keep newest lines that fit.
        let mut body = String::new();
        let overhead = header.len() + footer.len() + 1;
        for line in &mut lines {
            if body.len() + line.len() + 1 + overhead > budget && !body.is_empty() {
                break;
            }
            body.push_str(line);
            body.push('\n');
        }
        format!("{header}{body}{footer}")
    }

    /// Serialize to the on-disk format (`[source] text` per line).
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        for w in &self.wins {
            s.push_str(&format!("[{}] {}\n", w.source.tag(), w.text));
        }
        s
    }

    /// Parse from the on-disk format. Unknown/blank lines are skipped.
    pub fn from_text(text: &str) -> WinsLedger {
        let mut l = WinsLedger::new();
        for line in text.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix('[') {
                if let Some(close) = rest.find(']') {
                    if let Some(src) = Source::from_tag(&rest[..close]) {
                        l.add(src, rest[close + 1..].trim());
                    }
                }
            }
        }
        l
    }

    /// Persist to `path` (the per-project decisions file).
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, self.to_text())
    }

    /// Load from `path`; a missing file yields an empty ledger (not an error).
    pub fn load(path: &Path) -> WinsLedger {
        fs::read_to_string(path)
            .map(|t| WinsLedger::from_text(&t))
            .unwrap_or_default()
    }
}

/// Collapse to a single trimmed line (a win is one fact).
fn one_line(s: &str) -> String {
    s.lines().next().unwrap_or("").trim().to_string()
}

/// Normalize for dedup: lowercase + collapse whitespace.
fn norm(s: &str) -> String {
    s.split_whitespace()
        .map(|w| w.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_dedups_by_normalized_text() {
        let mut l = WinsLedger::new();
        assert!(l.add(Source::Commit, "feat: add the thing"));
        // Same text, different whitespace/case → not re-added.
        assert!(!l.add(Source::Pin, "FEAT:   add the   thing"));
        assert!(l.add(Source::Pin, "decided: we use sqlite, not a custom store"));
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn add_rejects_empty_and_takes_first_line() {
        let mut l = WinsLedger::new();
        assert!(!l.add(Source::Pin, "   "));
        assert!(l.add(Source::Pin, "keep this line\nand drop this one"));
        assert_eq!(l.wins()[0].text, "keep this line");
    }

    #[test]
    fn render_is_a_pinned_block_newest_first() {
        let mut l = WinsLedger::new();
        l.add(Source::Commit, "first win");
        l.add(Source::Evidence, "tests pass on the new path");
        l.add(
            Source::Pin,
            "we tried context-reset, it lost — do not rebuild",
        );
        let block = l.render(10_000);
        assert!(block.starts_with("<decisions>"));
        assert!(block.trim_end().ends_with("</decisions>"));
        // Newest first.
        let i_pin = block.find("context-reset").unwrap();
        let i_first = block.find("first win").unwrap();
        assert!(i_pin < i_first);
        // The negative result — the thing summaries drop — is preserved.
        assert!(block.contains("[pin] we tried context-reset, it lost"));
    }

    #[test]
    fn render_caps_to_budget_keeping_newest() {
        let mut l = WinsLedger::new();
        for i in 0..50 {
            l.add(
                Source::Commit,
                &format!("win number {i} with some padding text"),
            );
        }
        let block = l.render(200);
        assert!(
            block.len() <= 200 + 64,
            "roughly within budget: {}",
            block.len()
        );
        // The newest survived; the oldest was dropped.
        assert!(block.contains("win number 49"));
        assert!(!block.contains("win number 0 "));
    }

    #[test]
    fn empty_ledger_renders_nothing() {
        assert_eq!(WinsLedger::new().render(1000), "");
    }

    #[test]
    fn persists_and_reloads_across_sessions() {
        let dir = std::env::temp_dir().join(format!(
            "zero-wins-{}-{}",
            std::process::id(),
            crate::clock::unix_millis()
        ));
        let path = dir.join("decisions.md");
        let mut l = WinsLedger::new();
        l.add(Source::Commit, "feat: ship it");
        l.add(Source::Pin, "negative: approach A was abandoned");
        l.save(&path).unwrap();

        let reloaded = WinsLedger::load(&path);
        assert_eq!(reloaded.len(), 2);
        assert_eq!(reloaded.wins()[0].source, Source::Commit);
        assert_eq!(reloaded.wins()[1].source, Source::Pin);
        assert!(reloaded.wins()[1].text.contains("abandoned"));
        // The round-trip is stable.
        assert_eq!(reloaded, l);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_missing_file_is_empty_not_error() {
        let l = WinsLedger::load(Path::new("/no/such/zero/decisions.md"));
        assert!(l.is_empty());
    }

    #[test]
    fn from_text_skips_garbage_lines() {
        let l = WinsLedger::from_text("[commit] ok\nnot a win line\n[bogus] dropped\n[pin] yes");
        assert_eq!(l.len(), 2);
        assert_eq!(l.wins()[0].source, Source::Commit);
        assert_eq!(l.wins()[1].source, Source::Pin);
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! Recall over demoted content — the fix for the pointer-stub discovery hole (the
//! model must otherwise already know which stub it needs). Two staged tiers, both
//! std-only (PRD: *Recall over demoted content*):
//!
//! 1. **Lexical** ([`Lexical`]) — an inverted-index BM25 over demoted blocks, spill
//!    files, and old state rows. Kilobytes per session. Tracks already-returned ids
//!    per wake so a repeated query surfaces *new* material, not what was just sent.
//! 2. **Quantized semantic** ([`Semantic`]) — 1-bit sign-quantized embeddings
//!    (`[u64; D/64]`, 96 B per 768-dim chunk); similarity is Hamming distance via
//!    `count_ones()`, no float math at query time. Embeddings come from the
//!    backend's `/v1/embeddings` seam (the frontend fetches them; this module only
//!    quantizes + scores), and search is **allowlist-filtered** — the lexical stage
//!    narrows candidates, the dense stage reranks only within them.
//!
//! Results return external ids ([`SpillPtr`](crate::history::SpillPtr) keys) into
//! the lossless store: the index is quantized, the payload never is.

use std::collections::{HashMap, HashSet};

const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

/// Split text into lowercase alphanumeric terms.
fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            cur.extend(c.to_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// An inverted-index BM25 recall index over a growing set of documents, keyed by an
/// external `u64` id (a spill-file id, a state-row index, …).
#[derive(Debug, Default)]
pub struct Lexical {
    /// term → postings `(doc_index, term_freq)`.
    postings: HashMap<String, Vec<(usize, u32)>>,
    ids: Vec<u64>,
    lens: Vec<u32>,
    total_len: u64,
}

impl Lexical {
    pub fn new() -> Lexical {
        Lexical::default()
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Index a document under external id `id`. Online — no rebuild, so chunks can
    /// arrive one compaction at a time.
    pub fn add(&mut self, id: u64, text: &str) {
        let doc = self.ids.len();
        let terms = tokenize(text);
        let mut tf: HashMap<&str, u32> = HashMap::new();
        for t in &terms {
            *tf.entry(t.as_str()).or_insert(0) += 1;
        }
        for (term, freq) in tf {
            self.postings
                .entry(term.to_string())
                .or_default()
                .push((doc, freq));
        }
        self.ids.push(id);
        self.lens.push(terms.len() as u32);
        self.total_len += terms.len() as u64;
    }

    /// BM25 search. Returns up to `k` `(id, score)` pairs, highest score first,
    /// skipping any id in `exclude` (the per-wake already-returned set, so repeated
    /// queries surface new material). Ties break by id for determinism.
    pub fn search(&self, query: &str, k: usize, exclude: &HashSet<u64>) -> Vec<(u64, f32)> {
        if self.ids.is_empty() {
            return Vec::new();
        }
        let n = self.ids.len() as f32;
        let avgdl = (self.total_len as f32 / n).max(1.0);
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for term in dedup(tokenize(query)) {
            let Some(post) = self.postings.get(&term) else {
                continue;
            };
            let df = post.len() as f32;
            // BM25 idf with the +1 to keep it non-negative for common terms.
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(doc, freq) in post {
                let f = freq as f32;
                let dl = self.lens[doc] as f32;
                let denom = f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl);
                *scores.entry(doc).or_insert(0.0) += idf * (f * (BM25_K1 + 1.0)) / denom;
            }
        }

        let mut hits: Vec<(u64, f32)> = scores
            .into_iter()
            .map(|(doc, s)| (self.ids[doc], s))
            .filter(|(id, s)| *s > 0.0 && !exclude.contains(id))
            .collect();
        // Highest score first; stable tie-break on id.
        hits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        hits.truncate(k);
        hits
    }
}

/// Pack an embedding into a 1-bit **sign** quantization: bit set iff the component
/// is ≥ 0. A 768-dim f32 (3 KB) becomes `[u64; 12]` (96 B). Data-oblivious — no
/// train step, no rebuild.
pub fn quantize_sign(embedding: &[f32]) -> Vec<u64> {
    let words = embedding.len().div_ceil(64);
    let mut out = vec![0u64; words];
    for (i, &v) in embedding.iter().enumerate() {
        if v >= 0.0 {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

/// Hamming distance between two packed sign-vectors (lower = more similar). One
/// `count_ones()` per word — no float math.
pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x ^ y).count_ones())
        .sum::<u32>()
        // Penalize a length mismatch so a wrong-dim vector can't score well.
        + (a.len().abs_diff(b.len()) as u32 * 64)
}

/// A binary-quantized semantic index: external id → packed sign-vector. Search is
/// Hamming-nearest, optionally restricted to an allowlist (the lexical stage's
/// candidates) — the filter is honored *inside* the scan, not applied after.
#[derive(Debug, Default)]
pub struct Semantic {
    vecs: Vec<(u64, Vec<u64>)>,
}

impl Semantic {
    pub fn new() -> Semantic {
        Semantic::default()
    }
    pub fn len(&self) -> usize {
        self.vecs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vecs.is_empty()
    }

    /// Index `id`'s embedding (quantized on the way in).
    pub fn add(&mut self, id: u64, embedding: &[f32]) {
        self.vecs.push((id, quantize_sign(embedding)));
    }

    /// The `k` nearest ids to `query_embedding` by Hamming distance (closest
    /// first). If `allowlist` is `Some`, only those ids are scored — the two-stage
    /// pre-filter (lexical narrows, dense reranks within).
    pub fn search(
        &self,
        query_embedding: &[f32],
        k: usize,
        allowlist: Option<&HashSet<u64>>,
    ) -> Vec<(u64, u32)> {
        let q = quantize_sign(query_embedding);
        let mut hits: Vec<(u64, u32)> = self
            .vecs
            .iter()
            .filter(|(id, _)| allowlist.is_none_or(|a| a.contains(id)))
            .map(|(id, v)| (*id, hamming(&q, v)))
            .collect();
        // Closest (smallest Hamming) first; stable tie-break on id.
        hits.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        hits.truncate(k);
        hits
    }
}

fn dedup(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v.dedup();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(ids: &[u64]) -> HashSet<u64> {
        ids.iter().copied().collect()
    }

    #[test]
    fn tokenize_splits_on_non_alphanumeric_lowercase() {
        assert_eq!(
            tokenize("Fused QKV; cosine=0.99!"),
            vec!["fused", "qkv", "cosine", "0", "99"]
        );
    }

    #[test]
    fn bm25_ranks_the_relevant_doc_first() {
        let mut idx = Lexical::new();
        idx.add(10, "we fused the qkv projection and cosine improved");
        idx.add(11, "the mlp bucket is still slow, attack it next");
        idx.add(12, "unrelated notes about the build system");
        let hits = idx.search("qkv cosine", 3, &HashSet::new());
        assert_eq!(hits[0].0, 10, "the qkv/cosine doc ranks first");
        assert!(hits.iter().all(|(_, s)| *s > 0.0));
    }

    #[test]
    fn search_excludes_already_returned_ids() {
        let mut idx = Lexical::new();
        idx.add(1, "qkv fusion notes");
        idx.add(2, "qkv attention reshape");
        let first = idx.search("qkv", 1, &HashSet::new());
        let seen = set(&[first[0].0]);
        let second = idx.search("qkv", 1, &seen);
        assert_ne!(
            second[0].0, first[0].0,
            "repeat query surfaces new material"
        );
    }

    #[test]
    fn search_on_empty_index_is_empty() {
        assert!(Lexical::new()
            .search("anything", 5, &HashSet::new())
            .is_empty());
    }

    #[test]
    fn unknown_query_terms_score_nothing() {
        let mut idx = Lexical::new();
        idx.add(1, "alpha beta gamma");
        assert!(idx
            .search("zzzz nonexistent", 5, &HashSet::new())
            .is_empty());
    }

    #[test]
    fn quantize_sign_packs_bits() {
        let emb = [1.0f32, -1.0, 0.5, -0.2, 0.0];
        let q = quantize_sign(&emb);
        assert_eq!(q.len(), 1);
        // bits 0,2,4 set (>= 0); bits 1,3 clear.
        assert_eq!(q[0] & 0b11111, 0b10101);
    }

    #[test]
    fn hamming_is_zero_for_identical_and_grows_with_difference() {
        let a = quantize_sign(&[1.0, 1.0, 1.0, 1.0]);
        let b = quantize_sign(&[1.0, 1.0, 1.0, 1.0]);
        let c = quantize_sign(&[1.0, -1.0, 1.0, -1.0]);
        assert_eq!(hamming(&a, &b), 0);
        assert_eq!(hamming(&a, &c), 2);
    }

    #[test]
    fn semantic_search_returns_nearest_first() {
        let mut s = Semantic::new();
        s.add(1, &[1.0, 1.0, 1.0, 1.0]);
        s.add(2, &[1.0, 1.0, 1.0, -1.0]); // 1 bit off the query
        s.add(3, &[-1.0, -1.0, -1.0, -1.0]); // far
        let hits = s.search(&[1.0, 1.0, 1.0, 1.0], 3, None);
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[0].1, 0);
        assert_eq!(hits[1].0, 2);
        assert!(hits[2].0 == 3);
    }

    #[test]
    fn semantic_search_honors_the_allowlist() {
        let mut s = Semantic::new();
        s.add(1, &[1.0, 1.0, 1.0, 1.0]); // the closest
        s.add(2, &[1.0, 1.0, 1.0, -1.0]);
        // Restrict to {2}: even though 1 is closer, it's filtered out.
        let hits = s.search(&[1.0, 1.0, 1.0, 1.0], 5, Some(&set(&[2])));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 2);
    }

    #[test]
    fn two_stage_lexical_then_semantic() {
        // Lexical narrows to a candidate set; semantic reranks within it.
        let mut lex = Lexical::new();
        lex.add(1, "qkv fusion cosine win");
        lex.add(2, "qkv reshape attempt");
        lex.add(3, "totally unrelated mlp text");
        let cands: HashSet<u64> = lex
            .search("qkv", 5, &HashSet::new())
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(cands.contains(&1) && cands.contains(&2) && !cands.contains(&3));

        let mut sem = Semantic::new();
        sem.add(1, &[1.0, 1.0, -1.0]);
        sem.add(2, &[1.0, -1.0, -1.0]);
        sem.add(3, &[-1.0, -1.0, -1.0]);
        let reranked = sem.search(&[1.0, 1.0, -1.0], 5, Some(&cands));
        assert_eq!(reranked[0].0, 1); // closest within the allowlist
        assert!(reranked.iter().all(|(id, _)| *id != 3)); // 3 was filtered out
    }
}

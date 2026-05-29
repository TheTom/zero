//! The model-backend abstraction. Everything the TUI needs from "a model" is
//! behind the [`Backend`] trait, so the terminal and (future) app frontends are
//! identical from the engine's side, and the real OpenAI-compatible HTTP client
//! drops in without touching the UI.
//!
//! Streaming uses a plain callback sink rather than an iterator or channel: it
//! is std-only, zero-alloc per event, and lets a frontend render each delta the
//! instant it arrives.

use crate::message::Conversation;
use std::fmt;

/// One event in a streamed response.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// An incremental piece of assistant text.
    Token(String),
    /// The response finished; carries why it stopped.
    Done(StopReason),
}

/// Why a response ended. Mirrors OpenAI-compatible `finish_reason` values plus
/// room for tool-calling once that lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Natural end of the assistant turn.
    EndTurn,
    /// Truncated by the max-token limit.
    MaxTokens,
    /// The model wants to call a tool (reserved for the agentic-loop slice).
    ToolUse,
}

/// A backend failure. String-based for now; structured variants can come later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError(pub String);

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "backend error: {}", self.0)
    }
}

impl std::error::Error for BackendError {}

/// A source of model completions. Implementors stream their reply by invoking
/// `sink` for each [`StreamEvent`], ending with `StreamEvent::Done`.
///
/// `Send + Sync` so a frontend can run the stream on a background thread (via
/// `Arc<dyn Backend>`) and keep its input loop responsive — that is what lets
/// the terminal queue messages while a reply is still generating.
pub trait Backend: Send + Sync {
    /// Short human-readable identity, e.g. `"qwen-heretic (openai-compat)"`.
    fn name(&self) -> &str;

    /// Stream a completion for `conv`, calling `sink` per event. Returning `Ok`
    /// means the stream completed; a final `Done` event should have been sent.
    fn stream(
        &self,
        conv: &Conversation,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<(), BackendError>;
}

/// A dependency-free fake backend for the TUI-first slice. It echoes a canned
/// reply token-by-token, optionally pacing the stream with a *real* delay so the
/// streaming UX is visible — the delay is measured, never faked or estimated.
pub struct StubBackend {
    name: String,
    per_token_delay: std::time::Duration,
}

impl StubBackend {
    /// A stub that streams instantly — ideal for tests.
    pub fn instant() -> Self {
        StubBackend {
            name: "stub (instant)".to_string(),
            per_token_delay: std::time::Duration::ZERO,
        }
    }

    /// A stub that paces tokens by a real delay, for a lifelike demo stream.
    pub fn paced(delay: std::time::Duration) -> Self {
        StubBackend {
            name: "stub (paced)".to_string(),
            per_token_delay: delay,
        }
    }

    /// The canned reply, derived from the last user message so the echo feels
    /// responsive. Split into whitespace-preserving word tokens.
    fn reply_for(conv: &Conversation) -> String {
        let last_user = conv
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, crate::message::Role::User))
            .map(|m| m.content.as_str())
            .unwrap_or("");
        if last_user.trim().is_empty() {
            "I'm a stub backend — say something and I'll echo it back as a stream.".to_string()
        } else {
            format!("You said: \"{last_user}\". (stub reply — real model goes here.)")
        }
    }
}

impl Backend for StubBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn stream(
        &self,
        conv: &Conversation,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<(), BackendError> {
        let reply = Self::reply_for(conv);
        for tok in tokenize_keeping_spaces(&reply) {
            if !self.per_token_delay.is_zero() {
                std::thread::sleep(self.per_token_delay);
            }
            sink(StreamEvent::Token(tok));
        }
        sink(StreamEvent::Done(StopReason::EndTurn));
        Ok(())
    }
}

/// Split text into tokens that, concatenated, reproduce the input exactly —
/// each word carries its trailing whitespace. Streaming these reads naturally.
fn tokenize_keeping_spaces(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if ch == ' ' {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

    fn collect(conv: &Conversation, b: &dyn Backend) -> (Vec<String>, Option<StopReason>) {
        let mut tokens = Vec::new();
        let mut stop = None;
        b.stream(conv, &mut |ev| match ev {
            StreamEvent::Token(t) => tokens.push(t),
            StreamEvent::Done(r) => stop = Some(r),
        })
        .unwrap();
        (tokens, stop)
    }

    #[test]
    fn tokenize_is_lossless() {
        let s = "hello  world\nfoo ";
        assert_eq!(tokenize_keeping_spaces(s).concat(), s);
    }

    #[test]
    fn stub_echoes_last_user_message() {
        let mut conv = Conversation::new();
        conv.push(Message::user("ping"));
        let (tokens, stop) = collect(&conv, &StubBackend::instant());
        let full = tokens.concat();
        assert!(full.contains("ping"), "reply should echo user text: {full}");
        assert_eq!(stop, Some(StopReason::EndTurn));
    }

    #[test]
    fn stub_handles_empty_conversation() {
        let conv = Conversation::new();
        let (tokens, stop) = collect(&conv, &StubBackend::instant());
        assert!(!tokens.is_empty());
        assert_eq!(stop, Some(StopReason::EndTurn));
    }

    #[test]
    fn stream_always_ends_with_done() {
        let mut conv = Conversation::new();
        conv.push(Message::user("anything"));
        let mut events = Vec::new();
        StubBackend::instant()
            .stream(&conv, &mut |ev| events.push(ev))
            .unwrap();
        assert!(matches!(events.last(), Some(StreamEvent::Done(_))));
    }

    #[test]
    fn backend_name_is_reported() {
        assert!(StubBackend::instant().name().contains("stub"));
    }

    #[test]
    fn paced_backend_streams_and_names_itself() {
        // A 1ns delay exercises the non-zero pacing branch without slowing tests.
        let backend = StubBackend::paced(std::time::Duration::from_nanos(1));
        assert!(backend.name().contains("paced"));
        let mut conv = Conversation::new();
        conv.push(Message::user("hey"));
        let (tokens, stop) = collect(&conv, &backend);
        assert!(tokens.concat().contains("hey"));
        assert_eq!(stop, Some(StopReason::EndTurn));
    }

    #[test]
    fn empty_token_does_not_break_tokenizer() {
        assert!(tokenize_keeping_spaces("").is_empty());
        assert_eq!(tokenize_keeping_spaces("a").concat(), "a");
    }
}

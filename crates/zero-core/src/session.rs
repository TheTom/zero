// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! Append-only JSONL session transcripts. Every line is one self-contained
//! JSON object stamped with a **real** wall-clock millisecond timestamp — the
//! logging north star: rich, honest, replayable, no invented numbers.
//!
//! Generic over the sink so tests log to an in-memory buffer and production
//! logs to a file under the session directory.

use crate::clock::unix_millis;
use crate::json::Value;
use crate::message::{Conversation, Message, Role};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Writes structured events as newline-delimited JSON.
pub struct SessionLog<W: Write> {
    sink: W,
}

impl SessionLog<File> {
    /// Create `dir/zero-<unix_millis>.jsonl`, making `dir` if needed. The file stem
    /// is the **session id** (used by listing/resume). Millisecond granularity plus
    /// a collision suffix means two launches never clobber one another's transcript.
    /// Returns the log and the path it opened, so the frontend can show it.
    pub fn create_in(dir: impl AsRef<Path>) -> io::Result<(Self, PathBuf)> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        let base = crate::clock::unix_millis();
        // create_new fails if the path exists → bump a suffix instead of truncating
        // a live transcript (File::create would clobber it).
        for n in 0..1000 {
            let name = if n == 0 {
                format!("zero-{base}.jsonl")
            } else {
                format!("zero-{base}-{n}.jsonl")
            };
            let path = dir.join(name);
            match File::options().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok((SessionLog { sink: file }, path)),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(io::Error::other("could not allocate a unique session file"))
    }
}

/// The session id for a transcript path: the file stem (e.g. `zero-1718…`).
pub fn session_id(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// A one-line summary of a saved session, for `/sessions` listing and resume.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionInfo {
    /// The session id (transcript file stem).
    pub id: String,
    /// Full path to the transcript.
    pub path: PathBuf,
    /// Wall-clock start (first `ts_ms` seen), 0 if unknown.
    pub started_ms: u64,
    /// Number of completed turns (`turn_done` lines).
    pub turns: usize,
    /// The model/backend recorded at session start, if any.
    pub model: Option<String>,
    /// The first user prompt — a preview to recognize the session by.
    pub first_prompt: String,
}

/// List saved sessions under `dir`, newest first (by start time). Best-effort: an
/// unreadable or malformed transcript is skipped, never fatal. Returns empty if the
/// directory doesn't exist.
pub fn list_sessions(dir: &Path) -> Vec<SessionInfo> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "jsonl") {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        out.push(summarize(session_id(&path), path.clone(), &text));
    }
    // Newest first; stable tie-break on id so the order is deterministic.
    out.sort_by(|a, b| {
        b.started_ms
            .cmp(&a.started_ms)
            .then_with(|| b.id.cmp(&a.id))
    });
    out
}

/// Summarize one transcript's text into a [`SessionInfo`].
fn summarize(id: String, path: PathBuf, text: &str) -> SessionInfo {
    let mut started_ms = 0u64;
    let mut turns = 0usize;
    let mut model = None;
    let mut first_prompt = String::new();
    for line in text.lines() {
        let Ok(v) = Value::parse(line) else { continue };
        if started_ms == 0 {
            if let Some(ts) = v.get("ts_ms").and_then(Value::as_f64) {
                started_ms = ts as u64;
            }
        }
        match v.get("kind").and_then(Value::as_str) {
            Some("turn_done") => turns += 1,
            Some("meta") if v.get("key").and_then(Value::as_str) == Some("backend") => {
                model = v.get("value").and_then(Value::as_str).map(str::to_string);
            }
            Some("message")
                if first_prompt.is_empty()
                    && v.get("role").and_then(Value::as_str) == Some("user") =>
            {
                first_prompt = v
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
            }
            _ => {}
        }
    }
    SessionInfo {
        id,
        path,
        started_ms,
        turns,
        model,
        first_prompt,
    }
}

/// Resolve a user-typed session `id` to a transcript path under `dir`: an exact
/// stem match wins, otherwise a *unique* substring match. `Err` carries a
/// human-readable reason (not found / ambiguous) suitable for showing the user.
pub fn resolve_session(dir: &Path, id: &str) -> Result<PathBuf, String> {
    let sessions = list_sessions(dir);
    if let Some(s) = sessions.iter().find(|s| s.id == id) {
        return Ok(s.path.clone());
    }
    let matches: Vec<&SessionInfo> = sessions.iter().filter(|s| s.id.contains(id)).collect();
    match matches.as_slice() {
        [one] => Ok(one.path.clone()),
        [] => Err(format!("no session matching '{id}'")),
        many => Err(format!(
            "'{id}' is ambiguous — matches {} sessions; use a full id",
            many.len()
        )),
    }
}

/// Replay a saved transcript back into a [`Conversation`] for resume: the ordered
/// user + assistant **text** turns. Tool calls/results stay as audit detail in the
/// log and are not reconstructed into context (their request-id linkage isn't
/// recoverable, and the assistant replies already summarize what happened).
pub fn load_conversation(path: &Path) -> io::Result<Conversation> {
    let text = fs::read_to_string(path)?;
    let mut conv = Conversation::new();
    for line in text.lines() {
        let Ok(v) = Value::parse(line) else { continue };
        if v.get("kind").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let role = v
            .get("role")
            .and_then(Value::as_str)
            .and_then(Role::from_wire);
        let body = v.get("text").and_then(Value::as_str).unwrap_or("");
        match role {
            Some(Role::User) => conv.push(Message::user(body)),
            Some(Role::Assistant) => conv.push(Message::assistant(body)),
            _ => {} // skip system/tool — rebuild a clean user↔assistant thread
        }
    }
    Ok(conv)
}

impl<W: Write> SessionLog<W> {
    /// Wrap any writer (used by tests).
    pub fn from_writer(sink: W) -> Self {
        SessionLog { sink }
    }

    /// Record a chat message with its real timestamp.
    pub fn record_message(&mut self, role: Role, text: &str) -> io::Result<()> {
        self.write_object(vec![
            ("kind".to_string(), Value::Str("message".to_string())),
            ("role".to_string(), Value::Str(role.as_wire().to_string())),
            ("text".to_string(), Value::Str(text.to_string())),
        ])
    }

    /// Record a tool call the model requested — the tool name and its raw JSON
    /// arguments, exactly as emitted. Logged at call time (honest timestamp) so the
    /// transcript shows *what the agent actually did*, not just its final answer.
    pub fn record_tool_call(&mut self, name: &str, arguments: &str) -> io::Result<()> {
        self.write_object(vec![
            ("kind".to_string(), Value::Str("tool_call".to_string())),
            ("name".to_string(), Value::Str(name.to_string())),
            ("arguments".to_string(), Value::Str(arguments.to_string())),
        ])
    }

    /// Record a tool result fed back to the model. `result` is the model-facing
    /// (possibly capped) text; `raw_bytes`/`kept_bytes` make any capping visible —
    /// `raw_bytes > kept_bytes` means the full output was spilled to the artifact
    /// dir and only a prefix went to the model.
    pub fn record_tool_result(
        &mut self,
        name: &str,
        result: &str,
        raw_bytes: usize,
        kept_bytes: usize,
    ) -> io::Result<()> {
        self.write_object(vec![
            ("kind".to_string(), Value::Str("tool_result".to_string())),
            ("name".to_string(), Value::Str(name.to_string())),
            ("result".to_string(), Value::Str(result.to_string())),
            ("raw_bytes".to_string(), Value::Num(raw_bytes as f64)),
            ("kept_bytes".to_string(), Value::Num(kept_bytes as f64)),
        ])
    }

    /// Record that a turn finished, with the measured elapsed milliseconds and,
    /// when the server reported it, the real `(prompt, completion)` token usage —
    /// honest numbers in the transcript, never an estimate. `None` (e.g. the stub
    /// backend) omits the token fields rather than logging zeros.
    pub fn record_turn_done(
        &mut self,
        elapsed_ms: u128,
        usage: Option<(u64, u64)>,
    ) -> io::Result<()> {
        let mut fields = vec![
            ("kind".to_string(), Value::Str("turn_done".to_string())),
            ("elapsed_ms".to_string(), Value::Num(elapsed_ms as f64)),
        ];
        if let Some((prompt, completion)) = usage {
            fields.push(("prompt_tokens".to_string(), Value::Num(prompt as f64)));
            fields.push((
                "completion_tokens".to_string(),
                Value::Num(completion as f64),
            ));
            fields.push((
                "total_tokens".to_string(),
                Value::Num((prompt + completion) as f64),
            ));
        }
        self.write_object(fields)
    }

    /// Record a free-form metadata event (session start, model name, etc.).
    pub fn record_meta(&mut self, key: &str, value: &str) -> io::Result<()> {
        self.write_object(vec![
            ("kind".to_string(), Value::Str("meta".to_string())),
            ("key".to_string(), Value::Str(key.to_string())),
            ("value".to_string(), Value::Str(value.to_string())),
        ])
    }

    /// Prepend the shared `ts_ms` field, serialize, write one line, flush.
    fn write_object(&mut self, mut fields: Vec<(String, Value)>) -> io::Result<()> {
        let mut obj = Vec::with_capacity(fields.len() + 1);
        obj.push(("ts_ms".to_string(), Value::Num(unix_millis() as f64)));
        obj.append(&mut fields);
        let line = Value::Object(obj).to_json();
        self.sink.write_all(line.as_bytes())?;
        self.sink.write_all(b"\n")?;
        self.sink.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(buf: &[u8]) -> Vec<Value> {
        std::str::from_utf8(buf)
            .unwrap()
            .lines()
            .map(|l| Value::parse(l).expect("each line is valid json"))
            .collect()
    }

    #[test]
    fn message_line_has_timestamp_role_and_text() {
        let mut log = SessionLog::from_writer(Vec::new());
        log.record_message(Role::User, "hi there").unwrap();
        let rows = lines(&log.sink);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.get("kind").and_then(Value::as_str), Some("message"));
        assert_eq!(row.get("role").and_then(Value::as_str), Some("user"));
        assert_eq!(row.get("text").and_then(Value::as_str), Some("hi there"));
        // Real wall-clock timestamp, after 2020.
        let ts = row.get("ts_ms").and_then(Value::as_f64).unwrap();
        assert!(ts > 1_577_836_800_000.0);
    }

    #[test]
    fn turn_done_records_measured_elapsed() {
        let mut log = SessionLog::from_writer(Vec::new());
        log.record_turn_done(1234, None).unwrap();
        let rows = lines(&log.sink);
        assert_eq!(
            rows[0].get("elapsed_ms").and_then(Value::as_f64),
            Some(1234.0)
        );
        // No usage → token fields are omitted, not logged as zero.
        assert!(rows[0].get("total_tokens").is_none());
    }

    #[test]
    fn turn_done_records_real_token_usage_when_present() {
        let mut log = SessionLog::from_writer(Vec::new());
        log.record_turn_done(50, Some((1200, 340))).unwrap();
        let row = &lines(&log.sink)[0];
        assert_eq!(
            row.get("prompt_tokens").and_then(Value::as_f64),
            Some(1200.0)
        );
        assert_eq!(
            row.get("completion_tokens").and_then(Value::as_f64),
            Some(340.0)
        );
        assert_eq!(
            row.get("total_tokens").and_then(Value::as_f64),
            Some(1540.0)
        );
    }

    #[test]
    fn tool_call_records_name_and_raw_arguments() {
        let mut log = SessionLog::from_writer(Vec::new());
        log.record_tool_call("bash", r#"{"command":"cargo test"}"#)
            .unwrap();
        let row = &lines(&log.sink)[0];
        assert_eq!(row.get("kind").and_then(Value::as_str), Some("tool_call"));
        assert_eq!(row.get("name").and_then(Value::as_str), Some("bash"));
        assert_eq!(
            row.get("arguments").and_then(Value::as_str),
            Some(r#"{"command":"cargo test"}"#)
        );
        assert!(row.get("ts_ms").and_then(Value::as_f64).unwrap() > 1_577_836_800_000.0);
    }

    #[test]
    fn tool_result_records_bytes_and_makes_capping_visible() {
        let mut log = SessionLog::from_writer(Vec::new());
        // raw_bytes > kept_bytes ⇒ the result was capped (full output spilled).
        log.record_tool_result("grep", "a.rs:1: hit … [elided]", 50_000, 64)
            .unwrap();
        let row = &lines(&log.sink)[0];
        assert_eq!(row.get("kind").and_then(Value::as_str), Some("tool_result"));
        assert_eq!(row.get("name").and_then(Value::as_str), Some("grep"));
        assert_eq!(row.get("raw_bytes").and_then(Value::as_f64), Some(50_000.0));
        assert_eq!(row.get("kept_bytes").and_then(Value::as_f64), Some(64.0));
        assert!(row
            .get("result")
            .and_then(Value::as_str)
            .unwrap()
            .contains("hit"));
    }

    #[test]
    fn jsonl_is_one_object_per_line() {
        let mut log = SessionLog::from_writer(Vec::new());
        log.record_meta("model", "qwen").unwrap();
        log.record_message(Role::User, "a").unwrap();
        log.record_message(Role::Assistant, "b").unwrap();
        let text = String::from_utf8(log.sink).unwrap();
        assert_eq!(text.lines().count(), 3);
        assert!(text.ends_with('\n'));
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "zero-sess-{}-{}-{tag}",
            std::process::id(),
            crate::clock::unix_millis()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn create_in_never_clobbers_an_existing_transcript() {
        // Two sessions opened back-to-back get distinct files, even same-millisecond.
        let dir = temp_dir("noclobber");
        let (mut a, pa) = SessionLog::create_in(&dir).unwrap();
        let (mut b, pb) = SessionLog::create_in(&dir).unwrap();
        a.record_message(Role::User, "from a").unwrap();
        b.record_message(Role::User, "from b").unwrap();
        assert_ne!(pa, pb, "two sessions must not share a file");
        assert!(std::fs::read_to_string(&pa).unwrap().contains("from a"));
        assert!(std::fs::read_to_string(&pb).unwrap().contains("from b"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn list_sessions_summarizes_newest_first() {
        let dir = temp_dir("list");
        // Session 1: one turn, a prompt.
        {
            let (mut log, _) = SessionLog::create_in(&dir).unwrap();
            log.record_meta("backend", "qwen").unwrap();
            log.record_message(Role::User, "first session prompt")
                .unwrap();
            log.record_message(Role::Assistant, "ok").unwrap();
            log.record_turn_done(10, None).unwrap();
        }
        // Ensure a distinct, later start time for session 2.
        std::thread::sleep(std::time::Duration::from_millis(3));
        {
            let (mut log, _) = SessionLog::create_in(&dir).unwrap();
            log.record_message(Role::User, "second session prompt")
                .unwrap();
            log.record_turn_done(5, None).unwrap();
            log.record_turn_done(5, None).unwrap();
        }
        let sessions = list_sessions(&dir);
        assert_eq!(sessions.len(), 2);
        // Newest first.
        assert_eq!(sessions[0].first_prompt, "second session prompt");
        assert_eq!(sessions[0].turns, 2);
        assert_eq!(sessions[1].first_prompt, "first session prompt");
        assert_eq!(sessions[1].turns, 1);
        assert_eq!(sessions[1].model.as_deref(), Some("qwen"));
        assert!(sessions[0].id.starts_with("zero-"));
        assert!(sessions[0].started_ms > 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn list_sessions_skips_malformed_and_missing_dir() {
        assert!(list_sessions(std::path::Path::new("/no/such/zero/dir")).is_empty());
        let dir = temp_dir("malformed");
        std::fs::write(dir.join("zero-bad.jsonl"), "{not json\nalso bad").unwrap();
        std::fs::write(dir.join("ignore.txt"), "not a transcript").unwrap();
        // A malformed transcript still lists (as an empty-ish summary), the .txt is skipped.
        let s = list_sessions(&dir);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].turns, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_conversation_rebuilds_the_user_assistant_thread() {
        let dir = temp_dir("resume");
        let path = {
            let (mut log, path) = SessionLog::create_in(&dir).unwrap();
            log.record_meta("backend", "m").unwrap();
            log.record_message(Role::User, "what is 2+2").unwrap();
            log.record_tool_call("bash", r#"{"command":"echo 4"}"#)
                .unwrap();
            log.record_tool_result("bash", "4", 2, 2).unwrap();
            log.record_message(Role::Assistant, "It's 4.").unwrap();
            log.record_turn_done(7, None).unwrap();
            log.record_message(Role::User, "and 3+3").unwrap();
            log.record_message(Role::Assistant, "6.").unwrap();
            path
        };
        let conv = load_conversation(&path).unwrap();
        // Clean alternating thread; tool_call/result/meta/turn_done are dropped.
        assert_eq!(conv.messages.len(), 4);
        assert_eq!(conv.messages[0].role, Role::User);
        assert_eq!(conv.messages[0].content, "what is 2+2");
        assert_eq!(conv.messages[1].role, Role::Assistant);
        assert_eq!(conv.messages[1].content, "It's 4.");
        assert_eq!(conv.messages[3].content, "6.");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn session_id_is_the_file_stem() {
        assert_eq!(
            session_id(std::path::Path::new("/x/zero-123.jsonl")),
            "zero-123"
        );
    }

    #[test]
    fn create_in_writes_a_real_jsonl_file() {
        let dir = std::env::temp_dir().join(format!("zero-test-{}", crate::clock::unix_millis()));
        let (mut log, path) = SessionLog::create_in(&dir).unwrap();
        assert!(path.starts_with(&dir));
        assert!(path.to_string_lossy().ends_with(".jsonl"));
        log.record_meta("backend", "stub").unwrap();
        log.record_message(Role::User, "hi").unwrap();
        drop(log);

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents.lines().count(), 2);
        let first = Value::parse(contents.lines().next().unwrap()).unwrap();
        assert_eq!(first.get("kind").and_then(Value::as_str), Some("meta"));

        // Clean up.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_with_newlines_stays_on_one_jsonl_line() {
        let mut log = SessionLog::from_writer(Vec::new());
        log.record_message(Role::Assistant, "line1\nline2").unwrap();
        let text = String::from_utf8(log.sink).unwrap();
        // The embedded newline must be escaped, not split the record.
        assert_eq!(text.lines().count(), 1);
        let row = Value::parse(text.trim()).unwrap();
        assert_eq!(
            row.get("text").and_then(Value::as_str),
            Some("line1\nline2")
        );
    }
}

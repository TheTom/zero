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
use crate::message::Role;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Writes structured events as newline-delimited JSON.
pub struct SessionLog<W: Write> {
    sink: W,
}

impl SessionLog<File> {
    /// Create `dir/zero-<unix_seconds>.jsonl`, making `dir` if needed. Returns
    /// the log and the path it opened, so the frontend can show it to the user.
    pub fn create_in(dir: impl AsRef<Path>) -> io::Result<(Self, PathBuf)> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        let path = dir.join(format!("zero-{}.jsonl", crate::clock::unix_seconds()));
        let file = File::create(&path)?;
        Ok((SessionLog { sink: file }, path))
    }
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

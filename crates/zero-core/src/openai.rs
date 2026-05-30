//! An OpenAI-compatible streaming chat backend (llama.cpp / vLLM / Ollama shim).
//! Builds the `/v1/chat/completions` request from a [`Conversation`], POSTs it
//! with `stream: true` via [`crate::http`], and turns the SSE `data:` deltas
//! into [`StreamEvent`]s.
//!
//! Request building and SSE-line parsing are pure and unit-tested; the network
//! round-trip rides on the localhost-mock-tested `http` module.

use crate::backend::{Backend, BackendError, Completion, StopReason, StreamEvent, Usage};
use crate::config::Config;
use crate::http;
use crate::json::Value;
use crate::message::{Conversation, Message, Role};
use crate::tools::ToolDef;
use std::time::Duration;

/// A backend that streams from an OpenAI-compatible server.
pub struct OpenAiBackend {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    temperature: Option<f64>,
    system_prompt: Option<String>,
    name: String,
}

impl OpenAiBackend {
    /// Build from a [`Config`]. Returns `None` if no `base_url` is configured.
    pub fn from_config(cfg: &Config) -> Option<OpenAiBackend> {
        let base = cfg.base_url.as_ref()?;
        let endpoint = format!("{}/v1/chat/completions", base.trim_end_matches('/'));
        Some(OpenAiBackend {
            endpoint,
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            temperature: cfg.temperature,
            system_prompt: cfg.system_prompt.clone(),
            name: cfg.summary(),
        })
    }

    /// The JSON request body for `conv`.
    fn build_body(&self, conv: &Conversation) -> String {
        let mut messages: Vec<Value> = Vec::new();
        if let Some(sys) = &self.system_prompt {
            messages.push(Message::new(Role::System, sys.clone()).to_value());
        }
        messages.extend(conv.messages.iter().map(Message::to_value));

        let mut obj = vec![
            ("model".to_string(), Value::Str(self.model.clone())),
            ("stream".to_string(), Value::Bool(true)),
            // Ask the server to append a final usage chunk so the status line can
            // show real context usage (no client-side token estimation).
            (
                "stream_options".to_string(),
                Value::Object(vec![("include_usage".to_string(), Value::Bool(true))]),
            ),
            ("messages".to_string(), Value::Array(messages)),
        ];
        if let Some(t) = self.temperature {
            obj.push(("temperature".to_string(), Value::Num(t)));
        }
        Value::Object(obj).to_json()
    }

    /// Build a NON-streaming request body for the tool loop, advertising `tools`.
    /// Streaming is deliberately off: local servers' streaming tool-call parsers
    /// are buggy (calls split/lost/mis-typed across chunks), so the loop reads
    /// the whole completion at once.
    fn build_tool_body(&self, conv: &Conversation, tools: &[ToolDef]) -> String {
        let mut messages: Vec<Value> = Vec::new();
        if let Some(sys) = &self.system_prompt {
            messages.push(Message::new(Role::System, sys.clone()).to_value());
        }
        messages.extend(conv.messages.iter().map(Message::to_value));

        let mut obj = vec![
            ("model".to_string(), Value::Str(self.model.clone())),
            ("stream".to_string(), Value::Bool(false)),
            ("messages".to_string(), Value::Array(messages)),
        ];
        if !tools.is_empty() {
            obj.push(("tools".to_string(), crate::tools::tools_value(tools)));
        }
        if let Some(t) = self.temperature {
            obj.push(("temperature".to_string(), Value::Num(t)));
        }
        Value::Object(obj).to_json()
    }

    /// The non-streaming tool-loop turn, shared by the [`Backend::complete`]
    /// override below. Sends the structured `tools` array, reads back text +
    /// structured (or text-fallback) tool calls.
    fn complete_with_tools(
        &self,
        conv: &Conversation,
        tools: &[ToolDef],
        timeout: Duration,
    ) -> Result<Completion, BackendError> {
        let body = self.build_tool_body(conv, tools);
        let mut headers: Vec<(String, String)> = Vec::new();
        if let Some(key) = &self.api_key {
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
        }
        let (code, text) = http::post(&self.endpoint, &headers, &body, timeout)
            .map_err(|e| BackendError(e.to_string()))?;
        if !(200..300).contains(&code) {
            return Err(BackendError(format!("HTTP {code}: {}", text.trim())));
        }
        parse_completion(&text)
    }
}

/// Parse a non-streaming `/v1/chat/completions` response into a [`Completion`].
fn parse_completion(body: &str) -> Result<Completion, BackendError> {
    let v = Value::parse(body).map_err(|e| BackendError(format!("bad JSON: {e}")))?;
    let message = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(<[_]>::first)
        .and_then(|c| c.get("message"))
        .ok_or_else(|| BackendError("response has no choices[0].message".to_string()))?;
    let content = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let tool_calls = crate::tools::parse_tool_calls(message);
    Ok(Completion {
        content,
        tool_calls,
    })
}

impl Backend for OpenAiBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn complete(
        &self,
        conv: &Conversation,
        tools: &[ToolDef],
        timeout: Duration,
    ) -> Result<Completion, BackendError> {
        self.complete_with_tools(conv, tools, timeout)
    }

    fn stream(
        &self,
        conv: &Conversation,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<(), BackendError> {
        let body = self.build_body(conv);
        let mut headers: Vec<(String, String)> = Vec::new();
        if let Some(key) = &self.api_key {
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
        }

        http::post_stream(
            &self.endpoint,
            &headers,
            &body,
            &mut |line| match parse_sse_line(line) {
                Some(SseEvent::Token(t)) => sink(StreamEvent::Token(t)),
                Some(SseEvent::Usage(u)) => sink(StreamEvent::Usage(u)),
                _ => {}
            },
        )
        .map_err(|e| BackendError(e.to_string()))?;

        sink(StreamEvent::Done(StopReason::EndTurn));
        Ok(())
    }
}

/// Fetch the model's context window (`n_ctx`) from a llama.cpp-style server's
/// `/props` endpoint. Best-effort: any network/parse failure yields `None`.
pub fn fetch_context_window(base_url: &str, timeout: Duration) -> Option<u64> {
    let url = format!("{}/props", base_url.trim_end_matches('/'));
    let (status, body) = http::get(&url, timeout).ok()?;
    if status != 200 {
        return None;
    }
    parse_context_window(&body)
}

/// Pull `n_ctx` out of a `/props` JSON body. llama.cpp nests it under
/// `default_generation_settings`; some builds expose it at the top level.
fn parse_context_window(body: &str) -> Option<u64> {
    let v = Value::parse(body).ok()?;
    let n_ctx = v
        .get("default_generation_settings")
        .and_then(|g| g.get("n_ctx"))
        .or_else(|| v.get("n_ctx"))
        .and_then(Value::as_f64)?;
    if n_ctx > 0.0 {
        Some(n_ctx as u64)
    } else {
        None
    }
}

/// What an SSE line meant, if anything.
#[derive(Debug, PartialEq, Eq)]
enum SseEvent {
    Token(String),
    Usage(Usage),
    Done,
}

/// Parse one SSE line into an event. Returns `None` for blanks, comments,
/// keep-alives, role-only deltas, and anything unrecognized.
fn parse_sse_line(line: &str) -> Option<SseEvent> {
    let data = line.strip_prefix("data:")?.trim();
    if data == "[DONE]" {
        return Some(SseEvent::Done);
    }
    let v = Value::parse(data).ok()?;
    // A usage chunk (sent last when `stream_options.include_usage` is set) has an
    // empty `choices` array and a populated `usage` object.
    if let Some(usage) = v.get("usage").and_then(parse_usage) {
        return Some(SseEvent::Usage(usage));
    }
    let delta = v.get("choices")?.as_array()?.first()?.get("delta")?;
    let content = delta.get("content").and_then(Value::as_str)?;
    if content.is_empty() {
        None
    } else {
        Some(SseEvent::Token(content.to_string()))
    }
}

/// Read `{prompt_tokens, completion_tokens}` from a usage object.
fn parse_usage(v: &Value) -> Option<Usage> {
    let prompt = v.get("prompt_tokens").and_then(Value::as_f64)?;
    let completion = v.get("completion_tokens").and_then(Value::as_f64)?;
    Some(Usage {
        prompt_tokens: prompt as u64,
        completion_tokens: completion as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            base_url: Some("http://gx10:8000/".to_string()),
            model: "qwen".to_string(),
            api_key: Some("tok".to_string()),
            temperature: Some(0.3),
            system_prompt: Some("be terse".to_string()),
        }
    }

    #[test]
    fn from_config_requires_base_url() {
        assert!(OpenAiBackend::from_config(&Config::default()).is_none());
        let b = OpenAiBackend::from_config(&cfg()).unwrap();
        // Trailing slash on base_url is normalized.
        assert_eq!(b.endpoint, "http://gx10:8000/v1/chat/completions");
        assert!(b.name().contains("qwen"));
    }

    #[test]
    fn build_body_includes_system_prompt_and_params() {
        let b = OpenAiBackend::from_config(&cfg()).unwrap();
        let mut conv = Conversation::new();
        conv.push(Message::user("hi"));
        let body = b.build_body(&conv);
        let v = Value::parse(&body).unwrap();
        assert_eq!(v.get("model").and_then(Value::as_str), Some("qwen"));
        assert_eq!(v.get("stream").and_then(Value::as_bool), Some(true));
        assert_eq!(v.get("temperature").and_then(Value::as_f64), Some(0.3));
        let msgs = v.get("messages").and_then(Value::as_array).unwrap();
        assert_eq!(msgs.len(), 2); // system + user
        assert_eq!(msgs[0].get("role").and_then(Value::as_str), Some("system"));
        assert_eq!(msgs[1].get("content").and_then(Value::as_str), Some("hi"));
    }

    #[test]
    fn build_body_omits_optional_fields_when_unset() {
        let cfg = Config {
            base_url: Some("http://h:1".to_string()),
            model: "m".to_string(),
            ..Config::default()
        };
        let b = OpenAiBackend::from_config(&cfg).unwrap();
        let body = b.build_body(&Conversation::new());
        let v = Value::parse(&body).unwrap();
        assert!(v.get("temperature").is_none());
        // No system prompt → empty messages array.
        assert_eq!(
            v.get("messages").and_then(Value::as_array).map(<[_]>::len),
            Some(0)
        );
    }

    #[test]
    fn build_body_requests_usage_in_the_stream() {
        let b = OpenAiBackend::from_config(&cfg()).unwrap();
        let v = Value::parse(&b.build_body(&Conversation::new())).unwrap();
        let opt = v.get("stream_options").unwrap();
        assert_eq!(
            opt.get("include_usage").and_then(Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn parses_usage_chunk() {
        let line = r#"data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":5}}"#;
        assert_eq!(
            parse_sse_line(line),
            Some(SseEvent::Usage(Usage {
                prompt_tokens: 12,
                completion_tokens: 5,
            }))
        );
    }

    #[test]
    fn partial_usage_object_is_ignored() {
        // Missing completion_tokens → not a usable usage chunk.
        let line = r#"data: {"choices":[],"usage":{"prompt_tokens":12}}"#;
        assert_eq!(parse_sse_line(line), None);
    }

    #[test]
    fn build_tool_body_is_non_streaming_and_includes_tools() {
        let b = OpenAiBackend::from_config(&cfg()).unwrap();
        let defs = vec![ToolDef::new(
            "ls",
            "list",
            Value::Object(vec![("type".to_string(), Value::Str("object".to_string()))]),
        )];
        let v = Value::parse(&b.build_tool_body(&Conversation::new(), &defs)).unwrap();
        assert_eq!(v.get("stream").and_then(Value::as_bool), Some(false));
        let tools = v.get("tools").and_then(Value::as_array).unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools[0]
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str),
            Some("ls")
        );
    }

    #[test]
    fn build_tool_body_omits_tools_when_empty() {
        let b = OpenAiBackend::from_config(&cfg()).unwrap();
        let v = Value::parse(&b.build_tool_body(&Conversation::new(), &[])).unwrap();
        assert!(v.get("tools").is_none());
    }

    #[test]
    fn parse_completion_extracts_text_and_calls() {
        let body = r#"{"choices":[{"message":{"content":"sure","tool_calls":[
            {"id":"c1","function":{"name":"read_file","arguments":"{\"path\":\"a\"}"}}]}}]}"#;
        let c = parse_completion(body).unwrap();
        assert_eq!(c.content, "sure");
        assert_eq!(c.tool_calls.len(), 1);
        assert_eq!(c.tool_calls[0].name, "read_file");
    }

    #[test]
    fn parse_completion_text_only() {
        let body = r#"{"choices":[{"message":{"content":"just text"}}]}"#;
        let c = parse_completion(body).unwrap();
        assert_eq!(c.content, "just text");
        assert!(c.tool_calls.is_empty());
    }

    #[test]
    fn parse_completion_errors_on_no_choices() {
        assert!(parse_completion(r#"{"choices":[]}"#).is_err());
        assert!(parse_completion("not json").is_err());
    }

    #[test]
    fn complete_against_a_localhost_mock() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            sock.set_read_timeout(Some(Duration::from_millis(150))).ok();
            let mut buf = [0u8; 2048];
            while let Ok(n) = sock.read(&mut buf) {
                if n == 0 {
                    break;
                }
            }
            let body = r#"{"choices":[{"message":{"content":"","tool_calls":[{"id":"x","function":{"name":"grep","arguments":"{}"}}]}}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            sock.write_all(resp.as_bytes()).unwrap();
        });

        let cfg = Config {
            base_url: Some(format!("http://127.0.0.1:{port}")),
            model: "m".to_string(),
            ..Config::default()
        };
        let backend = OpenAiBackend::from_config(&cfg).unwrap();
        let c = backend
            .complete(&Conversation::new(), &[], Duration::from_millis(500))
            .unwrap();
        assert_eq!(c.tool_calls.len(), 1);
        assert_eq!(c.tool_calls[0].name, "grep");
    }

    #[test]
    fn complete_sends_auth_header_and_parses_content_fallback() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let got = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let got2 = got.clone();
        thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            sock.set_read_timeout(Some(Duration::from_millis(150))).ok();
            let mut req = Vec::new();
            let mut buf = [0u8; 2048];
            // Drain the whole request (headers + body) so the client's body
            // write always completes — the read timeout ends the loop. (Breaking
            // early at the header boundary can RST the client mid-body.)
            while let Ok(n) = sock.read(&mut buf) {
                if n == 0 {
                    break;
                }
                req.extend_from_slice(&buf[..n]);
            }
            *got2.lock().unwrap() = String::from_utf8_lossy(&req).into_owned();
            // Tool call hidden in content (quantized-model shape), no structured field.
            let body = r#"{"choices":[{"message":{"content":"<tool_call>{\"name\":\"ls\",\"arguments\":{}}</tool_call>"}}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            sock.write_all(resp.as_bytes()).unwrap();
        });
        let cfg = Config {
            base_url: Some(format!("http://127.0.0.1:{port}")),
            model: "m".to_string(),
            api_key: Some("sk-tok".to_string()),
            ..Config::default()
        };
        let backend = OpenAiBackend::from_config(&cfg).unwrap();
        let c = backend
            .complete(&Conversation::new(), &[], Duration::from_millis(500))
            .unwrap();
        assert_eq!(c.tool_calls.len(), 1);
        assert_eq!(c.tool_calls[0].name, "ls");
        assert!(got.lock().unwrap().contains("Authorization: Bearer sk-tok"));
    }

    #[test]
    fn complete_surfaces_http_error_status() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            sock.set_read_timeout(Some(Duration::from_millis(150))).ok();
            let mut buf = [0u8; 1024];
            while let Ok(n) = sock.read(&mut buf) {
                if n == 0 {
                    break;
                }
            }
            let resp = "HTTP/1.1 400 Bad Request\r\nContent-Length: 3\r\n\r\nbad";
            sock.write_all(resp.as_bytes()).unwrap();
        });
        let cfg = Config {
            base_url: Some(format!("http://127.0.0.1:{port}")),
            model: "m".to_string(),
            ..Config::default()
        };
        let backend = OpenAiBackend::from_config(&cfg).unwrap();
        let err = backend
            .complete(&Conversation::new(), &[], Duration::from_millis(500))
            .unwrap_err();
        assert!(err.0.contains("400"));
    }

    #[test]
    fn parse_context_window_reads_nested_and_top_level() {
        assert_eq!(
            parse_context_window(r#"{"default_generation_settings":{"n_ctx":32768}}"#),
            Some(32768)
        );
        assert_eq!(parse_context_window(r#"{"n_ctx":4096}"#), Some(4096));
        assert_eq!(parse_context_window(r#"{"foo":1}"#), None);
        assert_eq!(parse_context_window(r#"{"n_ctx":0}"#), None);
        assert_eq!(parse_context_window("not json"), None);
    }

    #[test]
    fn fetch_context_window_against_a_localhost_mock() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            sock.set_read_timeout(Some(Duration::from_millis(150))).ok();
            let mut buf = [0u8; 1024];
            while let Ok(n) = sock.read(&mut buf) {
                if n == 0 {
                    break;
                }
            }
            let body = r#"{"default_generation_settings":{"n_ctx":8192}}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            sock.write_all(resp.as_bytes()).unwrap();
        });

        let ctx = fetch_context_window(
            &format!("http://127.0.0.1:{port}"),
            Duration::from_millis(500),
        );
        assert_eq!(ctx, Some(8192));
    }

    #[test]
    fn fetch_context_window_on_dead_endpoint_is_none() {
        assert_eq!(
            fetch_context_window("http://127.0.0.1:1", Duration::from_millis(100)),
            None
        );
    }

    #[test]
    fn fetch_context_window_non_200_is_none() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            sock.set_read_timeout(Some(Duration::from_millis(150))).ok();
            let mut buf = [0u8; 1024];
            while let Ok(n) = sock.read(&mut buf) {
                if n == 0 {
                    break;
                }
            }
            let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
            sock.write_all(resp.as_bytes()).unwrap();
        });

        let ctx = fetch_context_window(
            &format!("http://127.0.0.1:{port}"),
            Duration::from_millis(500),
        );
        assert_eq!(ctx, None);
    }

    #[test]
    fn parses_token_delta() {
        let line = r#"data: {"choices":[{"delta":{"content":"Hel"}}]}"#;
        assert_eq!(
            parse_sse_line(line),
            Some(SseEvent::Token("Hel".to_string()))
        );
    }

    #[test]
    fn parses_done_marker() {
        assert_eq!(parse_sse_line("data: [DONE]"), Some(SseEvent::Done));
    }

    #[test]
    fn ignores_non_token_lines() {
        assert_eq!(parse_sse_line(""), None);
        assert_eq!(parse_sse_line(": keep-alive comment"), None);
        // Role-only opening delta has no content.
        assert_eq!(
            parse_sse_line(r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#),
            None
        );
        // Empty content string.
        assert_eq!(
            parse_sse_line(r#"data: {"choices":[{"delta":{"content":""}}]}"#),
            None
        );
        // Malformed JSON after data:.
        assert_eq!(parse_sse_line("data: {not json"), None);
    }

    #[test]
    fn end_to_end_against_a_localhost_mock() {
        use crate::backend::Backend;
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            // Drain the request fully (idle timeout) to avoid a RST-on-close race.
            sock.set_read_timeout(Some(std::time::Duration::from_millis(150)))
                .ok();
            let mut buf = [0u8; 1024];
            while let Ok(n) = sock.read(&mut buf) {
                if n == 0 {
                    break;
                }
            }
            let body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
                        data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n\
                        data: [DONE]\n\n";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            sock.write_all(resp.as_bytes()).unwrap();
        });

        let cfg = Config {
            base_url: Some(format!("http://127.0.0.1:{port}")),
            model: "m".to_string(),
            api_key: Some("tok".to_string()), // exercises the Authorization header
            ..Config::default()
        };
        let backend = OpenAiBackend::from_config(&cfg).unwrap();
        let mut conv = Conversation::new();
        conv.push(Message::user("hi"));

        let mut text = String::new();
        let mut stop = None;
        let mut usage = None;
        backend
            .stream(&conv, &mut |ev| match ev {
                StreamEvent::Token(t) => text.push_str(&t),
                StreamEvent::Usage(u) => usage = Some(u),
                StreamEvent::Done(r) => stop = Some(r),
            })
            .unwrap();
        assert_eq!(usage.map(|u| u.total()), Some(7)); // usage SSE chunk parsed
        assert_eq!(text, "Hello world");
        assert_eq!(stop, Some(StopReason::EndTurn));
    }

    #[test]
    fn stream_reports_connection_errors() {
        use crate::backend::Backend;
        let cfg = Config {
            base_url: Some("http://127.0.0.1:1".to_string()),
            model: "m".to_string(),
            ..Config::default()
        };
        let backend = OpenAiBackend::from_config(&cfg).unwrap();
        let err = backend
            .stream(&Conversation::new(), &mut |_| {})
            .unwrap_err();
        assert!(!err.0.is_empty());
    }
}

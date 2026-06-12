// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! Chat message types, modeled on the OpenAI-compatible chat schema since that
//! is Zero's first backend target (local qwen via llama.cpp / vLLM / Ollama
//! shim). Kept backend-agnostic enough that an Ollama-native adapter can map
//! onto the same types later.

use crate::json::Value;

/// A tool the model asked to invoke. `arguments` is kept as the raw JSON *string*
/// (OpenAI's wire shape) — never a parsed object — so it round-trips through the
/// conversation history without re-encoding surprises (a key pitfall: llama.cpp
/// hands back an object, the OpenAI SDK wants a string).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON arguments string (may be `"{}"` or, for a malformed call, empty).
    pub arguments: String,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// The OpenAI `tool_calls[]` entry for a request payload.
    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            ("id".to_string(), Value::Str(self.id.clone())),
            ("type".to_string(), Value::Str("function".to_string())),
            (
                "function".to_string(),
                Value::Object(vec![
                    ("name".to_string(), Value::Str(self.name.clone())),
                    ("arguments".to_string(), Value::Str(self.arguments.clone())),
                ]),
            ),
        ])
    }
}

/// Who authored a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// The wire string used in OpenAI-compatible `role` fields.
    pub fn as_wire(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    pub fn from_wire(s: &str) -> Option<Role> {
        match s {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "tool" => Some(Role::Tool),
            _ => None,
        }
    }
}

/// A single turn in the conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Tool calls the assistant requested this turn (empty for non-assistant or
    /// text-only turns). Must be replayed verbatim in history alongside the
    /// matching `tool` results, or compatible servers reject the next request.
    pub tool_calls: Vec<ToolCall>,
    /// For a `Tool`-role message: which `ToolCall.id` this is the result of.
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Message {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Message::new(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Message::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message::new(Role::Assistant, content)
    }

    /// An assistant turn that requested tool calls (content may be empty).
    pub fn assistant_tool_calls(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Message {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: calls,
            tool_call_id: None,
        }
    }

    /// The result of one tool call, fed back to the model. `id` MUST match the
    /// originating `ToolCall.id`.
    pub fn tool_result(id: impl Into<String>, content: impl Into<String>) -> Self {
        Message {
            role: Role::Tool,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(id.into()),
        }
    }

    /// Build the OpenAI message object for a request payload. Emits `tool_calls`
    /// on an assistant turn and `tool_call_id` on a tool result.
    pub fn to_value(&self) -> Value {
        let mut obj = vec![
            (
                "role".to_string(),
                Value::Str(self.role.as_wire().to_string()),
            ),
            ("content".to_string(), Value::Str(self.content.clone())),
        ];
        if !self.tool_calls.is_empty() {
            obj.push((
                "tool_calls".to_string(),
                Value::Array(self.tool_calls.iter().map(ToolCall::to_value).collect()),
            ));
        }
        if let Some(id) = &self.tool_call_id {
            obj.push(("tool_call_id".to_string(), Value::Str(id.clone())));
        }
        Value::Object(obj)
    }
}

/// A conversation: an ordered list of messages plus convenience builders.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Conversation {
    pub messages: Vec<Message>,
}

impl Conversation {
    pub fn new() -> Self {
        Conversation::default()
    }

    pub fn push(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// The `messages` array for an OpenAI-compatible request body.
    pub fn to_value(&self) -> Value {
        Value::Array(self.messages.iter().map(Message::to_value).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_wire_roundtrips() {
        for r in [Role::System, Role::User, Role::Assistant, Role::Tool] {
            assert_eq!(Role::from_wire(r.as_wire()), Some(r));
        }
        assert_eq!(Role::from_wire("bogus"), None);
    }

    #[test]
    fn message_serializes_to_openai_shape() {
        let m = Message::user("hello");
        let v = m.to_value();
        assert_eq!(v.get("role").and_then(Value::as_str), Some("user"));
        assert_eq!(v.get("content").and_then(Value::as_str), Some("hello"));
    }

    #[test]
    fn assistant_tool_call_serializes_openai_shape() {
        let m = Message::assistant_tool_calls(
            "",
            vec![ToolCall::new("call_1", "read_file", r#"{"path":"a.txt"}"#)],
        );
        let v = m.to_value();
        assert_eq!(v.get("role").and_then(Value::as_str), Some("assistant"));
        let calls = v.get("tool_calls").and_then(Value::as_array).unwrap();
        assert_eq!(calls.len(), 1);
        let f = calls[0].get("function").unwrap();
        assert_eq!(f.get("name").and_then(Value::as_str), Some("read_file"));
        // arguments stays a STRING, not a parsed object (round-trip safety).
        assert_eq!(
            f.get("arguments").and_then(Value::as_str),
            Some(r#"{"path":"a.txt"}"#)
        );
        assert_eq!(
            calls[0].get("type").and_then(Value::as_str),
            Some("function")
        );
    }

    #[test]
    fn tool_result_carries_id_and_no_tool_calls_field() {
        let m = Message::tool_result("call_1", "file contents");
        let v = m.to_value();
        assert_eq!(v.get("role").and_then(Value::as_str), Some("tool"));
        assert_eq!(
            v.get("tool_call_id").and_then(Value::as_str),
            Some("call_1")
        );
        assert_eq!(
            v.get("content").and_then(Value::as_str),
            Some("file contents")
        );
        assert!(v.get("tool_calls").is_none());
    }

    #[test]
    fn plain_message_omits_tool_fields() {
        let v = Message::user("hi").to_value();
        assert!(v.get("tool_calls").is_none());
        assert!(v.get("tool_call_id").is_none());
    }

    #[test]
    fn conversation_builds_message_array() {
        let mut c = Conversation::new();
        assert!(c.is_empty());
        c.push(Message::system("be terse"));
        c.push(Message::user("hi"));
        assert_eq!(c.len(), 2);
        let arr = c.to_value();
        assert_eq!(arr.as_array().map(<[_]>::len), Some(2));
    }
}

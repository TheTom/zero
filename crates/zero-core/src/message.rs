//! Chat message types, modeled on the OpenAI-compatible chat schema since that
//! is Zero's first backend target (local qwen via llama.cpp / vLLM / Ollama
//! shim). Kept backend-agnostic enough that an Ollama-native adapter can map
//! onto the same types later.

use crate::json::Value;

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
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Message {
            role,
            content: content.into(),
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

    /// Build the `{"role":..,"content":..}` object for a request payload.
    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            (
                "role".to_string(),
                Value::Str(self.role.as_wire().to_string()),
            ),
            ("content".to_string(), Value::Str(self.content.clone())),
        ])
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

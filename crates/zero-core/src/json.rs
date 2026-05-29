//! Minimal, std-only JSON. Enough for OpenAI-compatible chat payloads and
//! JSONL session logs — not a general-purpose library.
//!
//! Objects preserve insertion order (backed by a `Vec`), so serialized output
//! is deterministic. That matters for golden tests and content-addressed logs;
//! see the Rust/Metal playbook note on iteration-order stability.
//!
//! # Example
//! ```
//! use zero_core::json::Value;
//! let v = Value::parse(r#"{"model":"qwen","n":2}"#).unwrap();
//! assert_eq!(v.get("model").and_then(Value::as_str), Some("qwen"));
//! assert_eq!(v.get("n").and_then(Value::as_f64), Some(2.0));
//! ```

use std::fmt::{self, Write as _};

/// A JSON value. `Object` is an ordered list of pairs, not a map — payloads are
/// small and order-preservation buys deterministic serialization.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

/// A JSON parse failure, carrying the byte offset where parsing stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub at: usize,
    pub msg: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "json parse error at byte {}: {}", self.at, self.msg)
    }
}

impl std::error::Error for ParseError {}

impl Value {
    /// Parse a complete JSON document. Trailing non-whitespace is an error.
    pub fn parse(input: &str) -> Result<Value, ParseError> {
        let mut p = Parser {
            bytes: input.as_bytes(),
            pos: 0,
        };
        p.skip_ws();
        let v = p.parse_value()?;
        p.skip_ws();
        if p.pos != p.bytes.len() {
            return Err(p.err("trailing data after value"));
        }
        Ok(v)
    }

    /// Look up a key in an object. Returns `None` for non-objects or misses.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Serialize to compact JSON (no insignificant whitespace).
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    fn write_json(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(true) => out.push_str("true"),
            Value::Bool(false) => out.push_str("false"),
            Value::Num(n) => write_number(*n, out),
            Value::Str(s) => write_json_string(s, out),
            Value::Array(items) => {
                out.push('[');
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    it.write_json(out);
                }
                out.push(']');
            }
            Value::Object(pairs) => {
                out.push('{');
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_string(k, out);
                    out.push(':');
                    v.write_json(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_number(n: f64, out: &mut String) {
    // JSON has no NaN/Infinity; emit null rather than invalid output.
    if n.is_finite() {
        // Integral values print without a trailing ".0" for clean payloads.
        if n.fract() == 0.0 && n.abs() < 1e15 {
            let _ = write!(out, "{}", n as i64);
        } else {
            let _ = write!(out, "{n}");
        }
    } else {
        out.push_str("null");
    }
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn err(&self, msg: &str) -> ParseError {
        ParseError {
            at: self.pos,
            msg: msg.to_string(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Value::Str(self.parse_string()?)),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(_) => Err(self.err("unexpected character")),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), ParseError> {
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected '{}'", b as char)))
        }
    }

    fn parse_object(&mut self) -> Result<Value, ParseError> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(pairs));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            let val = self.parse_value()?;
            pairs.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
        Ok(Value::Object(pairs))
    }

    fn parse_array(&mut self) -> Result<Value, ParseError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
        Ok(Value::Array(items))
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            let b = self.peek().ok_or_else(|| self.err("unterminated string"))?;
            self.pos += 1;
            match b {
                b'"' => break,
                b'\\' => {
                    let esc = self.peek().ok_or_else(|| self.err("dangling escape"))?;
                    self.pos += 1;
                    match esc {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b't' => s.push('\t'),
                        b'r' => s.push('\r'),
                        b'b' => s.push('\u{08}'),
                        b'f' => s.push('\u{0c}'),
                        b'u' => s.push(self.parse_unicode_escape()?),
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                // A raw byte < 0x20 is invalid in JSON strings, but be lenient.
                _ => {
                    // Reconstruct UTF-8: back up and take the full char.
                    self.pos -= 1;
                    let rest = &self.bytes[self.pos..];
                    match std::str::from_utf8(rest) {
                        Ok(_) => {}
                        Err(e) if e.valid_up_to() > 0 => {}
                        Err(_) => return Err(self.err("invalid utf-8 in string")),
                    }
                    let ch = decode_utf8_char(rest).ok_or_else(|| self.err("invalid utf-8"))?;
                    self.pos += ch.len_utf8();
                    s.push(ch);
                }
            }
        }
        Ok(s)
    }

    fn parse_unicode_escape(&mut self) -> Result<char, ParseError> {
        let hi = self.parse_hex4()?;
        // Surrogate pair handling for code points above the BMP.
        if (0xD800..=0xDBFF).contains(&hi) {
            if self.peek() == Some(b'\\') {
                self.pos += 1;
                self.expect(b'u')?;
                let lo = self.parse_hex4()?;
                if (0xDC00..=0xDFFF).contains(&lo) {
                    let c = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                    return char::from_u32(c).ok_or_else(|| self.err("bad surrogate pair"));
                }
            }
            return Err(self.err("unpaired high surrogate"));
        }
        char::from_u32(hi).ok_or_else(|| self.err("invalid code point"))
    }

    fn parse_hex4(&mut self) -> Result<u32, ParseError> {
        let mut v = 0u32;
        for _ in 0..4 {
            let b = self.peek().ok_or_else(|| self.err("short \\u escape"))?;
            let d = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                _ => return Err(self.err("non-hex digit in \\u escape")),
            };
            v = v * 16 + d as u32;
            self.pos += 1;
        }
        Ok(v)
    }

    fn parse_bool(&mut self) -> Result<Value, ParseError> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Value::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Value::Bool(false))
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn parse_null(&mut self) -> Result<Value, ParseError> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Value::Null)
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn parse_number(&mut self) -> Result<Value, ParseError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let slice = &self.bytes[start..self.pos];
        let s = std::str::from_utf8(slice).map_err(|_| self.err("invalid number"))?;
        s.parse::<f64>().map(Value::Num).map_err(|_| ParseError {
            at: start,
            msg: "malformed number".to_string(),
        })
    }
}

/// Decode the first UTF-8 scalar from a byte slice, or `None` if malformed.
fn decode_utf8_char(bytes: &[u8]) -> Option<char> {
    let s = std::str::from_utf8(bytes).ok().or_else(|| {
        // Take the longest valid prefix so a single char still decodes.
        let err = std::str::from_utf8(bytes).err()?;
        std::str::from_utf8(&bytes[..err.valid_up_to().max(1).min(bytes.len())]).ok()
    })?;
    s.chars().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_primitives() {
        assert_eq!(Value::parse("null").unwrap(), Value::Null);
        assert_eq!(Value::parse("true").unwrap(), Value::Bool(true));
        assert_eq!(Value::parse("false").unwrap(), Value::Bool(false));
        assert_eq!(Value::parse("42").unwrap(), Value::Num(42.0));
        assert_eq!(Value::parse("-3.5e2").unwrap(), Value::Num(-350.0));
        assert_eq!(
            Value::parse("\"hi\"").unwrap(),
            Value::Str("hi".to_string())
        );
    }

    #[test]
    fn parses_nested_object() {
        let v = Value::parse(r#"{"a":[1,2,{"b":true}],"c":"x"}"#).unwrap();
        assert_eq!(v.get("c").and_then(Value::as_str), Some("x"));
        let a = v.get("a").and_then(Value::as_array).unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(a[2].get("b").and_then(Value::as_bool), Some(true));
    }

    #[test]
    fn handles_string_escapes() {
        let v = Value::parse(r#""line\nbreak\t\"quote\\""#).unwrap();
        assert_eq!(v.as_str(), Some("line\nbreak\t\"quote\\"));
    }

    #[test]
    fn handles_unicode_escapes() {
        assert_eq!(Value::parse(r#""A""#).unwrap().as_str(), Some("A"));
        // Surrogate pair: U+1F600 grinning face.
        assert_eq!(Value::parse(r#""😀""#).unwrap().as_str(), Some("😀"));
    }

    #[test]
    fn rejects_trailing_data() {
        assert!(Value::parse("1 2").is_err());
        assert!(Value::parse("{}garbage").is_err());
    }

    #[test]
    fn rejects_unterminated() {
        assert!(Value::parse(r#""no end"#).is_err());
        assert!(Value::parse("[1,2").is_err());
    }

    #[test]
    fn roundtrips_serialization() {
        let src = r#"{"model":"qwen","temperature":0.7,"messages":[{"role":"user","content":"hi"}],"stream":true}"#;
        let v = Value::parse(src).unwrap();
        let out = v.to_json();
        // Re-parse to confirm structural equality (formatting may differ).
        assert_eq!(Value::parse(&out).unwrap(), v);
    }

    #[test]
    fn serializes_integers_without_point() {
        assert_eq!(Value::Num(5.0).to_json(), "5");
        assert_eq!(Value::Num(5.5).to_json(), "5.5");
    }

    #[test]
    fn serializes_non_finite_as_null() {
        assert_eq!(Value::Num(f64::NAN).to_json(), "null");
        assert_eq!(Value::Num(f64::INFINITY).to_json(), "null");
    }

    #[test]
    fn escapes_control_chars_on_write() {
        let s = Value::Str("\u{01}\n".to_string()).to_json();
        assert!(s.contains("\\u0001"));
        assert!(s.contains("\\n"))
    }

    #[test]
    fn parses_unicode_content_directly() {
        let v = Value::parse("\"héllo 世界\"").unwrap();
        assert_eq!(v.as_str(), Some("héllo 世界"));
    }
}

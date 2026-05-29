//! User configuration — the `~/.zero/config.json` file, like pi/hermes use,
//! plus the merge with command-line overrides. Stored as JSON (parsed by our
//! own [`crate::json`]) so there are still zero dependencies.

use crate::json::Value;
use std::fs;
use std::io;
use std::path::Path;

/// Everything needed to talk to a model backend.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Config {
    /// Base URL of an OpenAI-compatible server, e.g. `http://host:8000`.
    /// `None` → run against the built-in stub backend.
    pub base_url: Option<String>,
    /// Model name sent in the `model` field.
    pub model: String,
    /// Optional bearer token (local servers usually need none).
    pub api_key: Option<String>,
    /// Optional sampling temperature.
    pub temperature: Option<f64>,
    /// Optional system prompt prepended to every conversation.
    pub system_prompt: Option<String>,
}

impl Config {
    /// Parse a config from JSON text. Unknown keys are ignored; missing keys
    /// take their defaults.
    pub fn from_json(text: &str) -> Result<Config, String> {
        let v = Value::parse(text).map_err(|e| e.to_string())?;
        let mut c = Config::default();
        if let Some(s) = v.get("base_url").and_then(Value::as_str) {
            if !s.is_empty() {
                c.base_url = Some(s.to_string());
            }
        }
        if let Some(s) = v.get("model").and_then(Value::as_str) {
            c.model = s.to_string();
        }
        if let Some(s) = v.get("api_key").and_then(Value::as_str) {
            if !s.is_empty() {
                c.api_key = Some(s.to_string());
            }
        }
        if let Some(t) = v.get("temperature").and_then(Value::as_f64) {
            c.temperature = Some(t);
        }
        if let Some(s) = v.get("system_prompt").and_then(Value::as_str) {
            if !s.is_empty() {
                c.system_prompt = Some(s.to_string());
            }
        }
        Ok(c)
    }

    /// Serialize to readable JSON (stable key order, one per line).
    pub fn to_json(&self) -> String {
        let q = |s: &str| Value::Str(s.to_string()).to_json();
        let mut out = String::from("{\n");
        out.push_str(&format!(
            "  \"base_url\": {},\n",
            q(self.base_url.as_deref().unwrap_or(""))
        ));
        out.push_str(&format!("  \"model\": {},\n", q(&self.model)));
        out.push_str(&format!(
            "  \"api_key\": {},\n",
            q(self.api_key.as_deref().unwrap_or(""))
        ));
        out.push_str(&format!(
            "  \"temperature\": {},\n",
            self.temperature
                .map(|t| Value::Num(t).to_json())
                .unwrap_or_else(|| "null".to_string())
        ));
        out.push_str(&format!(
            "  \"system_prompt\": {}\n",
            q(self.system_prompt.as_deref().unwrap_or(""))
        ));
        out.push('}');
        out.push('\n');
        out
    }

    /// Load from `path`. A missing file yields `Config::default()`; a malformed
    /// file is a hard error so the user notices.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Config> {
        match fs::read_to_string(path) {
            Ok(text) => Config::from_json(&text).map_err(io::Error::other),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Config::default()),
            Err(e) => Err(e),
        }
    }

    /// Write the config to `path`, creating parent directories.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, self.to_json())
    }

    /// True when a real backend is configured (vs. falling back to the stub).
    pub fn has_backend(&self) -> bool {
        self.base_url.is_some()
    }

    /// One-line human summary for `/config` and startup.
    pub fn summary(&self) -> String {
        match &self.base_url {
            Some(url) => {
                let model = if self.model.is_empty() {
                    "<no model set>"
                } else {
                    &self.model
                };
                format!("{model} @ {url}")
            }
            None => "stub backend (no base_url configured)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_no_backend() {
        let c = Config::default();
        assert!(!c.has_backend());
        assert!(c.summary().contains("stub"));
    }

    #[test]
    fn parses_full_config() {
        let json = r#"{
            "base_url": "http://gx10:8000",
            "model": "qwen",
            "api_key": "secret",
            "temperature": 0.7,
            "system_prompt": "be terse"
        }"#;
        let c = Config::from_json(json).unwrap();
        assert_eq!(c.base_url.as_deref(), Some("http://gx10:8000"));
        assert_eq!(c.model, "qwen");
        assert_eq!(c.api_key.as_deref(), Some("secret"));
        assert_eq!(c.temperature, Some(0.7));
        assert_eq!(c.system_prompt.as_deref(), Some("be terse"));
        assert!(c.has_backend());
        assert_eq!(c.summary(), "qwen @ http://gx10:8000");
    }

    #[test]
    fn empty_strings_become_none() {
        let c = Config::from_json(r#"{"base_url":"","api_key":"","model":"m"}"#).unwrap();
        assert!(c.base_url.is_none());
        assert!(c.api_key.is_none());
        assert_eq!(c.model, "m");
    }

    #[test]
    fn missing_keys_take_defaults() {
        let c = Config::from_json("{}").unwrap();
        assert_eq!(c, Config::default());
    }

    #[test]
    fn malformed_json_errors() {
        assert!(Config::from_json("{not json").is_err());
    }

    #[test]
    fn roundtrips_through_json() {
        let c = Config {
            base_url: Some("http://h:1".to_string()),
            model: "m".to_string(),
            api_key: None,
            temperature: Some(0.2),
            system_prompt: Some("sys".to_string()),
        };
        let reparsed = Config::from_json(&c.to_json()).unwrap();
        assert_eq!(reparsed, c);
    }

    #[test]
    fn summary_flags_missing_model() {
        let c = Config {
            base_url: Some("http://h:1".to_string()),
            ..Config::default()
        };
        assert!(c.summary().contains("<no model set>"));
    }

    #[test]
    fn load_missing_file_is_default() {
        let path =
            std::env::temp_dir().join(format!("zero-noexist-{}", crate::clock::unix_millis()));
        assert_eq!(Config::load(&path).unwrap(), Config::default());
    }

    #[test]
    fn save_then_load_roundtrips() {
        let dir = std::env::temp_dir().join(format!("zero-cfg-{}", crate::clock::unix_millis()));
        let path = dir.join("config.json");
        let c = Config {
            base_url: Some("http://h:8000".to_string()),
            model: "qwen".to_string(),
            api_key: Some("k".to_string()),
            temperature: Some(0.5),
            system_prompt: None,
        };
        c.save(&path).unwrap();
        assert_eq!(Config::load(&path).unwrap(), c);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_serializes_with_nulls_and_blanks() {
        let c = Config::default();
        let s = c.to_json();
        assert!(s.contains("\"temperature\": null"));
        assert_eq!(Config::from_json(&s).unwrap(), c);
    }

    #[test]
    fn load_of_a_directory_is_a_non_notfound_error() {
        // Reading a directory as a file errors with something other than
        // NotFound, exercising the catch-all error arm.
        let dir = std::env::temp_dir().join(format!("zero-isdir-{}", crate::clock::unix_millis()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(Config::load(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_malformed_file_errors() {
        let dir = std::env::temp_dir().join(format!("zero-bad-{}", crate::clock::unix_millis()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, "{ broken").unwrap();
        assert!(Config::load(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

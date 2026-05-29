//! The locally-saved list of known model servers (`~/.zero/servers.json`).
//!
//! Discovery writes here; the app reads it to let you re-attach without
//! re-scanning. [`ServerStore::upsert`] is the refresh path: re-probing a server
//! updates its model list in place, and a previously-selected model that has
//! since disappeared (the box now serves a different model) is dropped.

use crate::discovery::Discovered;
use crate::json::Value;
use std::fs;
use std::io;
use std::path::Path;

/// One known server: where it is, what it serves, and which model is selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerEntry {
    pub base_url: String,
    pub models: Vec<String>,
    /// The model chosen for this server, if any (kept across refreshes when
    /// still offered).
    pub model: Option<String>,
}

impl ServerEntry {
    fn to_value(&self) -> Value {
        Value::Object(vec![
            ("base_url".to_string(), Value::Str(self.base_url.clone())),
            (
                "model".to_string(),
                match &self.model {
                    Some(m) => Value::Str(m.clone()),
                    None => Value::Null,
                },
            ),
            (
                "models".to_string(),
                Value::Array(self.models.iter().cloned().map(Value::Str).collect()),
            ),
        ])
    }

    fn from_value(v: &Value) -> Option<ServerEntry> {
        let base_url = v.get("base_url")?.as_str()?.to_string();
        let models = v
            .get("models")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let model = v
            .get("model")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(String::from);
        Some(ServerEntry {
            base_url,
            models,
            model,
        })
    }
}

/// The persisted collection of known servers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServerStore {
    pub servers: Vec<ServerEntry>,
}

impl ServerStore {
    pub fn from_json(text: &str) -> Result<ServerStore, String> {
        let v = Value::parse(text).map_err(|e| e.to_string())?;
        let servers = v
            .get("servers")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(ServerEntry::from_value).collect())
            .unwrap_or_default();
        Ok(ServerStore { servers })
    }

    pub fn to_json(&self) -> String {
        let arr = Value::Array(self.servers.iter().map(ServerEntry::to_value).collect());
        Value::Object(vec![("servers".to_string(), arr)]).to_json()
    }

    /// Load from `path`; a missing file is an empty store.
    pub fn load(path: impl AsRef<Path>) -> io::Result<ServerStore> {
        match fs::read_to_string(path) {
            Ok(text) => ServerStore::from_json(&text).map_err(io::Error::other),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(ServerStore::default()),
            Err(e) => Err(e),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, self.to_json())
    }

    pub fn find(&self, base_url: &str) -> Option<&ServerEntry> {
        self.servers.iter().find(|s| s.base_url == base_url)
    }

    /// Add or refresh a discovered server. Updates its model list; keeps the
    /// selected model only if the server still offers it.
    pub fn upsert(&mut self, d: &Discovered) {
        if let Some(entry) = self.servers.iter_mut().find(|s| s.base_url == d.base_url) {
            entry.models = d.models.clone();
            if let Some(m) = &entry.model {
                if !entry.models.contains(m) {
                    entry.model = None; // the model under this IP changed
                }
            }
        } else {
            self.servers.push(ServerEntry {
                base_url: d.base_url.clone(),
                models: d.models.clone(),
                model: d.models.first().cloned(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc(url: &str, models: &[&str]) -> Discovered {
        Discovered {
            base_url: url.to_string(),
            models: models.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn upsert_adds_new_and_selects_first_model() {
        let mut store = ServerStore::default();
        store.upsert(&disc("http://a:8000", &["qwen", "llama"]));
        assert_eq!(store.servers.len(), 1);
        assert_eq!(store.servers[0].model.as_deref(), Some("qwen"));
    }

    #[test]
    fn upsert_refreshes_models_in_place() {
        let mut store = ServerStore::default();
        store.upsert(&disc("http://a:8000", &["qwen"]));
        store.upsert(&disc("http://a:8000", &["qwen", "new-model"]));
        assert_eq!(store.servers.len(), 1); // not duplicated
        assert_eq!(store.servers[0].models.len(), 2);
        assert_eq!(store.servers[0].model.as_deref(), Some("qwen")); // selection kept
    }

    #[test]
    fn upsert_drops_selection_when_model_disappears() {
        let mut store = ServerStore::default();
        store.upsert(&disc("http://a:8000", &["qwen"]));
        // The box now serves something else entirely.
        store.upsert(&disc("http://a:8000", &["completely-different"]));
        assert!(store.servers[0].model.is_none());
    }

    #[test]
    fn find_by_url() {
        let mut store = ServerStore::default();
        store.upsert(&disc("http://a:8000", &["m"]));
        assert!(store.find("http://a:8000").is_some());
        assert!(store.find("http://nope").is_none());
    }

    #[test]
    fn roundtrips_through_json() {
        let mut store = ServerStore::default();
        store.upsert(&disc("http://a:8000", &["m1", "m2"]));
        store.upsert(&disc("http://b:1234", &["x"]));
        let reparsed = ServerStore::from_json(&store.to_json()).unwrap();
        assert_eq!(reparsed, store);
    }

    #[test]
    fn from_json_tolerates_missing_and_bad() {
        assert_eq!(
            ServerStore::from_json("{}").unwrap(),
            ServerStore::default()
        );
        assert!(ServerStore::from_json("{bad").is_err());
    }

    #[test]
    fn load_missing_is_empty_save_then_load_roundtrips() {
        let dir = std::env::temp_dir().join(format!("zero-srv-{}", crate::clock::unix_millis()));
        let path = dir.join("servers.json");
        assert_eq!(ServerStore::load(&path).unwrap(), ServerStore::default());
        let mut store = ServerStore::default();
        store.upsert(&disc("http://a:8000", &["m"]));
        store.save(&path).unwrap();
        assert_eq!(ServerStore::load(&path).unwrap(), store);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn entry_with_null_model_parses() {
        let json = r#"{"servers":[{"base_url":"http://a:1","model":null,"models":["m"]}]}"#;
        let store = ServerStore::from_json(json).unwrap();
        assert!(store.servers[0].model.is_none());
    }
}

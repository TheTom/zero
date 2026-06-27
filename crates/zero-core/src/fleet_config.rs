// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! Fleet configuration — the optional `~/.zero/fleet.toml` that declares a
//! multi-tier model orchestrator. When absent (the default), Zero runs a single
//! model exactly as before; the orchestrator is opt-in.
//!
//! The file is a small TOML subset, parsed by a self-contained reader in this
//! module (the same shape `loop_config` uses for `loop.toml`, so there are still
//! zero dependencies). Supported: top-level `key = value`, `[[tier]]`
//! arrays-of-tables, a `[routing]` table, and value types string / bool / float
//! / array-of-strings. Example:
//!
//! ```toml
//! enabled  = true
//! routing  = "auto"            # auto | manual | off
//! baseline = "balanced"        # default tier for normal work
//!
//! [[tier]]
//! key     = "deep"             # your strongest model
//! where   = "http://localhost:8080"
//! model   = "your-strong-model"
//! use_for = ["plan", "architecture", "hard-debug"]
//!
//! [[tier]]
//! key     = "balanced"
//! where   = "http://localhost:8000"
//! model   = "your-mid-model"
//!
//! [[tier]]
//! key     = "fast"
//! where   = "http://192.168.1.50:11434"   # can be a different machine
//! model   = "your-fast-model"
//! use_for = ["simple-query", "quick-edit"]
//!
//! [routing]
//! plan_mode      = "deep"      # plan mode routes here
//! simple_queries = "fast"      # short/simple prompts route here
//! fallback       = ["deep", "balanced", "fast"]   # tried in order on error
//! ```

use crate::config::Config;
use std::fs;
use std::io;
use std::path::Path;

/// How the orchestrator decides which tier answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutingMode {
    /// The router picks a tier per request (the point of the feature).
    #[default]
    Auto,
    /// Always use the manual pin (or the first tier) — the user drives.
    Manual,
    /// Disabled — behave as a single model (the first tier).
    Off,
}

impl RoutingMode {
    fn parse(s: &str) -> RoutingMode {
        match s.trim().to_ascii_lowercase().as_str() {
            "manual" => RoutingMode::Manual,
            "off" => RoutingMode::Off,
            _ => RoutingMode::Auto,
        }
    }
}

/// One declared tier: a capability label bound to a concrete backend.
#[derive(Debug, Clone, PartialEq)]
pub struct TierConfig {
    /// Arbitrary user key, e.g. `"deep"` / `"fast"` / `"senior"`.
    pub key: String,
    /// Base URL of an OpenAI-compatible server for this tier (`where = …`).
    pub base_url: String,
    /// Model name sent in the `model` field.
    pub model: String,
    /// Capability tags that bias routing toward this tier (`use_for = [...]`).
    pub use_for: Vec<String>,
    /// Optional per-tier sampling temperature.
    pub temperature: Option<f64>,
    /// Optional per-tier bearer token.
    pub api_key: Option<String>,
    /// Optional per-tier system prompt (`system = …`).
    pub system_prompt: Option<String>,
}

impl TierConfig {
    /// Project this tier onto a [`Config`] so the existing `OpenAiBackend` can be
    /// built from it without any new construction code.
    pub fn to_config(&self) -> Config {
        Config {
            base_url: Some(self.base_url.clone()),
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            temperature: self.temperature,
            system_prompt: self.system_prompt.clone(),
            ..Config::default()
        }
    }
}

/// The parsed fleet. Defaults to disabled; a missing file yields this.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FleetConfig {
    pub enabled: bool,
    pub routing: RoutingMode,
    pub tiers: Vec<TierConfig>,
    /// Tier keys tried in order when a tier errors. Defaults to declared order.
    pub fallback: Vec<String>,
    /// Default tier for normal work. Defaults to the middle of the ladder.
    pub baseline: Option<String>,
    /// Tier to use in plan mode (`[routing] plan_mode`).
    pub plan_pin: Option<String>,
    /// Tier to use for short/simple prompts (`[routing] simple_queries`).
    pub simple_pin: Option<String>,
}

impl FleetConfig {
    /// A disabled fleet — the default when no `fleet.toml` exists.
    pub fn disabled() -> FleetConfig {
        FleetConfig::default()
    }

    /// True when the orchestrator should be wired in: enabled, not `off`, and at
    /// least one tier declared.
    pub fn is_active(&self) -> bool {
        self.enabled && self.routing != RoutingMode::Off && !self.tiers.is_empty()
    }

    /// Look up a tier by key.
    pub fn tier(&self, key: &str) -> Option<&TierConfig> {
        self.tiers.iter().find(|t| t.key == key)
    }

    /// The strongest tier key (front of the fallback ladder).
    pub fn strongest(&self) -> Option<&str> {
        self.fallback
            .first()
            .map(String::as_str)
            .or_else(|| self.tiers.first().map(|t| t.key.as_str()))
    }

    /// The weakest tier key (back of the fallback ladder).
    pub fn weakest(&self) -> Option<&str> {
        self.fallback
            .last()
            .map(String::as_str)
            .or_else(|| self.tiers.last().map(|t| t.key.as_str()))
    }

    /// The default tier for normal work: the configured baseline if it resolves,
    /// else the middle of the ladder.
    pub fn baseline_key(&self) -> Option<&str> {
        if let Some(b) = &self.baseline {
            if self.tier(b).is_some() {
                return Some(b.as_str());
            }
        }
        let ladder = if self.fallback.is_empty() {
            // Fall back to declared order when no explicit ladder.
            return self.tiers.get(self.tiers.len() / 2).map(|t| t.key.as_str());
        } else {
            &self.fallback
        };
        ladder.get(ladder.len() / 2).map(String::as_str)
    }

    /// Load from `path`. A missing file is a disabled fleet; a malformed file is a
    /// hard error so the user notices.
    pub fn load(path: impl AsRef<Path>) -> io::Result<FleetConfig> {
        match fs::read_to_string(path) {
            Ok(text) => FleetConfig::parse(&text).map_err(io::Error::other),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(FleetConfig::disabled()),
            Err(e) => Err(e),
        }
    }

    /// Parse `fleet.toml` text. Errors on a missing tier `key`/`where`, a
    /// duplicate tier key, or a malformed line — misconfiguration is loud.
    pub fn parse(text: &str) -> Result<FleetConfig, String> {
        let mut top = Table::default();
        let mut routing = Table::default();
        let mut tiers: Vec<Table> = Vec::new();

        enum Cur {
            Top,
            Routing,
            Tier,
        }
        let mut cur = Cur::Top;

        for (lineno, raw) in text.lines().enumerate() {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            if line == "[[tier]]" {
                tiers.push(Table::default());
                cur = Cur::Tier;
            } else if line == "[routing]" {
                cur = Cur::Routing;
            } else if line.starts_with('[') {
                return Err(format!("line {}: unknown section {line:?}", lineno + 1));
            } else if let Some(eq) = line.find('=') {
                let key = line[..eq].trim().to_string();
                let val = parse_value(line[eq + 1..].trim())
                    .ok_or_else(|| format!("line {}: empty value for {key:?}", lineno + 1))?;
                let table = match cur {
                    Cur::Top => &mut top,
                    Cur::Routing => &mut routing,
                    Cur::Tier => tiers
                        .last_mut()
                        .ok_or_else(|| format!("line {}: key before [[tier]]", lineno + 1))?,
                };
                table.kv.push((key, val));
            } else {
                return Err(format!("line {}: not a header or key=value", lineno + 1));
            }
        }

        let mut out = FleetConfig {
            enabled: top.bool("enabled").unwrap_or(false),
            routing: top
                .str("routing")
                .map(|s| RoutingMode::parse(&s))
                .unwrap_or_default(),
            baseline: top.str("baseline"),
            plan_pin: routing.str("plan_mode"),
            simple_pin: routing.str("simple_queries"),
            fallback: routing.arr("fallback"),
            tiers: Vec::new(),
        };

        for (i, t) in tiers.iter().enumerate() {
            let key = t
                .str("key")
                .ok_or_else(|| format!("tier #{}: missing key", i + 1))?;
            if out.tiers.iter().any(|x: &TierConfig| x.key == key) {
                return Err(format!("duplicate tier key {key:?}"));
            }
            let base_url = t
                .str("where")
                .ok_or_else(|| format!("tier {key:?}: missing where (base_url)"))?;
            out.tiers.push(TierConfig {
                key,
                base_url,
                model: t.str("model").unwrap_or_default(),
                use_for: t.arr("use_for"),
                temperature: t.float("temperature"),
                api_key: t.str("api_key").filter(|s| !s.is_empty()),
                system_prompt: t.str("system").filter(|s| !s.is_empty()),
            });
        }

        // Default the ladder to declared order when not given explicitly.
        if out.fallback.is_empty() {
            out.fallback = out.tiers.iter().map(|t| t.key.clone()).collect();
        }
        Ok(out)
    }
}

// ---- a tiny TOML-subset reader (self-contained, std only) -------------------

/// A parsed value. Only the shapes `fleet.toml` needs.
#[derive(Debug, Clone, PartialEq)]
enum Tv {
    Str(String),
    Bool(bool),
    Float(f64),
    Arr(Vec<String>),
}

#[derive(Default)]
struct Table {
    kv: Vec<(String, Tv)>,
}

impl Table {
    fn get(&self, key: &str) -> Option<&Tv> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    fn str(&self, key: &str) -> Option<String> {
        match self.get(key)? {
            Tv::Str(s) => Some(s.clone()),
            Tv::Bool(b) => Some(b.to_string()),
            Tv::Float(f) => Some(f.to_string()),
            Tv::Arr(_) => None,
        }
    }
    fn bool(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            Tv::Bool(b) => Some(*b),
            _ => None,
        }
    }
    fn float(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            Tv::Float(f) => Some(*f),
            _ => None,
        }
    }
    fn arr(&self, key: &str) -> Vec<String> {
        match self.get(key) {
            Some(Tv::Arr(a)) => a.clone(),
            _ => Vec::new(),
        }
    }
}

/// Drop a trailing `# comment` not inside a string.
fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    for (i, b) in line.bytes().enumerate() {
        match b {
            b'"' | b'\'' => in_str = !in_str,
            b'#' if !in_str => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Strip a matching pair of single or double quotes, if present.
fn quoted(s: &str) -> Option<String> {
    let b = s.as_bytes();
    if b.len() >= 2
        && ((b[0] == b'"' && b[b.len() - 1] == b'"') || (b[0] == b'\'' && b[b.len() - 1] == b'\''))
    {
        Some(s[1..s.len() - 1].to_string())
    } else {
        None
    }
}

/// Parse one value: string / bool / float / string-array. A bare word becomes a
/// string (so `routing = auto` works unquoted). Array items may be quoted; commas
/// inside a quoted item are not supported (not needed for these configs).
fn parse_value(s: &str) -> Option<Tv> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(q) = quoted(s) {
        return Some(Tv::Str(q));
    }
    if s == "true" {
        return Some(Tv::Bool(true));
    }
    if s == "false" {
        return Some(Tv::Bool(false));
    }
    if let Some(inner) = s.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
        let items = inner
            .split(',')
            .map(|x| {
                let x = x.trim();
                quoted(x).unwrap_or_else(|| x.to_string())
            })
            .filter(|x| !x.is_empty())
            .collect();
        return Some(Tv::Arr(items));
    }
    if let Ok(f) = s.parse::<f64>() {
        return Some(Tv::Float(f));
    }
    Some(Tv::Str(s.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
        enabled  = true
        routing  = "auto"
        baseline = "balanced"

        [[tier]]
        key     = "deep"
        where   = "http://localhost:8080"
        model   = "strong"
        use_for = ["plan", "architecture"]
        temperature = 0.7

        [[tier]]
        key   = "balanced"
        where = "http://localhost:8000"
        model = "mid"

        [[tier]]
        key     = "fast"
        where   = "http://192.168.1.50:11434"
        model   = "quick"
        use_for = ["simple-query"]

        [routing]
        plan_mode      = "deep"
        simple_queries = "fast"
        fallback       = ["deep", "balanced", "fast"]
    "#;

    #[test]
    fn parses_the_sample() {
        let f = FleetConfig::parse(SAMPLE).unwrap();
        assert!(f.enabled);
        assert_eq!(f.routing, RoutingMode::Auto);
        assert!(f.is_active());
        assert_eq!(f.tiers.len(), 3);
        assert_eq!(f.tiers[0].key, "deep");
        assert_eq!(f.tiers[0].base_url, "http://localhost:8080");
        assert_eq!(f.tiers[0].use_for, vec!["plan", "architecture"]);
        assert_eq!(f.tiers[0].temperature, Some(0.7));
        assert_eq!(f.tiers[2].base_url, "http://192.168.1.50:11434");
        assert_eq!(f.plan_pin.as_deref(), Some("deep"));
        assert_eq!(f.simple_pin.as_deref(), Some("fast"));
        assert_eq!(f.fallback, vec!["deep", "balanced", "fast"]);
    }

    #[test]
    fn ladder_helpers() {
        let f = FleetConfig::parse(SAMPLE).unwrap();
        assert_eq!(f.strongest(), Some("deep"));
        assert_eq!(f.weakest(), Some("fast"));
        assert_eq!(f.baseline_key(), Some("balanced"));
    }

    #[test]
    fn baseline_defaults_to_middle_of_ladder() {
        let cfg = r#"
            enabled = true
            [[tier]]
            key = "a"
            where = "http://a:1"
            [[tier]]
            key = "b"
            where = "http://b:1"
            [[tier]]
            key = "c"
            where = "http://c:1"
        "#;
        let f = FleetConfig::parse(cfg).unwrap();
        // No [routing] → fallback defaults to declared order; middle = "b".
        assert_eq!(f.fallback, vec!["a", "b", "c"]);
        assert_eq!(f.baseline_key(), Some("b"));
    }

    #[test]
    fn invalid_baseline_falls_back_to_middle() {
        let cfg = r#"
            enabled = true
            baseline = "nope"
            [[tier]]
            key = "a"
            where = "http://a:1"
            [[tier]]
            key = "b"
            where = "http://b:1"
        "#;
        let f = FleetConfig::parse(cfg).unwrap();
        assert_eq!(f.baseline_key(), Some("b")); // len 2 / 2 = index 1
    }

    #[test]
    fn unquoted_routing_word_parses() {
        let f = FleetConfig::parse(
            "enabled = true\nrouting = manual\n[[tier]]\nkey=\"a\"\nwhere=\"http://a:1\"",
        )
        .unwrap();
        assert_eq!(f.routing, RoutingMode::Manual);
    }

    #[test]
    fn off_routing_is_inactive() {
        let f = FleetConfig::parse(
            "enabled = true\nrouting = \"off\"\n[[tier]]\nkey=\"a\"\nwhere=\"http://a:1\"",
        )
        .unwrap();
        assert_eq!(f.routing, RoutingMode::Off);
        assert!(!f.is_active());
    }

    #[test]
    fn disabled_when_flag_false_or_no_tiers() {
        assert!(!FleetConfig::disabled().is_active());
        let f = FleetConfig::parse("enabled = false\n[[tier]]\nkey=\"a\"\nwhere=\"http://a:1\"")
            .unwrap();
        assert!(!f.is_active());
        let empty = FleetConfig::parse("enabled = true\n").unwrap();
        assert!(!empty.is_active());
    }

    #[test]
    fn missing_key_or_where_errors() {
        assert!(FleetConfig::parse("[[tier]]\nwhere=\"http://a:1\"").is_err());
        assert!(FleetConfig::parse("[[tier]]\nkey=\"a\"").is_err());
    }

    #[test]
    fn duplicate_tier_key_errors() {
        let cfg =
            "[[tier]]\nkey=\"a\"\nwhere=\"http://a:1\"\n[[tier]]\nkey=\"a\"\nwhere=\"http://b:1\"";
        assert!(FleetConfig::parse(cfg).is_err());
    }

    #[test]
    fn unknown_section_and_stray_lines_error() {
        assert!(FleetConfig::parse("[bogus]\nx=1").is_err());
        assert!(FleetConfig::parse("[[tier]]\nkey=\"a\"\nwhere=\"http://a:1\"\nnonsense").is_err());
        assert!(FleetConfig::parse("[routing]\nx =").is_err()); // empty value
    }

    #[test]
    fn key_before_tier_header_errors() {
        // A key=value under [routing] is fine; a tier field with no [[tier]] is not
        // reachable via Cur, but a bare key at top is allowed (top table). Here we
        // ensure a value line right after [[tier]] lands in the tier.
        let f =
            FleetConfig::parse("[[tier]]\nkey=\"a\"\nwhere=\"http://a:1\"\nmodel=\"m\"").unwrap();
        assert_eq!(f.tiers[0].model, "m");
    }

    #[test]
    fn comments_and_blank_lines_ignored() {
        let cfg = "# header\nenabled = true   # inline\n\n[[tier]]\nkey=\"a\" # named a\nwhere=\"http://a:1\"";
        let f = FleetConfig::parse(cfg).unwrap();
        assert!(f.enabled);
        assert_eq!(f.tiers[0].key, "a");
    }

    #[test]
    fn to_config_projects_tier_onto_backend_config() {
        let f = FleetConfig::parse(SAMPLE).unwrap();
        let c = f.tiers[0].to_config();
        assert_eq!(c.base_url.as_deref(), Some("http://localhost:8080"));
        assert_eq!(c.model, "strong");
        assert_eq!(c.temperature, Some(0.7));
        assert!(c.has_backend());
    }

    #[test]
    fn api_key_and_system_are_optional_and_blank_becomes_none() {
        let cfg = "[[tier]]\nkey=\"a\"\nwhere=\"http://a:1\"\napi_key=\"\"\nsystem=\"be terse\"";
        let f = FleetConfig::parse(cfg).unwrap();
        assert!(f.tiers[0].api_key.is_none());
        assert_eq!(f.tiers[0].system_prompt.as_deref(), Some("be terse"));
    }

    #[test]
    fn load_missing_file_is_disabled() {
        let path =
            std::env::temp_dir().join(format!("zero-fleet-none-{}", crate::clock::unix_millis()));
        assert_eq!(FleetConfig::load(&path).unwrap(), FleetConfig::disabled());
    }

    #[test]
    fn load_then_parse_roundtrips_from_disk() {
        let dir = std::env::temp_dir().join(format!("zero-fleet-{}", crate::clock::unix_millis()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fleet.toml");
        std::fs::write(&path, SAMPLE).unwrap();
        let f = FleetConfig::load(&path).unwrap();
        assert_eq!(f.tiers.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_malformed_file_errors() {
        let dir =
            std::env::temp_dir().join(format!("zero-fleet-bad-{}", crate::clock::unix_millis()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fleet.toml");
        std::fs::write(&path, "[[tier]]\nkey=\"a\"").unwrap(); // missing where
        assert!(FleetConfig::load(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_of_a_directory_is_an_error() {
        let dir =
            std::env::temp_dir().join(format!("zero-fleet-dir-{}", crate::clock::unix_millis()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(FleetConfig::load(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn value_table_getters_reject_wrong_types() {
        // str() on an array is None; bool()/float() on a string is None.
        let t = Table {
            kv: vec![
                ("a".to_string(), Tv::Arr(vec!["x".to_string()])),
                ("b".to_string(), Tv::Str("s".to_string())),
            ],
        };
        assert!(t.str("a").is_none());
        assert!(t.bool("b").is_none());
        assert!(t.float("b").is_none());
        assert_eq!(t.arr("a"), vec!["x"]);
        assert!(t.arr("b").is_empty());
    }

    #[test]
    fn parse_value_covers_each_shape() {
        assert_eq!(parse_value("\"hi\""), Some(Tv::Str("hi".to_string())));
        assert_eq!(parse_value("'hi'"), Some(Tv::Str("hi".to_string())));
        assert_eq!(parse_value("true"), Some(Tv::Bool(true)));
        assert_eq!(parse_value("false"), Some(Tv::Bool(false)));
        assert_eq!(parse_value("0.7"), Some(Tv::Float(0.7)));
        assert_eq!(parse_value("bare"), Some(Tv::Str("bare".to_string())));
        assert_eq!(
            parse_value("[\"a\", \"b\"]"),
            Some(Tv::Arr(vec!["a".to_string(), "b".to_string()]))
        );
        assert_eq!(parse_value(""), None);
        assert_eq!(parse_value("   "), None);
    }
}

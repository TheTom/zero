// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! MCP (Model Context Protocol) client over the stdio transport — zero-dep.
//!
//! An MCP server is a subprocess that speaks JSON-RPC 2.0 over its stdin/stdout,
//! one JSON message per line. This module: (1) parses the Claude-compatible
//! `mcpServers` config, (2) drives the JSON-RPC handshake + `tools/list` over any
//! line transport ([`Session`], fully unit-tested against in-memory pipes), and
//! (3) spawns a real server and wires its pipes into a [`Session`]
//! ([`Connection::connect`]).
//!
//! Connect, list tools ([`Connection::connect`]), and **call** them
//! ([`Connection::call_tool`] / `tools/call`). [`tool_routes`] turns a set of
//! live connections into namespaced [`ToolDef`]s the agentic loop advertises to
//! the model, with [`Connection::reconnect`] to recover a dropped server.
//!
//! [`ToolDef`]: crate::tools::ToolDef

use crate::json::Value;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// One configured MCP server (how to launch it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerSpec {
    pub command: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

/// Parsed `~/.zero/mcp.json`. Mirrors Claude's shape:
/// ```json
/// { "mcpServers": { "fs": { "command": "npx", "args": ["-y", "…"], "env": {} } } }
/// ```
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct McpConfig {
    /// (name, spec) pairs, in file order.
    pub servers: Vec<(String, ServerSpec)>,
}

impl McpConfig {
    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }

    /// Parse the JSON config text. Unknown/extra keys are ignored; a server
    /// entry missing `command` (e.g. an HTTP/SSE server) is skipped — Zero only
    /// speaks the stdio transport.
    pub fn parse(text: &str) -> Result<McpConfig, String> {
        let v = Value::parse(text).map_err(|e| e.to_string())?;
        Ok(McpConfig {
            servers: servers_from_value(&v),
        })
    }

    /// Load from a path, returning an empty config if the file does not exist.
    pub fn load(path: &std::path::Path) -> io::Result<McpConfig> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                McpConfig::parse(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(McpConfig::default()),
            Err(e) => Err(e),
        }
    }
}

/// Extract `(name, spec)` pairs from a value's `mcpServers` object. Stdio-only:
/// entries without a `command` are skipped. Shared by every config source —
/// Claude Desktop, Claude Code, Zero, and project `.mcp.json` all use this shape.
fn servers_from_value(v: &Value) -> Vec<(String, ServerSpec)> {
    let mut out = Vec::new();
    if let Some(Value::Object(entries)) = v.get("mcpServers") {
        for (name, spec) in entries {
            if let Some(s) = parse_spec(spec) {
                out.push((name.clone(), s));
            }
        }
    }
    out
}

/// Where a discovered server's config came from — for display and precedence.
/// Zero imports MCP servers you've already configured in other tools rather than
/// demanding its own file, the way Claude Code / pi / hermes do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// `./.mcp.json` in the working directory (highest precedence).
    Project,
    /// `~/.zero/mcp.json` — Zero's own file.
    Zero,
    /// Claude Desktop's `claude_desktop_config.json`.
    ClaudeDesktop,
    /// Claude Code's `~/.claude.json` (global + the current project's entry).
    ClaudeCode,
}

impl Source {
    /// Short human label for the `/mcp` summary.
    pub fn label(self) -> &'static str {
        match self {
            Source::Project => "project .mcp.json",
            Source::Zero => "~/.zero/mcp.json",
            Source::ClaudeDesktop => "Claude Desktop",
            Source::ClaudeCode => "Claude Code",
        }
    }
}

/// A server found during discovery, tagged with which config it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Discovered {
    pub name: String,
    pub spec: ServerSpec,
    pub source: Source,
}

/// The result of scanning all config sources: the merged server list plus any
/// per-source parse errors (a broken config from one tool never blocks the rest).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Discovery {
    pub servers: Vec<Discovered>,
    /// `(source, path-as-string, message)` for files that existed but failed to
    /// parse — surfaced as warnings, not hard failures.
    pub errors: Vec<(Source, String, String)>,
}

/// The standard config locations, highest precedence first. `home` is `$HOME`,
/// `cwd` the working directory, `zero_dir` Zero's dot-dir name (e.g. `.zero`).
pub fn default_sources(
    home: &std::path::Path,
    cwd: &std::path::Path,
    zero_dir: &str,
) -> Vec<(Source, std::path::PathBuf)> {
    vec![
        (Source::Project, cwd.join(".mcp.json")),
        (Source::Zero, home.join(zero_dir).join("mcp.json")),
        (
            Source::ClaudeDesktop,
            home.join("Library/Application Support/Claude/claude_desktop_config.json"),
        ),
        (Source::ClaudeCode, home.join(".claude.json")),
    ]
}

/// Read one config file. A missing file yields no servers and no error (it's just
/// absent); a present-but-malformed file yields no servers and a surfaced error.
/// For `ClaudeCode`, also pulls the per-project `projects.<cwd>.mcpServers` entry.
fn read_source(
    source: Source,
    path: &std::path::Path,
    cwd: &std::path::Path,
) -> (Vec<(String, ServerSpec)>, Option<String>) {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return (Vec::new(), None), // absent — silently skip
    };
    let v = match Value::parse(&text) {
        Ok(v) => v,
        Err(e) => return (Vec::new(), Some(e.to_string())),
    };
    let mut servers = servers_from_value(&v);
    if source == Source::ClaudeCode {
        // ~/.claude.json nests per-project config under projects.<abs-path>.
        let key = cwd.to_string_lossy();
        if let Some(proj) = v.get("projects").and_then(|p| p.get(&key)) {
            servers.extend(servers_from_value(proj));
        }
    }
    (servers, None)
}

/// Scan `candidates` (in precedence order) and merge into one server list,
/// first-seen-name wins. Stdio-only servers; collects per-source parse errors.
pub fn discover(candidates: &[(Source, std::path::PathBuf)], cwd: &std::path::Path) -> Discovery {
    let mut seen = std::collections::HashSet::new();
    let mut servers = Vec::new();
    let mut errors = Vec::new();
    for (source, path) in candidates {
        let (found, err) = read_source(*source, path, cwd);
        if let Some(msg) = err {
            errors.push((*source, path.to_string_lossy().into_owned(), msg));
        }
        for (name, spec) in found {
            if seen.insert(name.clone()) {
                servers.push(Discovered {
                    name,
                    spec,
                    source: *source,
                });
            }
        }
    }
    Discovery { servers, errors }
}

fn parse_spec(v: &Value) -> Option<ServerSpec> {
    let command = v.get("command")?.as_str()?.to_string();
    let mut args = Vec::new();
    if let Some(Value::Array(a)) = v.get("args") {
        for x in a {
            if let Some(s) = x.as_str() {
                args.push(s.to_string());
            }
        }
    }
    let mut env = Vec::new();
    if let Some(Value::Object(e)) = v.get("env") {
        for (k, val) in e {
            if let Some(s) = val.as_str() {
                env.push((k.clone(), s.to_string()));
            }
        }
    }
    Some(ServerSpec { command, args, env })
}

/// A tool advertised by a server.
#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// The tool's JSON-Schema for its arguments (MCP `inputSchema`), passed
    /// straight through as a [`crate::tools::ToolDef`]'s `parameters` so the model
    /// knows how to call it. Defaults to an open object schema when absent.
    pub input_schema: Value,
}

/// The name an MCP tool is advertised to the model under: `{server}__{tool}`,
/// sanitized to the `[A-Za-z0-9_-]` charset most function-calling APIs require.
/// The `__` delimiter mirrors Claude's MCP naming and keeps tools from distinct
/// servers from colliding.
pub fn mcp_tool_name(server: &str, tool: &str) -> String {
    let clean = |s: &str| -> String {
        s.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    };
    format!("{}__{}", clean(server), clean(tool))
}

/// A JSON-RPC 2.0 session over any newline-delimited transport. Generic over the
/// reader/writer so the protocol is testable without a real process.
pub struct Session<R: BufRead, W: Write> {
    reader: R,
    writer: W,
    next_id: u64,
}

impl<R: BufRead, W: Write> Session<R, W> {
    pub fn new(reader: R, writer: W) -> Self {
        Session {
            reader,
            writer,
            next_id: 0,
        }
    }

    fn send(&mut self, msg: &Value) -> io::Result<()> {
        self.writer.write_all(msg.to_json().as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()
    }

    /// Send a notification (no response expected).
    pub fn notify(&mut self, method: &str, params: Value) -> io::Result<()> {
        let msg = Value::Object(vec![
            ("jsonrpc".to_string(), Value::Str("2.0".to_string())),
            ("method".to_string(), Value::Str(method.to_string())),
            ("params".to_string(), params),
        ]);
        self.send(&msg)
    }

    /// Send a request and read until the matching response arrives, skipping any
    /// interleaved notifications / log lines. Returns the `result` value.
    pub fn request(&mut self, method: &str, params: Value) -> io::Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        let msg = Value::Object(vec![
            ("jsonrpc".to_string(), Value::Str("2.0".to_string())),
            ("id".to_string(), Value::Num(id as f64)),
            ("method".to_string(), Value::Str(method.to_string())),
            ("params".to_string(), params),
        ]);
        self.send(&msg)?;

        loop {
            let mut line = String::new();
            let n = self.reader.read_line(&mut line)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "MCP server closed the connection",
                ));
            }
            let Ok(v) = Value::parse(line.trim()) else {
                continue; // not JSON (a log line, blank, …)
            };
            if v.get("id").and_then(Value::as_f64) != Some(id as f64) {
                continue; // a notification or some other id
            }
            if let Some(err) = v.get("error") {
                let detail = err
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error");
                return Err(io::Error::other(format!("MCP error: {detail}")));
            }
            return Ok(v.get("result").cloned().unwrap_or(Value::Null));
        }
    }

    /// Perform the MCP handshake: `initialize` then the `initialized` notice.
    pub fn initialize(&mut self) -> io::Result<()> {
        let params = Value::Object(vec![
            (
                "protocolVersion".to_string(),
                Value::Str("2024-11-05".to_string()),
            ),
            ("capabilities".to_string(), Value::Object(vec![])),
            (
                "clientInfo".to_string(),
                Value::Object(vec![
                    ("name".to_string(), Value::Str("zero".to_string())),
                    ("version".to_string(), Value::Str("0.0.1".to_string())),
                ]),
            ),
        ]);
        self.request("initialize", params)?;
        self.notify("notifications/initialized", Value::Object(vec![]))
    }

    /// Ask the server for its tools.
    pub fn list_tools(&mut self) -> io::Result<Vec<Tool>> {
        let res = self.request("tools/list", Value::Object(vec![]))?;
        let mut tools = Vec::new();
        if let Some(arr) = res.get("tools").and_then(Value::as_array) {
            for t in arr {
                let Some(name) = t.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let description = t
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                // Pass the server's inputSchema straight through; default to an open
                // object schema so a tool without one is still callable.
                let input_schema = t.get("inputSchema").cloned().unwrap_or_else(|| {
                    Value::Object(vec![("type".to_string(), Value::Str("object".to_string()))])
                });
                tools.push(Tool {
                    name: name.to_string(),
                    description,
                    input_schema,
                });
            }
        }
        Ok(tools)
    }

    /// Call a tool (`tools/call`) and return its result as text. The MCP result is
    /// a `content` array of typed blocks; text blocks are concatenated and any
    /// non-text block (image/resource) is noted as `[<kind> content]`. A server
    /// that sets `isError: true` is surfaced as an `error: …` string the model can
    /// react to, never a hard failure.
    pub fn call_tool(&mut self, tool: &str, arguments: Value) -> io::Result<String> {
        let params = Value::Object(vec![
            ("name".to_string(), Value::Str(tool.to_string())),
            ("arguments".to_string(), arguments),
        ]);
        let res = self.request("tools/call", params)?;
        let mut out = String::new();
        if let Some(blocks) = res.get("content").and_then(Value::as_array) {
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        out.push_str(b.get("text").and_then(Value::as_str).unwrap_or(""));
                    }
                    Some(kind) => out.push_str(&format!("[{kind} content]")),
                    None => {}
                }
            }
        }
        if res.get("isError").and_then(Value::as_bool) == Some(true) {
            return Ok(format!("error: {out}"));
        }
        Ok(out)
    }
}

/// A live connection to a spawned MCP server: the child process, a session over
/// its pipes, and the tools it advertised at connect time.
pub struct Connection {
    pub name: String,
    pub tools: Vec<Tool>,
    /// How this server was launched — kept so the connection can [`reconnect`].
    ///
    /// [`reconnect`]: Connection::reconnect
    pub spec: ServerSpec,
    child: Child,
    session: Session<BufReader<ChildStdout>, ChildStdin>,
}

impl Connection {
    /// Spawn `spec`, handshake, and list its tools.
    pub fn connect(name: &str, spec: &ServerSpec) -> io::Result<Connection> {
        let mut child = Command::new(&spec.command)
            .args(&spec.args)
            .envs(spec.env.iter().map(|(k, v)| (k, v)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("no stdout"))?;
        let mut session = Session::new(BufReader::new(stdout), stdin);
        session.initialize()?;
        let tools = session.list_tools()?;
        Ok(Connection {
            name: name.to_string(),
            tools,
            spec: spec.clone(),
            child,
            session,
        })
    }

    /// Call one of this server's tools by its *raw* (un-namespaced) name.
    pub fn call_tool(&mut self, tool: &str, arguments: Value) -> io::Result<String> {
        self.session.call_tool(tool, arguments)
    }

    /// True if this server advertised a tool with the given raw name.
    pub fn has_tool(&self, tool: &str) -> bool {
        self.tools.iter().any(|t| t.name == tool)
    }

    /// Kill the current child and re-launch from the stored spec, refreshing the
    /// tool list. Used to recover a server that died or changed its tools.
    pub fn reconnect(&mut self) -> io::Result<()> {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let fresh = Connection::connect(&self.name, &self.spec)?;
        *self = fresh;
        Ok(())
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One MCP tool ready to expose to the model: its namespaced [`ToolDef`] plus how
/// to route a call back — the connection index and the tool's raw (un-namespaced)
/// name. [`crate::tools::ToolDef`].
#[derive(Debug, Clone, PartialEq)]
pub struct McpRoute {
    pub def: crate::tools::ToolDef,
    /// Index into the connections slice this route was built from.
    pub conn: usize,
    /// The tool's raw name as the server knows it (for `tools/call`).
    pub raw_name: String,
}

/// Build advertised tool defs + dispatch routes for a set of live connections.
/// Names are namespaced (`{server}__{tool}`) so tools from different servers can't
/// collide, and the description is tagged with the server for the model's benefit.
pub fn tool_routes(conns: &[Connection]) -> Vec<McpRoute> {
    let mut routes = Vec::new();
    for (i, c) in conns.iter().enumerate() {
        for t in &c.tools {
            let name = mcp_tool_name(&c.name, &t.name);
            let desc = if t.description.is_empty() {
                format!("[mcp:{}] {}", c.name, t.name)
            } else {
                format!("[mcp:{}] {}", c.name, t.description)
            };
            routes.push(McpRoute {
                def: crate::tools::ToolDef::new(name, desc, t.input_schema.clone()),
                conn: i,
                raw_name: t.name.clone(),
            });
        }
    }
    routes
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_claude_style_config() {
        let cfg = McpConfig::parse(
            r#"{ "mcpServers": {
                 "fs": { "command": "npx", "args": ["-y", "server-fs"], "env": { "ROOT": "/tmp" } },
                 "git": { "command": "uvx", "args": ["mcp-git"] }
               } }"#,
        )
        .unwrap();
        assert_eq!(cfg.servers.len(), 2);
        let fs = &cfg.servers.iter().find(|(n, _)| n == "fs").unwrap().1;
        assert_eq!(fs.command, "npx");
        assert_eq!(fs.args, vec!["-y", "server-fs"]);
        assert_eq!(fs.env, vec![("ROOT".to_string(), "/tmp".to_string())]);
    }

    #[test]
    fn config_skips_entry_without_command_and_handles_missing_key() {
        let cfg = McpConfig::parse(r#"{ "mcpServers": { "bad": { "args": ["x"] } } }"#).unwrap();
        assert!(cfg.is_empty());
        assert!(McpConfig::parse("{}").unwrap().is_empty());
    }

    #[test]
    fn config_rejects_malformed_json() {
        assert!(McpConfig::parse("{not json").is_err());
    }

    #[test]
    fn load_missing_file_is_empty() {
        let cfg = McpConfig::load(std::path::Path::new("/no/such/zero-mcp.json")).unwrap();
        assert!(cfg.is_empty());
    }

    #[test]
    fn load_reads_a_real_file() {
        let path = std::env::temp_dir().join(format!("zero-mcp-load-{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{"mcpServers":{"a":{"command":"true","args":["x"]}}}"#,
        )
        .unwrap();
        let cfg = McpConfig::load(&path).unwrap();
        assert_eq!(cfg.servers.len(), 1);
        assert_eq!(cfg.servers[0].0, "a");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_surfaces_malformed_file() {
        let path = std::env::temp_dir().join(format!("zero-mcp-bad-{}.json", std::process::id()));
        std::fs::write(&path, "{not json").unwrap();
        assert!(McpConfig::load(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn spec_without_args_or_env_defaults_to_empty() {
        let cfg = McpConfig::parse(r#"{"mcpServers":{"x":{"command":"c"}}}"#).unwrap();
        assert_eq!(cfg.servers[0].1.args, Vec::<String>::new());
        assert!(cfg.servers[0].1.env.is_empty());
    }

    #[test]
    fn session_notify_is_a_oneway_message() {
        let mut written = Vec::new();
        let mut s = Session::new(Cursor::new(Vec::new()), &mut written);
        s.notify("notifications/ping", Value::Object(vec![]))
            .unwrap();
        let sent = String::from_utf8(written).unwrap();
        assert!(sent.contains("notifications/ping"));
        assert!(!sent.contains("\"id\"")); // notifications carry no id
    }

    /// Drive a [`Session`] against canned response lines; assert the bytes it
    /// writes and the tools it parses.
    #[test]
    fn session_handshake_and_tools_list() {
        let server_out = concat!(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"capabilities\":{}}}\n",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{}}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[",
            "{\"name\":\"read_file\",\"description\":\"Read a file\"},",
            "{\"name\":\"write_file\"}]}}\n",
        );
        let mut written = Vec::new();
        let mut s = Session::new(Cursor::new(server_out.as_bytes().to_vec()), &mut written);
        s.initialize().unwrap();
        let tools = s.list_tools().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[0].description, "Read a file");
        assert_eq!(tools[1].description, ""); // missing description → empty

        let sent = String::from_utf8(written).unwrap();
        assert!(sent.contains("\"method\":\"initialize\""));
        assert!(sent.contains("notifications/initialized"));
        assert!(sent.contains("\"method\":\"tools/list\""));
    }

    #[test]
    fn session_call_tool_extracts_text_and_flags_errors() {
        // Text + a non-text block: text concatenated, non-text noted.
        let out = "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"content\":[\
             {\"type\":\"text\",\"text\":\"hello \"},\
             {\"type\":\"text\",\"text\":\"world\"},\
             {\"type\":\"image\",\"data\":\"…\"}]}}\n";
        let mut written = Vec::new();
        let mut s = Session::new(Cursor::new(out.as_bytes().to_vec()), &mut written);
        let r = s
            .call_tool(
                "echo",
                Value::Object(vec![("msg".to_string(), Value::Str("hi".to_string()))]),
            )
            .unwrap();
        assert_eq!(r, "hello world[image content]");
        let sent = String::from_utf8(written).unwrap();
        assert!(sent.contains("\"method\":\"tools/call\""));
        assert!(sent.contains("\"name\":\"echo\""));
        assert!(sent.contains("\"msg\":\"hi\""));

        // isError → surfaced as an `error:` string, not a hard failure.
        let err = "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"isError\":true,\
             \"content\":[{\"type\":\"text\",\"text\":\"boom\"}]}}\n";
        let mut w2 = Vec::new();
        let mut s2 = Session::new(Cursor::new(err.as_bytes().to_vec()), &mut w2);
        assert_eq!(
            s2.call_tool("x", Value::Object(vec![])).unwrap(),
            "error: boom"
        );
    }

    #[test]
    fn list_tools_captures_input_schema_with_default() {
        let out = "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[\
             {\"name\":\"add\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"number\"}}}},\
             {\"name\":\"bare\"}]}}\n";
        let mut written = Vec::new();
        let mut s = Session::new(Cursor::new(out.as_bytes().to_vec()), &mut written);
        let tools = s.list_tools().unwrap();
        assert!(tools[0].input_schema.get("properties").is_some());
        // Missing inputSchema → an open object schema so the tool is still callable.
        assert_eq!(
            tools[1].input_schema.get("type").and_then(Value::as_str),
            Some("object")
        );
    }

    #[test]
    fn mcp_tool_name_namespaces_and_sanitizes() {
        assert_eq!(mcp_tool_name("fs", "read_file"), "fs__read_file");
        // Non-identifier chars (space, slash, dot) collapse to underscores.
        assert_eq!(
            mcp_tool_name("my server", "do/it.now"),
            "my_server__do_it_now"
        );
    }

    #[test]
    fn connect_call_tool_and_routes_end_to_end() {
        // A tiny stdio MCP server in sh: initialize (id1), tools/list with a schema
        // (id2), then tools/call (id3). Exercises the real pipe path for call_tool
        // plus tool_routes / input_schema — no mocks.
        let script = "read a; \
             printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"capabilities\":{}}}'; \
             read b; read c; \
             printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"echoes\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"msg\":{\"type\":\"string\"}}}}]}}'; \
             read d; \
             printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"echoed: hi\"}]}}'";
        let spec = ServerSpec {
            command: "sh".to_string(),
            args: vec!["-c".to_string(), script.to_string()],
            env: Vec::new(),
        };
        let mut conn = Connection::connect("mock", &spec).unwrap();
        assert!(conn.has_tool("echo"));
        assert!(conn.tools[0].input_schema.get("properties").is_some());

        // Routes: namespaced name, schema carried through.
        let routes = tool_routes(std::slice::from_ref(&conn));
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].def.name, "mock__echo");
        assert_eq!(routes[0].raw_name, "echo");
        assert!(routes[0].def.description.contains("mcp:mock"));

        // The actual call over real pipes.
        let res = conn
            .call_tool(
                "echo",
                Value::Object(vec![("msg".to_string(), Value::Str("hi".to_string()))]),
            )
            .unwrap();
        assert_eq!(res, "echoed: hi");
    }

    #[test]
    fn session_skips_non_json_and_mismatched_id_lines() {
        // A log banner, a blank line, and a notification (no/other id) all
        // precede the real response — the request loop must skip them all.
        let out = concat!(
            "Listening on stdio…\n",
            "\n",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{}}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":99,\"result\":{\"stale\":true}}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}\n",
        );
        let mut written = Vec::new();
        let mut s = Session::new(Cursor::new(out.as_bytes().to_vec()), &mut written);
        let res = s.request("ping", Value::Object(vec![])).unwrap();
        assert_eq!(res.get("ok").and_then(Value::as_bool), Some(true));
    }

    #[test]
    fn session_surfaces_a_jsonrpc_error() {
        let out = "{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"message\":\"nope\"}}\n";
        let mut written = Vec::new();
        let mut s = Session::new(Cursor::new(out.as_bytes().to_vec()), &mut written);
        let err = s.request("initialize", Value::Object(vec![])).unwrap_err();
        assert!(err.to_string().contains("nope"));
    }

    #[test]
    fn session_eof_before_response_errors() {
        let mut written = Vec::new();
        let mut s = Session::new(Cursor::new(Vec::new()), &mut written);
        let err = s.request("ping", Value::Object(vec![])).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn connect_spawns_a_real_stdio_server_and_lists_tools() {
        // A tiny MCP server in sh: answer initialize (id 1), swallow the
        // initialized notice, then answer tools/list (id 2).
        let script = "read a; \
             printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"capabilities\":{}}}'; \
             read b; read c; \
             printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"echoes\"}]}}'";
        let spec = ServerSpec {
            command: "sh".to_string(),
            args: vec!["-c".to_string(), script.to_string()],
            env: Vec::new(),
        };
        let conn = Connection::connect("mock", &spec).unwrap();
        assert_eq!(conn.name, "mock");
        assert_eq!(conn.tools.len(), 1);
        assert_eq!(conn.tools[0].name, "echo");
        // Dropping kills the child (no assertion — just exercises Drop).
    }

    #[test]
    fn connect_reports_a_missing_command() {
        let spec = ServerSpec {
            command: "zero-no-such-binary-xyz".to_string(),
            args: Vec::new(),
            env: Vec::new(),
        };
        assert!(Connection::connect("x", &spec).is_err());
    }

    // --- fuzz: untrusted server stdout + config (std-only) ---------------

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn fuzz_session_request_never_panics_on_garbage_stdout() {
        use std::io::Cursor;
        // A misbehaving/hostile MCP server can emit anything on stdout. The
        // request loop must never panic — it returns Ok (a matching response) or
        // Err (EOF), skipping all non-JSON / mismatched-id lines in between.
        let mut rng = Rng(0xC0DE_F00D_1234_5678);
        const LINES: &[&str] = &[
            "garbage banner",
            "",
            "\x1b[2mANSI log\x1b[0m",
            "{not json",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/x\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":99,\"result\":{}}",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"message\":\"boom\"}}",
            "中😀\\\"",
        ];
        for _ in 0..5000 {
            let n = rng.below(6);
            let mut out = String::new();
            for _ in 0..n {
                out.push_str(LINES[rng.below(LINES.len() as u64) as usize]);
                out.push('\n');
            }
            let mut written = Vec::new();
            let mut s = Session::new(Cursor::new(out.into_bytes()), &mut written);
            let _ = s.request("probe", Value::Object(vec![])); // Ok or Err, never panic
        }
    }

    #[test]
    fn fuzz_mcp_config_parse_never_panics() {
        // ~/.zero/mcp.json is user/tool-authored — parse must not panic.
        let mut rng = Rng(0x4242_8888_0000_FFFF);
        const FRAG: &[&str] = &[
            "{",
            "}",
            "[",
            "]",
            "\"mcpServers\"",
            ":",
            ",",
            "\"command\"",
            "\"args\"",
            "\"env\"",
            "\"npx\"",
            "null",
            "true",
            "\\",
            "中",
        ];
        for _ in 0..10_000 {
            let n = rng.below(14);
            let text: String = (0..n)
                .map(|_| FRAG[rng.below(FRAG.len() as u64) as usize])
                .collect();
            let _ = McpConfig::parse(&text); // Ok or Err, never panic
        }
    }

    // --- discovery / multi-source import ---------------------------------

    fn write(dir: &std::path::Path, rel: &str, text: &str) -> std::path::PathBuf {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, text).unwrap();
        p
    }

    fn srv(name: &str, cmd: &str) -> String {
        format!(r#""{name}":{{"command":"{cmd}","args":[]}}"#)
    }

    #[test]
    fn default_sources_lists_known_locations_in_precedence_order() {
        let home = std::path::Path::new("/home/u");
        let cwd = std::path::Path::new("/work/proj");
        let s = default_sources(home, cwd, ".zero");
        let kinds: Vec<Source> = s.iter().map(|(k, _)| *k).collect();
        assert_eq!(
            kinds,
            vec![
                Source::Project,
                Source::Zero,
                Source::ClaudeDesktop,
                Source::ClaudeCode
            ]
        );
        assert_eq!(s[0].1, cwd.join(".mcp.json"));
        assert_eq!(s[1].1, home.join(".zero/mcp.json"));
    }

    #[test]
    fn discover_merges_sources_with_first_seen_name_winning() {
        let dir =
            std::env::temp_dir().join(format!("zero-disc-{}-{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        // Project and Zero both define `shared`; project (higher precedence) wins.
        let proj = write(
            &dir,
            "proj/.mcp.json",
            &format!(
                r#"{{"mcpServers":{{{},{}}}}}"#,
                srv("shared", "from_proj"),
                srv("only_proj", "p")
            ),
        );
        let zero = write(
            &dir,
            "zero/mcp.json",
            &format!(
                r#"{{"mcpServers":{{{},{}}}}}"#,
                srv("shared", "from_zero"),
                srv("only_zero", "z")
            ),
        );
        let cands = vec![(Source::Project, proj), (Source::Zero, zero)];
        let d = discover(&cands, &dir);
        let names: Vec<&str> = d.servers.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"shared")
                && names.contains(&"only_proj")
                && names.contains(&"only_zero")
        );
        let shared = d.servers.iter().find(|s| s.name == "shared").unwrap();
        assert_eq!(shared.spec.command, "from_proj"); // project won
        assert_eq!(shared.source, Source::Project);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discover_reads_claude_code_global_and_per_project() {
        let dir = std::env::temp_dir().join(format!("zero-cc-{}-{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = dir.join("myproj");
        // ~/.claude.json: a global server + a per-project server keyed by cwd.
        let claude = write(
            &dir,
            ".claude.json",
            &format!(
                r#"{{"mcpServers":{{{}}},"projects":{{"{}":{{"mcpServers":{{{}}}}}}}}}"#,
                srv("global_srv", "g"),
                cwd.display(),
                srv("proj_srv", "p"),
            ),
        );
        let d = discover(&[(Source::ClaudeCode, claude)], &cwd);
        let names: Vec<&str> = d.servers.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"global_srv"), "global: {names:?}");
        assert!(names.contains(&"proj_srv"), "per-project: {names:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discover_skips_missing_files_and_surfaces_parse_errors() {
        let dir = std::env::temp_dir().join(format!("zero-err-{}-{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = write(&dir, "bad.json", "{not json");
        let missing = dir.join("nope.json");
        let d = discover(&[(Source::Zero, bad), (Source::Project, missing)], &dir);
        assert!(d.servers.is_empty());
        assert_eq!(d.errors.len(), 1); // only the present-but-broken file
        assert_eq!(d.errors[0].0, Source::Zero);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discover_skips_non_stdio_http_servers() {
        let dir =
            std::env::temp_dir().join(format!("zero-http-{}-{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        // An HTTP server has a `url`, no `command` — Zero is stdio-only, skip it.
        let cfg = write(
            &dir,
            "mcp.json",
            r#"{"mcpServers":{"remote":{"type":"http","url":"https://x"},"local":{"command":"sh","args":[]}}}"#,
        );
        let d = discover(&[(Source::Zero, cfg)], &dir);
        let names: Vec<&str> = d.servers.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["local"]); // remote (no command) dropped
        std::fs::remove_dir_all(&dir).ok();
    }
}

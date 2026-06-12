// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! LAN discovery of OpenAI-compatible model servers.
//!
//! Scans the local `/24` subnet on a handful of common LLM-server ports, GETs
//! `/v1/models` on anything that answers, and reports what models it found. No
//! dependencies: raw `TcpStream` probes parallelized with `std::thread`.
//!
//! The model-list parsing and candidate generation are pure and unit-tested;
//! the socket scan is covered by pointing it at an in-process localhost server.

use crate::http;
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Ports commonly used by local LLM servers (llama.cpp, vLLM, Ollama, LM Studio,
/// text-generation-webui, …).
pub const DEFAULT_PORTS: &[u16] = &[8000, 8080, 11434, 1234, 5000, 8001, 1337, 8888, 80];

/// A reachable OpenAI-compatible server and the models it advertises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Discovered {
    pub base_url: String,
    pub models: Vec<String>,
}

/// Best-effort local IPv4 address, found by asking the OS which interface it
/// would use to reach a public address (no packets are actually sent).
pub fn local_ipv4() -> Option<[u8; 4]> {
    let sock = UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    match sock.local_addr().ok()?.ip() {
        IpAddr::V4(v4) => Some(v4.octets()),
        IpAddr::V6(_) => None,
    }
}

/// Loopback candidates — servers bound to `127.0.0.1` on this very device
/// (LM Studio, Ollama, llama.cpp run locally). Easy to miss otherwise.
pub fn loopback_candidates(ports: &[u16]) -> Vec<SocketAddr> {
    ports
        .iter()
        .map(|&p| SocketAddr::from(([127, 0, 0, 1], p)))
        .collect()
}

/// Build candidate `host:port` addresses across the local `/24` and `ports`.
pub fn local_candidates(ports: &[u16]) -> Vec<SocketAddr> {
    let Some([a, b, c, _]) = local_ipv4() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(254 * ports.len());
    for host in 1u8..=254 {
        for &port in ports {
            out.push(SocketAddr::from(([a, b, c, host], port)));
        }
    }
    out
}

/// Parse a `/v1/models` (OpenAI) or `/api/tags` (Ollama) response into model
/// names. Pure and tolerant of either shape.
pub fn parse_models(json: &str) -> Vec<String> {
    let Ok(v) = crate::json::Value::parse(json) else {
        return Vec::new();
    };
    let mut names = Vec::new();
    // OpenAI: { "data": [ { "id": "..." } ] }
    if let Some(arr) = v.get("data").and_then(crate::json::Value::as_array) {
        for item in arr {
            if let Some(id) = item.get("id").and_then(crate::json::Value::as_str) {
                names.push(id.to_string());
            }
        }
    }
    // Ollama: { "models": [ { "name": "..." } ] }
    if let Some(arr) = v.get("models").and_then(crate::json::Value::as_array) {
        for item in arr {
            if let Some(n) = item.get("name").and_then(crate::json::Value::as_str) {
                names.push(n.to_string());
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

/// Probe one base URL: GET `/v1/models` and parse it. `None` if it isn't a
/// reachable OpenAI-compatible server.
pub fn probe(base_url: &str, timeout: Duration) -> Option<Discovered> {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let (code, body) = http::get(&url, timeout).ok()?;
    if !(200..300).contains(&code) {
        return None;
    }
    let models = parse_models(&body);
    Some(Discovered {
        base_url: base_url.trim_end_matches('/').to_string(),
        models,
    })
}

/// Probe every candidate concurrently (bounded), returning the servers that
/// answered. `concurrency` caps simultaneous connections.
pub fn scan_candidates(
    candidates: &[SocketAddr],
    timeout: Duration,
    concurrency: usize,
) -> Vec<Discovered> {
    let concurrency = concurrency.max(1);
    let mut found = Vec::new();
    for batch in candidates.chunks(concurrency) {
        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();
        for &addr in batch {
            let tx = tx.clone();
            handles.push(thread::spawn(move || {
                // probe() connects with its own timeout, so closed ports reject
                // fast here too — no separate connect check needed.
                if let Some(d) = probe(&format!("http://{addr}"), timeout) {
                    let _ = tx.send(d);
                }
            }));
        }
        drop(tx);
        for d in rx {
            found.push(d);
        }
        for h in handles {
            let _ = h.join();
        }
    }
    found.sort_by(|a, b| a.base_url.cmp(&b.base_url));
    found
}

/// Scan this device (loopback) and the local subnet on the default ports.
pub fn scan(timeout: Duration) -> Vec<Discovered> {
    let mut candidates = loopback_candidates(DEFAULT_PORTS);
    candidates.extend(local_candidates(DEFAULT_PORTS));
    scan_candidates(&candidates, timeout, 64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn parse_openai_models() {
        let json = r#"{"data":[{"id":"qwen"},{"id":"llama"}],"object":"list"}"#;
        assert_eq!(parse_models(json), vec!["llama", "qwen"]);
    }

    #[test]
    fn parse_ollama_models() {
        let json = r#"{"models":[{"name":"mistral"}]}"#;
        assert_eq!(parse_models(json), vec!["mistral"]);
    }

    #[test]
    fn parse_models_tolerates_garbage_and_empties() {
        assert!(parse_models("not json").is_empty());
        assert!(parse_models("{}").is_empty());
        assert!(parse_models(r#"{"data":[]}"#).is_empty());
    }

    #[test]
    fn loopback_candidates_one_per_port() {
        let c = loopback_candidates(&[8000, 11434]);
        assert_eq!(c.len(), 2);
        assert!(c.iter().all(|a| a.ip().is_loopback()));
    }

    #[test]
    fn local_candidates_cover_the_subnet_when_ip_known() {
        let c = local_candidates(&[8000]);
        // Either we have a LAN IP (254 hosts) or none (CI with no route).
        assert!(c.len() == 254 || c.is_empty());
    }

    /// Spawn a localhost server that answers `/v1/models` once.
    fn serve_models(body: &'static str) -> (String, SocketAddr) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body = body.to_string();
        thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            sock.set_read_timeout(Some(Duration::from_millis(150))).ok();
            let mut buf = [0u8; 1024];
            while let Ok(n) = sock.read(&mut buf) {
                if n == 0 {
                    break;
                }
            }
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes());
        });
        (format!("http://{addr}"), addr)
    }

    #[test]
    fn probe_finds_a_real_server() {
        let (url, _) = serve_models(r#"{"data":[{"id":"qwen"}]}"#);
        let d = probe(&url, Duration::from_secs(2)).unwrap();
        assert_eq!(d.models, vec!["qwen"]);
        assert_eq!(d.base_url, url);
    }

    #[test]
    fn probe_returns_none_on_refused() {
        assert!(probe("http://127.0.0.1:1", Duration::from_millis(200)).is_none());
    }

    #[test]
    fn scan_candidates_finds_listening_server_ignores_closed() {
        let (_, addr) = serve_models(r#"{"data":[{"id":"m"}]}"#);
        let closed = SocketAddr::from(([127, 0, 0, 1], 1));
        let found = scan_candidates(&[addr, closed], Duration::from_secs(1), 8);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].models, vec!["m"]);
    }

    #[test]
    fn scan_candidates_empty_input_is_empty() {
        assert!(scan_candidates(&[], Duration::from_millis(100), 4).is_empty());
    }
}

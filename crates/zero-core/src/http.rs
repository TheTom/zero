// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! A tiny, std-only HTTP/1.1 client — just enough to POST a JSON body and read
//! a streaming (chunked) Server-Sent-Events response. No TLS: Zero is local-first
//! and talks to a model server on the LAN over plain `http://`.
//!
//! The URL parsing and chunk/line framing are pure and unit-tested; the socket
//! round-trip is covered by an in-process localhost mock server in the tests.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// A parsed `http://host[:port]/path` URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    pub host: String,
    pub port: u16,
    pub path: String,
}

/// Parse a plain-`http` URL. Returns an error for `https` or malformed input.
pub fn parse_url(url: &str) -> Result<Url, String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("only http:// URLs are supported (got {url:?})"))?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    if authority.is_empty() {
        return Err("missing host".to_string());
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h, p.parse::<u16>().map_err(|_| format!("bad port: {p:?}"))?),
        None => (authority, 80),
    };
    Ok(Url {
        host: host.to_string(),
        port,
        path: path.to_string(),
    })
}

/// Max connect attempts before giving up — a local model server that's still
/// loading (or just restarted) gets a couple of retries before the error surfaces.
const CONNECT_ATTEMPTS: u32 = 3;
/// Per-attempt connect timeout. Short on purpose: a refused connection returns
/// instantly, and a genuinely unreachable host shouldn't block for the (long) read
/// timeout × every attempt.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Connection-level failures worth retrying — the server may be starting up or
/// momentarily unavailable. (HTTP error *responses* and mid-stream read failures
/// are NOT retried; only the initial connect, before any bytes are exchanged.)
fn is_retryable_connect(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::TimedOut
    )
}

/// Resolve `host:port` and connect with a bounded timeout, retrying connection-
/// level errors up to [`CONNECT_ATTEMPTS`] times with a short increasing backoff.
/// Returns before any request is sent, so retrying is always safe (no duplicated
/// output). The final error is the last connect failure.
///
/// A hostname can resolve to several addresses (IPv4 + IPv6). We try **all** of
/// them each round, **IPv4 first** — an mDNS name like `train.local` often resolves
/// to a link-local IPv6 first, but a local model server typically binds IPv4 only,
/// so connecting to that first address would hang until timeout. (curl dodges this
/// with Happy-Eyeballs; the std-only client has to be explicit.)
fn connect(host: &str, port: u16) -> io::Result<TcpStream> {
    let mut addrs: Vec<std::net::SocketAddr> = (host, port).to_socket_addrs()?.collect();
    if addrs.is_empty() {
        return Err(io::Error::other("could not resolve host"));
    }
    // IPv4 (false → 0) sorts before IPv6 (true → 1); stable so order within a
    // family is preserved.
    addrs.sort_by_key(|a| a.is_ipv6());
    let mut last: Option<io::Error> = None;
    for attempt in 0..CONNECT_ATTEMPTS {
        for addr in &addrs {
            match TcpStream::connect_timeout(addr, CONNECT_TIMEOUT) {
                Ok(s) => return Ok(s),
                Err(e) => last = Some(e),
            }
        }
        // Every address failed this round. Retry (with backoff) only if the last
        // failure looks transient — a server still coming up.
        match &last {
            Some(e) if attempt + 1 < CONNECT_ATTEMPTS && is_retryable_connect(e) => {
                // 500ms, then 1000ms — give a loading server a moment to come up.
                std::thread::sleep(Duration::from_millis(500 * (attempt as u64 + 1)));
            }
            _ => break,
        }
    }
    Err(last.unwrap_or_else(|| io::Error::other("connect failed")))
}

/// POST `body` to `url` and invoke `on_line` for each line of the response body
/// (CRLF stripped), de-chunking a `Transfer-Encoding: chunked` stream on the fly.
/// `headers` are extra request headers (e.g. Authorization).
pub fn post_stream(
    url: &str,
    headers: &[(String, String)],
    body: &str,
    on_line: &mut dyn FnMut(&str),
) -> io::Result<()> {
    let u = parse_url(url).map_err(io::Error::other)?;
    let mut stream = connect(&u.host, u.port)?;

    let mut req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n",
        u.path,
        u.host,
        body.len()
    );
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    stream.write_all(req.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let (code, chunked) = read_status_and_headers(&mut reader)?;

    if !(200..300).contains(&code) {
        let mut err = String::new();
        let _ = reader.read_to_string(&mut err);
        return Err(io::Error::other(format!("HTTP {code}: {}", err.trim())));
    }

    let mut linebuf: Vec<u8> = Vec::new();
    if chunked {
        read_chunked(&mut reader, &mut linebuf, on_line)?;
    } else {
        let mut rest = Vec::new();
        reader.read_to_end(&mut rest)?;
        feed_lines(&rest, &mut linebuf, on_line);
    }
    // Flush a trailing line that had no terminator.
    if !linebuf.is_empty() {
        on_line(&String::from_utf8_lossy(&linebuf));
    }
    Ok(())
}

/// POST `body` to `url` and read the full response (with connect + read
/// timeouts), returning `(status_code, body)`. Used for non-streamed completions
/// — the agentic tool loop disables streaming to dodge the local-server
/// streaming-tool-call parser bugs, so it reads the whole JSON in one shot.
pub fn post(
    url: &str,
    headers: &[(String, String)],
    body: &str,
    timeout: Duration,
) -> io::Result<(u16, String)> {
    let u = parse_url(url).map_err(io::Error::other)?;
    let stream = connect(&u.host, u.port)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let mut stream = stream;

    let mut req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n",
        u.path,
        u.host,
        body.len()
    );
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    stream.write_all(req.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let (code, chunked) = read_status_and_headers(&mut reader)?;
    let mut buf = Vec::new();
    if chunked {
        read_chunked_raw(&mut reader, &mut buf)?;
    } else {
        reader.read_to_end(&mut buf)?;
    }
    Ok((code, String::from_utf8_lossy(&buf).into_owned()))
}

/// GET `url` (with connect + read timeouts) and return `(status_code, body)`.
/// Used for short, non-streamed responses like `/v1/models` during discovery.
pub fn get(url: &str, timeout: Duration) -> io::Result<(u16, String)> {
    let u = parse_url(url).map_err(io::Error::other)?;
    // Share the connect() helper so GET gets the same all-addresses, IPv4-first
    // behavior as POST (an mDNS host that resolves to IPv6 first must not stall).
    let stream = connect(&u.host, u.port)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let mut stream = stream;

    let req = format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        u.path, u.host
    );
    stream.write_all(req.as_bytes())?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let (code, chunked) = read_status_and_headers(&mut reader)?;

    let mut body = Vec::new();
    if chunked {
        read_chunked_raw(&mut reader, &mut body)?;
    } else {
        reader.read_to_end(&mut body)?;
    }
    Ok((code, String::from_utf8_lossy(&body).into_owned()))
}

/// Read the HTTP status line and headers; return `(status_code, is_chunked)`.
fn read_status_and_headers<R: BufRead>(reader: &mut R) -> io::Result<(u16, bool)> {
    let mut status = String::new();
    reader.read_line(&mut status)?;
    let code: u16 = status
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let mut chunked = false;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
            chunked = true;
        }
    }
    Ok((code, chunked))
}

/// Decode a chunked body into `out` as raw bytes (for non-streamed GETs).
fn read_chunked_raw<R: BufRead>(reader: &mut R, out: &mut Vec<u8>) -> io::Result<()> {
    loop {
        let mut size_line = String::new();
        if reader.read_line(&mut size_line)? == 0 {
            break;
        }
        let hex = size_line.trim().split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(hex, 16).unwrap_or(0);
        if size == 0 {
            break;
        }
        let mut chunk = vec![0u8; size];
        reader.read_exact(&mut chunk)?;
        out.extend_from_slice(&chunk);
        let mut crlf = [0u8; 2];
        let _ = reader.read_exact(&mut crlf);
    }
    Ok(())
}

/// Decode a chunked body, feeding decoded bytes through the line splitter.
fn read_chunked<R: BufRead>(
    reader: &mut R,
    linebuf: &mut Vec<u8>,
    on_line: &mut dyn FnMut(&str),
) -> io::Result<()> {
    loop {
        let mut size_line = String::new();
        if reader.read_line(&mut size_line)? == 0 {
            break;
        }
        let hex = size_line.trim().split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(hex, 16).unwrap_or(0);
        if size == 0 {
            break; // last chunk
        }
        let mut chunk = vec![0u8; size];
        reader.read_exact(&mut chunk)?;
        feed_lines(&chunk, linebuf, on_line);
        let mut crlf = [0u8; 2]; // trailing CRLF after the chunk data
        let _ = reader.read_exact(&mut crlf);
    }
    Ok(())
}

/// Append `bytes` to `buf`, emitting every complete `\n`-terminated line via
/// `on_line` (with trailing `\r`/`\n` stripped). Keeps any partial line in `buf`.
fn feed_lines(bytes: &[u8], buf: &mut Vec<u8>, on_line: &mut dyn FnMut(&str)) {
    buf.extend_from_slice(bytes);
    while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
        let line: Vec<u8> = buf.drain(..=pos).collect();
        let s = String::from_utf8_lossy(&line);
        on_line(s.trim_end_matches(['\r', '\n']));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn parses_urls() {
        assert_eq!(
            parse_url("http://gx10:8000/v1/chat").unwrap(),
            Url {
                host: "gx10".to_string(),
                port: 8000,
                path: "/v1/chat".to_string()
            }
        );
        // Default port and path.
        let u = parse_url("http://host").unwrap();
        assert_eq!(u.port, 80);
        assert_eq!(u.path, "/");
    }

    #[test]
    fn rejects_https_and_garbage() {
        assert!(parse_url("https://x").is_err());
        assert!(parse_url("ftp://x").is_err());
        assert!(parse_url("http://host:notaport/").is_err());
        assert!(parse_url("http:///path").is_err());
    }

    #[test]
    fn feed_lines_splits_and_keeps_partial() {
        let mut buf = Vec::new();
        let mut lines = Vec::new();
        feed_lines(b"a\r\nb\n", &mut buf, &mut |l| lines.push(l.to_string()));
        assert_eq!(lines, vec!["a", "b"]);
        assert!(buf.is_empty());
        feed_lines(b"par", &mut buf, &mut |l| lines.push(l.to_string()));
        assert_eq!(lines.len(), 2); // "par" stays buffered
        feed_lines(b"tial\n", &mut buf, &mut |l| lines.push(l.to_string()));
        assert_eq!(lines[2], "partial");
    }

    /// Spawn a one-shot localhost server that returns `response` verbatim.
    fn serve_once(response: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            drain_request(&mut sock);
            let _ = sock.write_all(&response);
        });
        format!("http://127.0.0.1:{}/v1/chat/completions", addr.port())
    }

    /// Read the whole request off the socket (until a short idle timeout) so the
    /// server never closes with unread bytes — which would RST the client.
    fn drain_request(sock: &mut std::net::TcpStream) {
        sock.set_read_timeout(Some(std::time::Duration::from_millis(150)))
            .ok();
        let mut tmp = [0u8; 1024];
        loop {
            match sock.read(&mut tmp) {
                Ok(0) => break,
                Ok(_) => continue,
                Err(_) => break, // WouldBlock (idle) or closed
            }
        }
    }

    /// Build a chunked HTTP body from byte parts (sizes computed for us).
    fn chunked(parts: &[&[u8]]) -> Vec<u8> {
        let mut out = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        for p in parts {
            out.extend_from_slice(format!("{:x}\r\n", p.len()).as_bytes());
            out.extend_from_slice(p);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(b"0\r\n\r\n");
        out
    }

    #[test]
    fn post_stream_reads_chunked_sse_body() {
        // SSE lines deliberately split across chunk boundaries.
        let url = serve_once(chunked(&[
            b"data: {\"n\":1}\n\ndata: {\"n\"",
            b":2}\n\ndone",
        ]));
        let mut lines = Vec::new();
        post_stream(&url, &[], "{}", &mut |l| {
            if !l.is_empty() {
                lines.push(l.to_string());
            }
        })
        .unwrap();
        assert_eq!(lines, vec!["data: {\"n\":1}", "data: {\"n\":2}", "done"]);
    }

    #[test]
    fn post_stream_reads_unchunked_body_to_eof() {
        let resp = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n\
            data: hello\ndata: world\n"
            .to_vec();
        let url = serve_once(resp);
        let mut lines = Vec::new();
        post_stream(&url, &[], "{}", &mut |l| lines.push(l.to_string())).unwrap();
        assert!(lines.contains(&"data: hello".to_string()));
        assert!(lines.contains(&"data: world".to_string()));
    }

    #[test]
    fn post_stream_surfaces_http_errors() {
        let resp = b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nno model!".to_vec();
        let url = serve_once(resp);
        let err = post_stream(&url, &[], "{}", &mut |_| {}).unwrap_err();
        assert!(err.to_string().contains("404"));
    }

    #[test]
    fn post_stream_sends_extra_headers() {
        let resp = b"HTTP/1.1 200 OK\r\n\r\ndata: ok\n".to_vec();
        let url = serve_once(resp);
        let headers = vec![("Authorization".to_string(), "Bearer t".to_string())];
        let mut lines = Vec::new();
        post_stream(&url, &headers, "{}", &mut |l| lines.push(l.to_string())).unwrap();
        assert!(lines.contains(&"data: ok".to_string()));
    }

    #[test]
    fn truncated_headers_end_cleanly() {
        // Server sends only a status line then closes (EOF mid-headers).
        let url = serve_once(b"HTTP/1.1 200 OK\r\n".to_vec());
        let mut lines = Vec::new();
        post_stream(&url, &[], "{}", &mut |l| lines.push(l.to_string())).unwrap();
        assert!(lines.is_empty());
    }

    #[test]
    fn chunked_body_that_ends_immediately() {
        // Chunked, but EOF before any chunk-size line arrives.
        let url = serve_once(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec());
        let mut lines = Vec::new();
        post_stream(&url, &[], "{}", &mut |l| lines.push(l.to_string())).unwrap();
        assert!(lines.is_empty());
    }

    #[test]
    fn get_reads_status_and_body() {
        let url = serve_once(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"data\":[]}".to_vec(),
        );
        let (code, body) = get(&url, Duration::from_secs(2)).unwrap();
        assert_eq!(code, 200);
        assert!(body.contains("data"));
    }

    #[test]
    fn get_reads_chunked_body() {
        let url = serve_once(chunked(&[b"{\"mod", b"els\":[]}"]));
        let (code, body) = get(&url, Duration::from_secs(2)).unwrap();
        assert_eq!(code, 200);
        assert!(body.contains("models"));
    }

    #[test]
    fn post_reads_status_and_body() {
        let url = serve_once(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"ok\":true}".to_vec(),
        );
        let (code, body) = post(&url, &[], "{\"q\":1}", Duration::from_secs(2)).unwrap();
        assert_eq!(code, 200);
        assert!(body.contains("ok"));
    }

    #[test]
    fn post_reads_chunked_body() {
        let url = serve_once(chunked(&[b"{\"cho", b"ices\":[]}"]));
        let (code, body) = post(&url, &[], "{}", Duration::from_secs(2)).unwrap();
        assert_eq!(code, 200);
        assert!(body.contains("choices"));
    }

    #[test]
    fn post_on_refused_port_errors() {
        assert!(post(
            "http://127.0.0.1:1/x",
            &[],
            "{}",
            Duration::from_millis(200)
        )
        .is_err());
    }

    #[test]
    fn get_on_refused_port_errors() {
        assert!(get("http://127.0.0.1:1/v1/models", Duration::from_millis(200)).is_err());
    }

    #[test]
    fn connect_reaches_ipv4_for_a_multi_address_host() {
        // `localhost` resolves to 127.0.0.1 and, on dual-stack hosts, ::1 too. The
        // server binds IPv4 only, so connect() must sort IPv4 first and reach it
        // rather than stalling on an unreachable IPv6 address — the regression that
        // made `train.local` (link-local IPv6 first) time out on every request.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            drain_request(&mut sock);
            let _ = sock.write_all(b"HTTP/1.1 200 OK\r\n\r\n{\"ok\":true}");
        });
        let url = format!("http://localhost:{port}/v1/models");
        let (code, body) = get(&url, Duration::from_secs(2)).unwrap();
        assert_eq!(code, 200);
        assert!(body.contains("ok"));
    }

    #[test]
    fn connection_refused_is_an_error() {
        // Nothing listening on this port.
        let err = post_stream("http://127.0.0.1:1/x", &[], "{}", &mut |_| {}).unwrap_err();
        assert!(err.kind() != io::ErrorKind::Other || !err.to_string().is_empty());
    }

    #[test]
    fn connect_retries_refused_before_giving_up() {
        use std::time::Instant;
        // Nothing listening → connection refused, retried CONNECT_ATTEMPTS times
        // with backoff (500ms + 1000ms). The elapsed time proves the retries ran;
        // the error still surfaces once they're exhausted.
        let t = Instant::now();
        let err = post("http://127.0.0.1:1/v1", &[], "{}", Duration::from_secs(5)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionRefused);
        assert!(
            t.elapsed() >= Duration::from_millis(1400),
            "connect did not retry (elapsed {:?})",
            t.elapsed()
        );
    }

    #[test]
    fn is_retryable_connect_classifies_kinds() {
        assert!(is_retryable_connect(&io::Error::from(
            io::ErrorKind::ConnectionRefused
        )));
        assert!(is_retryable_connect(&io::Error::from(
            io::ErrorKind::TimedOut
        )));
        // A protocol/HTTP-shaped error is not a connect retry.
        assert!(!is_retryable_connect(&io::Error::other("HTTP 500")));
    }
}

#!/usr/bin/env python3
"""Tiny pass-through proxy that records server-reported token usage.

Why this exists: the Zero-vs-Hermes bench needs *apples-to-apples* token counts
for the SAME gx10 model, but Hermes emits no usage line of its own. Both wrappers
speak the OpenAI-compatible HTTP contract, so we sit between them and the real
server, forward every request unchanged, and tee each response's `usage` object
(prompt_tokens / completion_tokens) to a log file. This is *measured*, not
estimated — the count comes straight from the upstream server, identical for
whichever wrapper made the call.

Handles both non-streaming (`usage` in the JSON body) and streaming (the final
`data: {... "usage": {...}}` SSE chunk that llama.cpp emits with
`stream_options.include_usage`). std-library only — no deps.

Usage:
    UPSTREAM=http://192.168.50.125:8000 TOKEN_LOG=/tmp/tok.log \\
        python3 scripts/token-proxy.py 8099
Then point a wrapper's base_url at http://127.0.0.1:8099 and read TOKEN_LOG:
each line is `<unix_ms> <prompt_tokens> <completion_tokens>` per upstream reply.
"""

import json
import os
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://192.168.50.125:8000").rstrip("/")
TOKEN_LOG = os.environ.get("TOKEN_LOG", "/tmp/zero-token-proxy.log")
# Optional: if set, each forwarded request body is dumped to DUMP_DIR/req-NNN.json
# so we can inspect what pre-context (system prompt, tool schemas) a wrapper sends.
DUMP_DIR = os.environ.get("DUMP_DIR", "")
_dump_n = [0]


def _log_usage(usage):
    """Append one measured usage record. Tolerant of missing fields."""
    if not isinstance(usage, dict):
        return
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)
    if pt == 0 and ct == 0:
        return
    with open(TOKEN_LOG, "a") as f:
        f.write(f"{int(time.time() * 1000)} {pt} {ct}\n")


def _extract_streaming_usage(body: bytes):
    """Scan SSE `data:` lines for the trailing usage chunk; return it or None."""
    found = None
    for line in body.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:"):].strip()
        if payload == b"[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("usage"):
            found = obj["usage"]  # last one wins — that's the cumulative total
    return found


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # silence access logging
        pass

    def _forward(self, method):
        length = int(self.headers.get("Content-Length", 0) or 0)
        req_body = self.rfile.read(length) if length else None
        if DUMP_DIR and req_body and self.path.endswith("/chat/completions"):
            _dump_n[0] += 1
            try:
                with open(os.path.join(DUMP_DIR, f"req-{_dump_n[0]:03d}.json"), "wb") as f:
                    f.write(req_body)
            except Exception:
                pass
        url = UPSTREAM + self.path
        req = urllib.request.Request(url, data=req_body, method=method)
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length"):
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = resp.read()
                status = resp.status
                headers = list(resp.getheaders())
        except urllib.error.HTTPError as e:
            body = e.read()
            status = e.code
            headers = list(e.headers.items())
        except Exception as e:
            self.send_error(502, f"upstream error: {e}")
            return

        # Tee usage — try a plain JSON body first, then the streaming form.
        try:
            obj = json.loads(body)
            if isinstance(obj, dict) and obj.get("usage"):
                _log_usage(obj["usage"])
            else:
                _log_usage(_extract_streaming_usage(body))
        except Exception:
            _log_usage(_extract_streaming_usage(body))

        self.send_response(status)
        for k, v in headers:
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self._forward("POST")

    def do_GET(self):
        self._forward("GET")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    # Truncate the log on startup so each bench run measures only its own calls.
    open(TOKEN_LOG, "w").close()
    print(f"token-proxy: 127.0.0.1:{port} → {UPSTREAM}  (usage → {TOKEN_LOG})", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", port), Proxy).serve_forever()

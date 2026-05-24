"""
server.py — Serve the RL dashboard and expose metrics as a JSON API.
Run:  python server.py
Then open:  http://localhost:8765
"""

import os
import json
import glob
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

METRICS_DIR = "metrics"
PORT = 8765


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Quiet server — only log errors
        if int(args[1]) >= 400:
            super().log_message(fmt, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")

        # ── /api/metrics?algo=DQN&env=CartPole_v1 ──
        if path == "/api/metrics":
            self._serve_metrics(parse_qs(parsed.query))

        # ── /api/list  — list all available metric files ──
        elif path == "/api/list":
            self._serve_list()

        # ── Static files ──
        elif path in ("", "/"):
            self._serve_file("dashboard.html", "text/html")
        elif path == "/dashboard.html":
            self._serve_file("dashboard.html", "text/html")
        else:
            # Try to serve any other static file
            fp = path.lstrip("/")
            if os.path.isfile(fp):
                mime = "text/plain"
                if fp.endswith(".js"):   mime = "application/javascript"
                elif fp.endswith(".css"): mime = "text/css"
                elif fp.endswith(".json"): mime = "application/json"
                self._serve_file(fp, mime)
            else:
                self._404()

    def _serve_metrics(self, qs):
        algo = qs.get("algo", [None])[0]
        env  = qs.get("env",  [None])[0]

        if not algo or not env:
            # Return all available data as a dict keyed by "ALGO_ENV"
            result = {}
            for fp in glob.glob(os.path.join(METRICS_DIR, "*.json")):
                key = os.path.splitext(os.path.basename(fp))[0]
                try:
                    with open(fp) as f:
                        result[key] = json.load(f)
                except Exception:
                    pass
            self._json(result)
        else:
            fname = os.path.join(METRICS_DIR, f"{algo}_{env}.json")
            if os.path.isfile(fname):
                try:
                    with open(fname) as f:
                        data = json.load(f)
                    self._json(data)
                except Exception as e:
                    self._json({"error": str(e)}, 500)
            else:
                self._json({"error": f"File not found: {fname}"}, 404)

    def _serve_list(self):
        files = []
        for fp in glob.glob(os.path.join(METRICS_DIR, "*.json")):
            files.append(os.path.splitext(os.path.basename(fp))[0])
        self._json({"files": files})

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filepath, mime):
        if not os.path.isfile(filepath):
            self._404()
            return
        with open(filepath, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _404(self):
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not found")


if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)
    server = HTTPServer(("localhost", PORT), Handler)
    print(f"\n  RL Dashboard server running!")
    print(f"  Open: http://localhost:{PORT}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")

"""Local stand-in for Vercel: serves public/ and the api/tessa.py function on one port.

    .venv/bin/python dev.py        # then open http://localhost:8000
"""

import importlib.util
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tessa_api", ROOT / "api" / "tessa.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class Dev(api.handler, SimpleHTTPRequestHandler):
    """One handler for both servers: /api/tessa goes to the function, every other path is
    a static file from public/."""
    def do_GET(self):
        """Route GET /api/tessa to the API handler, other paths to the static file server."""
        if self.path.startswith("/api/tessa"):
            return api.handler.do_GET(self)
        return SimpleHTTPRequestHandler.do_GET(self)

    def end_headers(self):
        """Ask browsers to revalidate every file, so edits show on the next reload
        (Vercel does the same for the deployed site)."""
        if not self.path.startswith("/api/tessa"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_POST(self):
        """Route POST /api/tessa to the API handler; any other POST gets 404."""
        if self.path.startswith("/api/tessa"):
            return api.handler.do_POST(self)
        self.send_error(404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Dev, directory=str(ROOT / "public")))
    print(f"Tessa Ripple web on http://localhost:{port}")
    server.serve_forever()

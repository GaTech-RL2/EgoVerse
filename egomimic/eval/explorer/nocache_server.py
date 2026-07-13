"""Static file server with caching DISABLED, for the chunkviz tunnel.

`python -m http.server` lets the browser disk-cache the 7 MB page, so it keeps
loading even after the SSH tunnel drops — making it impossible to tell whether
you're really hitting the server. This sends no-store/no-cache on every
response, so the browser revalidates each load and a dead tunnel fails at once.

    python nocache_server.py [port]   # default 8765, binds 0.0.0.0
"""
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    HTTPServer(("0.0.0.0", port), NoCacheHandler).serve_forever()

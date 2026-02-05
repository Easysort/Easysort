from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.parse import urlsplit
import json


@contextmanager
def minikeyvalue_server(host: str = "localhost", port: int = 3099) -> Iterator[str]:
  """Tiny minikeyvalue-compatible server backed by a temp folder."""
  tmp = TemporaryDirectory()
  try:
    root = Path(tmp.name)
    data, unlinked = root / "data", root / "unlinked"
    data.mkdir()
    unlinked.mkdir()

    def _keys(folder: Path, prefix: str = "") -> list[str]:
      return ["/" + str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and str(p.relative_to(folder)).startswith(prefix)]

    class H(BaseHTTPRequestHandler):
      def log_message(self, *_):
        pass

      def _send(self, code: int, body: bytes = b"", ctype: str = "text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
          self.wfile.write(body)

      def _parse(self):
        u = urlsplit(self.path)
        return u.path.lstrip("/"), u.query

      def do_PUT(self):
        key, _ = self._parse()
        if not key:
          return self._send(400, b"missing key")
        p = data / key
        if p.exists():
          return self._send(403, b"exists")
        p.parent.mkdir(parents=True, exist_ok=True)
        n = int(self.headers.get("Content-Length") or 0)
        p.write_bytes(self.rfile.read(n))
        self._send(201)

      def do_HEAD(self):
        key, _ = self._parse()
        if not key:
          return self._send(400)
        self._send(200 if (data / key).exists() else 404)

      def do_GET(self):
        key, q = self._parse()
        if q == "list":
          return self._send(200, json.dumps({"next": "", "keys": _keys(data, key)}).encode(), "application/json")
        if q == "unlinked":
          return self._send(200, json.dumps({"next": "", "keys": _keys(unlinked)}).encode(), "application/json")
        if not key:
          return self._send(400, b"missing key")
        p = data / key
        if not p.exists():
          return self._send(404, b"not found")
        b = p.read_bytes()
        self._send(200, b, "application/octet-stream")

      def do_UNLINK(self):
        key, _ = self._parse()
        src = data / key
        if not src.exists():
          return self._send(404, b"not found")
        dst = unlinked / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
          dst.unlink()
        src.rename(dst)
        self._send(204)

      def do_DELETE(self):
        key, _ = self._parse()
        for folder in (data, unlinked):
          p = folder / key
          if p.exists():
            p.unlink()
            return self._send(204)
        self._send(404, b"not found")

    httpd = ThreadingHTTPServer((host, port), H)
    t = Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    import time

    time.sleep(0.1)  # Give the server a moment to start listening
    try:
      yield f"http://{host}:{port}"
    finally:
      httpd.shutdown()
      t.join(2)
      httpd.server_close()
  finally:
    tmp.cleanup()

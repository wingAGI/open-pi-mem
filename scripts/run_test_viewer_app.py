from __future__ import annotations

import argparse
import http.server
import mimetypes
import os
import urllib.parse
from pathlib import Path


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str, repo_root: Path, **kwargs) -> None:
        self.repo_root = repo_root
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/repo/"):
            self._serve_repo_file()
            return
        super().do_GET()

    def _serve_repo_file(self) -> None:
        encoded = self.path[len("/repo/") :]
        decoded = urllib.parse.unquote(encoded).lstrip("/")
        raw_path = Path("/" + decoded) if self.path[len("/repo/") :].startswith("/") else Path(decoded)
        if raw_path.is_absolute():
            candidate = raw_path.resolve()
        else:
            candidate = (self.repo_root / raw_path).resolve()
        if self.repo_root not in candidate.parents and candidate != self.repo_root:
            self.send_error(403, "Forbidden")
            return
        if not candidate.exists() or not candidate.is_file():
            self.send_error(404, "Not Found")
            return
        content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(candidate.stat().st_size))
        self.end_headers()
        with candidate.open("rb") as f:
            self.wfile.write(f.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    web_root = repo_root / "web" / "viewer"

    def handler(*handler_args, **handler_kwargs):
        return ViewerHandler(*handler_args, directory=str(web_root), repo_root=repo_root, **handler_kwargs)

    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        print(f"Serving test viewer at http://{args.host}:{args.port}")
        print("Load reports with ?report=<repo-relative-path-to-report.json>")
        httpd.serve_forever()


if __name__ == "__main__":
    main()

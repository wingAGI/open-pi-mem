from __future__ import annotations

import argparse
import http.server
import json
import socketserver
from pathlib import Path


class AnnotatorHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str, repo_root: Path, **kwargs) -> None:
        self.repo_root = repo_root
        super().__init__(*args, directory=directory, **kwargs)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/save":
            self.send_error(404, "Not Found")
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            result = save_annotations(self.repo_root, payload)
        except Exception as exc:  # noqa: BLE001
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, **result}, ensure_ascii=False).encode("utf-8"))


def save_annotations(repo_root: Path, payload: dict) -> dict[str, str]:
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("episodes must be a non-empty list")

    base_dir = repo_root / "data" / "manual_annotations"
    episodes_dir = base_dir / "episodes"
    base_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for episode in episodes:
        episode_id = episode.get("episode_id") or episode.get("episodeId")
        if not episode_id:
            raise ValueError("episode missing episode_id")
        episode_path = episodes_dir / f"{episode_id}.json"
        episode_path.write_text(json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(str(episode_path.relative_to(repo_root)))

    jsonl_path = base_dir / "episodes.annotated.jsonl"
    jsonl_path.write_text("".join(json.dumps(ep, ensure_ascii=False) + "\n" for ep in episodes), encoding="utf-8")

    return {
        "jsonl_path": str(jsonl_path.relative_to(repo_root)),
        "episodes_dir": str(episodes_dir.relative_to(repo_root)),
        "written_count": str(len(written)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    web_root = repo_root / "web" / "annotator"

    def handler(*handler_args, **handler_kwargs):
        return AnnotatorHandler(*handler_args, directory=str(web_root), repo_root=repo_root, **handler_kwargs)

    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving annotator at http://{args.host}:{args.port}")
        print(f"Repository export target: {repo_root / 'data' / 'manual_annotations'}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()

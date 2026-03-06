from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_json(path: str | Path) -> dict | list:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_records(path: str | Path) -> list[dict]:
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload in {path}: expected object or array")


def read_json_or_jsonl(path: str | Path) -> list[dict]:
    resolved = Path(path)
    if resolved.suffix == ".jsonl":
        return read_jsonl(resolved)
    if resolved.suffix == ".json":
        return read_json_records(resolved)
    raise ValueError(f"Unsupported input file: {resolved}")


def read_records_from_path(path: str | Path) -> list[dict]:
    resolved = Path(path)
    if resolved.is_dir():
        rows: list[dict] = []
        for json_path in sorted(resolved.glob("*.json")):
            rows.extend(read_json_records(json_path))
        return rows
    return read_json_or_jsonl(resolved)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

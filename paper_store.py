"""Date-partitioned, integrity-checked paper archive (no external service needed)."""

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import tempfile

DEFAULT_DATA_PATH = "data/papers"
MAX_FILE_BYTES = 50 * 1024 * 1024


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def validate(data):
    if not isinstance(data, dict):
        raise ValueError("Paper archive must map dates to lists")
    for day, papers in data.items():
        if date.fromisoformat(day).isoformat() != day:
            raise ValueError("Invalid archive date")
        if not isinstance(papers, list) or any(not isinstance(p, dict) for p in papers):
            raise ValueError(f"Invalid paper list: {day}")


def load_papers(path=DEFAULT_DATA_PATH):
    path = Path(path)
    # Legacy input remains readable for migration and explicit exports.
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        validate(data)
        return data
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not isinstance(manifest.get("dates"), dict):
        raise ValueError("Unsupported archive manifest")
    expected = {f"{day}.json" for day in manifest["dates"]} | {"manifest.json"}
    if {p.name for p in path.glob("*.json")} != expected:
        raise ValueError("Archive file set does not match manifest")
    data = {}
    for day, metadata in manifest["dates"].items():
        validate({day: []})
        raw = (path / f"{day}.json").read_bytes()
        if len(raw) >= MAX_FILE_BYTES:
            raise ValueError(f"Archive shard exceeds size budget: {day}")
        if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {day}")
        data[day] = json.loads(raw)
        validate({day: data[day]})
        if len(data[day]) != metadata["count"]:
            raise ValueError(f"Archive count mismatch: {day}")
    return data


def atomic_write(path, raw):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_papers(data, path=DEFAULT_DATA_PATH):
    """Validate everything before writes; never silently discard historical rows.

    Files are replaced atomically and the manifest is written last. An interrupted
    multi-file save fails integrity checks, rather than loading partial history.
    """
    validate(data)
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        previous = load_papers(path)
        for day, papers in previous.items():
            # Retrying summaries may update a row, but may not remove its identity.
            old_ids = Counter(p.get("arxiv_html_link") for p in papers)
            new_ids = Counter(p.get("arxiv_html_link") for p in data.get(day, []))
            if day not in data or old_ids - new_ids:
                raise ValueError(f"Refusing to remove historical records: {day}")
    pending = {}
    manifest = {"version": 1, "dates": {}}
    for day, papers in sorted(data.items()):
        raw = encode(papers)
        if len(raw) >= MAX_FILE_BYTES:
            raise ValueError(f"Archive shard exceeds size budget: {day}")
        pending[f"{day}.json"] = raw
        manifest["dates"][day] = {
            "count": len(papers), "sha256": hashlib.sha256(raw).hexdigest()
        }
    pending["manifest.json"] = encode(manifest)
    path.mkdir(parents=True, exist_ok=True)
    for name, raw in pending.items():
        target = path / name
        if not target.exists() or target.read_bytes() != raw:
            atomic_write(target, raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["migrate", "validate", "export"])
    parser.add_argument("--archive", default=DEFAULT_DATA_PATH)
    parser.add_argument("--legacy", default="arxiv_cs_ro_papers_final.json")
    parser.add_argument("--output", help="Untracked export path (export only)")
    args = parser.parse_args()
    if args.command == "migrate":
        original = load_papers(args.legacy)
        save_papers(original, args.archive)
        if load_papers(args.archive) != original:
            raise ValueError("Migration round-trip changed historical records")
        print("Migration verified; the legacy input has NOT been deleted.")
    data = load_papers(args.archive)
    if args.command == "export":
        if not args.output:
            parser.error("export requires --output")
        atomic_write(Path(args.output), encode(data))
    print(f"Validated {len(data)} dates / {sum(map(len, data.values()))} records")


if __name__ == "__main__":
    main()

"""Mirror a local folder into a collection (full sync, client side).

    uv run python scripts/sync_folder.py --dir ./knowledge --collection default [--delete-missing] [--dry-run]

* Each file becomes a source with a deterministic ID: ``<prefix>:<relative:path>`` (``/`` → ``:``),
  so re-running only re-embeds files whose content changed (the service diffs by hash).
* ``--delete-missing`` deletes sources under the prefix whose files no longer exist — this is how
  "source deleted → embeddings removed" is automated for a folder.
* Supported: .md .markdown .txt .rst .log .html .htm .csv .json (text) and .pdf (base64).
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path
from typing import Any

import httpx

MIME = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".rst": "text/plain",
    ".log": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".csv": "text/csv",
    ".json": "application/json",
    ".pdf": "application/pdf",
}
_UNSAFE = re.compile(r"[^A-Za-z0-9_\-.:@=+~]")


def source_id(prefix: str, rel: Path) -> str:
    raw = f"{prefix}:{':'.join(rel.parts)}"
    return _UNSAFE.sub("-", raw)[:512]


def document(path: Path, root: Path, prefix: str) -> dict[str, Any]:
    rel = path.relative_to(root)
    mime = MIME[path.suffix.lower()]
    doc: dict[str, Any] = {
        "source_id": source_id(prefix, rel),
        "title": path.stem.replace("_", " ").replace("-", " "),
        "mime_type": mime,
        "filename": path.name,
        "metadata": {"path": rel.as_posix(), "folder": rel.parent.as_posix(), "extension": path.suffix.lower()},
    }
    if mime == "application/pdf":
        doc["content_base64"] = base64.b64encode(path.read_bytes()).decode()
    else:
        doc["content"] = path.read_text(encoding="utf-8", errors="replace")
    return doc


def existing_sources(client: httpx.Client, collection: str, prefix: str) -> set[str]:
    found: set[str] = set()
    offset = 0
    while True:
        body = client.get(f"/api/v1/collections/{collection}/documents", params={"limit": 500, "offset": offset}).json()
        found |= {s["source_id"] for s in body["data"] if s["source_id"].startswith(f"{prefix}:")}
        if not body["meta"]["pagination"]["has_more"]:
            return found
        offset += 500


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--collection", default="default")
    parser.add_argument("--prefix", default=None, help="source_id prefix (default: folder name)")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--delete-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root: Path = args.dir.resolve()
    prefix = args.prefix or _UNSAFE.sub("-", root.name)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in MIME)
    docs = [document(p, root, prefix) for p in files]
    print(f"{len(docs)} files under {root} → collection '{args.collection}' (prefix '{prefix}')")

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=900) as client:
        totals: dict[str, int] = {}
        if not args.dry_run:
            for start in range(0, len(docs), args.batch_size):
                batch = docs[start : start + args.batch_size]
                r = client.post(
                    f"/api/v1/collections/{args.collection}/documents",
                    json={"documents": batch, "options": {"mode": "sync"}},
                )
                if r.status_code not in (200, 207, 422):
                    print(f"batch failed: {r.status_code} {r.text}")
                    sys.exit(1)
                for result in r.json()["data"]["results"] or []:
                    totals[result["action"]] = totals.get(result["action"], 0) + 1
                    if result["action"] == "failed":
                        print(f"  failed {result['source_id']}: {result['error']}")
        print("upserts:", totals or "(dry run)")

        if args.delete_missing:
            stale = sorted(existing_sources(client, args.collection, prefix) - {d["source_id"] for d in docs})
            print(f"{len(stale)} source(s) no longer on disk" + (": " + ", ".join(stale[:10]) if stale else ""))
            if stale and not args.dry_run:
                r = client.post(f"/api/v1/collections/{args.collection}/documents:delete", json={"source_ids": stale})
                print("deleted:", sum(1 for x in r.json()["data"]["results"] if x["action"] == "deleted"))


if __name__ == "__main__":
    main()

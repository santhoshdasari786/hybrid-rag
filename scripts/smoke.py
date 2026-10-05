"""Smoke-test / demo a running service over HTTP (no service internals imported).

    uv run python scripts/smoke.py --base-url http://localhost:8080 [--api-key KEY] [--collection products]

Steps: health → ingest examples/ingest_products.json → retrieve → chat → re-sync one document →
evaluate retrieval variants. Exits non-zero on the first failed step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def step(title: str) -> None:
    print(f"\n=== {title}")


def check(response: httpx.Response, *ok: int) -> dict[str, Any]:
    if response.status_code not in (ok or (200,)):
        print(f"FAILED {response.request.method} {response.request.url} → {response.status_code}\n{response.text}")
        sys.exit(1)
    return response.json() if response.content else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--collection", default="products")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    base = f"/api/v1/collections/{args.collection}"
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=600) as client:
        step("health")
        print(check(client.get("/health/ready")))

        step("ingest sample catalog (sync)")
        payload = json.loads((EXAMPLES / "ingest_products.json").read_text())
        job = check(client.post(f"{base}/documents", json=payload), 200, 207)["data"]
        actions: dict[str, int] = {}
        for result in job["results"]:
            actions[result["action"]] = actions.get(result["action"], 0) + 1
        print(f"job {job['id']}: {job['status']} {actions}")

        step("retrieve 'revent 80 cfm' (hybrid)")
        data = check(client.post(f"{base}/retrieve", json=json.loads((EXAMPLES / "retrieve.json").read_text())))["data"]
        for r in data["results"]:
            print(f"  #{r['rank']} {r['score']:.4f} {r['source_id']:<28} {r['scores']}")

        step("chat")
        chat = check(client.post(f"{base}/chat", json=json.loads((EXAMPLES / "chat.json").read_text())))["data"]
        print(chat["message"]["content"])
        print("citations:", [(c["index"], c["source_id"]) for c in chat["citations"]])

        step("re-sync one document (only changed chunks are re-embedded)")
        doc = payload["documents"][0]
        body = {k: v for k, v in doc.items() if k != "source_id"}
        body["content"] = doc["content"] + f"\n\nUpdated {time.strftime('%Y-%m-%d %H:%M:%S')}."
        result = check(client.put(f"{base}/documents/{doc['source_id']}", json=body))["data"]["results"][0]
        print({k: result[k] for k in ("action", "version", "chunks_added", "chunks_removed", "chunks_unchanged")})

        if not args.skip_eval:
            step("evaluate retrieval variants")
            report = check(
                client.post(f"{base}/evaluate", json=json.loads((EXAMPLES / "evaluate_products.json").read_text()))
            )["data"]
            print(f"{'variant':<16}{'hit':>7}{'recall':>8}{'mrr':>7}{'ndcg':>7}{'p50 ms':>9}")
            for v in report["variants"]:
                m = v["metrics"]
                print(
                    f"{v['name']:<16}{m['hit_rate']:>7.2f}{m['recall']:>8.2f}{m['mrr']:>7.2f}{m['ndcg']:>7.2f}{m['latency_ms_p50']:>9.1f}"
                )
            print("best:", report["best"])
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()

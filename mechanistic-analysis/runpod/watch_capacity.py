#!/usr/bin/env python3
"""Poll RunPod for a multi-GPU configuration large enough to hold a model.

Qwen3.5-397B-A17B is ~794 GB in bf16, and every 8-GPU config that fits it was
out of stock when this was written. Run this to see current availability, or
with --watch to poll until something appears.

  python watch_capacity.py --min-vram 850
  python watch_capacity.py --min-vram 850 --watch --interval 600
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GRAPHQL = "https://api.runpod.io/graphql"


def api_key() -> str:
    env_path = ROOT / ".env"
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("RUNPOD_API_KEY missing from .env")


def gql(query: str) -> dict:
    out = subprocess.run(
        ["curl", "-s", "--max-time", "40", "-X", "POST", GRAPHQL,
         "-H", "Content-Type: application/json",
         "-H", f"Authorization: Bearer {api_key()}",
         "-d", json.dumps({"query": query})],
        capture_output=True, text=True,
    ).stdout
    payload = json.loads(out)
    if "errors" in payload:
        raise SystemExit("API error: " + json.dumps(payload["errors"]))
    return payload["data"]


def available(min_vram: int, gpu_count: int) -> list[tuple]:
    gpus = gql("query { gpuTypes { id memoryInGb } }")["gpuTypes"]
    hits = []
    for gpu in gpus:
        total = gpu["memoryInGb"] * gpu_count
        if total < min_vram:
            continue
        for cloud in ("true", "false"):
            query = (
                'query { gpuTypes(input:{id:"%s"}) { lowestPrice(input:{gpuCount:%d,secureCloud:%s}) '
                "{ uninterruptablePrice stockStatus } } }" % (gpu["id"], gpu_count, cloud)
            )
            try:
                low = gql(query)["gpuTypes"][0]["lowestPrice"]
            except SystemExit:
                continue
            price, stock = low.get("uninterruptablePrice"), low.get("stockStatus")
            if price and stock and stock != "none":
                hits.append((price, gpu["id"], total, "secure" if cloud == "true" else "community", stock))
    return sorted(hits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-vram", type=int, default=850, help="Total GB needed across the node.")
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between polls.")
    args = parser.parse_args()

    while True:
        hits = available(args.min_vram, args.gpu_count)
        stamp = time.strftime("%H:%M:%S")
        if hits:
            print(f"[{stamp}] AVAILABLE:")
            for price, gid, total, cloud, stock in hits:
                print(f"    {args.gpu_count}x {gid[:38]:38s} {total:5d} GB  ${price:6.2f}/hr  {cloud:9s} {stock}")
            if args.watch:
                print("    (capacity found, stopping)")
            return
        print(f"[{stamp}] nothing with >= {args.min_vram} GB at {args.gpu_count} GPUs", flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

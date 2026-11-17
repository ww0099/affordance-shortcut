"""下载 IffYuan/AGD 小样本（Phase 0 冒烟用，约 214MB）。

直接用 httpx 流式下载（支持断点续传），不依赖 huggingface_hub 的传输后端。
默认走本机代理 127.0.0.1:7892；也可用 --proxy 或 HF_PROXY 覆盖。

数据落在 `D:\\Myproject(AI)\\downloads\\affordance-datasets\\iffyuan-agd`，
可用环境变量 AGD_DATA_ROOT 覆盖。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

REPO_ID = "IffYuan/AGD"
FILES = [
    "data/train-00000-of-00001.parquet",
    "images.tar.gz",
    "masks.tar.gz",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("AGD_DATA_ROOT", str(PROJECT_ROOT.parent / "downloads" / "affordance-datasets"))
)
TARGET_DIR = DATA_ROOT / "iffyuan-agd"


def download_one(client: httpx.Client, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with client.stream("GET", url, headers=headers) as response:
        if response.status_code == 416:
            print(f"  [skip] 已完成：{dest.name}", flush=True)
            return
        response.raise_for_status()

        append = existing > 0 and response.status_code == 206
        if not append:
            existing = 0
        total = existing + int(response.headers.get("content-length", 0) or 0)
        mode = "ab" if append else "wb"

        with open(dest, mode) as handle:
            downloaded = existing
            last_report = time.time()
            for chunk in response.iter_bytes(1 << 18):
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report >= 2:
                    percent = f" ({downloaded / total * 100:.1f}%)" if total else ""
                    print(f"  {downloaded / 1048576:.1f} MB{percent}", flush=True)
                    last_report = now

    print(f"  [done] {dest.name}: {dest.stat().st_size / 1048576:.1f} MB", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--proxy", default=os.environ.get("HF_PROXY", "http://127.0.0.1:7892"))
    args = parser.parse_args()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"endpoint = {args.endpoint}", flush=True)
    print(f"proxy    = {args.proxy or '(none)'}", flush=True)
    print(f"target   = {TARGET_DIR}", flush=True)

    timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
    client_kwargs = {"follow_redirects": True, "timeout": timeout, "trust_env": False}
    if args.proxy:
        client_kwargs["proxy"] = args.proxy

    with httpx.Client(**client_kwargs) as client:
        for name in FILES:
            url = f"{args.endpoint.rstrip('/')}/datasets/{REPO_ID}/resolve/main/{name}"
            print(f"[download] {name}", flush=True)
            try:
                download_one(client, url, TARGET_DIR / name)
            except Exception as exc:
                print(f"[FAIL] {name}: {exc}", flush=True)
                return 1

    total_mb = sum((TARGET_DIR / name).stat().st_size for name in FILES) / 1048576
    print(f"ALL DONE, total {total_mb:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

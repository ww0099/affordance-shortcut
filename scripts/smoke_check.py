"""环境与数据自检。报告写入 results/smoke_report.txt（UTF-8），控制台只打 ASCII 摘要。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import resolve_data_root  # noqa: E402


def collect() -> list[str]:
    lines: list[str] = []
    lines.append(f"project_root: {PROJECT_ROOT}")
    lines.append(f"python: {sys.version.split()[0]}")

    for module in ["torch", "torchvision", "pyarrow", "PIL", "numpy", "yaml", "open_clip", "timm"]:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            lines.append(f"package {module}: {version}")
        except Exception:
            lines.append(f"package {module}: MISSING")

    try:
        import torch

        lines.append(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"torch.cuda.device_count: {torch.cuda.device_count()}")
            lines.append(f"torch.cuda.device0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        lines.append(f"torch import error: {exc}")

    data_root = resolve_data_root()
    lines.append(f"data_root: {data_root}")
    if data_root.exists():
        for child in sorted(data_root.iterdir()):
            if child.is_file():
                lines.append(f"data_file: {child.name} ({child.stat().st_size / 1024 / 1024:.1f} MB)")
            else:
                files = [item for item in child.rglob("*") if item.is_file()]
                total_mb = sum(item.stat().st_size for item in files) / 1024 / 1024
                lines.append(f"data_dir: {child.name} -> {len(files)} files, {total_mb:.1f} MB")
    else:
        lines.append("data_root: NOT FOUND")

    meta_path = PROJECT_ROOT / "data" / "processed" / "meta.csv"
    lines.append(f"meta_csv: {meta_path}")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        lines.append(f"meta_rows: {len(rows)}")
        splits: dict[str, int] = {}
        for row in rows:
            splits[row.get("split", "unknown")] = splits.get(row.get("split", "unknown"), 0) + 1
        lines.append(f"meta_splits: {splits}")
        missing = 0
        for row in rows:
            if not (data_root / row["image_path"]).exists() or not (data_root / row["mask_path"]).exists():
                missing += 1
        lines.append(f"missing_assets: {missing}")
    else:
        lines.append("meta_csv: NOT FOUND（先运行 scripts/prepare_meta.py）")

    return lines


def main() -> int:
    lines = collect()
    report = PROJECT_ROOT / "results" / "smoke_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"report_written: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""构建 Kaggle 冒烟包：从 meta.csv 选 5 个多样本，复制图像/掩码并打包 zip。

产物（默认在 AGD_DATA_ROOT 下）：
    smoke5/smoke_meta.csv
    smoke5/images/*.jpg|png
    smoke5/masks/*.png
    affordance-smoke5.zip

用法：python scripts/build_smoke_pack.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import resolve_data_root  # noqa: E402

PREFERRED_OBJECTS = [
    "knife",
    "wine_glass",
    "cup",
    "bicycle",
    "bottle",
    "skateboard",
    "banana",
    "suitcase",
    "frisbee",
    "motorcycle",
]
MASK_RATIO_MIN = 0.05
MASK_RATIO_MAX = 0.55
TARGET_COUNT = 5


def mask_ratio(path: Path) -> float:
    mask = Image.open(path).convert("L")
    array = np.asarray(mask, dtype=np.uint8)
    return float((array > 127).mean())


def main() -> int:
    data_root = resolve_data_root()
    rows = list(csv.DictReader(open(PROJECT_ROOT / "data" / "processed" / "meta.csv", encoding="utf-8-sig")))

    selected: list[dict[str, str]] = []
    used_objects: set[str] = set()
    used_affordances: set[str] = set()

    for object_name in PREFERRED_OBJECTS:
        if len(selected) >= TARGET_COUNT:
            break
        for row in rows:
            if row["object"] != object_name or object_name in used_objects:
                continue
            if row["affordance"] in used_affordances:
                continue
            mask_path = data_root / row["mask_path"]
            if not mask_path.exists():
                continue
            ratio = mask_ratio(mask_path)
            if MASK_RATIO_MIN <= ratio <= MASK_RATIO_MAX:
                selected.append(row)
                used_objects.add(object_name)
                used_affordances.add(row["affordance"])
                break

    if len(selected) < TARGET_COUNT:
        for row in rows:
            if len(selected) >= TARGET_COUNT:
                break
            if row["object"] in used_objects or row["affordance"] in used_affordances:
                continue
            mask_path = data_root / row["mask_path"]
            if mask_path.exists() and MASK_RATIO_MIN <= mask_ratio(mask_path) <= MASK_RATIO_MAX:
                selected.append(row)
                used_objects.add(row["object"])
                used_affordances.add(row["affordance"])

    if len(selected) < TARGET_COUNT:
        raise SystemExit(f"只选中 {len(selected)} 个样本，检查数据目录")

    out_dir = data_root / "smoke5"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "masks").mkdir(parents=True)

    meta_rows = []
    for index, row in enumerate(selected):
        image_src = data_root / row["image_path"]
        mask_src = data_root / row["mask_path"]
        stem = f"{index:02d}_{row['object'].replace(' ', '_')}_{row['affordance'].replace(' ', '_')}"
        image_name = f"{stem}{image_src.suffix.lower()}"
        mask_name = f"{stem}.png"
        shutil.copyfile(image_src, out_dir / "images" / image_name)
        Image.open(mask_src).convert("L").save(out_dir / "masks" / mask_name)
        meta_rows.append(
            {
                "id": row["id"],
                "image": f"images/{image_name}",
                "mask": f"masks/{mask_name}",
                "object": row["object"],
                "affordance": row["affordance"],
                "split": row["split"],
            }
        )

    meta_path = out_dir / "smoke_meta.csv"
    with open(meta_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image", "mask", "object", "affordance", "split"])
        writer.writeheader()
        writer.writerows(meta_rows)

    zip_path = data_root / "affordance-smoke5.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(out_dir.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(out_dir).as_posix())

    summary = {
        "count": len(meta_rows),
        "objects": [row["object"] for row in meta_rows],
        "affordances": [row["affordance"] for row in meta_rows],
        "dir": str(out_dir),
        "zip": str(zip_path),
        "zip_mb": round(zip_path.stat().st_size / 1024 / 1024, 2),
    }
    (out_dir / "smoke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""原始数据 → meta.csv（含完整性校验）。

当前支持：`iffyuan_agd`（IffYuan/AGD 子集，1,543 条）
- parquet 内嵌 image / mask 字节（HF Image 特征），无需 tar 包；
- 若列里只有文件名（未来换数据源），再回退到 images.tar.gz / masks.tar.gz。
后续补充：`agd20k`（Seen/Unseen 目录树，Phase 1）

输出列：
    id, image_path, mask_path, object, affordance, split, subset, instruction

image_path / mask_path 相对 AGD_DATA_ROOT，可直接被 src/data.py 解析。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import resolve_data_root  # noqa: E402

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


OBJECT_KEYS = ["object", "noun", "obj", "category", "object_name", "cls", "label"]
AFFORDANCE_KEYS = ["affordance", "verb", "action", "aff", "function"]
IMAGE_KEYS = ["image_path", "image_name", "image", "img", "file_name", "filename", "image_file"]
MASK_KEYS = ["mask_path", "mask_name", "mask", "mask_file", "segmentation"]
INSTRUCTION_KEYS = ["instruction", "text", "prompt", "description"]
ID_KEYS = ["id", "question_id", "image_id"]
SPLIT_KEYS = ["split", "subset_split", "phase"]


def find_column(names: list[str], candidates: list[str]) -> str | None:
    lower = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return None


def safe_extract(tar_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as archive:
        members = []
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                continue
            members.append(member)
        try:
            archive.extractall(dest, members=members, filter="data")
        except TypeError:  # Python < 3.11.4
            archive.extractall(dest, members=members)
    count = sum(1 for item in dest.rglob("*") if item.is_file())
    print(f"[extract] {tar_path.name} -> {dest}（{count} 个文件）", flush=True)


def build_file_index(roots: list[Path]) -> dict[str, tuple[Path, str]]:
    """key（相对路径 / 文件名 / 去扩展名）→ (实际根目录, 相对路径)。"""
    index: dict[str, tuple[Path, str]] = {}
    for root in roots:
        if not root.exists():
            continue
        for item in root.rglob("*"):
            if not item.is_file():
                continue
            relative = item.relative_to(root).as_posix()
            index.setdefault(relative, (root, relative))
            index.setdefault(item.name, (root, relative))
            index.setdefault(Path(relative).stem, (root, relative))
    return index


def is_inline(value) -> bool:
    return isinstance(value, dict) and bool(value.get("bytes"))


def materialize(value, target_dir: Path, fallback_stem: str) -> str:
    """把 parquet 字段落成文件名；已是路径/文件名的原样返回。

    内嵌字节一律用 fallback_stem + 原扩展名命名，避免不同样本的 path 重名互相覆盖。
    """
    if isinstance(value, dict):
        path = value.get("path")
        payload = value.get("bytes")
        if payload:
            target_dir.mkdir(parents=True, exist_ok=True)
            suffix = Path(path).suffix if path else ".png"
            name = f"{fallback_stem}{suffix}"
            (target_dir / name).write_bytes(payload)
            return name
        if path:
            return str(path)
        return ""

    if isinstance(value, (bytes, bytearray)):
        target_dir.mkdir(parents=True, exist_ok=True)
        name = f"{fallback_stem}.png"
        (target_dir / name).write_bytes(bytes(value))
        return name

    return "" if value is None else str(value)


def resolve_value(materialized: str, index: dict[str, tuple[Path, str]], target_dir: Path, data_root: Path) -> str | None:
    if materialized:
        direct = target_dir / materialized
        if direct.exists():
            return direct.relative_to(data_root).as_posix()
    if not materialized:
        return None
    normalized = materialized.replace("\\", "/").lstrip("./")
    hit = index.get(normalized) or index.get(Path(normalized).name) or index.get(Path(normalized).stem)
    if hit is None:
        return None
    root, relative = hit
    return (root / relative).relative_to(data_root).as_posix()


def smoke_split(sample_id: str) -> str:
    """HF 子集没有官方划分：按 id 稳定分配 80/20，仅用于冒烟。"""
    digest = hashlib.md5(sample_id.encode("utf-8")).digest()
    return "train" if digest[0] % 10 < 8 else "val"


def prepare_iffyuan_agd(data_root: Path, force_extract: bool) -> tuple[list[dict[str, str]], dict]:
    if pq is None:
        raise SystemExit("需要 pyarrow：pip install pyarrow")

    asset_root = data_root / "iffyuan-agd"
    parquet_files = sorted(asset_root.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"未找到 parquet：{asset_root}")
    print(f"[parquet] {parquet_files[0]}", flush=True)

    table = pq.read_table(parquet_files[0])
    columns = table.column_names
    print(f"[schema] columns = {columns}", flush=True)
    rows = table.to_pylist()

    object_col = find_column(columns, OBJECT_KEYS)
    affordance_col = find_column(columns, AFFORDANCE_KEYS)
    image_col = find_column(columns, IMAGE_KEYS)
    mask_col = find_column(columns, MASK_KEYS)
    instruction_col = find_column(columns, INSTRUCTION_KEYS)
    id_col = find_column(columns, ID_KEYS)
    split_col = find_column(columns, SPLIT_KEYS)

    required = {"object": object_col, "affordance": affordance_col, "image": image_col, "mask": mask_col}
    missing_required = [key for key, value in required.items() if value is None]
    if missing_required:
        raise SystemExit(f"parquet 缺少必要列 {missing_required}；实际列：{columns}")

    inline_images_dir = asset_root / "extracted" / "inline_images"
    inline_masks_dir = asset_root / "extracted" / "inline_masks"
    images_root = asset_root / "extracted" / "images"
    masks_root = asset_root / "extracted" / "masks"

    first_row = rows[0]
    inline = is_inline(first_row.get(image_col)) and is_inline(first_row.get(mask_col))
    if inline:
        print("[layout] image/mask 内嵌于 parquet，跳过 tar 解压", flush=True)
        image_index = build_file_index([inline_images_dir])
        mask_index = build_file_index([inline_masks_dir])
    else:
        print("[layout] 使用 tar 包中的图片/掩码", flush=True)
        for tar_name, dest in [("images.tar.gz", images_root), ("masks.tar.gz", masks_root)]:
            tar_path = asset_root / tar_name
            if not tar_path.exists():
                raise SystemExit(f"缺少 {tar_path}")
            if force_extract and dest.exists():
                import shutil

                shutil.rmtree(dest)
            if not dest.exists() or not any(dest.iterdir()):
                safe_extract(tar_path, dest)
            else:
                print(f"[skip] 已解压：{dest}", flush=True)
        image_index = build_file_index([images_root, inline_images_dir])
        mask_index = build_file_index([masks_root, inline_masks_dir])

    meta_rows: list[dict[str, str]] = []
    missing_assets = 0

    for index, row in enumerate(rows):
        sample_id = str(row.get(id_col, index)) if id_col else str(index)
        image_name = materialize(row.get(image_col), inline_images_dir, f"{sample_id}_image")
        mask_name = materialize(row.get(mask_col), inline_masks_dir, f"{sample_id}_mask")

        image_rel = resolve_value(image_name, image_index, inline_images_dir, data_root)
        mask_rel = resolve_value(mask_name, mask_index, inline_masks_dir, data_root)
        if image_rel is None or mask_rel is None:
            missing_assets += 1
            continue

        split = str(row.get(split_col, "") or "") if split_col else ""
        if not split:
            split = smoke_split(sample_id)

        meta_rows.append(
            {
                "id": sample_id,
                "image_path": image_rel,
                "mask_path": mask_rel,
                "object": str(row.get(object_col, "") or "").strip(),
                "affordance": str(row.get(affordance_col, "") or "").strip(),
                "split": split,
                "subset": "iffyuan_agd",
                "instruction": str(row.get(instruction_col, "") or "").strip() if instruction_col else "",
            }
        )

    stats = {"parsed_rows": len(rows), "paired": len(meta_rows), "missing_assets": missing_assets}
    print(f"[meta] {stats}", flush=True)
    return meta_rows, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="iffyuan_agd", choices=["iffyuan_agd"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "processed" / "meta.csv"))
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    started = time.time()
    data_root = resolve_data_root(args.data_root)
    print(f"[paths] data_root = {data_root}", flush=True)

    rows, parse_stats = prepare_iffyuan_agd(data_root, args.force_extract)
    if not rows:
        raise SystemExit("没有生成任何样本，检查 parquet 列名与压缩包内容")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "image_path", "mask_path", "object", "affordance", "split", "subset", "instruction"]
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    splits: dict[str, int] = {}
    objects = set()
    affordances = set()
    for row in rows:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
        objects.add(row["object"])
        affordances.add(row["affordance"])

    summary = {
        "total": len(rows),
        "splits": splits,
        "objects": len(objects),
        "affordances": len(affordances),
        "parse": parse_stats,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    summary_path = out_path.with_name("meta_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] {out_path}", flush=True)
    print(f"[summary] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

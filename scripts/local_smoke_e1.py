"""本机 CPU 冒烟：用 src 里的真实代码跑 5 个样本的 E1 零样本。

与 Kaggle Notebook 的区别：这里直接调用仓库模块（FrozenCLIPBackbone / e1_clip_zeroshot /
mean_metrics），同时验证这三个文件本身；Kaggle 版是自包含的，便于快速排错。

用法：
    python scripts/local_smoke_e1.py
环境变量：
    HF_HOME / HF_ENDPOINT   权重缓存与下载镜像
    SMOKE_DIR               默认 <AGD_DATA_ROOT>/smoke5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import e1_clip_zeroshot  # noqa: E402
from src.data import CLIP_MEAN, CLIP_STD  # noqa: E402
from src.eval import mean_metrics  # noqa: E402
from src.model import FrozenCLIPBackbone  # noqa: E402
from src.paths import resolve_data_root  # noqa: E402

IMAGE_SIZE = 224


def load_rows(smoke_dir: Path) -> list[dict[str, str]]:
    meta_path = smoke_dir / "smoke_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}; run scripts/build_smoke_pack.py first")
    with open(meta_path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_split(smoke_dir: Path, rows: list[dict[str, str]], transform) -> tuple[torch.Tensor, list[np.ndarray]]:
    images = []
    masks = []
    for row in rows:
        image = Image.open(smoke_dir / row["image"]).convert("RGB")
        images.append(transform(image))
        mask = Image.open(smoke_dir / row["mask"]).convert("L").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
        )
        masks.append((np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32))
    return torch.stack(images), masks


def norm01(array: np.ndarray) -> np.ndarray:
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-8:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b88k",
        help="open_clip 权重标签，或本地 .safetensors / .pt 路径",
    )
    args = parser.parse_args()

    smoke_dir = Path(args.smoke_dir) if args.smoke_dir else resolve_data_root() / "smoke5"
    rows = load_rows(smoke_dir)
    print(f"[data] {smoke_dir} | samples={len(rows)}", flush=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}", flush=True)

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    images, masks = load_split(smoke_dir, rows, transform)
    images = images.to(device)

    started = time.time()
    backbone = FrozenCLIPBackbone("ViT-B-16", args.pretrained).to(device)
    print(f"[model] loaded in {time.time() - started:.1f}s", flush=True)

    with torch.no_grad():
        heatmaps = e1_clip_zeroshot(
            backbone,
            images,
            [row["object"] for row in rows],
            [row["affordance"] for row in rows],
            output_size=IMAGE_SIZE,
        )
    print(f"[forward] {time.time() - started:.1f}s | shape={tuple(heatmaps.shape)}", flush=True)

    preds = [norm01(item) for item in heatmaps[:, 0].cpu().numpy()]
    metrics = mean_metrics(preds, masks, ["iou", "nss", "kld", "sim"], iou_threshold=0.5)
    print(f"[metrics] {json.dumps(metrics, ensure_ascii=True)}", flush=True)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "smoke_e1_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["object", "affordance", "iou", "nss", "kld", "sim"])
        writer.writeheader()
        for row, pred, gt in zip(rows, preds, masks):
            sample = mean_metrics([pred], [gt], ["iou", "nss", "kld", "sim"], iou_threshold=0.5)
            writer.writerow(
                {
                    "object": row["object"],
                    "affordance": row["affordance"],
                    "iou": round(sample["iou"], 6),
                    "nss": round(sample["nss"], 6),
                    "kld": round(sample["kld"], 6),
                    "sim": round(sample["sim"], 6),
                }
            )

    summary = {
        "device": str(device),
        "samples": len(rows),
        "objects": [row["object"] for row in rows],
        "affordances": [row["affordance"] for row in rows],
        "metrics": {key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    summary_path = results_dir / "smoke_e1_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.6 * len(rows)))
    for index, (row, pred, gt) in enumerate(zip(rows, preds, masks)):
        image = Image.open(smoke_dir / row["image"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        axes[index, 0].imshow(image)
        axes[index, 0].set_title(f"{row['object']} / {row['affordance']}", fontsize=9)
        axes[index, 1].imshow(gt, cmap="gray")
        axes[index, 1].set_title("GT mask", fontsize=9)
        axes[index, 2].imshow(image)
        axes[index, 2].imshow(pred, cmap="jet", alpha=0.5)
        axes[index, 2].set_title(f"E1 IoU={mean_metrics([pred], [gt], ['iou'], 0.5)['iou']:.2f}", fontsize=9)
        for axis in axes[index]:
            axis.axis("off")
    plt.tight_layout()
    figure_path = results_dir / "smoke_e1_heatmaps.png"
    plt.savefig(figure_path, dpi=120, bbox_inches="tight")

    print(f"[saved] {csv_path}", flush=True)
    print(f"[saved] {summary_path}", flush=True)
    print(f"[saved] {figure_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""基线：E0 类别平均掩码、E1 CLIP 零样本相似度。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import build_prompt
from .paths import resolve_data_root


def _prior_key(object_name: str, affordance: str) -> str:
    return f"{object_name}::{affordance}"


def build_category_prior(
    rows: list[dict[str, str]],
    data_root: str | Path | None = None,
    out_npz: str | Path | None = None,
    train_split: str = "train",
    size: int = 224,
) -> dict[str, np.ndarray]:
    """E0：每个 (object, affordance) 取训练集 GT 掩码的平均值。"""
    from PIL import Image

    root = resolve_data_root(str(data_root) if data_root else None)
    sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((size, size), dtype=np.float64))
    counts: dict[str, int] = defaultdict(int)

    used = 0
    for row in rows:
        if train_split and row.get("split") != train_split:
            continue
        mask = Image.open(root / row["mask_path"]).convert("L")
        mask = mask.resize((size, size), Image.Resampling.NEAREST)
        key = _prior_key(row["object"], row["affordance"])
        sums[key] += np.asarray(mask, dtype=np.float32) / 255.0
        counts[key] += 1
        used += 1

    if used == 0:
        raise ValueError(f"train_split={train_split!r} 下没有可用于构建先验的样本")

    priors = {key: (sums[key] / max(counts[key], 1)).astype(np.float32) for key in sums}

    if out_npz:
        out_path = Path(out_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, keys=np.array(list(priors.keys())), masks=np.stack(list(priors.values())))

    print(f"[E0] 构建类别先验：{len(priors)} 个 (object, affordance) 组合，来自 {used} 张训练图", flush=True)
    return priors


def load_category_prior(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    keys = [str(key) for key in data["keys"]]
    return {key: mask for key, mask in zip(keys, data["masks"])}


def e0_predict(priors: dict[str, np.ndarray], object_name: str, affordance: str, size: int = 224) -> np.ndarray:
    mask = priors.get(_prior_key(object_name, affordance))
    if mask is None:
        return np.zeros((size, size), dtype=np.float32)
    if mask.shape[0] != size:
        from PIL import Image

        mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST), dtype=np.float32) / 255.0
    return mask


@torch.no_grad()
def e1_clip_zeroshot(backbone, images: torch.Tensor, objects: list[str], affordances: list[str], output_size: int = 224) -> torch.Tensor:
    """E1：patch 特征与文本嵌入的余弦相似度，上采样成热图。"""
    grid, _ = backbone.patch_grid(images)  # B, D, S, S
    grid = backbone.project_to_text_space(grid)  # 768 -> 512，和文本同空间
    prompts = [build_prompt(obj, aff) for obj, aff in zip(objects, affordances)]
    text = backbone.encode_text(prompts)  # B, D

    batch, dim, side, _ = grid.shape
    flat = grid.reshape(batch, dim, side * side)
    flat = F.normalize(flat, dim=1)
    text = F.normalize(text, dim=-1)

    similarity = torch.einsum("bd,bdn->bn", text, flat).reshape(batch, 1, side, side)
    return F.interpolate(similarity, size=(output_size, output_size), mode="bilinear", align_corners=False)

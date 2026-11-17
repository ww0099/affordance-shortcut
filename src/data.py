"""数据集、prompt 构造与划分读取。

meta.csv 约定列（prepare_meta.py 生成）：
    id, image_path, mask_path, object, affordance, split, subset, instruction

其中 image_path / mask_path 相对 AGD_DATA_ROOT 解析。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .paths import resolve_data_root

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

PROMPT_TEMPLATE = "the part of a {object} you can {affordance}"


def build_prompt(object_name: str, affordance: str) -> str:
    """AGD 系列常用的文本模板。"""
    obj = object_name.strip().replace("_", " ")
    aff = affordance.strip().replace("_", " ")
    return PROMPT_TEMPLATE.format(object=obj, affordance=aff)


def read_meta(meta_csv: str | Path) -> list[dict[str, str]]:
    path = Path(meta_csv)
    if not path.is_absolute():
        from .paths import project_path

        path = project_path(str(meta_csv))
    if not path.exists():
        raise FileNotFoundError(f"未找到 meta.csv：{path}（先运行 scripts/prepare_meta.py）")
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class AffordanceDataset(Dataset):
    """图像 + 文本 prompt → 可供性掩码。"""

    def __init__(
        self,
        rows: Iterable[dict[str, str]],
        data_root: str | Path | None = None,
        image_size: int = 224,
        split: str | None = None,
    ) -> None:
        self.rows = [row for row in rows if not split or row.get("split") == split]
        if not self.rows:
            raise ValueError(f"split={split!r} 下没有样本")
        self.root = resolve_data_root(str(data_root) if data_root else None)
        self.image_size = image_size

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, relative: str) -> Image.Image:
        return Image.open(self.root / relative).convert("RGB")

    def _load_mask(self, relative: str) -> np.ndarray:
        mask = Image.open(self.root / relative).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        return np.asarray(mask, dtype=np.float32) / 255.0

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image = self._load_image(row["image_path"])
        mask = self._load_mask(row["mask_path"])

        return {
            "image": self.image_transform(image),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "prompt": build_prompt(row["object"], row["affordance"]),
            "object": row["object"],
            "affordance": row["affordance"],
            "split": row.get("split", "unknown"),
            "id": row.get("id", str(index)),
        }


def group_by_split(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("split", "unknown")] = counts.get(row.get("split", "unknown"), 0) + 1
    return counts

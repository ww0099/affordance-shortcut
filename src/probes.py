"""三组诊断探针（KAGGLE_RESEARCH_PLAN 2.4 节）。"""

from __future__ import annotations

import numpy as np
import torch

from .data import build_prompt


@torch.no_grad()
def text_swap_probe(model, images: torch.Tensor, objects: list[str], affordances: list[str], swapped_objects: list[str]) -> list[float]:
    """同一张图，把查询里的物体名换掉，看掩码相对正确 prompt 的 IoU 变化。"""
    correct_prompts = [build_prompt(obj, aff) for obj, aff in zip(objects, affordances)]
    swapped_prompts = [build_prompt(obj, aff) for obj, aff in zip(swapped_objects, affordances)]

    correct = torch.sigmoid(model(images, correct_prompts)).cpu().numpy()[:, 0]
    swapped = torch.sigmoid(model(images, swapped_prompts)).cpu().numpy()[:, 0]

    scores = []
    for a, b in zip(correct, swapped):
        binary_a = a >= 0.5
        binary_b = b >= 0.5
        union = np.logical_or(binary_a, binary_b).sum()
        scores.append(float(np.logical_and(binary_a, binary_b).sum() / union) if union else float("nan"))
    return scores


@torch.no_grad()
def same_object_affordance_probe(model, images: torch.Tensor, object_name: str, affordance_a: str, affordance_b: str) -> list[float]:
    """同一物体、不同功能词，看掩码是否真的分开（IoU 越低说明分得越开）。"""
    prompt_a = [build_prompt(object_name, affordance_a)] * images.shape[0]
    prompt_b = [build_prompt(object_name, affordance_b)] * images.shape[0]

    map_a = torch.sigmoid(model(images, prompt_a)).cpu().numpy()[:, 0]
    map_b = torch.sigmoid(model(images, prompt_b)).cpu().numpy()[:, 0]

    scores = []
    for a, b in zip(map_a, map_b):
        binary_a = a >= 0.5
        binary_b = b >= 0.5
        union = np.logical_or(binary_a, binary_b).sum()
        scores.append(float(np.logical_and(binary_a, binary_b).sum() / union) if union else float("nan"))
    return scores


def prior_ratio(iou_e0: float, iou_e2: float) -> float:
    """Prior Ratio = IoU(E0)/IoU(E2)；>0.7 说明大部分性能可由类别先验解释。"""
    if iou_e2 <= 0:
        return float("nan")
    return iou_e0 / iou_e2

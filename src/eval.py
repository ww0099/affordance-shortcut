"""可供性热图评测指标：IoU / NSS / KLD / SIM。

约定：pred 与 gt 均为 [0,1] 区间的二维数组（热图概率或归一化显著性）。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

EPS = 1e-8


def _as_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    low, high = float(heatmap.min()), float(heatmap.max())
    if high - low < EPS:
        return np.zeros_like(heatmap)
    return (heatmap - low) / (high - low)


def iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    binary_pred = pred >= threshold
    binary_gt = gt >= 0.5
    union = np.logical_or(binary_pred, binary_gt).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(binary_pred, binary_gt).sum() / union)


def nss(pred: np.ndarray, gt: np.ndarray) -> float:
    heatmap = _normalize_heatmap(pred)
    mask = (gt >= 0.5).astype(np.float32)
    total = mask.sum()
    if total < EPS:
        return float("nan")
    standardized = (heatmap - heatmap.mean()) / (heatmap.std() + EPS)
    return float((standardized * mask).sum() / total)


def kld(pred: np.ndarray, gt: np.ndarray) -> float:
    heatmap = _normalize_heatmap(pred)
    mask = (gt >= 0.5).astype(np.float32)
    if mask.sum() < EPS:
        return float("nan")
    p = mask / (mask.sum() + EPS)
    q = heatmap / (heatmap.sum() + EPS)
    return float((p * np.log((p + EPS) / (q + EPS))).sum())


def sim(pred: np.ndarray, gt: np.ndarray) -> float:
    heatmap = _normalize_heatmap(pred)
    mask = (gt >= 0.5).astype(np.float32)
    if mask.sum() < EPS:
        return float("nan")
    mask = mask / (mask.sum() + EPS)
    q = heatmap / (heatmap.sum() + EPS)
    return float(np.minimum(mask, q).sum())


METRIC_FUNCTIONS = {"iou": iou, "nss": nss, "kld": kld, "sim": sim}


def mean_metrics(preds: Iterable[np.ndarray], gts: Iterable[np.ndarray], metrics: list[str], iou_threshold: float = 0.5) -> dict[str, float]:
    preds = [_as_array(p) for p in preds]
    gts = [_as_array(g) for g in gts]
    if len(preds) != len(gts):
        raise ValueError("preds 与 gts 数量不一致")
    if not preds:
        return {}

    results: dict[str, float] = {}
    for name in metrics:
        func = METRIC_FUNCTIONS[name]
        values = []
        for pred, gt in zip(preds, gts):
            value = func(pred, gt, iou_threshold) if name == "iou" else func(pred, gt)
            if not np.isnan(value):
                values.append(value)
        results[name] = float(np.mean(values)) if values else float("nan")
    results["count"] = len(preds)
    return results

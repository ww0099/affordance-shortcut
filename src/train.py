"""E2 主模型训练入口。

用法（Kaggle / 有 GPU 的环境）：
    python -m src.train --config configs/base.yaml
    python -m src.train --config configs/base.yaml --train-split smoke_train --val-split smoke_val
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import AffordanceDataset, read_meta
from .eval import mean_metrics
from .model import AffordanceModel, count_trainable_parameters
from .paths import load_config, project_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: AffordanceModel, loader: DataLoader, device: torch.device, metrics: list[str], threshold: float) -> dict[str, float]:
    model.eval()
    preds, gts = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images, batch["prompt"])
        probs = torch.sigmoid(logits.float()).cpu()
        preds.extend(probs[:, 0].numpy())
        gts.extend(batch["mask"][:, 0].numpy())
    return mean_metrics(preds, gts, metrics, threshold)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="调试用：限制每轮步数")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    rows = read_meta(project_path(config["data"]["meta_csv"]))
    train_rows = [row for row in rows if row.get("split") == args.train_split]
    val_rows = [row for row in rows if row.get("split") == args.val_split]
    if not train_rows:
        raise SystemExit(f"没有 split={args.train_split} 的样本，先检查 meta.csv")

    data_root = config["data"].get("root")
    image_size = int(config["data"]["image_size"])
    train_set = AffordanceDataset(train_rows, data_root=data_root, image_size=image_size)
    val_set = AffordanceDataset(val_rows, data_root=data_root, image_size=image_size) if val_rows else None

    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=int(config["data"]["num_workers"]), drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=int(config["data"]["num_workers"])) if val_set else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AffordanceModel(
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        decoder_channels=int(config["model"]["decoder_channels"]),
        decoder_dropout=float(config["model"]["decoder_dropout"]),
    ).to(device)

    print(f"device={device} | train={len(train_set)} | val={len(val_set) if val_set else 0}", flush=True)
    print(f"trainable_params={count_trainable_parameters(model.decoder):,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    epochs = args.epochs or int(config["train"]["epochs"])
    use_amp = bool(config["train"].get("amp")) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = project_path(config["output"]["dir"], time.strftime("e2_%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = project_path(config["output"]["save_metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        model.backbone.eval()
        running = 0.0
        steps = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images, batch["prompt"])
                loss = F.binary_cross_entropy_with_logits(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), float(config["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach())
            steps += 1
            progress.set_postfix(loss=running / steps)

            if args.max_steps and step >= args.max_steps:
                break

        row = {"epoch": epoch, "train_loss": round(running / max(steps, 1), 6)}
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, list(config["eval"]["metrics"]), float(config["eval"]["iou_threshold"]))
            row.update({key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()})
        print(row, flush=True)

        torch.save(
            {
                "decoder": model.decoder.state_dict(),
                "config": config,
                "epoch": epoch,
                "metrics": row,
            },
            out_dir / "decoder.pth",
        )

        file_exists = metrics_path.exists()
        with open(metrics_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    print(f"训练完成，产物：{out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

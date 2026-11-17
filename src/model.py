"""E2 主模型：冻结 CLIP 图像塔 + 轻量 FiLM 解码器。

设计目标（来自 KAGGLE_RESEARCH_PLAN 2.3 节）：
- backbone 冻结，只训练 2–5M 参数解码器；
- 文本用 CLIP text encoder 编码后调制图像 patch 特征；
- 输出 224×224 的可供性热图 logits。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrozenCLIPBackbone(nn.Module):
    """open_clip 的冻结图像/文本塔，暴露 patch tokens。"""

    def __init__(self, backbone: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k") -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("需要 open_clip_torch：pip install open_clip_torch") from exc

        self.model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(backbone)

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

        with torch.no_grad():
            probe = self.model.encode_text(self.tokenizer(["a probe prompt"]))
        self.text_dim = int(probe.shape[-1])
        self.image_dim = int(self.model.visual.conv1.out_channels)

    def train(self, mode: bool = True) -> "FrozenCLIPBackbone":
        # backbone 永远保持 eval（冻结 + 不更新 BN 统计）
        super().train(False)
        return self

    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(prompts))
        tokens = tokens.to(next(self.model.parameters()).device)
        return self.model.encode_text(tokens)

    def patch_grid(self, images: torch.Tensor) -> tuple[torch.Tensor, int]:
        """返回 (B, D, S, S) 的 patch 特征网格。

        open_clip 不同版本的视觉塔实现不同，按可用性依次回退：
        1) TimmModel：visual.trunk.forward_features；
        2) 自定义 VisionTransformer：visual.forward_intermediates（>= 2.32）；
        3) 旧版 CLIPVisionTransformer：手动 patch 化 + transformer。
        """
        visual = self.model.visual

        trunk = getattr(visual, "trunk", None)
        if trunk is not None and hasattr(trunk, "forward_features"):
            features = trunk.forward_features(images)
            if isinstance(features, dict):
                features = features.get("x")
            if features is None:
                raise RuntimeError("无法从 CLIP trunk 取 patch 特征")
            num_prefix = int(getattr(trunk, "num_prefix_tokens", 1))
            return self._tokens_to_grid(features[:, num_prefix:, :])

        if hasattr(visual, "forward_intermediates"):
            output = visual.forward_intermediates(
                images,
                indices=1,
                normalize_intermediates=True,
                intermediates_only=True,
                output_fmt="NCHW",
            )
            grid = output["image_intermediates"][-1]
            side = int(math.isqrt(grid.shape[-2] * grid.shape[-1]))
            return grid, side

        if (
            hasattr(visual, "conv1")
            and hasattr(visual, "transformer")
            and hasattr(visual, "class_embedding")
        ):
            x = visual.conv1(images)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x).permute(1, 0, 2)
            x = visual.transformer(x)
            if isinstance(x, (tuple, list)):
                x = x[0]
            return self._tokens_to_grid(x.permute(1, 0, 2)[:, 1:, :])

        raise RuntimeError("无法从当前 open_clip 视觉塔取 patch 特征")

    @staticmethod
    def _tokens_to_grid(tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
        batch, num_tokens, dim = tokens.shape
        side = int(math.isqrt(num_tokens))
        if side * side != num_tokens:
            raise RuntimeError(f"patch 数量 {num_tokens} 不是完全平方数，无法还原网格")
        return tokens.transpose(1, 2).reshape(batch, dim, side, side), side


class AffordanceDecoder(nn.Module):
    """文本调制的轻量卷积解码器。"""

    def __init__(self, image_dim: int, text_dim: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden * 2),
        )
        self.proj_in = nn.Conv2d(image_dim, hidden, 1)
        self.block1 = ConvBlock(hidden, hidden)
        self.block2 = ConvBlock(hidden, hidden)
        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, grid: torch.Tensor, text_embedding: torch.Tensor, output_size: int | None = None) -> torch.Tensor:
        x = self.proj_in(grid)
        gamma, beta = self.text_mlp(text_embedding).chunk(2, dim=1)
        x = x * (1 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)

        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dropout(x)
        logits = self.head(x)

        target = output_size or logits.shape[-1]
        if logits.shape[-1] != target:
            logits = F.interpolate(logits, size=(target, target), mode="bilinear", align_corners=False)
        return logits


class AffordanceModel(nn.Module):
    def __init__(
        self,
        backbone: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        decoder_channels: int = 128,
        decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = FrozenCLIPBackbone(backbone, pretrained)
        self.decoder = AffordanceDecoder(
            image_dim=self.backbone.image_dim,
            text_dim=self.backbone.text_dim,
            hidden=decoder_channels,
            dropout=decoder_dropout,
        )

    def forward(self, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        with torch.no_grad():
            grid, _ = self.backbone.patch_grid(images)
            text_embedding = self.backbone.encode_text(prompts)
        return self.decoder(grid, text_embedding, output_size=images.shape[-1])

    def trainable_parameters(self):
        return [p for p in self.decoder.parameters() if p.requires_grad]


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

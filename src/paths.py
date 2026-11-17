"""路径与配置解析。项目内不硬编码绝对路径。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_root(explicit: str | None = None) -> Path:
    """原始数据根目录：显式参数 > AGD_DATA_ROOT > 工作区 downloads/affordance-datasets。"""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("AGD_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (PROJECT_ROOT.parent / "downloads" / "affordance-datasets").resolve()


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = project_path(str(path))
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式错误：{path}")
    return config

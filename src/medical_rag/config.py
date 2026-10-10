# -*- coding: utf-8 -*-
"""配置读取：把 configs/config.yaml 读成支持点号访问的嵌套字典。

用法：
    from medical_rag.config import load_config
    cfg = load_config()
    cfg.get("chunking.chunk_size", 400)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

# 仓库根目录：src/medical_rag/config.py 往上三级
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


class Config(Mapping):
    """只读的配置树，支持 "a.b.c" 路径取值与命令行覆盖。"""

    def __init__(self, data: Dict[str, Any], source: Optional[Path] = None):
        self._data = data
        self.source = source

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在：{config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls(data, config_path)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """按 "section.key" 取值，缺失时返回 default（而不是抛异常）。"""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"配置项缺失：{dotted_key}（检查 {self.source}）")
        return value

    def set(self, dotted_key: str, value: Any) -> None:
        """命令行覆盖写场景；路径上的中间节点按需创建。"""
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    # Mapping 接口，只读
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _to_path(value: str | Path, root: Path = ROOT_DIR) -> Path:
    """配置里的相对路径统一按仓库根目录解析。"""
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def load_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """读取配置并应用命令行覆盖项。overrides 形如 {"paths.data_dir": "data/raw"}。"""
    cfg = Config.load(path)
    for dotted_key, value in (overrides or {}).items():
        if value is not None:
            cfg.set(dotted_key, value)
    return cfg


def resolve_path(value: str | Path, root: Path = ROOT_DIR) -> Path:
    """公开版本，供脚本层复用相对路径解析规则。"""
    return _to_path(value, root)

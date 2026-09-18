"""公共工具: 配置加载 / 日志 / 目录管理"""
import logging
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    """读取 config.yaml, 返回 dict"""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def ensure_dirs(cfg):
    """按配置创建 results 目录, 返回绝对路径 dict"""
    paths = {}
    for key, rel in cfg["paths"].items():
        p = PROJECT_ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        paths[key] = p
    return paths


def subject_list(cfg):
    """批量处理的被试编号列表(排除采样率异常者)"""
    exclude = set(cfg["data"]["exclude_subjects"])
    return [s for s in range(1, cfg["data"]["n_subjects"] + 1) if s not in exclude]


def epochs_path(paths, subject):
    return paths["processed_dir"] / f"S{subject:03d}-epo.fif"


def raw_path(paths, subject):
    return paths["processed_dir"] / f"S{subject:03d}_clean_raw.fif"

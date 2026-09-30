"""数据管线：edf → 标准化通道名 → 电极坐标 → z-score/截幅 → 切窗缓存。

两种数据集：
- PretrainSet：每个样本 (x[C,T,P], coords[C,3])，无标签，全部 run 都用。
- ProbeSet：只用 run 4/8/12（左右手想象），提示后 probe_tmax 秒，标签 0=左 1=右。

首次运行按受试者缓存到 cache/，之后直接读 npy。
"""
from pathlib import Path

import mne
import numpy as np
import torch
from torch.utils.data import Dataset

from config import Cfg

mne.set_log_level("ERROR")

MI_RUNS = (4, 8, 12)              # 左右手想象
_MONTAGE = None


def _montage():
    global _MONTAGE
    if _MONTAGE is None:
        _MONTAGE = mne.channels.make_standard_montage("colin27_1005")
    return _MONTAGE


def load_run(cfg: Cfg, subj: int, run: int):
    """返回 data[C,N]（已 z-score 截幅）、coords[C,3]（单位 dm，量级 ±1）、events。"""
    f = cfg.data_root / f"S{subj:03d}" / f"S{subj:03d}R{run:02d}.edf"
    raw = mne.io.read_raw_edf(f, preload=True)
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage(_montage())
    assert int(raw.info["sfreq"]) == cfg.fs, (f, raw.info["sfreq"])
    x = raw.get_data().astype(np.float32)                  # [C,N]，单位 V
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    x = np.clip(x, -cfg.clip_sigma, cfg.clip_sigma)
    coords = np.array([c["loc"][:3] for c in raw.info["chs"]], dtype=np.float32) * 10  # m → dm
    events, ids = mne.events_from_annotations(raw)
    return x, coords, events, ids


# ---------------------------------------------------------------- 预训练

def _cache_pretrain(cfg: Cfg, subj: int) -> Path:
    # 本地只有部分 run 时只用已有的；文件名带 run 数，补下载后缓存自动失效
    runs = [r for r in range(1, 15) if (cfg.data_root / f"S{subj:03d}" / f"S{subj:03d}R{r:02d}.edf").exists()]
    assert runs, f"S{subj:03d} 没有任何 edf"
    out = cfg.cache_dir / f"pre_S{subj:03d}_r{len(runs)}.npz"
    if out.exists():
        return out
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    wins, coords = [], None
    L = cfg.T * cfg.P
    for run in runs:
        x, coords, _, _ = load_run(cfg, subj, run)
        C, N = x.shape
        n = N // L
        if n == 0:
            continue
        wins.append(x[:, : n * L].reshape(C, n, cfg.T, cfg.P).transpose(1, 0, 2, 3))  # [n,C,T,P]
    np.savez(out, x=np.concatenate(wins, 0), coords=coords)
    return out


class PretrainSet(Dataset):
    def __init__(self, cfg: Cfg, which: str = "train"):
        self.cfg = cfg
        subs = cfg.subjects(which)
        if which == "train" and cfg.n_pretrain_subjects > 0:
            subs = subs[: cfg.n_pretrain_subjects]   # 数据量消融：只缩预训练人数，探针池不变
        xs, cs = [], []
        for s in subs:
            z = np.load(_cache_pretrain(cfg, s))
            xs.append(z["x"])
            cs.append(np.repeat(z["coords"][None], len(z["x"]), 0))
        self.x = np.concatenate(xs, 0)          # [N,C,T,P]
        self.coords = np.concatenate(cs, 0)     # [N,C,3]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.coords[i])


# ---------------------------------------------------------------- 探针

def _cache_probe(cfg: Cfg, subj: int) -> Path:
    out = cfg.cache_dir / f"mi_S{subj:03d}.npz"
    if out.exists():
        return out
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    xs, ys, coords = [], [], None
    n_tok = int(cfg.probe_tmax)
    L = n_tok * cfg.P
    for run in MI_RUNS:
        x, coords, events, ids = load_run(cfg, subj, run)
        for onset, _, code in events:
            if code == ids.get("T0"):
                continue
            seg = x[:, onset : onset + L]
            if seg.shape[1] < L:
                continue
            xs.append(seg.reshape(x.shape[0], n_tok, cfg.P))
            ys.append(0 if code == ids["T1"] else 1)
    np.savez(out, x=np.stack(xs), y=np.array(ys, dtype=np.int64), coords=coords)
    return out


class ProbeSet(Dataset):
    def __init__(self, cfg: Cfg, which: str):
        xs, ys, cs, subj = [], [], [], []
        for s in cfg.subjects(which):
            z = np.load(_cache_probe(cfg, s))
            xs.append(z["x"]); ys.append(z["y"])
            cs.append(np.repeat(z["coords"][None], len(z["x"]), 0))
            subj += [s] * len(z["x"])
        self.x = np.concatenate(xs, 0)          # [N,C,T,P]
        self.y = np.concatenate(ys, 0)
        self.coords = np.concatenate(cs, 0)
        self.subj = np.array(subj)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.coords[i]), int(self.y[i])


if __name__ == "__main__":
    cfg = Cfg()
    ds = PretrainSet(cfg, "train")
    print("pretrain", ds.x.shape, ds.coords.shape, "GB", ds.x.nbytes / 1e9)
    ps = ProbeSet(cfg, "test")
    print("probe", ps.x.shape, ps.y.mean())

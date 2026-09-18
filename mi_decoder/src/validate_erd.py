"""模块2: ERD 神经生理验证(解码前的质量门禁)

目的: 证明信号里真的存在可解码的神经活动, 且不是伪迹。
验收标准:
- 左手想象 -> C4(右脑, 对侧)mu/beta 功率下降(时频图蓝色区域)
- 右手想象 -> C3 下降
- 地形图上 ERD 聚焦中央区(全头弥漫说明伪迹未除净)

ERD属于感应(induced)活动: 相位不锁定, 必须"逐试次算功率再平均",
时域平均(ERP)看不到它 —— 所以这里用时频分析而非evoked。
"""
import matplotlib
matplotlib.use("Agg")  # 批量出图不弹窗
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC",
                                   "PingFang SC"]  # 中文字体, 避免标题豆腐块
plt.rcParams["axes.unicode_minus"] = False           # 负号正常显示
import mne
import numpy as np

from .utils import (ensure_dirs, epochs_path, get_logger, load_config,
                    subject_list)

log = get_logger("validate_erd")


def compute_tfr(epochs, freqs, n_cycles):
    """Morlet小波时频变换(兼容新旧MNE API)"""
    try:                                          # MNE >= 1.7
        return epochs.compute_tfr("morlet", freqs=freqs, n_cycles=n_cycles,
                                  return_itc=False, average=True)
    except (AttributeError, TypeError):           # 旧版本
        from mne.time_frequency import tfr_morlet
        return tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                          return_itc=False)


def plot_erd(subject, epochs, cfg, fig_dir):
    """输出两张图: C3/C4时频图(左右手对比) + mu频段地形图"""
    e = cfg["erd"]
    freqs = np.arange(e["freq_min"], e["freq_max"], 1)
    n_cycles = freqs / 2.0                        # 低频窗长、高频窗短(小波自适应)
    baseline = tuple(e["baseline"])
    chans = e["plot_channels"]                    # [C3, C4]

    # ---- 图1: 2x2 时频矩阵(行=想象侧, 列=电极) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for i, cond in enumerate(["left", "right"]):
        tfr = compute_tfr(epochs[cond], freqs, n_cycles)
        tfr.apply_baseline(baseline, mode="percent")   # 相对基线的功率变化
        for j, ch in enumerate(chans):
            tfr.plot(picks=[ch], axes=axes[i, j], colorbar=(j == 1),
                     show=False, verbose=False)
            expect = " <-- 期望ERD" if (cond, ch) in [("left", "C4"),
                                                      ("right", "C3")] else ""
            axes[i, j].set_title(f"{cond}想象 @ {ch}{expect}")
    fig.suptitle(f"S{subject:03d} ERD验证: 对侧mu/beta功率应下降(蓝色)")
    fig.tight_layout()
    f1 = fig_dir / f"S{subject:03d}_erd_tfr.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # ---- 图2: mu频段(8-13Hz)想象期地形图 ----
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, cond in enumerate(["left", "right"]):
        tfr = compute_tfr(epochs[cond], freqs, n_cycles)
        tfr.apply_baseline(baseline, mode="percent")
        tfr.plot_topomap(tmin=0.5, tmax=3.5, fmin=8, fmax=13,
                         axes=axes[i], show=False)
        axes[i].set_title(f"{cond}想象 mu-ERD地形")
    fig.tight_layout()
    f2 = fig_dir / f"S{subject:03d}_erd_topo.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    log.info("S%03d 图已存: %s / %s", subject, f1.name, f2.name)


def main():
    cfg = load_config()
    paths = ensure_dirs(cfg)
    for sub in subject_list(cfg):
        fpath = epochs_path(paths, sub)
        if not fpath.exists():
            log.warning("S%03d epochs不存在, 先跑 python -m src.preprocess", sub)
            continue
        epochs = mne.read_epochs(fpath, preload=True, verbose=False)
        # 空/试次过少的被试跳过(如全部被350uV阈值剔除者 kept=0),
        # 否则 epochs[cond] 会因 event_id 为空抛 KeyError, 中断整个批量
        eid = epochs.event_id or {}
        n_by_cond = {c: int((epochs.events[:, -1] == eid[c]).sum())
                     for c in ("left", "right") if c in eid}
        if len(epochs) == 0 or min(n_by_cond.values(), default=0) < 5:
            log.warning("S%03d 有效试次过少 %s, 跳过ERD验证", sub, n_by_cond)
            continue
        plot_erd(sub, epochs, cfg, paths["fig_dir"])


if __name__ == "__main__":
    main()

"""模块2b: ERD 定量表(把"看图判蓝"换成"数值门禁")

validate_erd.py 只出时频图/地形图, 肉眼受 colorbar 自动缩放影响易误判;
本模块对每个被试计算 C3/C4 在 mu(8-13)/beta(13-30) 频段、想象期(0.5-3.5s)
相对基线(-1~0s)的功率变化百分比, 并按对侧/同侧配对给出侧化指标。

计算流程(ERD是非锁相感应活动, 顺序不可颠倒):
  逐试次Morlet功率 -> 跨试次平均 -> percent基线归一 -> 取频段×时间窗均值 -> ×100

对侧/同侧定义(左脑C3支配右手, 右脑C4支配左手):
  contra = mean(左手@C4, 右手@C3)   # 期望显著为负(功率下降)
  ipsi   = mean(左手@C3, 右手@C4)
  lat    = contra - ipsi            # 期望为负(对侧比同侧更抑制)

输出: results/metrics/erd_quant.csv
"""
import mne
import numpy as np
import pandas as pd

from .utils import (ensure_dirs, epochs_path, get_logger, load_config,
                    subject_list)

log = get_logger("quantify_erd")

# 目标频段(与config的erd扫频6-32Hz一致, 这里只取判读用的mu/beta)
BANDS = {"mu": (8, 13), "beta": (13, 30)}
TIME_WIN = (0.5, 3.5)          # 想象期稳定ERD窗
BASELINE = (-1.0, 0.0)         # 基线窗(想象开始前)
MIN_TRIALS_PER_COND = 5        # 单条件最少试次, 少于此判为不可靠


def band_erd_pct(epochs, cond, ch, band):
    """某条件、某电极、某频段的平均ERD百分比。

    返回 float(百分比, 负=功率下降/ERD); 试次不足返回 nan。
    """
    e = epochs[cond]
    if len(e) < 3:
        return np.nan
    freqs = np.arange(band[0], band[1] + 1, 1.0)
    n_cycles = freqs / 2.0                     # 小波自适应窗长
    tfr = e.compute_tfr("morlet", freqs=freqs, n_cycles=n_cycles,
                        return_itc=False, average=True, decim=2)
    tfr.apply_baseline(BASELINE, mode="percent")   # 返回分数, ×100转百分比
    ti = np.where((tfr.times >= TIME_WIN[0]) & (tfr.times <= TIME_WIN[1]))[0]
    ci = tfr.ch_names.index(ch)
    return float(tfr.data[ci][:, ti].mean()) * 100.0


def quantify_subject(epochs, sub):
    """单被试定量表一行: mu/beta 的左右手C3C4 + 对侧/同侧/侧化"""
    eid = epochs.event_id or {}
    if "left" not in eid or "right" not in eid:
        return None
    n_l = int((epochs.events[:, -1] == eid["left"]).sum())
    n_r = int((epochs.events[:, -1] == eid["right"]).sum())
    if min(n_l, n_r) < MIN_TRIALS_PER_COND:
        log.warning("S%03d 单条件试次过少(L=%d,R=%d), 跳过", sub, n_l, n_r)
        return None

    row = dict(subject=sub, n_left=n_l, n_right=n_r)
    for bname, band in BANDS.items():
        # 四格: 左/右手 × C3/C4
        lc4 = band_erd_pct(epochs, "left", "C4", band)   # 左手对侧
        rc3 = band_erd_pct(epochs, "right", "C3", band)  # 右手对侧
        lc3 = band_erd_pct(epochs, "left", "C3", band)   # 左手同侧
        rc4 = band_erd_pct(epochs, "right", "C4", band)  # 右手同侧
        contra = np.nanmean([lc4, rc3])
        ipsi = np.nanmean([lc3, rc4])
        row[f"{bname}_contra_pct"] = round(float(contra), 2)
        row[f"{bname}_ipsi_pct"] = round(float(ipsi), 2)
        row[f"{bname}_lat_pct"] = round(float(contra - ipsi), 2)
    return row


def classify(contra_pct):
    """按对侧mu-ERD强度分档(供报告筛选有效被试)"""
    if np.isnan(contra_pct):
        return "nan"
    if contra_pct <= -20:
        return "strong"          # 强ERD
    if contra_pct <= -10:
        return "medium"          # 中ERD
    if contra_pct <= 0:
        return "weak"            # 弱ERD
    return "none_or_reversed"    # 无/反向: illiteracy或伪迹


def main():
    cfg = load_config()
    paths = ensure_dirs(cfg)
    mne.set_log_level("ERROR")

    records = []
    for sub in subject_list(cfg):
        fpath = epochs_path(paths, sub)
        if not fpath.exists():
            continue
        epochs = mne.read_epochs(fpath, preload=True, verbose=False)
        row = quantify_subject(epochs, sub)
        if row is not None:
            records.append(row)

    df = pd.DataFrame(records)
    df["mu_class"] = df["mu_contra_pct"].apply(classify)
    out = paths["metrics_dir"] / "erd_quant.csv"
    df.to_csv(out, index=False)

    # 汇总打印
    c = df["mu_contra_pct"].dropna()
    log.info("有效被试 %d | 对侧mu-ERD 均值%.1f%% 中位%.1f%% std%.1f%%",
             len(df), c.mean(), c.median(), c.std())
    log.info("分档: %s", df["mu_class"].value_counts().to_dict())
    log.info("侧化为负(方向正确): %d/%d",
             int((df["mu_lat_pct"] < 0).sum()), len(df))
    log.info("定量表已存: %s", out)


if __name__ == "__main__":
    main()

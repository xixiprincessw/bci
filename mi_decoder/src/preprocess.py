"""模块1: 自动化预处理流水线(批量)

七步顺序: 电极定位 -> 插值坏导 -> 重参考 -> 滤波 -> ICA去伪迹 -> Epoch
(第7步"平均"属于ERP分析, MI解码不需要; ERD验证见模块2)

关键设计:
- 坏导用通道对数方差z分数自动检测(批量处理不靠肉眼)
- 插值必须在平均参考之前, 防止坏导拉偏参考
- MI任务不做基线校正(特征是频带功率, 不是电位)
- 同时保存清洗后的连续raw(供模块5伪在线重放)和epochs(供模块3/4)
"""
import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci
from mne.preprocessing import ICA

from .utils import (ensure_dirs, epochs_path, get_logger, load_config,
                    raw_path, subject_list)

log = get_logger("preprocess")


def auto_detect_bads(raw, z_thresh=3.0):
    """基于通道对数方差z分数的坏导自动检测。

    原理: 接触不良的通道方差异常大(噪声)或异常小(近乎断连),
    对数方差在好通道间近似正态, 偏离中位数超过阈值即判坏导。
    """
    data = raw.get_data(picks="eeg")
    ch_var = np.log(np.var(data, axis=1) + 1e-20)
    z = (ch_var - np.median(ch_var)) / (np.std(ch_var) + 1e-20)
    picks = mne.pick_types(raw.info, eeg=True)
    return [raw.ch_names[picks[i]] for i in np.where(np.abs(z) > z_thresh)[0]]


def preprocess_subject(subject, cfg):
    """单被试完整流水线, 返回 (raw_clean, epochs, 日志dict)"""
    p = cfg["preprocess"]

    # ---- 加载 + 电极定位 ----
    # 直接从 data_root 按文件名读取已下载的EDF(离线, 不依赖MNE_DATA/联网)
    root = Path(cfg["data"]["data_root"])
    files = [root / f"S{subject:03d}" / f"S{subject:03d}R{r:02d}.edf"
             for r in cfg["data"]["runs"]]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"S{subject:03d} 缺少EDF: {missing}")
    raw = mne.concatenate_raws(
        [mne.io.read_raw_edf(f, preload=True) for f in files])
    eegbci.standardize(raw)                       # 规范通道名(如 Fc5. -> FC5)
    raw.set_montage("standard_1005")              # 三维坐标: 插值/地形图的几何基础

    # ---- 自动坏导检测 + 插值(必须在重参考之前) ----
    raw.info["bads"] = auto_detect_bads(raw, p["bad_z_thresh"])
    bads = list(raw.info["bads"])
    if bads:
        raw.interpolate_bads(reset_bads=True)     # 球面样条插值

    # ---- 平均参考 ----
    raw.set_eeg_reference("average", projection=False)

    # ---- 宽带滤波(1Hz高通兼顾去漂移与ICA分解质量) ----
    raw.filter(l_freq=p["l_freq"], h_freq=p["h_freq"])

    # ---- ICA 去眼电(连续数据上拟合, 分解更稳定) ----
    ica = ICA(n_components=p["ica_n_components"], random_state=42,
              max_iter="auto")
    ica.fit(raw)
    eog_idx = []
    for ch in p["eog_proxy_channels"]:            # 额区电极做EOG代理
        idx, _ = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
        eog_idx.extend(idx)
    eog_idx = sorted(set(eog_idx))
    ica.exclude = eog_idx
    ica.apply(raw)

    # ---- Epoch(以事件为零点对齐试次; MI不做基线校正) ----
    events, _ = mne.events_from_annotations(raw, verbose=False)
    event_id = dict(cfg["data"]["event_id"])      # {'left':2, 'right':3}
    epochs = mne.Epochs(raw, events, event_id,
                        tmin=p["epoch_tmin"], tmax=p["epoch_tmax"],
                        baseline=None, preload=True,
                        reject=dict(eeg=p["reject_eeg"]), verbose=False)

    n_total = int(np.isin(events[:, -1], list(event_id.values())).sum())
    info = dict(subject=subject, n_bads=len(bads), bads=";".join(bads),
                n_ica_excluded=len(eog_idx),
                n_epochs_kept=len(epochs), n_epochs_total=n_total,
                sfreq=raw.info["sfreq"])
    log.info("S%03d | bads=%d %s | ICA excl=%d | epochs %d/%d",
             subject, len(bads), bads, len(eog_idx), len(epochs), n_total)
    return raw, epochs, info


def main():
    parser = argparse.ArgumentParser(description="模块1: 批量预处理")
    parser.add_argument("--subjects", type=int, nargs="*", default=None,
                        help="指定被试编号; 缺省按config批量")
    args = parser.parse_args()

    cfg = load_config()
    paths = ensure_dirs(cfg)
    subjects = args.subjects or subject_list(cfg)

    records = []
    for sub in subjects:
        try:
            raw, epochs, info = preprocess_subject(sub, cfg)
            raw.save(raw_path(paths, sub), overwrite=True)
            epochs.save(epochs_path(paths, sub), overwrite=True)
            info["status"] = "ok"
        except Exception as e:                    # 单被试失败不阻断批量
            log.error("S%03d 失败: %s", sub, e)
            info = dict(subject=sub, status=f"error: {e}")
        records.append(info)

    df = pd.DataFrame(records)
    out = paths["metrics_dir"] / "preprocess_log.csv"
    df.to_csv(out, index=False)
    log.info("完成 %d 个被试, 日志已存 %s", len(subjects), out)


if __name__ == "__main__":
    main()

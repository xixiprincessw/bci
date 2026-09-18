"""模块5: 伪在线验证 —— 离线冠军方案在因果约束下还剩多少

核心思想: 在线系统只能看到"过去", 所以
- 滤波必须换成因果实现(sosfilt), 不能用离线的零相位filtfilt
- 训练也在因果滤波后的数据上做(训练/推理一致性, 定版原则)
- 测试试次按时间顺序流式重放, 滑动窗口盲判

被试内协议: 每人前 70% 试次做校准训练, 后 30% 试次 + 同期静息段做流式测试。
模型不落盘(训->评->记数字), 部署需另加 joblib.dump。

回答四个问题:
1. 因果约束下性能存活率(对比模块3/4的离线acc)
2. 延迟-精度曲线: 想象开始后多久能判对(offset扫描)
3. 静息期误触发率: 被试没想象时系统会不会乱输出
4. ITR(信息传输率, bits/min): BCI通信效率的通用指标

运行:
    python -m src.pseudo_online                   # CSP+LDA, 88人约2分钟
    python -m src.pseudo_online --model eegnet    # EEGNet, 每人训一次, 约20-30分钟
    python -m src.pseudo_online --max-subjects 3  # 冒烟

--model eegnet 与 CSP 的三处不同(其余口径完全一致):
- 输入长度固定 -> 训练窗与测试窗同为 win_sec, 在 crop 内滑窗取样(每试次约5窗)
- 频带 eegnet.band(1-40Hz 宽带), CSP 用 decode.band(8-30Hz)
- 早停验证集从 70% 训练试次中**按试次**留出(同一试次的滑窗不能一半训一半验)
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC",
                                   "PingFang SC"]  # 中文字体, 避免标题豆腐块
plt.rcParams["axes.unicode_minus"] = False           # 负号正常显示
import mne
import pandas as pd

from .online_models import (CSPLDA, EEGNetModel, load_stream, mi_events,
                            replay_stream, split_val_by_trial, summarize,
                            train_windows)
from .utils import (ensure_dirs, get_logger, load_config, raw_path,
                    subject_list)

log = get_logger("pseudo_online")
mne.set_log_level("WARNING")


def fit_online_model(stream, sfreq, train_ev, event_id, cfg, model_name, device):
    """在因果滤波数据的前70%试次上训练(与在线推理同一套预处理口径)"""
    if model_name == "csp":
        X, y, _ = train_windows(stream, sfreq, train_ev, event_id, cfg, slide=False)
        return CSPLDA(cfg["decode"]["csp_components"]).fit(X, y)
    X, y, tid = train_windows(stream, sfreq, train_ev, event_id, cfg, slide=True)
    fit_m, val_m = split_val_by_trial(tid, y, cfg["eegnet"]["val_ratio"],
                                      cfg["decode"]["random_state"])
    return EEGNetModel(cfg, device).fit(X[fit_m], y[fit_m], X[val_m], y[val_m])


def run_subject(subject, cfg, paths, model_name, device):
    """单被试完整伪在线评估, 返回汇总dict和曲线DataFrame"""
    band = cfg["decode"]["band"] if model_name == "csp" else cfg["eegnet"]["band"]
    stream, events, sfreq = load_stream(subject, cfg, paths, [band])
    event_id = dict(cfg["data"]["event_id"])

    # MI事件按时间顺序切分, 不能shuffle: 在线场景训练数据永远在测试数据之前
    trials = mi_events(events, event_id)
    n_train = int(len(trials) * cfg["pseudo_online"]["train_ratio"])
    train_ev, test_ev = trials[:n_train], trials[n_train:]
    if len(test_ev) < 5:
        raise ValueError(f"测试试次太少({len(test_ev)})")

    model = fit_online_model(stream, sfreq, train_ev, event_id, cfg,
                             model_name, device)
    curve, fpr, n_rest = replay_stream(stream, events, sfreq, event_id, model,
                                       cfg, test_start=test_ev[0][0])
    summary = dict(subject=subject, n_train=len(train_ev), n_test=len(test_ev),
                   **summarize(curve, fpr, n_rest))
    return summary, curve


def plot_curve(subject, curve, summary, fig_dir, suffix):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve.offset, curve.acc, "o-")
    ax.axhline(0.5, ls="--", c="gray", label="随机水平")
    ax.axvline(summary["best_offset"], ls=":", c="r",
               label=f"最佳 {summary['best_offset']:.2f}s "
                     f"acc={summary['best_acc']:.2f}")
    ax.set_xlabel("决策时刻(想象开始后, 秒)")
    ax.set_ylabel("流式准确率")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f"S{subject:03d} 延迟-精度曲线 | "
                 f"ITR={summary['itr_bits_min']:.1f} bits/min | "
                 f"静息误触发={summary['rest_fpr']:.2f}")
    ax.legend()
    fig.tight_layout()
    out = fig_dir / f"S{subject:03d}_pseudo_online{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["csp", "eegnet"], default="csp")
    ap.add_argument("--max-subjects", type=int, default=None, help="冒烟测试")
    args = ap.parse_args()

    matplotlib.use("Agg")                          # 批量出图不弹窗
    cfg = load_config()
    paths = ensure_dirs(cfg)
    suffix = "" if args.model == "csp" else f"_{args.model}"   # csp 沿用旧文件名

    device = None
    if args.model == "eegnet":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
        log.info("device = %s", device)

    records, curves = [], []
    for sub in subject_list(cfg):
        if not raw_path(paths, sub).exists():
            log.warning("S%03d clean raw不存在, 先跑 python -m src.preprocess",
                        sub)
            continue
        try:
            summary, curve = run_subject(sub, cfg, paths, args.model, device)
        except Exception as e:                     # 单被试失败不阻断
            log.error("S%03d 失败: %s", sub, e)
            continue
        plot_curve(sub, curve, summary, paths["fig_dir"], suffix)
        curve["subject"] = sub
        curves.append(curve)
        records.append(summary)
        log.info("S%03d | %s | acc_at_end=%.3f | best acc=%.3f @%.2fs | "
                 "ITR=%.1f | rest FPR=%.2f", sub, args.model,
                 summary["acc_at_end"], summary["best_acc"],
                 summary["best_offset"], summary["itr_bits_min"],
                 summary["rest_fpr"])
        if args.max_subjects and len(records) >= args.max_subjects:
            break

    df = pd.DataFrame(records)
    out = paths["metrics_dir"] / f"pseudo_online{suffix}_results.csv"
    df.to_csv(out, index=False)
    if curves:
        pd.concat(curves).to_csv(
            paths["metrics_dir"] / f"pseudo_online{suffix}_curves.csv", index=False)
    if len(df):
        log.info("均值(%s, n=%d): acc_at_end=%.3f | best acc=%.3f | "
                 "ITR=%.1f bits/min | rest FPR=%.2f -> %s", args.model, len(df),
                 df.acc_at_end.mean(), df.best_acc.mean(),
                 df.itr_bits_min.mean(), df.rest_fpr.mean(), out)


if __name__ == "__main__":
    main()

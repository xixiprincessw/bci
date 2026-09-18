"""文章配图: 中篇《批量验证、ERD 定量与离线-在线衔接》

只读 results/metrics/ 下已有的 csv, 不重跑任何解码。输出到 results/figures/:
  erd_vs_csp_acc.png                  §3   ERD 强度 vs CSP 准确率, 88 人散点
  causal_vs_zerophase.png             §6.2 因果 / 零相位滤波: 脉冲响应 + 包络延迟
  pseudo_online_csp_vs_eegnet.png     §6.6 106 人伪在线延迟-精度曲线, CSP vs EEGNet

运行: python -m src.plot_article
"""
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, hilbert, sosfilt, sosfiltfilt

from .utils import ensure_dirs, get_logger, load_config

log = get_logger("plot_article")

CLASS_STYLE = {  # mu_class -> (颜色, 图例)
    "strong":           ("tab:blue",   "strong (≤ −20%)"),
    "medium":           ("tab:green",  "medium (−20 ~ −10%)"),
    "weak":             ("tab:orange", "weak (−10 ~ 0%)"),
    "none_or_reversed": ("tab:red",    "none / reversed (> 0)"),
}


def plot_erd_vs_acc(paths):
    erd = pd.read_csv(paths["metrics_dir"] / "erd_quant.csv")
    csp = pd.read_csv(paths["metrics_dir"] / "csp_results.csv")
    d = erd.merge(csp, on="subject")
    r, p = stats.pearsonr(d.mu_contra_pct, d.csp_acc)

    fig, ax = plt.subplots(figsize=(7, 5))
    for cls, (c, label) in CLASS_STYLE.items():
        sub = d[d.mu_class == cls]
        ax.scatter(sub.mu_contra_pct, sub.csp_acc, c=c, s=40, alpha=0.8,
                   label=f"{label}  n={len(sub)}, acc={sub.csp_acc.mean():.3f}")
    k, b = np.polyfit(d.mu_contra_pct, d.csp_acc, 1)
    xs = np.linspace(d.mu_contra_pct.min(), d.mu_contra_pct.max(), 50)
    ax.plot(xs, k * xs + b, "k--", lw=1, alpha=0.6)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("mu_contra_pct  (对侧 C3/C4, 8~13 Hz, 0.5~3.5 s 功率变化 %)")
    ax.set_ylabel("CSP+LDA 5 折准确率")
    ax.set_title(f"ERD 越深, 解码越准 | {len(d)} 人, Pearson r = {r:.2f}")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = paths["fig_dir"] / "erd_vs_csp_acc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_causal_vs_zerophase(paths, fs=160):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # 左: 单个脉冲过 8~30 Hz 二阶巴特沃斯 (与正文 6.2 的数值表同一滤波器)
    sos = butter(2, [8, 30], btype="band", fs=fs, output="sos")
    n = 41
    x = np.zeros(n); x[n // 2] = 1.0
    t = np.arange(n) - n // 2
    ax = axes[0]
    ax.stem(t, sosfilt(sos, x), linefmt="C0-", markerfmt="C0o", basefmt=" ",
            label="因果 sosfilt")
    ax.stem(t + 0.25, sosfiltfilt(sos, x), linefmt="C3-", markerfmt="C3s",
            basefmt=" ", label="零相位 filtfilt")
    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlim(-8, 14)
    ax.set_xlabel("采样点 (脉冲在 t=0)")
    ax.set_ylabel("滤波输出")
    ax.set_title("脉冲响应: 零相位在 t<0 已有输出, 因果的全在 t≥0")
    ax.legend()
    ax.grid(alpha=0.3)

    # 右: 10 Hz 振荡在 1.0 s 幅度 1 -> 0.4, 看包络掉一半的时刻 (正文 6.4)
    sos4 = butter(4, [8, 30], btype="band", fs=fs, output="sos")
    tt = np.arange(0, 3, 1 / fs)
    amp = np.where(tt < 1.0, 1.0, 0.4)
    sig = amp * np.sin(2 * np.pi * 10 * tt)
    ax = axes[1]
    for y, c, label in [(sosfilt(sos4, sig), "C0", "因果 sosfilt"),
                        (sosfiltfilt(sos4, sig), "C3", "零相位 filtfilt")]:
        env = np.abs(hilbert(y))
        half = 0.5 * (env[(tt > 0.5) & (tt < 0.9)].mean() + env[(tt > 1.5) & (tt < 2.5)].mean())
        t_half = tt[(tt > 0.9) & (env < half)][0]
        ax.plot(tt, env, c, label=f"{label} 包络, 掉一半 @ {t_half*1000:.0f} ms")
        ax.axvline(t_half, color=c, ls=":", lw=1)
    ax.axvline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlim(0.7, 1.4)
    ax.set_xlabel("时间 (s)  |  真实幅度在 1.000 s 由 1 降到 0.4")
    ax.set_ylabel("Hilbert 包络")
    ax.set_title("包络延迟约 60 ms, 远小于 250 ms 决策步长")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = paths["fig_dir"] / "causal_vs_zerophase.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_pseudo_online_curves(paths):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for fname, c, label in [("pseudo_online_curves.csv", "C0", "CSP+LDA"),
                            ("pseudo_online_eegnet_curves.csv", "C3", "EEGNet")]:
        cur = pd.read_csv(paths["metrics_dir"] / fname)
        g = cur.groupby("offset").acc
        m, se = g.mean(), g.std() / np.sqrt(g.count())
        n = cur.subject.nunique()
        ax.plot(m.index, m.values, c, marker="o", ms=4,
                label=f"{label}  n={n}, 末端 {m.iloc[-1]:.3f}")
        ax.fill_between(m.index, m - se, m + se, color=c, alpha=0.15)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylim(0.45, 0.7)
    ax.set_xlabel("决策时刻 (想象开始后, 秒)")
    ax.set_ylabel("流式准确率 (被试均值 ± SEM)")
    ax.set_title("被试内伪在线: 同一条因果路径, CSP 稳、EEGNet 低且抖")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = paths["fig_dir"] / "pseudo_online_csp_vs_eegnet.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    cfg = load_config()
    paths = ensure_dirs(cfg)
    for fn in (plot_erd_vs_acc, plot_causal_vs_zerophase, plot_pseudo_online_curves):
        log.info("-> %s", fn(paths))


if __name__ == "__main__":
    main()

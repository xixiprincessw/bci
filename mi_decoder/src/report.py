"""模块6: 跨被试汇总报告

合并模块3/4/5的指标, 回答:
- CSP vs FBCSP vs EEGNet 谁赢? 差异显著吗(配对检验)?
- 被试间差异有多大(BCI illiteracy: 约15-20%的人MI信号弱)?
- 伪在线相比离线掉了多少点?

产出: 汇总csv + 配对散点图 + 分布直方图 + report.md
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC",
                                   "PingFang SC"]  # 中文字体, 避免标题豆腐块
plt.rcParams["axes.unicode_minus"] = False           # 负号正常显示
import numpy as np
import pandas as pd
from scipy import stats

from .utils import PROJECT_ROOT, ensure_dirs, get_logger, load_config

log = get_logger("report")


def load_all_metrics(metrics_dir):
    """合并三个模块的结果csv(缺哪个就跳过哪个)"""
    dfs = []
    for name in ["csp_results.csv", "eegnet_results.csv",
                 "pseudo_online_results.csv"]:
        f = metrics_dir / name
        if f.exists():
            dfs.append(pd.read_csv(f))
        else:
            log.warning("%s 不存在, 跳过(对应模块还没跑)", name)
    if not dfs:
        raise FileNotFoundError("metrics目录下没有任何结果csv")
    df = dfs[0]
    for other in dfs[1:]:
        df = df.merge(other, on="subject", how="outer")
    return df.sort_values("subject").reset_index(drop=True)


def paired_test(df, col_a, col_b, name_a, name_b):
    """配对Wilcoxon检验(小样本、不假设正态), 返回描述字符串"""
    sub = df[[col_a, col_b]].dropna()
    if len(sub) < 6:
        return f"{name_a} vs {name_b}: n={len(sub)} 太少, 不做检验"
    stat, p = stats.wilcoxon(sub[col_a], sub[col_b])
    diff = (sub[col_a] - sub[col_b]).mean()
    return (f"{name_a} vs {name_b}: 均值差={diff:+.3f}, "
            f"Wilcoxon p={p:.4f} (n={len(sub)})"
            + (" *显著*" if p < 0.05 else " 不显著"))


def plot_paired_scatter(df, fig_dir):
    """CSP vs EEGNet 配对散点(对角线以上=EEGNet赢)"""
    sub = df[["subject", "csp_acc", "eegnet_acc"]].dropna()
    if not len(sub):
        return None
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(sub.csp_acc, sub.eegnet_acc, s=50, alpha=0.8)
    for _, r in sub.iterrows():
        ax.annotate(f"S{int(r.subject)}", (r.csp_acc, r.eegnet_acc),
                    fontsize=7, alpha=0.6)
    lims = [0.4, 1.0]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlim(lims), ax.set_ylim(lims)
    ax.set_xlabel("CSP+LDA accuracy")
    ax.set_ylabel("EEGNet accuracy")
    ax.set_title("对角线以上: 深度学习赢; 以下: 经典方法赢")
    ax.legend()
    fig.tight_layout()
    out = fig_dir / "paired_csp_vs_eegnet.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_accuracy_hist(df, fig_dir):
    """被试准确率分布(观察BCI illiteracy现象)"""
    cols = [c for c in ["csp_acc", "fbcsp_acc", "eegnet_acc"]
            if c in df.columns]
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in cols:
        ax.hist(df[c].dropna(), bins=np.arange(0.4, 1.05, 0.05),
                alpha=0.5, label=c)
    ax.axvline(0.5, ls="--", c="gray")
    ax.axvline(0.7, ls=":", c="r", label="0.7 (BCI可用性常用门槛)")
    ax.set_xlabel("accuracy")
    ax.set_ylabel("被试数")
    ax.set_title("跨被试准确率分布")
    ax.legend()
    fig.tight_layout()
    out = fig_dir / "accuracy_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_report(df, tests, fig_dir):
    """生成 markdown 报告"""
    lines = ["# MI-Decoder 跨被试汇总报告\n"]
    lines.append(f"被试数: {len(df)}\n")

    lines.append("## 各方法均值\n")
    for col, name in [("csp_acc", "CSP+LDA"), ("fbcsp_acc", "FBCSP"),
                      ("eegnet_acc", "EEGNet"),
                      ("best_acc", "伪在线(最佳时刻)")]:
        if col in df.columns and df[col].notna().any():
            s = df[col].dropna()
            lines.append(f"- {name}: {s.mean():.3f} ± {s.std():.3f} "
                         f"(min={s.min():.3f}, max={s.max():.3f})")
    lines.append("")

    lines.append("## 配对检验\n")
    lines.extend(f"- {t}" for t in tests)
    lines.append("")

    if "best_acc" in df.columns and "csp_acc" in df.columns:
        sub = df[["csp_acc", "best_acc"]].dropna()
        if len(sub):
            drop = (sub.csp_acc - sub.best_acc).mean()
            lines.append("## 离线 -> 伪在线 性能存活\n")
            lines.append(f"- 平均掉点: {drop:+.3f} "
                         "(离线CV acc - 伪在线最佳acc)")
            lines.append("- 掉点来源: 因果滤波群延迟 + 按时间切分(无shuffle) "
                         "+ 训练数据只有70%\n")

    if "itr_bits_min" in df.columns and df["itr_bits_min"].notna().any():
        s = df["itr_bits_min"].dropna()
        lines.append(f"## ITR\n\n- 均值 {s.mean():.1f} bits/min "
                     f"(max={s.max():.1f})\n")

    lines.append("## 每被试明细\n")
    lines.append(df.round(3).to_markdown(index=False))
    lines.append("\n图表见: " + str(fig_dir))
    return "\n".join(lines)


def main():
    cfg = load_config()
    paths = ensure_dirs(cfg)
    df = load_all_metrics(paths["metrics_dir"])

    tests = []
    if {"csp_acc", "eegnet_acc"} <= set(df.columns):
        tests.append(paired_test(df, "csp_acc", "eegnet_acc",
                                 "CSP", "EEGNet"))
    if {"csp_acc", "fbcsp_acc"} <= set(df.columns):
        tests.append(paired_test(df, "fbcsp_acc", "csp_acc",
                                 "FBCSP", "CSP"))

    plot_paired_scatter(df, paths["fig_dir"])
    plot_accuracy_hist(df, paths["fig_dir"])

    df.to_csv(paths["metrics_dir"] / "all_results.csv", index=False)
    report = build_report(df, tests, paths["fig_dir"])
    out = PROJECT_ROOT / "results" / "report.md"
    out.write_text(report, encoding="utf-8")
    log.info("报告已生成: %s", out)
    for t in tests:
        log.info(t)


if __name__ == "__main__":
    main()

"""模块3b: 跨被试在线解码 —— 70 人训 / 18 人测, 因果流式重放

与被试内伪在线(模块5)同一条数据路径、同一套指标, 只换训练集:
- W4/W5 被试内: 目标自己前 70% 试次训 -> 后 30% 试次测(模块5已算好, 这里只读)
- E1~E8 跨被试: 其他 70 人全部试次合训 -> 18 个测试被试的全部试次流式测;
  测试被试的任何数据(含无标签信号)都不进训练

实验矩阵(--exps):
  E1  CSP+LDA                 零校准
  E2  FBCSP+LDA               零校准(5 子带, 每子带一条因果流)
  E3  黎曼切空间+LR           零校准
  E4  EEGNet                  零校准(70 人中按人留 8 人做早停验证)
  E5  CSP+LDA  + 在线EA       训练被试各自按整条流的平均协方差白化后合训;
                              测试时每个窗用目标流 t 之前递归估计的 R_t 白化(无标签, 因果)
  E6  黎曼+LR  + 在线重定心   同 E5, 参考点换成测地线递归黎曼均值
  E7  EEGNet   + 在线EA       同 E5 的输入侧对齐
  E8  E5 + 自适应LDA          每个试次最后一个决策点后用伪标签递归更新类均值(Vidaurre 2011)

E5~E8 的训练集与 E1~E4 完全相同(同 70 人), 只改推理; 对齐/自适应只用目标 t 之前的流。

划分: 可用被试(有 clean_raw 且 epo>=20) 用 decode.random_state 随机排列,
      cross_subject.test_ratio 做测试(88 -> 18 人), 写入 cross_subject_split.json
输出: results/metrics/cross_subject_online.csv        每行一个测试被试; 多次运行按列合并
      results/metrics/cross_subject_online_curves.csv 各实验的延迟-精度曲线
      results/figures/cross_subject_latency_curves.png 已跑实验 + 被试内对照 画在一张图

运行:
    python -m src.decode_cross_subject                       # E1,E3,E5,E6,E8 (CSP/黎曼族, 几分钟)
    python -m src.decode_cross_subject --exps E2             # FBCSP
    python -m src.decode_cross_subject --exps E4,E7          # EEGNet, 十几分钟到半小时
    python -m src.decode_cross_subject --exps E1 --max-train 5 --max-test 2   # 冒烟
    python -m src.decode_cross_subject --plot-only           # 只用已有 csv 重画汇总图
"""
import argparse
import copy
import json

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC",
                                   "PingFang SC"]  # 中文字体, 避免标题豆腐块
plt.rcParams["axes.unicode_minus"] = False           # 负号正常显示
import mne
import numpy as np
import pandas as pd

from .online_models import (CSPLDA, EEGNetModel, FBCSPLDA, RiemannTSLR,
                            StreamAligner, align, load_stream, mi_events,
                            replay_stream, summarize, train_windows)
from .utils import (ensure_dirs, epochs_path, get_logger, load_config,
                    raw_path, subject_list)

log = get_logger("decode_cross_subject")
mne.set_log_level("WARNING")

MIN_EPOCHS = 20                                   # 与模块3/4一致

EXPS = {
    "E1": dict(model="csp",     align=None,      adaptive=False),
    "E2": dict(model="fbcsp",   align=None,      adaptive=False),
    "E3": dict(model="riemann", align=None,      adaptive=False),
    "E4": dict(model="eegnet",  align=None,      adaptive=False),
    "E5": dict(model="csp",     align="euclid",  adaptive=False),
    "E6": dict(model="riemann", align="riemann", adaptive=False),
    "E7": dict(model="eegnet",  align="euclid",  adaptive=False),
    "E8": dict(model="csp",     align="euclid",  adaptive=True),
}

# 汇总图样式: 同一模型族同色, 零校准实线 / 在线对齐虚线 / 自适应点线
EXP_STYLE = {
    "E1": ("CSP+LDA",            "tab:blue",   "-"),
    "E5": ("CSP+LDA +在线EA",     "tab:blue",   "--"),
    "E8": ("CSP+LDA +EA+自适应LDA", "tab:blue",   ":"),
    "E2": ("FBCSP+LDA",          "tab:purple", "-"),
    "E3": ("黎曼TS+LR",           "tab:green",  "-"),
    "E6": ("黎曼TS+LR +在线重定心", "tab:green",  "--"),
    "E4": ("EEGNet",             "tab:red",    "-"),
    "E7": ("EEGNet +在线EA",      "tab:red",    "--"),
}


def model_bands(cfg, model):
    d = cfg["decode"]
    return {"csp": [d["band"]], "riemann": [d["band"]],
            "fbcsp": d["fbcsp_bands"], "eegnet": [cfg["eegnet"]["band"]]}[model]


def available_subjects(cfg, paths):
    """有 clean_raw 且 epo.fif 试次>=20 的被试(与模块3/4的88人同一池)"""
    subs = []
    for s in subject_list(cfg):
        if not raw_path(paths, s).exists() or not epochs_path(paths, s).exists():
            continue
        n = len(mne.read_epochs(epochs_path(paths, s), preload=False, verbose=False))
        if n >= MIN_EPOCHS:
            subs.append(s)
    return subs


def split_subjects(subs, cfg):
    """按人一次随机切分: 训练被试 / 测试被试, 互不重叠"""
    rng = np.random.RandomState(cfg["decode"]["random_state"])
    order = rng.permutation(subs)
    n_test = int(round(len(subs) * cfg["cross_subject"]["test_ratio"]))
    return sorted(int(s) for s in order[n_test:]), sorted(int(s) for s in order[:n_test])


def build_model(spec, cfg, device):
    m = spec["model"]
    if m == "csp":
        return CSPLDA(cfg["decode"]["csp_components"], adaptive=spec["adaptive"],
                      eta=cfg["pseudo_online"]["adapt_eta"])
    if m == "fbcsp":
        return FBCSPLDA(cfg["decode"]["fbcsp_components"])
    if m == "riemann":
        return RiemannTSLR()
    return EEGNetModel(cfg, device)


def load_train(subs, cfg, paths, bands, slide, align_kinds, win_sec):
    """训练被试: 只留训练窗(不留整条流) + 各对齐方式下整条流的 R^{-1/2}"""
    event_id = dict(cfg["data"]["event_id"])
    out = {}
    for s in subs:
        stream, events, sfreq = load_stream(s, cfg, paths, bands)
        X, y, _ = train_windows(stream, sfreq, mi_events(events, event_id),
                                event_id, cfg, slide)
        isq = {k: StreamAligner(k, round(win_sec * sfreq)).fit_stream(stream[0]).final()
               for k in align_kinds}
        out[s] = (X, y, isq)
    log.info("训练被试 %d 人, 共 %d 个训练窗", len(out),
             sum(len(v[1]) for v in out.values()))
    return out


def load_test(subs, cfg, paths, bands, align_kinds, win_sec):
    """测试被试: 整条流 + 因果递归对齐器(只在推理时用, 不进训练)"""
    out = {}
    for s in subs:
        stream, events, sfreq = load_stream(s, cfg, paths, bands)
        aligners = {k: StreamAligner(k, round(win_sec * sfreq)).fit_stream(stream[0])
                    for k in align_kinds}
        out[s] = (stream, events, sfreq, aligners)
    log.info("测试被试 %d 人", len(out))
    return out


def assemble_train(train, align_kind):
    """拼 70 人训练集; align_kind 给出时每人先用自己整条流的 R^{-1/2} 白化"""
    Xs, ys, gs = [], [], []
    for s, (X, y, isq) in train.items():
        Xs.append(align(X, isq[align_kind]) if align_kind else X)
        ys.append(y)
        gs.append(np.full(len(y), s))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def run_exp(name, spec, train, test, cfg, device):
    """一个实验: 合训一次 -> 18 个测试被试各自流式重放"""
    event_id = dict(cfg["data"]["event_id"])
    X_tr, y_tr, g_tr = assemble_train(train, spec["align"])
    model = build_model(spec, cfg, device)

    if spec["model"] == "eegnet":
        # 早停验证集按被试留出, 不按窗切: 同一人的窗放两边会让验证虚高
        rng = np.random.RandomState(cfg["decode"]["random_state"])
        val_subs = rng.choice(sorted(train), cfg["cross_subject"]["eegnet_val_subjects"],
                              replace=False)
        vm = np.isin(g_tr, val_subs)
        log.info("[%s] EEGNet 拟合 %d 人/%d 窗, 验证 %d 人/%d 窗", name,
                 len(train) - len(val_subs), (~vm).sum(), len(val_subs), vm.sum())
        model.fit(X_tr[~vm], y_tr[~vm], X_tr[vm], y_tr[vm])
    else:
        model.fit(X_tr, y_tr)
    log.info("[%s] 训练完成: %s | %d 人 / %d 窗 | align=%s adaptive=%s", name,
             spec["model"], len(train), len(y_tr), spec["align"], spec["adaptive"])
    del X_tr

    rows, curves = [], []
    for s, (stream, events, sfreq, aligners) in test.items():
        m = copy.deepcopy(model) if spec["adaptive"] else model   # 自适应状态按人重置
        curve, fpr, n_rest = replay_stream(
            stream, events, sfreq, event_id, m, cfg, test_start=0,
            aligner=aligners.get(spec["align"]) if spec["align"] else None)
        sm = summarize(curve, fpr, n_rest)
        n_trials = len(mi_events(events, event_id))
        rows.append({"subject": s, "n_trials": n_trials,
                     **{f"{name}_{k}": v for k, v in sm.items()
                        if k in ("acc_at_end", "best_acc", "best_offset",
                                 "itr_bits_min", "rest_fpr")}})
        curve["subject"], curve["exp"] = s, name
        curves.append(curve)
        log.info("[%s] S%03d | acc_at_end=%.3f | best=%.3f @%.2fs | rest FPR=%.2f",
                 name, s, sm["acc_at_end"], sm["best_acc"], sm["best_offset"],
                 sm["rest_fpr"])
    return pd.DataFrame(rows), pd.concat(curves)


def merge_results(paths, df, curves, exps):
    """多次运行(不同 --exps)按被试合并到同一张表; 曲线按 exp 覆盖"""
    out = paths["metrics_dir"] / "cross_subject_online.csv"
    if out.exists():
        old = pd.read_csv(out)
        drop = [c for c in old.columns if c.split("_")[0] in exps]
        old = old.drop(columns=drop)
        df = old.merge(df.drop(columns=["n_trials"], errors="ignore"),
                       on="subject", how="outer")
    # 拼被试内在线结果(同一批测试被试, 后 30% 试次)作对照列
    for fname, tag in [("pseudo_online_results.csv", "within_csp"),
                       ("pseudo_online_eegnet_results.csv", "within_eegnet")]:
        p = paths["metrics_dir"] / fname
        if p.exists():
            w = pd.read_csv(p)[["subject", "acc_at_end", "rest_fpr"]]
            w = w.rename(columns={"acc_at_end": f"{tag}_acc_at_end",
                                  "rest_fpr": f"{tag}_rest_fpr"})
            df = df.drop(columns=[c for c in w.columns if c != "subject"],
                         errors="ignore").merge(w, on="subject", how="left")
    df = df.sort_values("subject")
    df.to_csv(out, index=False)

    cpath = paths["metrics_dir"] / "cross_subject_online_curves.csv"
    if cpath.exists():
        oldc = pd.read_csv(cpath)
        curves = pd.concat([oldc[~oldc.exp.isin(exps)], curves])
    curves.to_csv(cpath, index=False)
    return df, out


def plot_curves(paths, test_subs):
    """所有已跑实验的 18 人平均延迟-精度曲线画在一张图, 叠同一批人的被试内曲线作对照"""
    cpath = paths["metrics_dir"] / "cross_subject_online_curves.csv"
    if not cpath.exists():
        return None
    cv = pd.read_csv(cpath)
    cv = cv[cv.subject.isin(test_subs)]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    # 被试内对照: 同一批测试被试, 各自前70%训 -> 后30%测
    for fname, label, c in [("pseudo_online_curves.csv", "被试内 CSP+LDA(W4)", "black"),
                            ("pseudo_online_eegnet_curves.csv", "被试内 EEGNet(W5)", "gray")]:
        p = paths["metrics_dir"] / fname
        if not p.exists():
            continue
        w = pd.read_csv(p)
        w = w[w.subject.isin(test_subs)].groupby("offset").acc.mean()
        ax.plot(w.index, w.values, "-.", c=c, lw=1.5, alpha=0.8,
                label=f"{label}  末端 {w.iloc[-1]:.3f}")
    # 跨被试各实验
    for e, (label, c, ls) in EXP_STYLE.items():
        sub = cv[cv.exp == e]
        if sub.empty:
            continue
        m = sub.groupby("offset").acc.mean()
        ax.plot(m.index, m.values, ls, c=c, lw=2, marker="o", ms=3.5,
                label=f"{e} {label}  末端 {m.iloc[-1]:.3f}")
    ax.axhline(0.5, ls="--", c="lightgray", lw=1)
    ax.set_xlabel("决策时刻(想象开始后, 秒)")
    ax.set_ylabel("流式准确率(测试被试均值)")
    ax.set_ylim(0.45, 0.75)
    ax.set_title(f"跨被试在线: 70 人合训 -> {len(test_subs)} 人流式测 | 延迟-精度曲线")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    out = paths["fig_dir"] / "cross_subject_latency_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", default="E1,E3,E5,E6,E8",
                    help="逗号分隔, 见文件头实验矩阵; E4/E7(EEGNet)与E2(FBCSP)建议单独跑")
    ap.add_argument("--max-train", type=int, default=None, help="冒烟: 只用前N个训练被试")
    ap.add_argument("--max-test", type=int, default=None, help="冒烟: 只测前N个测试被试")
    ap.add_argument("--plot-only", action="store_true", help="不训练, 只用已有 csv 重画汇总图")
    args = ap.parse_args()
    exps = [e.strip() for e in args.exps.split(",")]
    unknown = [e for e in exps if e not in EXPS]
    assert not unknown, f"未知实验 {unknown}, 可选 {list(EXPS)}"

    cfg = load_config()
    paths = ensure_dirs(cfg)
    win_sec = cfg["pseudo_online"]["win_sec"]

    if args.plot_only:
        with open(paths["metrics_dir"] / "cross_subject_split.json") as f:
            test_subs = json.load(f)["test"]
        log.info("汇总图 -> %s", plot_curves(paths, test_subs))
        return

    device = None
    if any(EXPS[e]["model"] == "eegnet" for e in exps):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
        log.info("device = %s", device)

    subs = available_subjects(cfg, paths)
    train_subs, test_subs = split_subjects(subs, cfg)
    with open(paths["metrics_dir"] / "cross_subject_split.json", "w") as f:
        json.dump(dict(train=train_subs, test=test_subs), f)
    log.info("可用 %d 人 -> 训练 %d 人 / 测试 %d 人: %s", len(subs),
             len(train_subs), len(test_subs), test_subs)
    if args.max_train:
        train_subs = train_subs[:args.max_train]
    if args.max_test:
        test_subs = test_subs[:args.max_test]

    # 同一模型族的实验共享一次数据加载(频带相同)
    all_df, all_curves = None, []
    for model in ["csp", "riemann", "fbcsp", "eegnet"]:
        group = [e for e in exps if EXPS[e]["model"] == model]
        if not group:
            continue
        kinds = sorted({EXPS[e]["align"] for e in group if EXPS[e]["align"]})
        bands = model_bands(cfg, model)
        log.info("== 模型族 %s: %s | 频带 %s | 对齐 %s", model, group, bands, kinds)
        train = load_train(train_subs, cfg, paths, bands, model == "eegnet", kinds, win_sec)
        test = load_test(test_subs, cfg, paths, bands, kinds, win_sec)
        for e in group:
            df_e, cv_e = run_exp(e, EXPS[e], train, test, cfg, device)
            all_df = df_e if all_df is None else all_df.merge(
                df_e.drop(columns=["n_trials"]), on="subject")
            all_curves.append(cv_e)
        del train, test

    df, out = merge_results(paths, all_df, pd.concat(all_curves), exps)
    parts = []
    for col, name in [("within_csp_acc_at_end", "被试内CSP(W4)"),
                      ("within_eegnet_acc_at_end", "被试内EEGNet(W5)")]:
        if col in df:
            parts.append("%s=%.3f" % (name, df.loc[df.subject.isin(test_subs), col].mean()))
    for e in exps:
        sub = df[df[f"{e}_acc_at_end"].notna()]
        parts.append("%s acc_at_end=%.3f best=%.3f restFPR=%.2f" % (
            e, sub[f"{e}_acc_at_end"].mean(), sub[f"{e}_best_acc"].mean(),
            sub[f"{e}_rest_fpr"].mean()))
    log.info("均值(测试 %d 人): %s -> %s", len(test_subs), " | ".join(parts), out)
    log.info("汇总图 -> %s", plot_curves(paths, test_subs))


if __name__ == "__main__":
    main()

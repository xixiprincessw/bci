"""模块3: 经典解码 —— CSP+LDA 基线 与 Filter-Bank CSP

CSP(共空间模式): 找一组空间滤波器, 使两类信号投影后的方差差异最大化
—— 方差=频带功率, 所以CSP解码的正是ERD的空间差异(C3 vs C4)。

设计要点:
- 先窄化到8-30Hz(mu+beta, ERD所在频段), 时域波形对MI无用
- 特征窗口0.5-3.5s: 落在稳定ERD期, 避开想象结束后的beta反弹
- FBCSP: 多个子带各自提CSP特征后拼接, 手动逐fold训练防止数据泄露
- 指标: accuracy + Cohen's kappa(排除随机基线)
"""
import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .utils import (ensure_dirs, epochs_path, get_logger, load_config,
                    subject_list)

log = get_logger("decode_csp")


def get_xy(epochs, band, crop):
    """窄带滤波 + 裁剪特征窗口, 返回 X(试次,通道,时间), y(0/1)"""
    ep = epochs.copy().filter(band[0], band[1], verbose=False)
    ep.crop(crop[0], crop[1])
    X = ep.get_data(copy=True)
    y = (ep.events[:, -1] == ep.event_id["right"]).astype(int)
    return X, y


def run_csp(epochs, cfg):
    """CSP+LDA 交叉验证(整个pipeline在CV内拟合, 无泄露)"""
    d = cfg["decode"]
    X, y = get_xy(epochs, d["band"], d["crop"])
    clf = Pipeline([
        # reg="ledoit_wolf": 平均参考/ICA/插值使有效秩<64, 协方差奇异,
        # 需协方差收缩恢复正定, 否则广义特征分解报 non-positive definite
        ("csp", CSP(n_components=d["csp_components"], log=True,
                    reg="ledoit_wolf")),
        ("lda", LinearDiscriminantAnalysis()),
    ])
    cv = StratifiedKFold(d["cv_folds"], shuffle=True,
                         random_state=d["random_state"])
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    return dict(acc=float(np.mean(y_pred == y)),
                kappa=float(cohen_kappa_score(y, y_pred)))


def run_fbcsp(epochs, cfg):
    """Filter-Bank CSP: 每个子带的CSP在训练fold内拟合(手动CV防泄露)"""
    d = cfg["decode"]
    bands = d["fbcsp_bands"]
    # 预先按子带滤波(滤波本身无监督, 不泄露标签)
    Xs, y = [], None
    for lo, hi in bands:
        X_b, y = get_xy(epochs, (lo, hi), d["crop"])
        Xs.append(X_b)

    cv = StratifiedKFold(d["cv_folds"], shuffle=True,
                         random_state=d["random_state"])
    y_pred = np.zeros_like(y)
    for tr, te in cv.split(Xs[0], y):
        feats_tr, feats_te = [], []
        for X_b in Xs:                            # 每个子带独立提CSP功率特征
            csp = CSP(n_components=d["fbcsp_components"], log=True,
                      reg="ledoit_wolf")
            feats_tr.append(csp.fit_transform(X_b[tr], y[tr]))
            feats_te.append(csp.transform(X_b[te]))
        lda = LinearDiscriminantAnalysis()
        lda.fit(np.hstack(feats_tr), y[tr])
        y_pred[te] = lda.predict(np.hstack(feats_te))
    return dict(acc=float(np.mean(y_pred == y)),
                kappa=float(cohen_kappa_score(y, y_pred)))


def main():
    cfg = load_config()
    paths = ensure_dirs(cfg)

    records = []
    for sub in subject_list(cfg):
        fpath = epochs_path(paths, sub)
        if not fpath.exists():
            log.warning("S%03d epochs不存在, 跳过", sub)
            continue
        epochs = mne.read_epochs(fpath, preload=True, verbose=False)
        if len(epochs) < 20:                      # 试次太少不做CV
            log.warning("S%03d 仅%d个epoch, 跳过", sub, len(epochs))
            continue
        res_csp = run_csp(epochs, cfg)
        res_fb = run_fbcsp(epochs, cfg)
        records.append(dict(subject=sub, n_epochs=len(epochs),
                            csp_acc=res_csp["acc"], csp_kappa=res_csp["kappa"],
                            fbcsp_acc=res_fb["acc"], fbcsp_kappa=res_fb["kappa"]))
        log.info("S%03d | CSP acc=%.3f k=%.3f | FBCSP acc=%.3f k=%.3f",
                 sub, res_csp["acc"], res_csp["kappa"],
                 res_fb["acc"], res_fb["kappa"])

    df = pd.DataFrame(records)
    out = paths["metrics_dir"] / "csp_results.csv"
    df.to_csv(out, index=False)
    if len(df):
        log.info("均值: CSP acc=%.3f | FBCSP acc=%.3f (n=%d) -> %s",
                 df.csp_acc.mean(), df.fbcsp_acc.mean(), len(df), out)


if __name__ == "__main__":
    main()

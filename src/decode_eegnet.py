"""模块4: 深度学习解码 —— EEGNet(braindecode)

与模块3使用同一套CV划分, 保证与CSP的对比公平。
预期发现: 单被试小样本(~45试次)下EEGNet未必赢CSP
—— 数据量与模型容量的匹配是实际工作中的核心命题。

工程细节:
- 标准化只用训练fold统计量(防泄露)
- 训练fold内部再切验证集做早停
- 兼容braindecode不同版本的模型参数名
"""
import inspect

import mne
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .decode_csp import get_xy
from .utils import (ensure_dirs, epochs_path, get_logger, load_config,
                    subject_list)

log = get_logger("decode_eegnet")


def build_eegnet(n_chans, n_times, n_classes=2):
    """兼容braindecode新旧版本的EEGNet构造"""
    try:
        from braindecode.models import EEGNetv4 as Model
    except ImportError:
        from braindecode.models import EEGNet as Model
    params = inspect.signature(Model.__init__).parameters
    if "n_chans" in params:                       # 新版命名
        return Model(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
    return Model(in_chans=n_chans, n_classes=n_classes,   # 旧版命名
                 input_window_samples=n_times)


def fit_eegnet(X_fit, y_fit, X_val, y_val, cfg, device):
    """在已标准化的数据上训练EEGNet, 验证损失早停, 返回最优权重的模型(eval模式)。
    模块4(按试次切验证)/模块5(按试次)/模块3b(按被试)共用, 验证集由调用方切好传入"""
    e = cfg["eegnet"]

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=e["batch_size"], shuffle=shuffle)

    model = build_eegnet(X_fit.shape[1], X_fit.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=e["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val, best_state, wait = np.inf, None, 0
    fit_loader = to_loader(X_fit, y_fit, shuffle=True)
    val_loader = to_loader(X_val, y_val, shuffle=False)
    for _ in range(e["max_epochs"]):
        model.train()
        for xb, yb in fit_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        # 验证损失早停
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([
                loss_fn(model(xb.to(device)), yb.to(device)).item()
                for xb, yb in val_loader]))
        if val_loss < best_val - 1e-4:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= e["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_one_fold(X_tr, y_tr, X_te, y_te, cfg, device):
    """单fold: 标准化->早停训练->测试集预测, 返回y_pred"""
    e = cfg["eegnet"]
    # 通道级z-score, 统计量只来自训练fold
    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    X_tr, X_te = (X_tr - mean) / std, (X_te - mean) / std

    # 训练fold内部切验证集(早停依据)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=e["val_ratio"], stratify=y_tr,
        random_state=cfg["decode"]["random_state"])

    model = fit_eegnet(X_fit, y_fit, X_val, y_val, cfg, device)
    with torch.no_grad():
        logits = model(torch.tensor(X_te, dtype=torch.float32).to(device))
    return logits.argmax(dim=1).cpu().numpy()


def run_eegnet(epochs, cfg, device):
    """与模块3同参数的StratifiedKFold, 保证对比公平"""
    d = cfg["decode"]
    # 用较宽频带(数据已1-45Hz), 让网络自己学频率特征
    X, y = get_xy(epochs, cfg["eegnet"]["band"], d["crop"])
    X = X.astype(np.float32) * 1e6                # V->uV, 数值尺度更友好

    cv = StratifiedKFold(d["cv_folds"], shuffle=True,
                         random_state=d["random_state"])
    y_pred = np.zeros_like(y)
    for tr, te in cv.split(X, y):
        y_pred[te] = train_one_fold(X[tr], y[tr], X[te], y[te], cfg, device)
    return dict(acc=float(np.mean(y_pred == y)),
                kappa=float(cohen_kappa_score(y, y_pred)))


def main():
    cfg = load_config()
    paths = ensure_dirs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    log.info("device = %s", device)

    records = []
    for sub in subject_list(cfg):
        fpath = epochs_path(paths, sub)
        if not fpath.exists():
            continue
        epochs = mne.read_epochs(fpath, preload=True, verbose=False)
        if len(epochs) < 20:
            continue
        res = run_eegnet(epochs, cfg, device)
        records.append(dict(subject=sub, eegnet_acc=res["acc"],
                            eegnet_kappa=res["kappa"]))
        log.info("S%03d | EEGNet acc=%.3f kappa=%.3f",
                 sub, res["acc"], res["kappa"])

    df = pd.DataFrame(records)
    out = paths["metrics_dir"] / "eegnet_results.csv"
    df.to_csv(out, index=False)
    if len(df):
        log.info("均值: EEGNet acc=%.3f (n=%d) -> %s",
                 df.eegnet_acc.mean(), len(df), out)


if __name__ == "__main__":
    main()

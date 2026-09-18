"""在线解码公共件: 因果流加载 / 流式对齐 / 在线模型 / 时间顺序重放评估

被试内伪在线(模块5)与跨被试在线(模块3b)共用, 保证两者口径完全一致:
- 数据: clean_raw.fif -> 因果 sosfilt(每个频带一条流), 不用离线 filtfilt 的 epo.fif
- 训练窗: CSP/FBCSP/黎曼 取 crop 整段(协方差与窗长无关);
          EEGNet 输入长度固定, 训练窗必须与测试窗同长(win_sec), 在 crop 内滑窗取样
- 测试: 试次按时间顺序重放, 每个试次在 offset 扫描点回看 win_sec 窗盲判;
        同期静息段滑窗测误触发
- 在线自适应(可选): 只用目标 t 之前的流, 无标签; 训练集里没有目标任何试次

注: ICA 在预处理阶段对整段录音拟合(无监督, 不看标签), clean_raw 已去眼电,
    这里不再重做; 严格的在线系统应在校准段拟合 ICA 后冻结。
"""
import numpy as np
import pandas as pd
import mne
from mne.decoding import CSP
from scipy.signal import butter, sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .utils import raw_path


# ----------------------------------------------------------------- 数据流
def causal_bandpass(data, sfreq, band, order=4):
    """因果IIR带通: 只用过去样本, 可直接搬到在线系统。

    与离线filtfilt的区别: filtfilt正反各滤一遍实现零相位, 反向那遍用了"未来",
    在线不存在。因果滤波有群延迟, 这是在线要接受的代价。
    """
    sos = butter(order, band, btype="bandpass", fs=sfreq, output="sos")
    return sosfilt(sos, data, axis=-1)


def load_stream(subject, cfg, paths, bands):
    """读清洗后的连续raw, 每个频带各做一次整段因果滤波(合法: 因果滤波 t 时刻输出
    只依赖 t 之前, 整段先滤与逐样本滤结果相同)。
    返回 stream(n_bands, n_ch, T) float32, events, sfreq"""
    raw = mne.io.read_raw_fif(raw_path(paths, subject), preload=True,
                              verbose=False)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks="eeg")
    stream = np.stack([causal_bandpass(data, sfreq, b) for b in bands])
    return stream.astype(np.float32), events, sfreq


def mi_events(events, event_id):
    """只留左/右手运动想象事件(按时间顺序)"""
    return events[np.isin(events[:, -1], list(event_id.values()))]


def extract_window(stream, sfreq, onset_sample, t_start, t_end):
    """从连续流中取一个窗口(相对onset的秒数), 越界返回None"""
    s0 = onset_sample + int(round(t_start * sfreq))
    s1 = onset_sample + int(round(t_end * sfreq))
    if s0 < 0 or s1 > stream.shape[-1]:
        return None
    return stream[..., s0:s1]


def train_windows(stream, sfreq, trials, event_id, cfg, slide):
    """训练样本。
    slide=False: 每试次一个 crop 整段窗(CSP族, 协方差不管窗长);
    slide=True : 每试次在 crop 内按 eegnet_train_step_sec 滑 win_sec 窗
                 (EEGNet 输入长度固定, 训练窗必须与测试窗同长)。
    返回 X(n, n_bands, ch, T), y, trial_idx(每个窗所属试次序号, 供按试次切验证集)"""
    d, po = cfg["decode"], cfg["pseudo_online"]
    crop = d["crop"]
    if slide:
        win, step = po["win_sec"], po["eegnet_train_step_sec"]
        ends = np.arange(crop[0] + win, crop[1] + 1e-9, step)
        spans = [(float(e - win), float(e)) for e in ends]
    else:
        spans = [tuple(crop)]
    X, y, tid = [], [], []
    for i, ev in enumerate(trials):
        for t0, t1 in spans:
            w = extract_window(stream, sfreq, ev[0], t0, t1)
            if w is None:
                continue
            X.append(w)
            y.append(int(ev[-1] == event_id["right"]))
            tid.append(i)
    return np.array(X), np.array(y), np.array(tid)


def split_val_by_trial(tid, y, val_ratio, seed):
    """按试次(而非按窗)切验证集: 同一试次的滑窗高度重叠, 不能一半训一半验。
    每类按比例抽试次, 至少1个。返回窗级布尔掩码 (fit_mask, val_mask)"""
    rng = np.random.RandomState(seed)
    trials = np.unique(tid)
    label = np.array([y[tid == t][0] for t in trials])
    val_trials = []
    for c in np.unique(label):
        pool = trials[label == c]
        k = max(1, int(round(len(pool) * val_ratio)))
        val_trials.extend(rng.choice(pool, k, replace=False))
    val_mask = np.isin(tid, val_trials)
    return ~val_mask, val_mask


# ----------------------------------------------------------------- 流式对齐
def _window_covs(x, win):
    """整条流按 win 切不重叠窗, Ledoit-Wolf 协方差(平均参考后秩亏, 需收缩保证正定)"""
    from pyriemann.utils.covariance import covariances
    n = x.shape[-1] // win
    segs = x[:, :n * win].reshape(x.shape[0], n, win).transpose(1, 0, 2)
    return covariances(segs.astype(np.float64), estimator="lwf"), \
        np.arange(1, n + 1) * win


class StreamAligner:
    """流式参考协方差: 沿整条流按 win_sec 切不重叠窗, 递归更新参考 R_k。
    - euclid : 算术均值递归(在线 EA, He & Wu 2020)
    - riemann: 测地线递归黎曼均值(在线重定心, Zanini 2018)
    对齐都是 w <- R^{-1/2} w。
    isqrt_at(sample) 只用结束时刻 <= sample 的窗 -> 严格因果(测试被试用);
    final() 用整条流 -> 训练被试用, 与测试侧收敛到同一统计量。"""

    def __init__(self, kind, win_samples):
        assert kind in ("euclid", "riemann")
        self.kind, self.win = kind, int(win_samples)

    def fit_stream(self, x):
        from pyriemann.utils.base import invsqrtm, powm, sqrtm
        covs, self.ends = _window_covs(x, self.win)
        self.isqrts, M = [], None
        for k, C in enumerate(covs, start=1):
            if M is None:
                M = C
            elif self.kind == "euclid":
                M = M + (C - M) / k
            else:                                  # 沿测地线朝 C 走 1/k
                Ms, Mi = sqrtm(M), invsqrtm(M)
                M = Ms @ powm(Mi @ C @ Mi, 1.0 / k) @ Ms
            self.isqrts.append(invsqrtm(M).astype(np.float32))
        return self

    def isqrt_at(self, sample):
        k = int(np.searchsorted(self.ends, sample, side="right"))
        return None if k == 0 else self.isqrts[k - 1]   # 流最开头2s内: 不对齐

    def final(self):
        return self.isqrts[-1]


def align(w, isqrt):
    """w(..., ch, T) <- isqrt @ w, 所有频带用同一个空间变换"""
    if isqrt is None:
        return w
    return np.einsum("ij,...jt->...it", isqrt, w).astype(w.dtype)


# ----------------------------------------------------------------- 在线模型
# 统一接口: fit(X, y) / predict(w) -> (pred, conf, feat) / update(feat, pred)
# X: (n, n_bands, ch, T); w: (n_bands, ch, T)
class CSPLDA:
    """CSP(log方差)+LDA, 用第0个频带。
    adaptive=True: LDA 类均值按伪标签递归更新(Vidaurre 2011 无监督自适应),
    协方差和先验冻结; 每次 update 后重算判别方向。"""

    def __init__(self, n_components, adaptive=False, eta=0.05):
        self.n_components, self.adaptive, self.eta = n_components, adaptive, eta

    def fit(self, X, y):
        X = X[:, 0].astype(np.float64)
        self.csp = CSP(n_components=self.n_components, reg="ledoit_wolf",
                       log=True).fit(X, y)
        F = self.csp.transform(X)
        self.lda = LinearDiscriminantAnalysis(store_covariance=True).fit(F, y)
        self.mu = self.lda.means_.copy()
        self.P = np.linalg.pinv(self.lda.covariance_)
        self.logprior = float(np.log(self.lda.priors_[1] / self.lda.priors_[0]))
        return self

    def predict(self, w):
        f = self.csp.transform(w[0][np.newaxis].astype(np.float64))[0]
        if self.adaptive:
            d = self.mu[1] - self.mu[0]
            s = d @ self.P @ (f - self.mu.mean(0)) + self.logprior
            p1 = 1.0 / (1.0 + np.exp(-s))
            proba = np.array([1 - p1, p1])
        else:
            proba = self.lda.predict_proba(f[np.newaxis])[0]
        return int(proba.argmax()), float(proba.max()), f

    def update(self, feat, pred):
        if self.adaptive:
            self.mu[pred] = (1 - self.eta) * self.mu[pred] + self.eta * feat


class FBCSPLDA:
    """Filter-Bank CSP: 每个频带独立 CSP, 特征拼接进 LDA"""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        self.csps = [CSP(n_components=self.n_components, reg="ledoit_wolf",
                         log=True).fit(X[:, b].astype(np.float64), y)
                     for b in range(X.shape[1])]
        self.lda = LinearDiscriminantAnalysis().fit(self._feats(X), y)
        return self

    def _feats(self, X):
        return np.hstack([c.transform(X[:, b].astype(np.float64))
                          for b, c in enumerate(self.csps)])

    def predict(self, w):
        f = self._feats(w[np.newaxis])
        proba = self.lda.predict_proba(f)[0]
        return int(proba.argmax()), float(proba.max()), f[0]

    def update(self, feat, pred):
        pass


class RiemannTSLR:
    """协方差 -> 黎曼切空间投影 -> 逻辑回归(跨被试 MI 的经典基线), 用第0个频带"""

    def fit(self, X, y):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        self.clf = make_pipeline(Covariances(estimator="lwf"),
                                 TangentSpace(metric="riemann"),
                                 LogisticRegression(max_iter=2000))
        self.clf.fit(X[:, 0].astype(np.float64), y)
        return self

    def predict(self, w):
        proba = self.clf.predict_proba(w[0][np.newaxis].astype(np.float64))[0]
        return int(proba.argmax()), float(proba.max()), None

    def update(self, feat, pred):
        pass


class EEGNetModel:
    """EEGNet(宽带, 输入长度=win_sec)。V->uV 后通道级 z-score, 统计量只来自训练窗。
    fit 的验证集由调用方切好: 被试内按试次, 跨被试按被试。"""

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device

    def fit(self, X_fit, y_fit, X_val, y_val):
        import torch
        from .decode_eegnet import fit_eegnet
        X_fit = X_fit[:, 0].astype(np.float32) * 1e6
        X_val = X_val[:, 0].astype(np.float32) * 1e6
        self.mean = X_fit.mean(axis=(0, 2), keepdims=True)
        self.std = X_fit.std(axis=(0, 2), keepdims=True) + 1e-8
        self.model = fit_eegnet((X_fit - self.mean) / self.std, y_fit,
                                (X_val - self.mean) / self.std, y_val,
                                self.cfg, self.device)
        self._torch = torch
        return self

    def predict(self, w):
        torch = self._torch
        x = (w[0].astype(np.float32) * 1e6 - self.mean[0]) / self.std[0]
        with torch.no_grad():
            logits = self.model(torch.tensor(x[np.newaxis]).to(self.device))
            proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return int(proba.argmax()), float(proba.max()), None

    def update(self, feat, pred):
        pass


# ----------------------------------------------------------------- 重放评估
def replay_stream(stream, events, sfreq, event_id, model, cfg,
                  test_start=0, aligner=None):
    """按时间顺序重放 test_start 之后的流:
    - MI 试次: 决策点 = 想象开始后 offset 秒, 回看 win_sec 窗盲判, 记 acc(offset);
               最后一个决策点后用伪标签做一次在线更新(非自适应模型为空操作)
    - 静息 T0 段: 滑窗, 记高置信输出比例(误触发率)
    aligner 给出时每个窗先 w <- R_t^{-1/2} w, R_t 只用窗结束时刻之前的流。
    返回 curve DataFrame(offset, acc, n), rest_fpr, n_rest_windows"""
    po = cfg["pseudo_online"]
    win, step = po["win_sec"], po["step_sec"]
    start, stop, ostep = po["offsets"]
    offsets = np.arange(start, stop + 1e-9, ostep)
    mi_codes = set(event_id.values())

    def judge(onset, t0, t1):
        w = extract_window(stream, sfreq, onset, t0, t1)
        if w is None:
            return None
        if aligner is not None:
            w = align(w, aligner.isqrt_at(onset + int(round(t1 * sfreq))))
        return model.predict(w)

    correct, n = np.zeros(len(offsets)), np.zeros(len(offsets))
    n_fire = n_win = 0
    for ev in events[events[:, 0] >= test_start]:
        code = ev[-1]
        if code in mi_codes:
            truth, last = int(code == event_id["right"]), None
            for i, off in enumerate(offsets):
                r = judge(ev[0], off - win, off)
                if r is None:
                    continue
                correct[i] += int(r[0] == truth)
                n[i] += 1
                last = r
            if last is not None:
                model.update(last[2], last[0])
        elif code == 1:                                 # T0 静息, 约4.2s
            for off in np.arange(win, 4.0 + 1e-9, step):
                r = judge(ev[0], off - win, off)
                if r is None:
                    continue
                n_fire += int(r[1] >= po["conf_threshold"])
                n_win += 1

    keep = n > 0
    curve = pd.DataFrame(dict(offset=offsets[keep], acc=correct[keep] / n[keep],
                              n=n[keep].astype(int)))
    return curve, (n_fire / n_win if n_win else np.nan), n_win


def compute_itr(acc, n_classes, decision_sec):
    """ITR(bits/min), Wolpaw公式。decision_sec=从提示到出结果的时间。
    (乐观估计: 未计入试次间隔, 真实系统会更低)"""
    p = float(np.clip(acc, 1e-9, 1 - 1e-9))
    if p <= 1.0 / n_classes:
        return 0.0
    bits = (np.log2(n_classes) + p * np.log2(p)
            + (1 - p) * np.log2((1 - p) / (n_classes - 1)))
    return bits * 60.0 / decision_sec


def summarize(curve, fpr, n_rest):
    """曲线 -> 汇总指标。best_* 是在测试集上挑 offset 的乐观数字, acc_at_end 更保守"""
    best = curve.loc[curve.acc.idxmax()]
    return dict(best_offset=float(best.offset), best_acc=float(best.acc),
                acc_at_end=float(curve.acc.iloc[-1]),
                itr_bits_min=compute_itr(best.acc, 2, best.offset),
                rest_fpr=fpr, n_rest_windows=n_rest)

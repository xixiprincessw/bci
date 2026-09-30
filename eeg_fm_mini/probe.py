"""三档评估，对齐简版方案一期口径：冻结探针 / 全量微调 / 标注砍到 10-30-100%。

读出口与方案触点头同形态：一个可学习 query 对每个导联沿时间轴做 attention
pooling（C 维保留，出 C 个向量），线性层逐导联出 logit；MI 数据只有试次级
标签、没有导联级监督，所以最后一步把 C 个 logit 取平均当试次分数。

对照线（各自同口径跑）：
- pretrained : runs/<tag>/ckpt.pt 的权重
- random     : 同架构随机初始化（负控制）
- bandpower  : μ(8–13)/β(13–30) 对数带功率 + 逻辑回归（手工特征基线，
               同样砍标注跑 10/30/100；冻结档无骨干可言，记 —）

数量关系与方案一致：100% 微调既是口径二也是口径三曲线末点。
探针在受试者 1–80 上拟合，81–109 上测。报告 ACC + 按受试者 bootstrap 95% CI。

用法：
    python probe.py --tag base2
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import Cfg
from data import ProbeSet
from model import EEGFM
from pretrain import cfg_from_dict, pick_device

FRACS = (0.1, 0.3, 1.0)
SUBSET_SEED = 1234          # 标注子集三家共用，保证同题同卷


class ProbeHead(nn.Module):
    """按导联 T 轴 attention pooling + 线性投票，导联轴最后取平均。"""

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.normal(torch.zeros(d), 0.02))
        self.lin = nn.Linear(d, 1)

    def forward(self, h):                              # [B,C,T,d] → [B]
        w = torch.softmax(h @ self.q / math.sqrt(h.shape[-1]), -1)   # [B,C,T]
        v = (w[..., None] * h).sum(-2)                 # [B,C,d] 每导联一个向量
        return self.lin(v).mean(1).squeeze(-1)         # [B] 导联投票取平均


def acc_ci(pred, yte, subje, seed=0):
    """准确率 + 按受试者 bootstrap 的 95% CI。"""
    acc = (pred == yte).mean()
    rng = np.random.default_rng(seed)
    subs = np.unique(subje)
    accs = []
    for _ in range(1000):
        pick = rng.choice(subs, len(subs), replace=True)
        idx = np.concatenate([np.where(subje == s)[0] for s in pick])
        accs.append((pred[idx] == yte[idx]).mean())
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return acc, lo, hi


@torch.no_grad()
def extract_h(model, ds, dev, bs=32):
    """冻结骨干提全部 token 表征 → CPU 张量 [N,C,T,d]。"""
    model.eval()
    hs = []
    for i in range(0, len(ds), bs):
        x = torch.from_numpy(ds.x[i : i + bs]).to(dev)
        c = torch.from_numpy(ds.coords[i : i + bs]).to(dev)
        hs.append(model.encode(x, c).float().cpu())
    return torch.cat(hs)


def frozen_probe(htr, ytr, hte, yte, subje, cfg, seed=0):
    """口径一：骨干冻住，只训 ProbeHead（query + 线性层，129 个参数级别）。"""
    torch.manual_seed(seed)
    head = ProbeHead(htr.shape[-1])
    opt = torch.optim.Adam(head.parameters(), lr=cfg.head_lr)
    ytr = torch.from_numpy(np.ascontiguousarray(ytr)).float()
    n = len(htr)
    for _ in range(cfg.head_epochs):
        for i in torch.randperm(n).split(256):
            loss = F.binary_cross_entropy_with_logits(head(htr[i]), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (head(hte) > 0).long().numpy()
    return acc_ci(pred, yte, subje)


@torch.no_grad()
def predict(model, head, ds, dev, bs=64):
    model.eval(); head.eval()
    preds = []
    for i in range(0, len(ds), bs):
        x = torch.from_numpy(ds.x[i : i + bs]).to(dev)
        c = torch.from_numpy(ds.coords[i : i + bs]).to(dev)
        preds.append((head(model.encode(x, c)) > 0).cpu().numpy())
    return np.concatenate(preds).astype(int)


def finetune(cfg, tr, te, dev, idx, init_state=None, seed=0):
    """口径二/三：骨干 + 头一起训，标注只用 idx 指定的子集。"""
    torch.manual_seed(seed)
    model = EEGFM(cfg).to(dev)
    if init_state is not None:
        model.load_state_dict(init_state)
    head = ProbeHead(cfg.d_model).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=cfg.ft_lr, weight_decay=cfg.weight_decay)
    x = torch.from_numpy(tr.x[idx]); c = torch.from_numpy(tr.coords[idx])
    y = torch.from_numpy(np.ascontiguousarray(tr.y[idx])).float()
    for _ in range(cfg.ft_epochs):
        model.train(); head.train()
        for i in torch.randperm(len(idx)).split(cfg.batch_size * 2):
            h = model.encode(x[i].to(dev), c[i].to(dev))
            loss = F.binary_cross_entropy_with_logits(head(h), y[i].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    pred = predict(model, head, te, dev)
    return acc_ci(pred, te.y, te.subj)


def bandpower(ds, fs):
    x = ds.x.reshape(len(ds), ds.x.shape[1], -1)              # [N,C,T*P]
    f, pxx = welch(x, fs=fs, nperseg=fs, axis=-1)
    mu = pxx[..., (f >= 8) & (f < 13)].mean(-1)
    beta = pxx[..., (f >= 13) & (f < 30)].mean(-1)
    return np.log(np.stack([mu, beta], -1)).reshape(len(ds), -1)


def bandpower_arm(ftr, ytr, idx, fte, yte, subje):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(ftr[idx], ytr[idx])
    return acc_ci(clf.predict(fte), yte, subje)


def fmt(cell):
    return "—" if cell is None else f"{cell[0]:.3f} [{cell[1]:.3f}, {cell[2]:.3f}]"


def say(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="base2")
    a = p.parse_args()

    ck = torch.load(Cfg().out_dir / a.tag / "ckpt.pt", map_location="cpu")
    cfg = cfg_from_dict(ck["cfg"])
    cfg.prefix_S = -1                                          # 探针取双向表征
    cfg.n_pretrain_subjects = 0                                # 数据量消融只缩预训练，探针训练池恒为全量 1–80
    if "sec_head.q" not in ck["model"]:
        cfg.sec_lambda = 0.0                                   # 旧权重没训次级头
    dev = pick_device(cfg.device)

    tr, te = ProbeSet(cfg, "train"), ProbeSet(cfg, "test")
    say(f"probe train {len(tr)}  test {len(te)}  chance {max(te.y.mean(), 1 - te.y.mean()):.3f}")

    order = np.random.default_rng(SUBSET_SEED).permutation(len(tr))
    idx_f = {f: order[: max(1, int(round(f * len(tr))))] for f in FRACS}

    # ---- 骨干两份：pretrained / random ----
    pre = EEGFM(cfg).to(dev)
    pre.load_state_dict(ck["model"])
    torch.manual_seed(cfg.seed + 1)
    rnd = EEGFM(cfg).to(dev)

    # ---- 口径一：冻结探针（全标注训头）----
    res = {}
    for name, m in (("pretrained", pre), ("random", rnd)):
        t0 = time.time()
        htr = extract_h(m, tr, dev); hte = extract_h(m, te, dev)
        res[("frozen", name)] = frozen_probe(htr, tr.y, hte, te.y, te.subj, cfg)
        say(f"frozen {name} {fmt(res[('frozen', name)])} {time.time() - t0:.0f}s")
        del htr, hte

    # ---- 口径二/三：微调 × 标注比例 × 骨干两份 ----
    for f in FRACS:
        for name, st, sd in (("pretrained", ck["model"], cfg.seed), ("random", None, cfg.seed + 1)):
            t0 = time.time()
            res[("ft", f, name)] = finetune(cfg, tr, te, dev, idx_f[f], st, sd)
            say(f"ft {f:.0%} {name} {fmt(res[('ft', f, name)])} {time.time() - t0:.0f}s")

    # ---- bandpower 基线（同比例砍标注）----
    ftr, fte = bandpower(tr, cfg.fs), bandpower(te, cfg.fs)
    for f in FRACS:
        res[("bp", f)] = bandpower_arm(ftr, tr.y, idx_f[f], fte, te.y, te.subj)

    # ---- 汇总表 ----
    print(f"\n{'口径':<10}{'标注%':>7}  {'pretrained':<26}{'random':<26}bandpower")
    print(f"{'冻结探针':<8}{'100%':>8}  {fmt(res[('frozen', 'pretrained')]):<26}"
          f"{fmt(res[('frozen', 'random')]):<26}—")
    for f in FRACS:
        print(f"{'微调':<10}{f:>7.0%}  {fmt(res[('ft', f, 'pretrained')]):<26}"
              f"{fmt(res[('ft', f, 'random')]):<26}{fmt(res[('bp', f)])}")


if __name__ == "__main__":
    main()

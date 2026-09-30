"""ERD 分档增益分析：base2 微调 10%/30% 档的预训练增益，按中篇 ERD 四档拆开。

读数回答下篇 3.4 的问题：预训练增益是只落在 strong 档（样本效率工具），
还是 weak/none 档也动（表示看到了 mu 指标看不到的调制）。
被试分档取自 mi_decoder 的 erd_quant.csv（mu_class 列，口径与中篇一致）。

用法：
    python erd_tier_eval.py --tag base2
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import Cfg
from data import ProbeSet
from model import EEGFM
from probe import ProbeHead, predict, SUBSET_SEED
from pretrain import cfg_from_dict, pick_device


def finetune_preds(cfg, tr, te, dev, idx, init_state, seed):
    """与 probe.finetune 同构，但返回逐试次预测而非汇总 acc。"""
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
    return predict(model, head, te, dev)


def say(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="base2")
    p.add_argument("--erd_csv", default="/Users/erqian/Desktop/util_udf/bci/mi_decoder/results/metrics/erd_quant.csv")
    a = p.parse_args()

    ck = torch.load(Cfg().out_dir / a.tag / "ckpt.pt", map_location="cpu")
    cfg = cfg_from_dict(ck["cfg"])
    cfg.prefix_S = -1
    cfg.n_pretrain_subjects = 0
    if "sec_head.q" not in ck["model"]:
        cfg.sec_lambda = 0.0
    dev = pick_device(cfg.device)
    tr, te = ProbeSet(cfg, "train"), ProbeSet(cfg, "test")

    tier = pd.read_csv(a.erd_csv).set_index("subject")["mu_class"]
    te_tier = tier.reindex(te.subj).to_numpy()
    say(f"test {len(te)} trials, {len(np.unique(te.subj))} subjects, tiers:",
        {t: int(((te_tier == t) & True).sum()) for t in ("strong", "medium", "weak", "none_or_reversed")})

    order = np.random.default_rng(SUBSET_SEED).permutation(len(tr))
    tiers = ("strong", "medium", "weak", "none_or_reversed")
    for f in (0.1, 0.3):
        idx = order[: max(1, int(round(f * len(tr))))]
        preds = {}
        for name, st, sd in (("pre", ck["model"], cfg.seed), ("rnd", None, cfg.seed + 1)):
            t0 = time.time()
            preds[name] = finetune_preds(cfg, tr, te, dev, idx, st, sd)
            say(f"ft {f:.0%} {name} done {time.time() - t0:.0f}s")
        print(f"\n== ft {f:.0%}  按 ERD 分档（acc_pre / acc_rnd / 增益）==")
        print(f"{'tier':<18}{'被试数':>6}{'试次数':>7}  {'pre':>7}{'rnd':>7}{'增益':>8}{'chance':>8}")
        for t in tiers:
            m = te_tier == t
            if m.sum() == 0:
                continue
            acc_p = (preds["pre"][m] == te.y[m]).mean()
            acc_r = (preds["rnd"][m] == te.y[m]).mean()
            ch = max(te.y[m].mean(), 1 - te.y[m].mean())
            ns = len(np.unique(te.subj[m]))
            print(f"{t:<18}{ns:>6}{m.sum():>7}  {acc_p:>7.3f}{acc_r:>7.3f}{acc_p - acc_r:>+8.3f}{ch:>8.3f}")
        acc_p = (preds["pre"] == te.y).mean(); acc_r = (preds["rnd"] == te.y).mean()
        print(f"{'ALL':<18}{len(np.unique(te.subj)):>6}{len(te.y):>7}  {acc_p:>7.3f}{acc_r:>7.3f}{acc_p - acc_r:>+8.3f}")


if __name__ == "__main__":
    main()

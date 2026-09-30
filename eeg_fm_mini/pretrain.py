"""预训练：掩码波形重建。

用法：
    python pretrain.py                       # 默认配置
    python pretrain.py --epochs 3 --attn flat --tag flat
命令行参数名与 Cfg 字段同名，任意字段都能覆盖。
输出到 runs/<tag>/：ckpt.pt、log.txt。
"""
import argparse
import dataclasses
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Cfg
from data import PretrainSet
from model import EEGFM, random_mask


def parse_cfg() -> tuple[Cfg, str]:
    def conv(ftype):
        # argparse 的 type=bool 会把 "false" 解成 True，bool 字段需自定义解析
        if ftype is bool:
            return lambda s: str(s).strip().lower() not in ("false", "0", "no")
        return ftype

    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="base")
    for f in dataclasses.fields(Cfg):
        if f.type in (int, float, str, bool):
            p.add_argument(f"--{f.name}", type=conv(f.type), default=None)
    a = p.parse_args()
    cfg = Cfg()
    for f in dataclasses.fields(Cfg):
        v = getattr(a, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg, a.tag


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lr_at(step, total, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    r = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * r))


def cfg_to_dict(cfg: Cfg) -> dict:
    """Path 转 str，保证 torch.save/load(weights_only=True) 能过。"""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(cfg).items()}


def cfg_from_dict(d: dict) -> Cfg:
    names = {f.name for f in dataclasses.fields(Cfg)}
    cfg = Cfg(**{k: v for k, v in d.items() if k in names})
    for k in ("data_root", "cache_dir", "out_dir"):
        setattr(cfg, k, Path(getattr(cfg, k)))
    return cfg


def main():
    cfg, tag = parse_cfg()
    torch.manual_seed(cfg.seed)
    dev = pick_device(cfg.device)
    out = cfg.out_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s); log.write(s + "\n"); log.flush()

    say("cfg", {k: v for k, v in cfg_to_dict(cfg).items() if not isinstance(v, tuple)})
    ds = PretrainSet(cfg, "train")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
    say(f"windows {len(ds)}  tokens/epoch {len(ds) * ds.x.shape[1] * ds.x.shape[2] / 1e6:.1f}M  device {dev}")

    model = EEGFM(cfg).to(dev)
    say(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    total = cfg.epochs * len(dl)
    step = 0

    for ep in range(cfg.epochs):
        model.train()
        t0, run, secrun = time.time(), 0.0, 0.0
        for i, (x, coords) in enumerate(dl):
            x, coords = x.to(dev), coords.to(dev)
            B, C, T, _ = x.shape
            mask = random_mask(B, C, T, cfg.mask_ratio, dev)
            if cfg.chan_drop:
                # BIOT 式增广：每窗随机丢 10%–70% 整导联，并入 mask（被丢导联也参与重建损失）
                r = torch.rand(B, 1, 1, device=dev) * 0.6 + 0.1
                mask = mask | (torch.rand(B, C, 1, device=dev) < r)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, cfg)
            main_l, sec_l = model(x, coords, mask)
            loss = main_l + cfg.sec_lambda * sec_l
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += main_l.item(); secrun += sec_l.item(); step += 1
            if (i + 1) % 50 == 0:
                say(f"ep {ep} it {i + 1}/{len(dl)} main {run / 50:.4f} sec {secrun / 50:.4f} lr {lr_at(step, total, cfg):.2e} {(time.time() - t0) / (i + 1):.2f}s/it")
                run, secrun = 0.0, 0.0
        torch.save({"cfg": cfg_to_dict(cfg), "model": model.state_dict(), "epoch": ep}, out / "ckpt.pt")
        say(f"== epoch {ep} done, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

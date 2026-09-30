"""模型：1D 卷积 stem → 坐标 Fourier 编码 → criss-cross Transformer → 重建头。

张量约定：x[B,C,T,P]，C 导联、T 秒、P 每秒采样点；token 网格 [B,C,T,d]。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Cfg


# ---------------------------------------------------------------- stem

class ConvStem(nn.Module):
    """P 个采样点 → d 维。三层 1D 卷积，长度 P → 1，通道 1 → d。

    160 点 = 4 × 4 × 10。第一层核 16 点 = 100 ms，覆盖 α/β 半个周期以上。
    """

    def __init__(self, P: int, d: int):
        super().__init__()
        assert P == 160, "stem 的步长按 160 点写死，换采样率要改 schedule"
        c1, c2 = d // 4, d // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=16, stride=4, padding=6), nn.GroupNorm(4, c1), nn.GELU(),   # 160 → 40
            nn.Conv1d(c1, c2, kernel_size=8, stride=4, padding=2), nn.GroupNorm(4, c2), nn.GELU(),    # 40 → 10
            nn.Conv1d(c2, d, kernel_size=10, stride=10),                                              # 10 → 1
        )

    def forward(self, x):                         # x[B,C,T,P]
        B, C, T, P = x.shape
        h = self.net(x.reshape(B * C * T, 1, P))  # [BCT,d,1]
        return h.squeeze(-1).reshape(B, C, T, -1)


# ---------------------------------------------------------------- 坐标编码

class FourierCoord(nn.Module):
    """coords[B,C,3] → [B,C,d]。每轴 n_freq 把频率取 sin/cos，拼成 6·n_freq 维后线性到 d。"""

    def __init__(self, d: int, n_freq: int):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * math.pi)   # π, 2π, 4π ...
        self.proj = nn.Linear(3 * 2 * n_freq, d)

    def forward(self, coords):
        ang = coords.unsqueeze(-1) * self.freqs               # [B,C,3,n_freq]
        feat = torch.cat([ang.sin(), ang.cos()], -1)          # [B,C,3,2n]
        return self.proj(feat.flatten(-2))                    # [B,C,d]


# ---------------------------------------------------------------- RoPE

def rope(q, k, pos):
    """对最后一维成对旋转。q,k[..., L, hd]，pos[L]。"""
    hd = q.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=q.device).float() / hd))
    ang = pos.float()[:, None] * inv[None]                    # [L,hd/2]
    sin, cos = ang.sin(), ang.cos()

    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

    return rot(q), rot(k)


# ---------------------------------------------------------------- 注意力

class CrissCrossAttention(nn.Module):
    """一半头沿导联（同一秒），一半头沿时间（同一导联）。attn='flat' 时退化为平铺全注意力。"""

    def __init__(self, cfg: Cfg):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        assert d % H == 0 and H % 2 == 0
        self.H, self.hd, self.mode, self.use_rope = H, d // H, cfg.attn, cfg.rope
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = cfg.dropout

    def _attend(self, q, k, v, mask=None):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)

    def forward(self, x, time_mask=None):        # x[B,C,T,d]；time_mask[T,T] bool，True=可见
        B, C, T, d = x.shape
        q, k, v = self.qkv(x).reshape(B, C, T, 3, self.H, self.hd).unbind(3)   # 各 [B,C,T,H,hd]

        if self.mode == "flat":
            q, k, v = (t.reshape(B, C * T, self.H, self.hd).transpose(1, 2) for t in (q, k, v))
            mask = None
            if time_mask is not None:
                mask = time_mask.repeat(C, C)                   # [CT,CT]，只按时间限制
            o = self._attend(q, k, v, mask).transpose(1, 2).reshape(B, C, T, d)
            return self.out(o)

        h = self.H // 2
        # 空间头：同一秒内跨导联。序列长 C，批 B·T
        qs, ks, vs = (t[..., :h, :].permute(0, 2, 3, 1, 4).reshape(B * T, h, C, self.hd) for t in (q, k, v))
        os_ = self._attend(qs, ks, vs).reshape(B, T, h, C, self.hd).permute(0, 3, 1, 2, 4)   # [B,C,T,h,hd]
        # 时间头：同一导联内跨秒。序列长 T，批 B·C
        qt, kt, vt = (t[..., h:, :].permute(0, 1, 3, 2, 4).reshape(B * C, h, T, self.hd) for t in (q, k, v))
        if self.use_rope:
            qt, kt = rope(qt, kt, torch.arange(T, device=x.device))
        ot = self._attend(qt, kt, vt, time_mask).reshape(B, C, h, T, self.hd).permute(0, 1, 3, 2, 4)  # [B,C,T,h,hd]
        return self.out(torch.cat([os_, ot], -2).reshape(B, C, T, d))


class Block(nn.Module):
    """Pre-LN：x = x + Attn(LN(x))；x = x + FFN(LN(x))。FFN 中间维 ffn_mult·d，GELU。"""

    def __init__(self, cfg: Cfg):
        super().__init__()
        d = cfg.d_model
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CrissCrossAttention(cfg)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.ffn_mult * d), nn.GELU(), nn.Linear(cfg.ffn_mult * d, d), nn.Dropout(cfg.dropout))

    def forward(self, x, time_mask=None):
        x = x + self.attn(self.ln1(x), time_mask)
        return x + self.ffn(self.ln2(x))


# ---------------------------------------------------------------- 整模型

def sin_time(T: int, d: int, device) -> torch.Tensor:
    """固定正弦时间编码 [T,d]。时间轴走 RoPE 没有可加向量，次级头自带一份。"""
    t = torch.arange(T, device=device).float()[:, None]
    i = torch.arange(0, d, 2, device=device).float()[None] / d
    ang = t / (10000 ** i)
    return torch.cat([ang.sin(), ang.cos()], -1)


class SecondaryHead(nn.Module):
    """REVE 式次级损失头：跨层输出 attention pooling 成全局向量 g，
    在被遮位置加坐标 + 时间编码后过两层 FFN 重建波形。训完与主头一起丢弃。"""

    def __init__(self, cfg: Cfg):
        super().__init__()
        d = cfg.d_model
        self.q = nn.Parameter(torch.normal(torch.zeros(d), 0.02))
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, cfg.P))

    def forward(self, layers, coord_enc, mask, x):
        # layers: L 个 [B,C,T,d]（每层 FFN 后的输出）；coord_enc [B,C,d]；mask [B,C,T] bool
        pool = torch.stack(layers, 2).flatten(1, 3)            # [B, L*C*T, d]
        w = torch.softmax(pool @ self.q / math.sqrt(pool.shape[-1]), 1)
        g = (w[..., None] * pool).sum(1)                       # [B,d] 信息瓶颈
        pos = coord_enc[:, :, None, :] + sin_time(x.shape[2], coord_enc.shape[-1], x.device)[None, None]
        inp = (g[:, None, None, :] + pos)[mask]                # g 复制 M 份 + 各被遮格位置
        return F.mse_loss(self.ffn(inp), x[mask])


def prefix_mask(T: int, S: int, device) -> torch.Tensor | None:
    """前缀掩码：前 S 秒互相全可见，之后每秒只看自己和之前。S<0 或 S>=T 返回 None（全双向），S=0 纯因果。"""
    if S < 0 or S >= T:
        return None
    i = torch.arange(T, device=device)
    return (i[None, :] < S) | (i[None, :] <= i[:, None])       # [T,T]，True=可见


class EEGFM(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.stem = ConvStem(cfg.P, d)
        self.coord = FourierCoord(d, cfg.n_freq)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = (nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, cfg.P))
                     if cfg.head_ffn else nn.Linear(d, cfg.P))   # 重建头：两层 FFN（方案中间档）或一层线性
        self.sec_head = SecondaryHead(cfg) if cfg.sec_lambda > 0 else None
        nn.init.normal_(self.mask_token, std=0.02)

    def _coord_enc(self, coords, ref):
        """坐标编码；use_coord=False 时置零（空间位置消融，主路径与次级头同步失去坐标）。"""
        if self.cfg.use_coord:
            return self.coord(coords)
        return torch.zeros(coords.shape[0], coords.shape[1], self.cfg.d_model,
                           device=ref.device, dtype=ref.dtype)

    def encode(self, x, coords, mask=None, return_layers=False):
        """x[B,C,T,P], coords[B,C,3], mask[B,C,T] bool（True=遮）→ 表征 [B,C,T,d]；
        return_layers 时额外返回各 Block 输出列表（供次级损失跨层池化）。"""
        h = self.stem(x)
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_token.to(h.dtype), h)
        h = h + self._coord_enc(coords, h)[:, :, None, :]     # 坐标广播到每一秒
        tm = prefix_mask(x.shape[2], self.cfg.prefix_S, x.device)
        layers = []
        for blk in self.blocks:
            h = blk(h, tm)
            if return_layers:
                layers.append(h)
        out = self.ln_f(h)
        return (out, layers) if return_layers else out

    def forward(self, x, coords, mask):
        """返回 (主损失, 次级损失)：主 = 被遮 token 逐格重建 MSE；
        次 = 跨层池化向量 g 再重建一遍被遮格（sec_lambda=0 时次级为 0）。"""
        use_sec = self.sec_head is not None
        out = self.encode(x, coords, mask, return_layers=use_sec)
        h, layers = out if use_sec else (out, None)      # sec_lambda=0 时 encode 只返回表征
        main = F.mse_loss(self.head(h)[mask], x[mask])        # [B,C,T,P]
        if not use_sec:
            return main, main.new_zeros(())
        return main, self.sec_head(layers, self._coord_enc(coords, x), mask, x)

    @torch.no_grad()
    def features(self, x, coords):
        """冻结特征：对 C、T 平均池化 → [B,d]。"""
        return self.encode(x, coords).mean((1, 2))


def random_mask(B, C, T, ratio, device):
    n = int(round(C * T * ratio))
    scores = torch.rand(B, C * T, device=device)
    idx = scores.argsort(1)[:, :n]
    m = torch.zeros(B, C * T, dtype=torch.bool, device=device)
    m.scatter_(1, idx, True)
    return m.reshape(B, C, T)


if __name__ == "__main__":
    cfg = Cfg()
    m = EEGFM(cfg)
    print("params", sum(p.numel() for p in m.parameters()) / 1e6, "M")
    x = torch.randn(2, 64, cfg.T, cfg.P)
    c = torch.rand(2, 64, 3) * 2 - 1
    mk = random_mask(2, 64, cfg.T, cfg.mask_ratio, x.device)
    mloss, sloss = m(x, c, mk)
    print("main", mloss.item(), "sec", sloss.item(), "feat", m.features(x, c).shape)
    cfg.attn = "flat"
    print("flat main", EEGFM(cfg)(x, c, mk)[0].item())
    # 消融开关自检：FFN 重建头 + 关坐标 + 关 RoPE
    cfg2 = Cfg(head_ffn=True, use_coord=False, rope=False)
    m2 = EEGFM(cfg2)
    print("head_ffn+nocoord+norope params", sum(p.numel() for p in m2.parameters()) / 1e6,
          "M  main", m2(x, c, mk)[0].item(), " sec", m2(x, c, mk)[1].item())

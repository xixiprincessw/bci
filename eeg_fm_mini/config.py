"""所有超参放这里，改参数不改代码。

对应简版方案的缩小版：patch 1 秒、1D 卷积 stem、坐标 Fourier 编码、
criss-cross 骨干（Pre-LN / GELU / FFN 4d）、时间头 RoPE、前缀掩码、掩码波形重建。
数据只有 eegmmidb 一个源（160 Hz、64 导），所以不做重采样，P = 160。
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/Users/erqian/Desktop/util_udf/physionet.org/files/eegmmidb/1.0.0")


@dataclass
class Cfg:
    # ---------- 数据 ----------
    data_root: Path = DATA_ROOT
    cache_dir: Path = ROOT / "cache"
    out_dir: Path = ROOT / "runs"
    exclude_subjects: tuple = (88, 92, 100)          # 采样率 128 Hz，跳过
    pretrain_subjects: tuple = tuple(range(1, 81))    # 预训练与探针训练用 1–80
    probe_test_subjects: tuple = tuple(range(81, 110))  # 探针测试用 81–109，预训练没见过
    n_subjects: int = 0           # 0 = 全部；>0 每组只取前 n 个受试者，冒烟测试用
    n_pretrain_subjects: int = 0  # 0 = 跟 n_subjects；>0 只缩预训练人数（数据量消融，探针池不变）
    fs: int = 160
    patch_sec: float = 1.0        # 每导联每秒一个 token
    window_sec: int = 8           # 预训练窗口 T（秒）
    clip_sigma: float = 20.0      # 每通道 z-score 后截幅

    # ---------- 模型 ----------
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4              # criss-cross 下一半看空间、一半看时间
    attn: str = "criss"           # "criss" | "flat"（平铺全注意力，作对照）
    rope: bool = True             # 时间头 Q/K 用 RoPE
    use_coord: bool = True        # 坐标 Fourier 编码开关（False = 空间位置消融）
    head_ffn: bool = False        # 重建头两层 FFN（方案中间档）vs 一层线性
    n_freq: int = 6               # 坐标 Fourier 每轴频率数
    ffn_mult: int = 4
    dropout: float = 0.0

    # ---------- 预训练 ----------
    mask_ratio: float = 0.5
    chan_drop: bool = False       # BIOT 式增广：每窗随机丢 10%–70% 整导联
    sec_lambda: float = 1.0       # 次级损失（跨层 attention pooling 重建）权重，0 = 关闭
    prefix_S: int = -1            # -1 = S=T 全双向；0 = 纯因果；0<S<T 前缀
    batch_size: int = 16
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 200
    seed: int = 0
    device: str = "auto"          # auto → mps > cuda > cpu

    # ---------- 探针 ----------
    probe_tmax: float = 4.0       # 提示后取 4 秒 = 4 个 token
    head_lr: float = 1e-3         # 冻结口径：只训 attention pooling 头
    head_epochs: int = 100
    ft_lr: float = 1e-4           # 微调口径：骨干 + 头一起训
    ft_epochs: int = 10

    @property
    def P(self) -> int:
        return int(self.fs * self.patch_sec)

    @property
    def T(self) -> int:
        return self.window_sec

    def subjects(self, which: str) -> list:
        base = self.pretrain_subjects if which == "train" else self.probe_test_subjects
        subs = [s for s in base if s not in self.exclude_subjects]
        return subs[: self.n_subjects] if self.n_subjects > 0 else subs

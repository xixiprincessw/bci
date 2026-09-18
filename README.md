# MI-Decoder：运动想象 EEG 解码与伪在线评估

基于 PhysioNet EEGBCI（109 被试、64 导、160 Hz）的左手/右手运动想象解码全链路：
批量预处理 → ERD 生理验证与定量 → CSP / FBCSP / EEGNet 被试内解码 → 伪在线因果模拟（CSP / EEGNet） → 跨被试在线解码（70 人训 / 18 人测） → 汇总报告。

每个模块是一个独立可运行的脚本，只依赖上游模块落盘的文件；所有参数集中在 `config.yaml`。

```
EDF ─► 1 preprocess ─► Sxxx-epo.fif ──┬─► 2 validate_erd   ─► 时频图/地形图        ┐
                     Sxxx_clean_raw.fif│   2b quantify_erd  ─► erd_quant.csv        │ 旁路诊断
                                       ├─► 3 decode_csp     ─► csp_results.csv      ┐
                                       └─► 4 decode_eegnet  ─► eegnet_results.csv   │
                     Sxxx_clean_raw.fif ─┬─► 5 pseudo_online  ─► pseudo_online_*.csv  │  (--model csp|eegnet)
                                         └─► 3b decode_cross_subject ─► cross_subject_online.csv
                                                              6 report ─► all_results.csv + report.md
```

模块 5 和 3b 共用 `src/online_models.py`（因果流加载、流式对齐、在线模型、时间顺序重放），保证被试内与跨被试的在线口径完全一致。

## 目录

```
mi_decoder/
├── config.yaml                    # 全部参数：频带 / 窗长 / 模型超参 / 路径
├── requirements.txt
├── src/
│   ├── utils.py                   # 配置加载、日志、目录与文件路径约定
│   ├── preprocess.py              # 1  批量预处理（坏导→插值→重参考→滤波→ICA→Epoch）
│   ├── validate_erd.py            # 2  ERD 时频图 + 地形图（肉眼质检）
│   ├── quantify_erd.py            # 2b ERD 数值化（C3/C4 mu/beta 功率变化百分比）
│   ├── decode_csp.py              # 3  CSP+LDA / FBCSP 被试内 5 折 CV
│   ├── decode_eegnet.py           # 4  EEGNet 被试内 5 折 CV（与模块 3 同划分）
│   ├── online_models.py           # 5/3b 公共件：因果流、流式 EA/RA 对齐、在线模型、重放评估
│   ├── pseudo_online.py           # 5  被试内伪在线：前 70% 训 → 后 30% 流式测（CSP / EEGNet）
│   ├── decode_cross_subject.py    # 3b 跨被试在线：70 人合训 → 18 人流式测，E1~E8
│   └── report.py                  # 6  合并指标、配对检验、出图、report.md
├── notebooks/
│   └── single_subject_demo.ipynb  # 单被试教学版（逐步可视化）
└── results/                       # 运行后生成
    ├── processed/                 # Sxxx-epo.fif, Sxxx_clean_raw.fif
    ├── figures/                   # 每被试 ERD 图、伪在线曲线、汇总图
    ├── metrics/                   # 各模块 csv
    └── report.md
```

## 快速开始

```bash
cd mi_decoder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 把 config.yaml 里 data.data_root 指向已下载的 EEGBCI 目录（<root>/Sxxx/SxxxRxx.edf）

python -m src.preprocess              # 1   约 1 min/被试
python -m src.validate_erd            # 2   出图
python -m src.quantify_erd            # 2b  出数
python -m src.decode_csp              # 3   几分钟
python -m src.decode_eegnet           # 4   CPU 数十分钟；MPS/CUDA 更快
python -m src.pseudo_online           # 5   CSP，几分钟
python -m src.pseudo_online --model eegnet          # 5   EEGNet，约 15 min（MPS）
python -m src.decode_cross_subject                  # 3b  E1,E3,E5,E6,E8，几分钟
python -m src.decode_cross_subject --exps E2        # 3b  FBCSP
python -m src.decode_cross_subject --exps E4,E7     # 3b  EEGNet，十几分钟到半小时
python -m src.report                  # 6
```

单独处理指定被试：`python -m src.preprocess --subjects 1 2 3`
冒烟：`python -m src.pseudo_online --max-subjects 3`，`python -m src.decode_cross_subject --exps E1 --max-train 5 --max-test 2`

## 模块说明

每个模块按 **功能 / 输入 / 输出 / 关键设计** 四项说明。

### 1. `preprocess.py` — 批量预处理

| 项 | 内容 |
|---|---|
| 功能 | 把每个被试 runs 4/8/12 的 EDF 拼成一条连续数据，清洗后切成 epoch |
| 输入 | `<data_root>/Sxxx/SxxxR{04,08,12}.edf`；事件来自 EDF+ annotations（T1=左手, T2=右手） |
| 输出 | `results/processed/Sxxx-epo.fif`（−1~4 s epoch，无基线校正）<br>`results/processed/Sxxx_clean_raw.fif`（ICA 清洗后的连续数据，供模块 5）<br>`results/metrics/preprocess_log.csv`：坏导数、剔除的 ICA 成分数、保留/总试次、采样率 |
| 处理顺序 | ① 电极定位 standard_1005 → ② 坏导检测（通道对数方差 z>2.6）→ ③ 球面样条插值 → ④ 平均参考 → ⑤ 1~45 Hz 零相位 FIR → ⑥ ICA 20 成分，用 Fp1/Fp2 做 EOG 代理自动定位眼电成分并剔除 → ⑦ Epoch，>350 µV 试次整个丢弃 |

顺序约束：坏导必须在重参考前修好，否则平均参考被坏导拉偏；ICA 必须在宽带数据上做，眨眼主能量在 1~5 Hz，先窄带再 ICA 就找不到眼电成分了。350 µV 阈值作用在 ICA 之后，只拦漏网的极端伪迹。

数据链路（本次运行）：109 被试 → 排除 3 个 128 Hz 采样的 → 106 完成预处理 → 4 人（S009/S048/S053/S091）全部试次被幅值阈值剔光 → 18 人保留试次 <20 → **88 人进入解码**。

### 2. `validate_erd.py` — ERD 时频验证

| 项 | 内容 |
|---|---|
| 功能 | 证明 epoch 里有可解码的神经活动而不是伪迹 |
| 输入 | `Sxxx-epo.fif` |
| 输出 | `results/figures/Sxxx_erd_tfr.png`（C3/C4 时频图，percent 基线归一）<br>`results/figures/Sxxx_erd_topo.png`（左/右手 mu 段 ERD 地形图） |
| 关键设计 | Morlet 小波 6~32 Hz；**逐试次算功率再跨试次平均**（ERD 非锁相，先平均波形会抵消掉）；基线 −1~0 s，除法口径消 1/f |

验收四维证据：空间上聚焦 C3/C4、时间上锁定任务期、频率上落在 mu/beta、左右手对侧翻转。全头弥漫的"ERD"通常是肌电。

### 2b. `quantify_erd.py` — ERD 数值化

| 项 | 内容 |
|---|---|
| 功能 | 把"看图判蓝"换成数字，避免 colorbar 自动缩放误导 |
| 输入 | `Sxxx-epo.fif` |
| 输出 | `results/metrics/erd_quant.csv` |
| 列 | `mu_contra_pct`：左手@C4 与右手@C3 在 8~13 Hz、0.5~3.5 s 的平均功率变化 %（期望显著为负）<br>`mu_ipsi_pct`：同侧电极同口径<br>`mu_lat_pct` = contra − ipsi（期望为负）<br>`beta_*`：同上，13~30 Hz<br>`mu_class`：按 mu_contra_pct 分四档 strong(≤−20) / medium(≤−10) / weak(≤0) / none_or_reversed(>0)，经验阈值，非文献标准 |

`erd_quant.csv` **不被任何下游模块读取**，纯诊断用途。本次 92 人有效：strong 40 / medium 16 / weak 12 / none_or_reversed 24。

已知局限：8~13 Hz 扁平平均对 ERD 峰值落在 beta 的被试（本数据集约 46%）会稀释甚至翻转符号；`beta_contra_pct` 列即为此设的对照。

### 3. `decode_csp.py` — CSP+LDA / FBCSP 被试内解码

| 项 | 内容 |
|---|---|
| 功能 | 每个被试独立训练、5 折分层 CV 评估 |
| 输入 | `Sxxx-epo.fif`（试次 <20 跳过） |
| 输出 | `results/metrics/csp_results.csv`：`csp_acc/kappa`, `fbcsp_acc/kappa` |
| CSP | 8~30 Hz 窄带 → 裁 0.5~3.5 s → CSP 6 分量（头 3 尾 3，log 方差）→ LDA；整条 Pipeline 在 `cross_val_predict` 内拟合，测试折不参与任何拟合 |
| FBCSP | 子带 [4-8, 8-12, 12-16, 16-22, 22-30]，每子带各提 4 个 CSP 特征后拼接 → LDA，手动逐折训练 |
| 正则 | `CSP(reg="ledoit_wolf")`：平均参考 / 插值 / ICA 使 64 导协方差有效秩 <64，不收缩会报 `LinAlgError: not positive definite` |

报告口径为折外预测池化后的 acc（41 个试次每个恰好被预测一次），与"五折 acc 简单平均"略有差异。每折 CSP 学出的 64×64 矩阵各不相同，属正常现象。

### 3b. `decode_cross_subject.py` — 跨被试在线解码

| 项 | 内容 |
|---|---|
| 功能 | 回答"用别人的数据训的模型，放到一个新人身上实时用还剩多少"。与模块 5 同一条因果路径、同一套指标，只换训练集 |
| 划分 | 88 个可用被试（有 clean_raw 且 ≥20 试次）按 `random_state` 随机排列，18 人测试、70 人训练，写入 `cross_subject_split.json`。测试被试的任何数据（含无标签信号）都不进训练 |
| 输入 | `Sxxx_clean_raw.fif` → 因果 `sosfilt`；训练取每试次 0.5~3.5 s 窗（EEGNet 在其中滑 2 s 窗）；测试被试**全部试次**流式重放 |
| 输出 | `results/metrics/cross_subject_online.csv`：每行一个测试被试，每个实验 `Ex_acc_at_end / Ex_best_acc / Ex_best_offset / Ex_itr_bits_min / Ex_rest_fpr`，并拼入同一人的 `within_csp_*` / `within_eegnet_*`（模块 5 后 30% 试次结果）作对照<br>`cross_subject_online_curves.csv`：各实验延迟-精度曲线<br>`results/figures/cross_subject_latency_curves.png`：已跑实验 + 同一批人的被试内曲线画在一张图（`--plot-only` 可用已有 csv 重画）|

实验矩阵（`--exps`，多次运行按列合并到同一张表）：

| # | 模型 | 训练侧 | 推理时用目标什么 |
|---|---|---|---|
| E1 | CSP(6)+LDA | 70 人合训 | 无（零校准） |
| E2 | FBCSP+LDA | 同上，5 子带各一条因果流 | 无 |
| E3 | 黎曼切空间+LR | 同上 | 无 |
| E4 | EEGNet 1~40 Hz | 同上，70 人中按人留 8 人早停 | 无 |
| E5 | CSP+LDA + 在线 EA | 每人用自己整条流的平均协方差白化后合训 | 每个窗用目标流 t 之前递归估计的 R_t 做 `R_t^{-1/2} w`（无标签、因果） |
| E6 | 黎曼+LR + 在线重定心 | 同上，参考点为测地线递归黎曼均值 | 同上 |
| E7 | EEGNet + 在线 EA | 同 E5 输入侧 | 同 E5 |
| E8 | E5 + 自适应 LDA | 同 E5 | 每个试次最后一个决策点后用伪标签递归更新类均值（Vidaurre 2011） |

E5~E8 训练集与 E1~E4 完全相同，只改推理，对比出"在线自适应值多少"。参考协方差沿整条流按 2 s 切不重叠窗、Ledoit-Wolf 估计后递归更新；训练被试用整条流的终值，测试被试在时刻 t 只用结束于 t 之前的窗，两侧收敛到同一统计量。

### 4. `decode_eegnet.py` — EEGNet 被试内解码

| 项 | 内容 |
|---|---|
| 功能 | 端到端 CNN，与模块 3 用同一 CV 划分，配对可比 |
| 输入 | 同一份 `Sxxx-epo.fif`（ICA 清洗后），1~40 Hz 宽带，裁 0.5~3.5 s，V→µV |
| 输出 | `results/metrics/eegnet_results.csv`：`eegnet_acc/kappa` |
| 模型 | braindecode EEGNetv4，2610 参数；时间卷积 [1×64]（学滤波器）→ 深度空间卷积 [64×1]（学空间滤波器）→ 可分离卷积 → 池化 → 全连接 |
| 防泄漏 | 通道级 z-score 只用训练折统计量；训练折内再切 20% 做早停验证 |

### 5. `pseudo_online.py` — 伪在线模拟

| 项 | 内容 |
|---|---|
| 功能 | 把离线冠军方案放到因果约束下，看能不能实时用 |
| 输入 | `Sxxx_clean_raw.fif`（连续数据，不是 epoch） |
| 输出 | `results/metrics/pseudo_online_results.csv`：`best_offset, best_acc, acc_at_end, itr_bits_min, rest_fpr`<br>`results/metrics/pseudo_online_curves.csv`：每被试每个 offset 的 acc<br>`results/figures/Sxxx_pseudo_online.png`<br>`--model eegnet` 时文件名加 `_eegnet` 后缀 |
| 因果化 | 滤波用 `sosfilt`（只用过去样本）替代离线的零相位 `filtfilt`；**训练也在因果滤波后的数据上做**，保证训练/推理特征分布一致 |
| 流程 | 前 70% 试次训练、后 30% 按时间顺序流式测试（不 shuffle）；决策时刻从想象开始后 0.5 s 扫到 4.0 s，步长 0.25 s，每个时刻回看 2 s 窗口盲判 |
| EEGNet | 输入长度固定 = 2 s 窗，训练窗在 0.5~3.5 s 内按 0.25 s 滑动取样（每试次 5 窗）；1~40 Hz；早停验证集从 70% 试次中**按试次**留出 |
| ITR | Wolpaw 公式，bits/min |
| 误触发 | 静息期滑窗，后验 ≥0.7 计为一次误触发 |

CSP / LDA / ICA 这类空间投影逐采样点独立计算，天然因果，离线权重直接复用；只有滤波是沿时间轴用了未来样本的操作，必须改成因果实现。因果 IIR 的群延迟实测约 60 ms，小于 250 ms 决策步长，不是在线掉点的主因。ICA 在预处理阶段对整段录音拟合（无监督），未按 70/30 切分；严格的在线系统应在校准段拟合后冻结。模型不落盘。

### 6. `report.py` — 汇总报告

| 项 | 内容 |
|---|---|
| 输入 | `csp_results.csv`, `eegnet_results.csv`, `pseudo_online_results.csv` |
| 输出 | `results/metrics/all_results.csv`（按 subject outer merge）<br>`results/figures/paired_csp_vs_eegnet.png`, `accuracy_distribution.png`<br>`results/report.md`：均值 ± 标准差、Wilcoxon 配对检验、离线→伪在线掉点、每被试明细表 |

## 当前结果（88 被试，被试内 5 折 CV）

| 方法 | acc 均值 | 备注 |
|---|---|---|
| CSP+LDA | 0.598 ± 0.170 | min 0.333, max 1.000 |
| FBCSP | 0.584 ± 0.130 | vs CSP：Wilcoxon p=0.42，不显著 |
| EEGNet | 0.573 ± 0.119 | vs CSP：p=0.35，不显著；88 人中 41 人 EEGNet 更高 |
| 伪在线 CSP acc_at_end | 0.592 | 4.0 s 决策点，106 人；rest FPR 0.89 |
| 伪在线 EEGNet acc_at_end | 0.537 | 同上；rest FPR 0.17，best_acc 0.706 |

ERD 与可解码性：`csp_acc` 与 `mu_contra_pct` 的 Pearson r = −0.44（ERD 越深，解码越准）；按 mu_class 分组 strong 0.674 > medium 0.575 > none_or_reversed 0.537 > weak 0.493。这个相关性没有泄漏——ERD 量化和解码用的是不同的量、且都没看对方。

CSP 与 EEGNet 互补：CSP 强的经典被试 EEGNet 往往更低（S029 1.00→0.64），CSP 接近随机的被试 EEGNet 能拉回（S101 0.36→0.78, S105 0.48→0.86）。

### 跨被试在线（70 人训 / 18 人测，测试被试全部 45 试次流式重放）

同一 18 人的对照：离线 5 折 CSP 0.589 / EEGNet 0.601；被试内在线（后 30% 共 14 试次）CSP 0.615 ± 0.159 / EEGNet 0.536 ± 0.089。

| # | 方法 | acc_at_end (4.0 s) | 1.5~3.0 s 均值 | best_acc | rest FPR | ≥0.6 人数 |
|---|---|---|---|---|---|---|
| E1 | CSP+LDA 零校准 | 0.521 ± 0.063 | 0.549 | 0.605 | 0.23 | 4/18 |
| E2 | FBCSP+LDA | 0.502 ± 0.066 | 0.566 | 0.633 | 0.37 | 0/18 |
| E3 | 黎曼切空间+LR | 0.544 ± 0.083 | 0.588 | 0.653 | 0.84 | 4/18 |
| E4 | EEGNet | 0.568 ± 0.093 | 0.605 | 0.719 | 0.34 | 9/18 |
| E5 | CSP+LDA + 在线 EA | 0.572 ± 0.131 | **0.663** | 0.725 | 0.38 | 8/18 |
| E6 | 黎曼+LR + 在线重定心 | 0.562 ± 0.107 | 0.612 | 0.688 | 0.80 | 6/18 |
| E7 | EEGNet + 在线 EA | **0.583** ± 0.092 | 0.610 | 0.693 | 0.38 | 8/18 |
| E8 | E5 + 自适应 LDA | 0.560 ± 0.128 | 0.631 | 0.699 | 0.56 | 8/18 |

方法层（18 人配对 Wilcoxon，p 只作参考）：

- **零校准掉点**：CSP 0.615→0.521；FBCSP 最差（无人过 0.6），子带切细后跨人频带差异被放大。
- **EEGNet 是零校准下唯一均匀的增益**：E4−E1 均值 +0.047、中位 +0.044，12/18 人（p=0.06）。
- **EA 增益高方差**：E5−E1 均值 +0.051 但中位仅 +0.02；S094/S012/S015 +0.18~0.27，S038 −0.24，其余人基本不动。对 EEGNet 增益更小（E7−E4 +0.015）。
- **排名依决策时刻而变**：末端 E7 最高，中段 1.5~3.0 s E5 一枝独秀（0.663，峰 0.681 @2.0 s，整段高于被试内 CSP）。EEGNet 逐点抖动是 CSP 的两倍（相邻 offset 差 E4 0.061 vs E1 0.026），固定 2 s 输入对窗口落点敏感。E4~E8 之间 ≤0.02 的末端差别不能当结论。
- **自适应 LDA 无益**（E8−E5 −0.011，FPR 0.38→0.56）；**黎曼后验不可用**（FPR 0.80+，acc 不差但静息期频繁 ≥0.7）。
- **EEGNet 对在线约束比 CSP 敏感**：同 18 人离线 0.601 → 被试内在线 0.536（训练试次 36→31 再切验证、窗 3 s→2 s、因果滤波），CSP 同样约束下 0.589→0.615 不掉。跨被试 E4 0.568 高于被试内在线 0.536，但低于离线 0.601：70 人数据量补回了在线约束的一部分，不是"跨被试优于被试内"。

被试层（为什么方法差异 ≤0.06 而人与人差 0.4~0.87）：

- **ERD 强度预测迁移**：`mu_contra_pct` 与 E7 r=−0.58 (p=0.01)、E5 −0.45、E4 −0.43，与被试内 W4 −0.53，与上方 88 人的 r=−0.44 同一条线。
- **6 人 none_or_reversed（占测试集 1/3）所有方法都在 0.51**：被试内 0.52、E1/E5/E4 0.51。不是迁移失败，是没有可迁移的信号；跨被试的天花板由目标被试自己的信号质量定。
- **离线可解码性预测可迁移性**：离线 5 折 CSP acc 与 E1 / E4 的 Spearman 均为 0.65 (p<0.01)；被试内在线 W4 仅 14 试次，与 E1 只有 0.30，不宜做逐人对照。
- EA 唯一大负例 S038 的 `mu_contra_pct`=+1.3（无 ERD），白化可能在放大噪声；单人证据，作假设。
- rest FPR 与 acc 逐人无相关（rho ≈ ±0.2）：FPR 是分类器校准属性，拒识阈值应按模型标定而非按人。

图：`figures/cross_subject_latency_curves.png`（E1~E8 + 同 18 人被试内曲线）。

## 阅读结果时的注意事项

- **`best_acc` 是乐观数字**：它是在 14 个测试试次上、15 个 offset 里挑最大值，含选择偏差，均值 0.732 高于离线 0.598 就是这个原因。看在线性能请用 `acc_at_end` 或整条 `pseudo_online_curves.csv`。`report.md` 里"离线→伪在线掉点"一栏基于 best_acc，同样偏乐观。
- **`rest_fpr` 均值 0.89**（被试内 CSP）：静息期几乎每个窗口都被判为"有意图"，说明 LDA 后验在 0.7 阈值下几乎不拒识。上线需要拒识策略，这个数字如实反映了当前方案的缺口。跨被试 CSP 的 rest FPR 反而只有 0.23~0.38，不是变好了，而是合训的 LDA 判别边界更钝、后验整体向 0.5 收缩——acc 和 FPR 要一起看。
- **跨被试只做了一次 70/18 切分**：18 人均值标准误约 0.02~0.03，稳的只有方向：EEGNet 零校准最好、EA 对部分人大幅有效、自适应 LDA 无益、FBCSP 最差。被试内数字基于每人后 30% 的 14 试次，跨被试基于 45 试次，两者直接比大小时要带上这句。
- **S029 CSP/FBCSP 均为 1.000**：过于完美，需要看 CSP pattern 地形图排除肌电伪迹冒充。
- **ERD 不能用作试次级过滤**：按 ERD 强弱挑试次再评估 = 用与标签相关的量做选择，是泄漏。被试级剔除只能作为诚实披露的 BCI illiteracy 分析，不能当主结果。

## 配置速查（config.yaml）

| 段 | 关键参数 |
|---|---|
| `data` | `data_root`, `n_subjects=109`, `exclude_subjects=[88,92,100]`, `runs=[4,8,12]` |
| `preprocess` | `l_freq=1, h_freq=45, bad_z_thresh=2.6, ica_n_components=20, epoch −1~4 s, reject_eeg=350 µV` |
| `erd` | `freq 6~32 Hz, baseline [−1,0], plot_channels [C3,C4]` |
| `decode` | `band [8,30], crop [0.5,3.5], csp_components=6, fbcsp_bands, fbcsp_components=4, cv_folds=5` |
| `eegnet` | `band [1,40], max_epochs=100, patience=15, lr=1e-3, batch_size=16, val_ratio=0.2` |
| `pseudo_online` | `win_sec=2.0, step_sec=0.25, train_ratio=0.7, offsets [0.5,4.0,0.25], conf_threshold=0.7, eegnet_train_step_sec=0.25, adapt_eta=0.05` |
| `cross_subject` | `test_ratio=0.2, eegnet_val_subjects=8` |

## 环境

Python 3.12，MNE 1.13，scikit-learn，PyTorch，braindecode ≥0.8，pyriemann（模块 3b 的黎曼方法与流式对齐）。macOS 上 EEGNet 自动使用 MPS。

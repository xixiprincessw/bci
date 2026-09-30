# bci

脑机接口信号解码。

| 目录 | 内容 |
|---|---|
| [`mi_decoder/`](mi_decoder/README.md) | PhysioNet EEGBCI 左/右手运动想象解码全链路：批量预处理 → ERD 验证 → CSP / FBCSP / EEGNet 被试内解码 → 伪在线因果模拟 → 跨被试在线解码（70 人训 / 18 人测，E1~E8） |
| `mi_decoder/notebooks/single_subject_demo.ipynb` | 单被试 S001 教学版 notebook（逐步可视化，配套文章《运动想象 EEG 解码·上篇》） |
| [`eeg_fm_mini/`](eeg_fm_mini/) | EEG 基础模型缩小实现（1.07M 参数，criss-cross 骨干）：eegmmidb 80 人无标注预训练 → 26 人探针/微调（10%/30%/100% 标注档）与 11 组组件消融，配套文章《运动想象 EEG 解码·下篇》；`cache/`（预处理窗口）与 `runs/`（checkpoint）不进版本库 |
| `eeg_fm_mini/eeg_fm_mini.ipynb`、`eeg_fm_mini/walkthrough.ipynb` | 教学版 notebook：前者全流程跑通（配置 → 数据 → 模型 → 预训练 → 探针/微调，每段先讲做什么再跑），后者按数据流逐步拆解（切窗 → stem → 坐标编码 → 掩码 → 注意力 → 重建，每步打印形状并作图） |

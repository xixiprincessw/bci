# bci

脑机接口信号解码。

| 目录 | 内容 |
|---|---|
| [`mi_decoder/`](mi_decoder/README.md) | PhysioNet EEGBCI 左/右手运动想象解码全链路：批量预处理 → ERD 验证 → CSP / FBCSP / EEGNet 被试内解码 → 伪在线因果模拟 → 跨被试在线解码（70 人训 / 18 人测，E1~E8） |
| `mi_decoder/notebooks/single_subject_demo.ipynb` | 单被试 S001 教学版 notebook（逐步可视化，配套文章《运动想象 EEG 解码·上篇》） |
| `mi_decoder/results/figs/` | 上篇文章引用的 12 张图 |

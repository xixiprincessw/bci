#!/bin/sh
# 消融实验批量脚本：每臂 = 一次预训练 + 一套三档探针，串行执行。
# 主读数：微调 10%/30% 档 pretrained-random 差值；副读数：冻结档（看 nosec/nocoord/headffn）。
# 监控：总进度看 runs/ablations_master.log，单臂明细看 runs/<tag>_run.log 与 runs/<tag>/log.txt。
PY=/Users/erqian/Desktop/util_udf/bci/mi_decoder/.venv/bin/python
cd /Users/erqian/Desktop/util_udf/bci/eeg_fm_mini

run() {
  tag=$1; shift
  echo "==== $tag start $(date '+%F %T') flags: $* ===="
  { $PY pretrain.py --tag "$tag" "$@" && $PY probe.py --tag "$tag"; } > "runs/${tag}_run.log" 2>&1
  rc=$?
  echo "==== $tag done  $(date '+%F %T') rc=$rc ===="
}

# --- 零代码开关臂 ---
run abl_norope   --rope false          # 2 时间位置编码价值
run abl_nosec    --sec_lambda 0        # 3 次级损失价值（预期动冻结档）
run abl_mask75   --mask_ratio 0.75     # 4 掩码率 0.5 vs 0.75
run abl_prefix3  --prefix_S 3          # 5 前缀掩码（三期形态预演）

# --- 小改代码臂 ---
run abl_nocoord  --use_coord false     # 6 电极坐标编码价值
run abl_headffn  --head_ffn true       # 7 重建头两层 FFN（方案选型）
run abl_chandrop --chan_drop true      # 8 BIOT 式通道丢弃增广

# --- 数据量 scaling（探针池恒为全量 1-80）---
run abl_sub10    --n_pretrain_subjects 10
run abl_sub20    --n_pretrain_subjects 20
run abl_sub40    --n_pretrain_subjects 40

# --- 最重的放最后：平铺全注意力（512 token 全互见，比 criss 慢）---
run abl_flat     --attn flat           # 1 criss-cross vs 平铺

echo "ALL_DONE $(date '+%F %T')"

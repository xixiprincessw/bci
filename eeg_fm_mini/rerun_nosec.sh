#!/bin/sh
# 等主消融链（PID $1）结束后补跑修复过的 abl_nosec 臂
PY=/Users/erqian/Desktop/util_udf/bci/mi_decoder/.venv/bin/python
cd /Users/erqian/Desktop/util_udf/bci/eeg_fm_mini
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "==== abl_nosec rerun start $(date '+%F %T') ====" >> runs/ablations_master.log
{ $PY pretrain.py --tag abl_nosec --sec_lambda 0 && $PY probe.py --tag abl_nosec; } > runs/abl_nosec_run.log 2>&1
rc=$?
echo "==== abl_nosec rerun done $(date '+%F %T') rc=$rc ====" >> runs/ablations_master.log

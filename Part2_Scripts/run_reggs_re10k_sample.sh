#!/bin/bash
# 简单脚本：按顺序运行 RegGS 的 infer/refine/metric（re10k sample）
set -e

REGGS_DIR="/root/autodl-tmp/AIAA3201_3DGS_Project/third_party/RegGS"

cd "${REGGS_DIR}"

echo ">>> [1/3] Inference stage: run_infer.py"
CUDA_VISIBLE_DEVICES=0 python3 run_infer.py config/re10k.yaml

echo ">>> [2/3] Refinement stage: run_refine.py"
CUDA_VISIBLE_DEVICES=0 python3 run_refine.py --checkpoint_path output/re10k/000c3ab189999a83

echo ">>> [3/3] Evaluation stage: run_metric.py"
CUDA_VISIBLE_DEVICES=0 python3 run_metric.py --checkpoint_path output/re10k/000c3ab189999a83

echo "✅ Done."


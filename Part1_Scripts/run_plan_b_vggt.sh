#!/bin/bash
# 【脚本说明】方案 B：VGGT 初始化的后半段自动化流水线

DATASET_BASE_DIR="/root/autodl-fs"
EXP_BASE_DIR="/root/autodl-tmp/AIAA3201_3DGS_Project/experiments"
CODE_DIR="/root/autodl-tmp/AIAA3201_3DGS_Project/third_party/gaussian-splatting"

# 同样遍历这 5 个数据集
DATASETS=("Re10k-1" "405841" "DL3DV-2" "360_extra_scenes" "tandt_db")

for DATASET_NAME in "${DATASETS[@]}"; do
    echo "================================================================"
    echo " 🚀 开始处理数据集: ${DATASET_NAME} | 方案: VGGT 初始化"
    echo "================================================================"
    
    # 注意这里的输入路径！我们在名字后面加了 _vggt，去找队友的数据
    INPUT_PATH="${DATASET_BASE_DIR}/${DATASET_NAME}_vggt"
    OUTPUT_PATH="${EXP_BASE_DIR}/exp_Part1_${DATASET_NAME}_vggt"
    
    # 跳过第一站，直接进入第二站：3DGS 训练
    echo ">>> [1/3] 使用队友的 VGGT 点云开始 3DGS 训练..."
    time python ${CODE_DIR}/train.py -s ${INPUT_PATH} -m ${OUTPUT_PATH} --eval
    
    # 第三站：离线渲染照片
    echo ">>> [2/3] 开始渲染测试集照片..."
    python ${CODE_DIR}/render.py -m ${OUTPUT_PATH}
    
    # 第四站：计算 PSNR 等指标
    echo ">>> [3/3] 开始计算评估指标..."
    python ${CODE_DIR}/metrics.py -m ${OUTPUT_PATH}
    
    echo "✅ ${DATASET_NAME} (VGGT) 处理完毕！模型保存在: ${OUTPUT_PATH}"
    echo "----------------------------------------------------------------"
done
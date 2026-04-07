#!/bin/bash
# 遇到报错立刻停止
set -e

CODE_DIR="/root/autodl-tmp/AIAA3201_3DGS_Project/third_party/Scaffold-GS"
EXP_BASE_DIR="/root/autodl-tmp/AIAA3201_3DGS_Project/experiments_Scaffold_GS"

# 确保输出主目录存在
mkdir -p "${EXP_BASE_DIR}"

PATHS=(
    # "/root/autodl-fs/DL3DV-2"
    # "/root/autodl-fs/Re10k-1"
     "/root/autodl-fs/405841/FRONT"
    # "/root/autodl-fs/tandt_train_ours"
    "/root/autodl-fs/tandt_truck_ours"
    "/root/autodl-fs/360_flowers_ours_4"
    "/root/autodl-fs/360_treehill_ours_4"
)

NAMES=(
    # "DL3DV-2"
    # "Re10k-1"
    "405841"
    # "tandt_train_ours"
    "tandt_truck_ours"
    "360_flowers_ours_4"
    "360_treehill_ours_4"
)

echo "======================================================="
echo " 🚀 开始批量训练 Scaffold-GS "
echo " 🕒 时间: $(date)"
echo "======================================================="

for i in "${!PATHS[@]}"; do    
    TARGET_PATH="${PATHS[$i]}"
    DATASET_NAME="${NAMES[$i]}"
    
    # --- 核心逻辑：自动检查并生成不冲突的文件夹名 ---
    BASE_OUTPUT_NAME="exp_Scaffold_${DATASET_NAME}_colmap"
    OUTPUT_DIR="${EXP_BASE_DIR}/${BASE_OUTPUT_NAME}"
    
    COUNTER=2
    # 如果文件夹已存在，则进入循环寻找新名字
    while [ -d "$OUTPUT_DIR" ]; do
        NEW_NAME="exp_${COUNTER}_Scaffold_${DATASET_NAME}_colmap"
        OUTPUT_DIR="${EXP_BASE_DIR}/${NEW_NAME}"
        ((COUNTER++))
    done
    
    echo ""
    echo "======================================================="
    echo " 🚀 正在处理 [${DATASET_NAME}]"
    echo " 📍 数据路径: ${TARGET_PATH}"
    echo " 📍 输出路径: ${OUTPUT_DIR}"
    echo "======================================================="
    
    echo ">>> [1/3] 执行 Scaffold-GS 训练..."
    #time python "${CODE_DIR}/train.py" -s "${TARGET_PATH}" -m "${OUTPUT_DIR}" 
    time python "${CODE_DIR}/train.py" -s "${TARGET_PATH}" -m "${OUTPUT_DIR}" --eval --voxel_size 0.001 --appearance_dim 0
    
    echo ">>> [2/3] 离线渲染测试视角..."
    python "${CODE_DIR}/render.py" -m "${OUTPUT_DIR}"
    
    echo ">>> [3/3] 计算 PSNR/SSIM/LPIPS..."
    python "${CODE_DIR}/metrics.py" -m "${OUTPUT_DIR}"
    
    echo "✅ [${DATASET_NAME}] 全部流水线跑完！"
done

echo "🎉 所有 7 个数据集全部通过 Scaffold-GS 处理完毕！"

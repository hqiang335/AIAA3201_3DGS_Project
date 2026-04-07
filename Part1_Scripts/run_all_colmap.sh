#!/bin/bash
# 【精准制导沙盒版】全自动流水线：从零开始初始化、训练、评估所有 5 个场景
# 👇 [新增 1] 安全锁：遇到任何一个报错（哪怕是找不到文件），立刻终止整个脚本！不会执行到最后的关机。
set -e

CODE_DIR="/root/autodl-tmp/AIAA3201_3DGS_Project/third_party/gaussian-splatting"
EXP_BASE_DIR="/root/autodl-tmp/AIAA3201_3DGS_Project/experiments"

# 极其关键：第一行的 405841 路径现在深入到了 FRONT！
PATHS=(
    # "/root/autodl-fs/405841/FRONT"
    # "/root/autodl-fs/DL3DV-2"
    "/root/autodl-fs/Re10k-1"
    # "/root/autodl-fs/tandt_truck_ours"
    # "/root/autodl-fs/tandt_train_ours"
    # "/root/autodl-fs/360_treehill_ours_4"
    # "/root/autodl-fs/360_flowers_ours_4"
)

NAMES=(
    # "405841"
    # "DL3DV-2"
    "Re10k-1"
    # "tandt_truck"
    # "tandt_train"
    # "360_treehill_4"
    # "360_flowers_4"
)

for i in "${!PATHS[@]}"; do    
    TARGET_PATH="${PATHS[$i]}"
    DATASET_NAME="${NAMES[$i]}"
    
    # --- 核心逻辑：自动检查并生成不冲突的文件夹名 ---
    BASE_OUTPUT_NAME="exp_Part1_${DATASET_NAME}_colmap"
    OUTPUT_DIR="${EXP_BASE_DIR}/${BASE_OUTPUT_NAME}"
    
    COUNTER=2
    # 如果文件夹已存在，则进入循环寻找新名字
    while [ -d "$OUTPUT_DIR" ]; do
        # 构造新名字，如：exp_2_Part1_tandt_truck_colmap
        NEW_NAME="exp_${COUNTER}_Part1_${DATASET_NAME}_colmap"
        OUTPUT_DIR="${EXP_BASE_DIR}/${NEW_NAME}"
        ((COUNTER++))
    done
    
    echo "======================================================="
    echo " 🚀 正在处理 [${DATASET_NAME}]"
    echo " 📍 数据路径: ${TARGET_PATH}"
    echo " 📍 输出路径: ${OUTPUT_DIR}"
    echo "======================================================="
    
    # echo ">>> [1/4] 执行 COLMAP 初始化提取点云..."
    # time python ${CODE_DIR}/convert.py -s ${TARGET_PATH}
    
    echo ">>> [2/4] 执行 3DGS 训练..."
    time python ${CODE_DIR}/train.py -s ${TARGET_PATH} -m ${OUTPUT_DIR} --eval
    
    echo ">>> [3/4] 离线渲染测试视角..."
    python ${CODE_DIR}/render.py -m ${OUTPUT_DIR}
    
    echo ">>> [4/4] 计算 PSNR/SSIM/LPIPS..."
    python ${CODE_DIR}/metrics.py -m ${OUTPUT_DIR}
    
    echo "✅ [${DATASET_NAME}] 全部流水线跑完！"
done
# echo "🎉 所有 5 个数据集全部处理完毕！准备自动关机省钱..."
# shutdown
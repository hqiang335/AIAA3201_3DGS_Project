#!/bin/bash
# 按显式列表批量运行 RegGS 配置
# 顺序：run_infer -> run_refine -> run_metric
set -e

REGGS_DIR="/root/autodl-tmp/AIAA3201_3DGS_Project/third_party/RegGS"
CONFIG_DIR="${REGGS_DIR}/config"
GPU_ID="${GPU_ID:-0}"
DO_REFINE="${DO_REFINE:-1}"   # 1=运行优化阶段, 0=跳过
DO_METRIC="${DO_METRIC:-1}"   # 1=运行评估阶段, 0=跳过

# 显式配置列表：通过注释/取消注释灵活选择本次训练数据
CONFIG_FILES=(
  # "flowers_360.yaml"
  # "treehill_360.yaml"
  # "tandt_train.yaml"
  # "tandt_truck.yaml"
  # "re10k.yaml"
  "dl3dv.yaml"
  # "front_405841.yaml"
)

if [ ! -d "$CONFIG_DIR" ]; then
  echo "❌ 配置目录不存在: $CONFIG_DIR"
  exit 1
fi

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
  echo "❌ CONFIG_FILES 为空，请至少保留一个配置名。"
  exit 1
fi

cd "$REGGS_DIR"

echo "======================================================="
echo " 🚀 开始按列表批量运行 RegGS configs"
echo " 📁 Config dir: ${CONFIG_DIR}"
echo " 🖥️  CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo " ⚙️  DO_REFINE=${DO_REFINE}, DO_METRIC=${DO_METRIC}"
echo "======================================================="

for cfg_name in "${CONFIG_FILES[@]}"; do
  cfg="${CONFIG_DIR}/${cfg_name}"
  if [ ! -f "$cfg" ]; then
    echo "❌ 配置文件不存在: $cfg"
    continue
  fi

  # 从 yaml 中抽取 data.output_path（当前配置格式固定为两空格缩进）
  checkpoint_path="$(sed -n 's/^  output_path:[[:space:]]*//p' "$cfg" | head -n 1)"
  checkpoint_path="${checkpoint_path%/}"  # 去掉末尾 /

  if [ -z "$checkpoint_path" ]; then
    echo "❌ ${cfg_name} 中未找到 data.output_path，跳过。"
    continue
  fi

  echo ""
  echo "======================================================="
  echo " 📌 当前配置: ${cfg_name}"
  echo " 📌 checkpoint_path: ${checkpoint_path}"
  echo "======================================================="

  echo ">>> [1/3] The inference stage:"
  echo "CUDA_VISIBLE_DEVICES=${GPU_ID} python3 run_infer.py config/${cfg_name}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 run_infer.py "config/${cfg_name}"

  if [ "$DO_REFINE" -eq 1 ]; then
    echo ">>> [2/3] The refinement stage:"
    echo "CUDA_VISIBLE_DEVICES=${GPU_ID} python3 run_refine.py --checkpoint_path ${checkpoint_path}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 run_refine.py --checkpoint_path "${checkpoint_path}"
  else
    echo ">>> [2/3] The refinement stage: [跳过]"
  fi

  if [ "$DO_METRIC" -eq 1 ]; then
    echo ">>> [3/3] The evaluation stage:"
    echo "CUDA_VISIBLE_DEVICES=${GPU_ID} python3 run_metric.py --checkpoint_path ${checkpoint_path}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 run_metric.py --checkpoint_path "${checkpoint_path}"
  else
    echo ">>> [3/3] The evaluation stage: [跳过]"
  fi

  echo "✅ ${cfg_name} 处理完成"
done

echo ""
echo "🎉 所有配置处理完成"

# shutdown


# 用法：
# cd /root/autodl-tmp/AIAA3201_3DGS_Project/Part2_Scripts
# chmod +x run_all_reggs_configs.sh
# ./run_all_reggs_configs.sh
# 或指定显卡：GPU_ID=1 ./run_all_reggs_configs.sh
# 或跳过优化/评估：DO_REFINE=0 DO_METRIC=0 ./run_all_reggs_configs.sh

# chmod +x run_all_reggs_configs.sh
# nohup ./run_all_reggs_configs.sh > training_log_dl3dv_mixckpt.txt 2>&1 &
# tail -f /root/autodl-tmp/AIAA3201_3DGS_Project/Part2_Scripts/training_log_dl3dv_mixckpt.txt

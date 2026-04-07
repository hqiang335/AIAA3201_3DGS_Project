#!/bin/bash
# 【精准制导沙盒版】数据准备脚本：针对不同数据集的“方言”进行精准清洗

echo "======================================================="
echo " 🧹 阶段一：精准清洗强制数据集"
echo "======================================================="

# # 1. 精准处理 405841 (注意它的真实根目录在 FRONT 里面，图片叫 rgb)
# DIR_405841="/root/autodl-fs/405841/FRONT"
# echo "[*] 正在清洗 405841..."
# if [ -d "$DIR_405841/rgb" ]; then
#     mv "$DIR_405841/rgb" "$DIR_405841/input"
#     echo "  [+] 已将 rgb 重命名为 input"
# fi

# # 2. 精准处理 DL3DV-2 (图片叫 rgb)
# DIR_DL3DV="/root/autodl-fs/DL3DV-2"
# echo "[*] 正在清洗 DL3DV-2..."
# if [ -d "$DIR_DL3DV/rgb" ]; then
#     mv "$DIR_DL3DV/rgb" "$DIR_DL3DV/input"
#     echo "  [+] 已将 rgb 重命名为 input"
# fi

# # 3. 精准处理 Re10k-1 (图片叫 images)
# DIR_RE10K="/root/autodl-fs/Re10k-1"
# echo "[*] 正在清洗 Re10k-1..."
# if [ -d "$DIR_RE10K/images" ]; then
#     mv "$DIR_RE10K/images" "$DIR_RE10K/input"
#     echo "  [+] 已将 images 重命名为 input"
# fi

echo "======================================================="
echo " 📦 阶段二：为拓展数据集创建【安全沙盒】"
echo "======================================================="

# # 1. 创建 tandt 的 truck 场景沙盒
# TRUCK_SANDBOX="/root/autodl-fs/tandt_truck_ours"
# mkdir -p $TRUCK_SANDBOX
# if [ ! -d "$TRUCK_SANDBOX/input" ]; then
#     echo "  [+] 正在提取 truck 场景的纯净图片到沙盒..."
#     cp -r /root/autodl-fs/tandt_db/tandt/truck/images $TRUCK_SANDBOX/input
# else
#     echo "  [*] truck 沙盒已存在，仅清理历史生成的点云。"
#     rm -rf "$TRUCK_SANDBOX/sparse" "$TRUCK_SANDBOX/database.db"
# fi

# 1. 创建 tandt 的 train 场景沙盒
TRUCK_SANDBOX="/root/autodl-fs/tandt_train_ours"
mkdir -p $TRUCK_SANDBOX
if [ ! -d "$TRUCK_SANDBOX/input" ]; then
    echo "  [+] 正在提取 train 场景的纯净图片到沙盒..."
    cp -r /root/autodl-fs/tandt_db/tandt/train/images $TRUCK_SANDBOX/input
else
    echo "  [*] train 沙盒已存在，仅清理历史生成的点云。"
    rm -rf "$TRUCK_SANDBOX/sparse" "$TRUCK_SANDBOX/database.db"
fi

# 2. 创建 360_extra_scenes 的 treehill 场景沙盒
# TREEHILL_SANDBOX="/root/autodl-fs/360_treehill_ours_4"
# mkdir -p $TREEHILL_SANDBOX
# if [ ! -d "$TREEHILL_SANDBOX/input" ]; then
#     echo "  [+] 正在提取 treehill 场景的纯净图片到沙盒..."
#     cp -r /root/autodl-fs/360_extra_scenes/treehill/images_4 $TREEHILL_SANDBOX/images
# else
#     echo "  [*] treehill 沙盒已存在，仅清理历史生成的点云。"
#     rm -rf "$TREEHILL_SANDBOX/sparse" "$TREEHILL_SANDBOX/database.db"
# fi

# TREEHILL_SANDBOX="/root/autodl-fs/360_flowers_ours_4"
# mkdir -p $TREEHILL_SANDBOX
# if [ ! -d "$TREEHILL_SANDBOX/input" ]; then
#     echo "  [+] 正在提取 flowers 场景的纯净图片到沙盒..."
#     cp -r /root/autodl-fs/360_extra_scenes/flowers/images_4 $TREEHILL_SANDBOX/images
# else
#     echo "  [*] flowers 沙盒已存在，仅清理历史生成的点云。"
#     rm -rf "$TREEHILL_SANDBOX/sparse" "$TREEHILL_SANDBOX/database.db"
# fi

echo "✅ 数据准备完毕！所有历史痕迹已清除，图片均已归位到 input 文件夹！"
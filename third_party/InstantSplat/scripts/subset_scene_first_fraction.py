#!/usr/bin/env python3
"""
从完整 COLMAP 场景中取出「按 InstantSplat 排序规则」的前若干帧，生成新场景目录。

用途：例如只用前 1/3 图像做 init_geo / train / eval，划分逻辑仍是 split_train_eval_views
（在「新场景」的短序列上抽 n_train / n_test）；同时让 sparse/0 与 images/ 一一对应，
metrics.py 里 read_colmap_gt_pose + split_train_test 才正确。

说明：
  - MASt3R / init_geo 只读新目录下的 images/，并在 source_path 下写 sparse_{n_train}/0、/1；
    原 sparse/0 不参与几何初始化。
  - 若不做子集 sparse/0：全序列 images.bin 与少量 images/ 索引错位，位姿 ATE/RPE 无意义；
    图像 PSNR 等也可能在 metrics 里因异常整段跳过。

用法（在 InstantSplat 根目录下）:
  python scripts/subset_scene_first_fraction.py \\
    --src assets/part3/405841_front \\
    --dst assets/part3/405841_front_third \\
    --fraction 0.3333333
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scene.colmap_loader import (
    Image,
    read_extrinsics_binary,
    read_intrinsics_binary,
    write_cameras_binary,
    write_images_binary,
)
from utils.sfm_utils import get_sorted_image_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="完整场景根（含 images/ 与 sparse/0/）")
    parser.add_argument("--dst", type=Path, required=True, help="输出场景根（不存在则创建）")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0 / 3.0,
        help="保留排序后前 floor(N * fraction) 张（默认 1/3）",
    )
    parser.add_argument(
        "--n-keep",
        type=int,
        default=None,
        help="若指定则覆盖 fraction，直接保留前 N 张（仍需 >=3）",
    )
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    sparse0_src = src / "sparse" / "0"
    if not (src / "images").is_dir():
        raise SystemExit(f"缺少 {src / 'images'}")
    if not (sparse0_src / "images.bin").is_file():
        raise SystemExit(f"缺少 GT 位姿文件 {sparse0_src / 'images.bin'}")

    sorted_files, _ = get_sorted_image_files(src / "images")
    n = len(sorted_files)
    if args.n_keep is not None:
        n_keep = max(3, min(n, args.n_keep))
    else:
        n_keep = max(3, int(math.floor(n * args.fraction)))

    subset_paths = sorted_files[:n_keep]
    allowed = {Path(p).name for p in subset_paths}

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "sparse" / "0").mkdir(parents=True, exist_ok=True)

    for p in subset_paths:
        shutil.copy2(p, dst / "images" / Path(p).name)

    imgs = read_extrinsics_binary(str(sparse0_src / "images.bin"))
    kept = [img for _, img in sorted(imgs.items(), key=lambda kv: kv[1].name) if img.name in allowed]
    if len(kept) != len(subset_paths):
        missing = allowed - {img.name for img in kept}
        extra = len(subset_paths) - len(kept)
        raise SystemExit(
            f"COLMAP 与 images 不一致: 复制了 {len(subset_paths)} 张图, images.bin 只匹配到 {len(kept)}。"
            f" 请检查文件名是否与 sparse/0 中登记一致。missing_in_bin={missing or 'none'}"
        )

    cams = read_intrinsics_binary(str(sparse0_src / "cameras.bin"))
    used_ids = {img.camera_id for img in kept}
    sub_cams = {cid: cams[cid] for cid in used_ids if cid in cams}
    if len(sub_cams) != len(used_ids):
        raise SystemExit("子集相机 ID 在 cameras.bin 中缺失")

    new_images: dict[int, Image] = {}
    for new_id, img in enumerate(kept, start=1):
        new_images[new_id] = Image(
            id=new_id,
            qvec=img.qvec,
            tvec=img.tvec,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,
        )

    write_images_binary(new_images, str(dst / "sparse" / "0" / "images.bin"))
    write_cameras_binary(sub_cams, str(dst / "sparse" / "0" / "cameras.bin"))

    # points3D 未同步裁剪；metrics 的 read_colmap_gt_pose 只读 images.bin。
    # 若需要 points3D.ply 给其它工具，可从原场景自行复制（可能与子图不一致）。
    print(f"源序列: {n} 张 → 保留前 {n_keep} 张（sorted 与 get_sorted_image_files 一致）")
    print(f"已写入: {dst}")
    print(f"  {dst / 'images'}  ({len(list((dst / 'images').iterdir()))} files)")
    print(f"  {dst / 'sparse' / '0' / 'images.bin'} ({len(new_images)} cameras)")


if __name__ == "__main__":
    main()

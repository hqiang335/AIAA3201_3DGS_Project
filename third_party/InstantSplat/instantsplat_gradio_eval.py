"""
InstantSplat：Eval 评测链路 Gradio 界面（无 --infer_video）。

流程与 run_eval_pipeline.py 一致：init_geo → train → [可选 init_test_pose] → render（训练视角）
→ render --eval → metrics。

训练 / 测试划分：split_train_eval_views（--n_views=训练张数，--n_test=测试张数）。
"""

from __future__ import annotations

import json
import os
import sys

# 必须在 import numpy/torch 前设置，并会被子进程继承
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import gradio as gr
import subprocess
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# 传给 train.py 的学习率：与 instantsplat_gradio_no_metric.py 第 87–91 行相同写法。
# · 某行用 # 注释掉后，该项 **不会** 传给 train.py → 使用 arguments.OptimizationParams 内置默认。
# · 需要 feature / rotation 时，取消对应两行的注释即可。
# · 界面勾选「逐项覆盖学习率」后，以下表单项非空则覆盖、**留空则与不传该 flag 等效**（内置默认）。
# ---------------------------------------------------------------------------


def _lr(flag: str, value: str) -> list[str]:
    return [flag, value]


_DEFAULT_TRAIN_LR: list[str] = []
_DEFAULT_TRAIN_LR += _lr("--position_lr_init", "0.000016")
_DEFAULT_TRAIN_LR += _lr("--position_lr_final", "0.00000016")
# _DEFAULT_TRAIN_LR += _lr("--feature_lr", "0.00025")
_DEFAULT_TRAIN_LR += _lr("--scaling_lr", "0.0005")
# _DEFAULT_TRAIN_LR += _lr("--rotation_lr", "0.0001")


def _train_lr_defaults_map() -> dict[str, str]:
    m: dict[str, str] = {}
    it = iter(_DEFAULT_TRAIN_LR)
    for flag in it:
        m[flag] = next(it)
    return m


_LR_FLAG_ORDER = [
    "--position_lr_init",
    "--position_lr_final",
    "--feature_lr",
    "--scaling_lr",
    "--rotation_lr",
]


def _build_train_lr_cli(
    use_ui_override: bool,
    position_lr_init: str,
    position_lr_final: str,
    feature_lr: str,
    scaling_lr: str,
    rotation_lr: str,
) -> list[str]:
    if not use_ui_override:
        return list(_DEFAULT_TRAIN_LR)
    m = _train_lr_defaults_map()
    for flag, raw in (
        ("--position_lr_init", position_lr_init),
        ("--position_lr_final", position_lr_final),
        ("--feature_lr", feature_lr),
        ("--scaling_lr", scaling_lr),
        ("--rotation_lr", rotation_lr),
    ):
        v = (raw or "").strip()
        if v:
            m[flag] = v
        else:
            m.pop(flag, None)
    out: list[str] = []
    for k in _LR_FLAG_ORDER:
        if k in m:
            out.extend([k, m[k]])
    return out


INSTANTSPLAT_ROOT = Path(__file__).resolve().parent
# 与 instantsplat_gradio.py 一致：仓库根目录，供 launch(allowed_paths=…) 使用
REPO_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

sys.path.insert(0, str(INSTANTSPLAT_ROOT))
from utils.sfm_utils import get_sorted_image_files, split_train_eval_views

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def run_process(cmd: list[str], cwd: Path) -> bool:
    print(f"Running command: {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=str(cwd),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    return process.returncode == 0


def get_image_count(input_dir: str) -> int:
    image_dir = Path(input_dir) / "images"
    if not image_dir.is_dir():
        return 0
    return sum(
        1
        for item in image_dir.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def _read_metrics_summary(model_path: Path) -> str:
    results = model_path / "results.json"
    if not results.is_file():
        return "（未找到 results.json；若跳过了 metrics 或运行失败则正常）"
    try:
        data = json.loads(results.read_text(encoding="utf-8"))
    except Exception as e:
        return f"读取 results.json 失败: {e}"
    lines = [f"results.json: {results}"]
    for scene_key, methods in data.items():
        if not isinstance(methods, dict):
            continue
        lines.append(f"\n[{scene_key}]")
        for method, m in methods.items():
            if not isinstance(m, dict):
                continue
            lines.append(f"  {method}:")
            for k in ("SSIM", "PSNR", "LPIPS", "RPE_t", "RPE_r", "ATE"):
                if k in m:
                    lines.append(f"    {k}: {m[k]}")
    return "\n".join(lines) if len(lines) > 1 else str(data)


def run_eval_pipeline_ui(
    input_dir: str,
    output_dir: str,
    n_train: float,
    n_test: float,
    iterations: float,
    resolution: float,
    scene_graph: str,
    optim_test_pose_iter: float,
    use_ui_lr_override: bool,
    position_lr_init: str,
    position_lr_final: str,
    feature_lr: str,
    scaling_lr: str,
    rotation_lr: str,
    use_pp_optimizer: bool,
    use_optim_pose: bool,
    run_init_test_pose: bool,
    skip_init: bool,
    skip_train: bool,
    skip_render_train: bool,
    skip_render_eval: bool,
    skip_metrics: bool,
    progress=gr.Progress(),
):
    if not torch.cuda.is_available():
        return "Error: CUDA not available", ""

    input_dir = str(input_dir).strip()
    output_dir = str(output_dir).strip()
    nt = int(n_train)
    nte = int(n_test)
    iters = int(iterations)
    res = int(resolution)
    opt_pose_iter = int(optim_test_pose_iter)

    log_lines: list[str] = []

    def log(msg: str):
        log_lines.append(msg)
        print(msg, flush=True)

    if nt < 2:
        return "Error: 训练视图数 n_train 至少为 2", ""
    if nte < 1:
        return "Error: 测试视图数 n_test 至少为 1", ""
    ic = get_image_count(input_dir)
    if ic == 0:
        return "Error: 输入目录下需要 images/ 且含有支持的图片", ""

    try:
        image_files, _ = get_sorted_image_files(Path(input_dir) / "images")
        split_train_eval_views(image_files, n_train=nt, n_test=nte, verbose=False)
    except ValueError as e:
        return f"Error: 划分不合法 — {e}", ""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log(f"输出目录: {output_path.resolve()}")

    nv_train = ["--n_views", str(nt)]
    nv_init = nv_train + ["--n_test", str(nte)]

    train_lr_args = _build_train_lr_cli(
        use_ui_lr_override,
        position_lr_init,
        position_lr_final,
        feature_lr,
        scaling_lr,
        rotation_lr,
    )

    steps = [
        ("init_geo", not skip_init, 0.15),
        ("train", not skip_train, 0.35),
        ("init_test_pose", run_init_test_pose and not skip_train, 0.48),
        ("render_train", not skip_render_train, 0.62),
        ("render_eval", not skip_render_eval, 0.78),
        ("metrics", not skip_metrics, 0.92),
    ]

    for name, do, p in steps:
        if not do:
            continue
        progress(p, f"运行 {name}...")

        if name == "init_geo":
            cmd = [
                PY,
                "init_geo.py",
                "-s",
                input_dir,
                "-m",
                str(output_path),
                *nv_init,
                "--focal_avg",
                "--co_vis_dsp",
                "--conf_aware_ranking",
                "--scene_graph",
                scene_graph,
            ]
        elif name == "train":
            cmd = [
                PY,
                "train.py",
                "-s",
                input_dir,
                "-m",
                str(output_path),
                "-r",
                str(res),
                *nv_train,
                "--iterations",
                str(iters),
                "--test_iterations",
                str(iters),
                *train_lr_args,
            ]
            if use_pp_optimizer:
                cmd.append("--pp_optimizer")
            if use_optim_pose:
                cmd.append("--optim_pose")
        elif name == "init_test_pose":
            cmd = [
                PY,
                "init_test_pose.py",
                "-s",
                input_dir,
                "-m",
                str(output_path),
                *nv_init,
                "--focal_avg",
            ]
        elif name == "render_train":
            cmd = [
                PY,
                "render.py",
                "-s",
                input_dir,
                "-m",
                str(output_path),
                "-r",
                str(res),
                *nv_train,
                "--iterations",
                str(iters),
            ]
        elif name == "render_eval":
            cmd = [
                PY,
                "render.py",
                "-s",
                input_dir,
                "-m",
                str(output_path),
                "-r",
                str(res),
                *nv_train,
                "--iterations",
                str(iters),
                "--eval",
                "--optim_test_pose_iter",
                str(opt_pose_iter),
            ]
        else:  # metrics
            cmd = [
                PY,
                "metrics.py",
                "-s",
                input_dir,
                "-m",
                str(output_path),
                "--n_views",
                str(nt),
                "--n_test",
                str(nte),
            ]

        if not run_process(cmd, INSTANTSPLAT_ROOT):
            return "\n".join(log_lines) + f"\n\nError in {name}", _read_metrics_summary(output_path)

        log(f"完成: {name}")
        if name == "init_geo":
            mf = output_path / "split_manifest.json"
            if mf.is_file():
                log(f"划分已写入: {mf}")

    log("全部所选步骤执行结束。")
    summary = _read_metrics_summary(output_path)
    return "\n".join(log_lines), summary


_dm = _train_lr_defaults_map()

with gr.Blocks() as demo:
    gr.Markdown(
        "# InstantSplat — Eval 管线（train/test 显式划分，无 infer_video）\n"
        "与 `run_eval_pipeline.py` 一致；**请勿**在本流程中使用 `--infer_video`。\n"
        "**学习率**：默认完全由本文件中 `_DEFAULT_TRAIN_LR`（可注释行，同 `instantsplat_gradio_no_metric.py`）决定；"
        "勾选「逐项覆盖」后可用下方文本框修改（某项留空 = 不传该参数，与注释掉行等效）。"
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_dir = gr.Textbox(label="输入目录 (-s)", placeholder="含 images/ 的场景根目录")
            output_dir = gr.Textbox(label="输出目录 (-m)", placeholder="模型与 split_manifest、评测输出")
            n_train = gr.Number(value=6, precision=0, minimum=2, label="训练视图数 n_train (--n_views)")
            n_test = gr.Number(value=12, precision=0, minimum=1, label="测试视图数 n_test")
            iterations = gr.Slider(
                minimum=1000, maximum=30000, value=1000, step=1000, label="训练迭代次数"
            )
            resolution = gr.Number(value=1, precision=0, minimum=1, label="分辨率 -r（train/render）")
            scene_graph = gr.Dropdown(
                choices=[
                    "complete",
                    "swin-2-noncyclic",
                    "swin-3-noncyclic",
                    "logwin-3-noncyclic",
                ],
                value="complete",
                label="MASt3R Scene Graph (init_geo)",
            )
            optim_test_pose_iter = gr.Number(
                value=500, precision=0, minimum=0, label="render --eval 每视角位姿优化步数"
            )
            use_ui_lr_override = gr.Checkbox(
                value=False,
                label="逐项覆盖学习率（关闭=仅用文件中 _DEFAULT_TRAIN_LR；开启=下方非空项覆盖、空项=不传）",
            )
            gr.Markdown("**学习率覆盖（仅在上项开启时生效；默认值便于对照文件）**")
            position_lr_init = gr.Textbox(
                label="position_lr_init", value=_dm.get("--position_lr_init", "")
            )
            position_lr_final = gr.Textbox(
                label="position_lr_final", value=_dm.get("--position_lr_final", "")
            )
            feature_lr = gr.Textbox(label="feature_lr", value=_dm.get("--feature_lr", ""))
            scaling_lr = gr.Textbox(label="scaling_lr", value=_dm.get("--scaling_lr", ""))
            rotation_lr = gr.Textbox(label="rotation_lr", value=_dm.get("--rotation_lr", ""))
            use_pp_optimizer = gr.Checkbox(value=True, label="--pp_optimizer")
            use_optim_pose = gr.Checkbox(value=True, label="--optim_pose")
            run_init_test_pose = gr.Checkbox(value=True, label="训练后运行 init_test_pose.py")
            gr.Markdown("**跳过步骤（用于断点续跑）**")
            skip_init = gr.Checkbox(value=False, label="跳过 init_geo")
            skip_train = gr.Checkbox(value=False, label="跳过 train")
            skip_render_train = gr.Checkbox(value=False, label="跳过 render（训练视角）")
            skip_render_eval = gr.Checkbox(value=False, label="跳过 render --eval")
            skip_metrics = gr.Checkbox(value=False, label="跳过 metrics")
            run_btn = gr.Button("运行 Eval 管线")
        with gr.Column(scale=1):
            log_out = gr.Textbox(label="运行日志", lines=18)
            metrics_out = gr.Textbox(label="metrics 摘要 (results.json)", lines=12)

    run_btn.click(
        fn=run_eval_pipeline_ui,
        inputs=[
            input_dir,
            output_dir,
            n_train,
            n_test,
            iterations,
            resolution,
            scene_graph,
            optim_test_pose_iter,
            use_ui_lr_override,
            position_lr_init,
            position_lr_final,
            feature_lr,
            scaling_lr,
            rotation_lr,
            use_pp_optimizer,
            use_optim_pose,
            run_init_test_pose,
            skip_init,
            skip_train,
            skip_render_train,
            skip_render_eval,
            skip_metrics,
        ],
        outputs=[log_out, metrics_out],
    )

if __name__ == "__main__":
    print("Starting Gradio eval (import torch/gradio may take tens of seconds)...", flush=True)
    demo.launch(allowed_paths=[str(REPO_ROOT.resolve())])

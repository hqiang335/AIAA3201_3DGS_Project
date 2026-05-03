import os
import sys
# 必须在 import numpy/torch 前设置，并会被子进程继承；否则子进程里 MKL 与 libgomp 冲突会导致 init_geo 直接退出
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
# 通过管道/tee 重定向时避免 stdout 全缓冲，否则日志长时间空白、像“未启动”
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import gradio as gr
import subprocess
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
PY = sys.executable

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

def run_process(cmd):
    print(f"Running command: {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    return process.returncode == 0

def get_image_count(input_dir):
    image_dir = Path(input_dir) / "images"
    if not image_dir.is_dir():
        return 0
    return sum(1 for item in image_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)

def process_scene(input_dir, output_dir, n_views, iterations, progress=gr.Progress()):
    if not torch.cuda.is_available():
        return "Error: CUDA not available"

    n_views = int(n_views)
    iterations = int(iterations)
    if n_views < 2:
        return "Error: Number of Views must be at least 2"

    image_count = get_image_count(input_dir)
    if image_count == 0:
        return "Error: Input Directory must contain an images/ folder with images"

    if n_views != image_count:
        return f"Error: Number of Views should match the number of files in images/. Found {image_count} images, got {n_views}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created at: {output_path}")
    
    init_cmd = [
        PY, "init_geo.py",
        "--source_path", input_dir,
        "--model_path", str(output_path),
        "--n_views", str(n_views),
        "--focal_avg",
        "--co_vis_dsp",
        "--conf_aware_ranking",
        "--infer_video"
    ]
    
    train_cmd = [
        PY, "train.py",
        "-s", input_dir,
        "-m", str(output_path),
        "--n_views", str(n_views),
        "--iterations", str(iterations),
        "--test_iterations", str(iterations),

        # "--position_lr_init", "0.000016",
        # "--position_lr_final", "0.00000016",
        # "--feature_lr", "0.00025",
        # "--scaling_lr", "0.0005",
        # "--rotation_lr", "0.0001",
        
        "--pp_optimizer",
        "--optim_pose"
    ]
    
    render_cmd = [
        PY, "render.py",
        "-s", input_dir,
        "-m", str(output_path),
        "--n_views", str(n_views),
        "--iterations", str(iterations),
        "--infer_video"
    ]

    commands = [
        (init_cmd, 0.2, "Initialization"),
        (train_cmd, 0.4, "Training"),
        (render_cmd, 0.8, "Rendering")
    ]

    for cmd, prog, name in commands:
        progress(prog, f"Running {name}...")
        if not run_process(cmd):
            return f"Error in {name}"
        print(f"After {name}, contents of output dir:")
        for item in os.listdir(output_path):
            print(f"  - {item}")

    video_path = output_path / f"interp/ours_{iterations}/interp_{n_views}_view.mp4"
    if video_path.exists():
        print(f"Found video at: {video_path}")
        return str(video_path)
    else:
        print("Video not found, searching for alternatives...")
        for mp4_file in output_path.rglob("*.mp4"):
            print(f"Found video at: {mp4_file}")
            return str(mp4_file)
    return "Video not found"

with gr.Blocks() as demo:
    gr.Markdown("# InstantSplat Demo")
    with gr.Row():
        with gr.Column():
            input_dir = gr.Textbox(label="Input Directory")
            output_dir = gr.Textbox(label="Output Directory")
            n_views = gr.Number(value=3, precision=0, minimum=2, label="Number of Views")
            iterations = gr.Slider(minimum=1000, maximum=30000, value=1000, step=1000, label="Training Iterations")
            process_btn = gr.Button("Process Scene")
        with gr.Column():
            output_video = gr.Video(label="Output Video")
    
    process_btn.click(fn=process_scene, inputs=[input_dir, output_dir, n_views, iterations], outputs=output_video)

if __name__ == "__main__":
    print("Starting Gradio (import torch/gradio may take tens of seconds)...", flush=True)
    demo.launch()
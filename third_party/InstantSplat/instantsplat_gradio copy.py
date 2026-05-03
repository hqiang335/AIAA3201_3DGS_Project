import os
# 必须在 import numpy/torch 前设置，并会被子进程继承；否则子进程里 MKL 与 libgomp 冲突会导致 init_geo 直接退出
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import gradio as gr
import json
import subprocess
from pathlib import Path
import torch

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PART2_EVAL_SCRIPT = PROJECT_ROOT / "Part2_Scripts" / "prepare_instantsplat_part2_eval.py"

def run_process(cmd):
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    return process.returncode == 0

def get_image_count(input_dir):
    image_dir = Path(input_dir) / "images"
    if not image_dir.is_dir():
        return 0
    return sum(1 for item in image_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)

def load_source_info(input_dir):
    info_path = Path(input_dir) / "source_info.txt"
    info = {}
    if not info_path.exists():
        return info
    for line in info_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()
    return info

def is_part2_scene(source_info):
    return source_info.get("split_method") == "official_instantsplat_split_train_test_with_pdf_rate_n_views"

def resolve_original_data_dir(source_info):
    source_images = source_info.get("source_images")
    if not source_images:
        return None
    return str(Path(source_images).parent)

def summarize_part2_sampling(source_info, n_views):
    if not source_info:
        return ""
    return (
        "Part2 sampling:\n"
        f"- scene: {source_info.get('scene_name', 'unknown')}\n"
        f"- source images: {source_info.get('source_image_count', 'unknown')}\n"
        f"- train candidates after official split: {source_info.get('train_candidate_count', 'unknown')}\n"
        f"- test views: {source_info.get('test_count', 'unknown')}\n"
        f"- PDF stride: 1/{source_info.get('p2_stride', 'unknown')}\n"
        f"- n_views used for init/train: {n_views}\n"
    )

def load_metric_summary(output_path):
    results_path = Path(output_path) / "results.json"
    if not results_path.exists():
        return f"Metrics file not found: {results_path}"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not data:
        return f"Metrics file is empty: {results_path}"
    method, metrics = next(iter(data.items()))
    lines = [f"Metrics ({method}):"]
    for key in ["PSNR", "SSIM", "LPIPS", "RPE_t", "RPE_r", "ATE"]:
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    lines.extend([
        f"results: {results_path}",
        f"per-view: {Path(output_path) / 'per_view.json'}",
    ])
    return "\n".join(lines)

def process_scene(input_dir, output_dir, n_views, iterations, progress=gr.Progress()):
    if not torch.cuda.is_available():
        return None, "Error: CUDA not available"

    input_dir = str(input_dir).strip()
    output_dir = str(output_dir).strip()
    requested_n_views = int(n_views or 0)
    iterations = int(iterations)
    source_info = load_source_info(input_dir)
    part2_scene = is_part2_scene(source_info)
    if requested_n_views <= 0 and part2_scene:
        n_views = int(source_info["sampled_count"])
    elif requested_n_views <= 0:
        return None, (
            "Error: Number of Views is 0, but this input was not recognized as a Part2 scene.\n"
            f"Expected source_info.txt at: {Path(input_dir) / 'source_info.txt'}\n"
            "Use a Part2 scene directory, or manually enter Number of Views >= 2."
        )
    else:
        n_views = requested_n_views

    if n_views < 2:
        return None, "Error: Number of Views must be at least 2"

    image_count = get_image_count(input_dir)
    if image_count == 0:
        return None, "Error: Input Directory must contain an images/ folder with images"

    if not part2_scene and n_views != image_count:
        return None, f"Error: Number of Views should match the number of files in images/. Found {image_count} images, got {n_views}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created at: {output_path}")
    
    init_cmd = [
        "python", "init_geo.py",
        "--source_path", input_dir,
        "--model_path", str(output_path),
        "--n_views", str(n_views),
        "--focal_avg",
        "--co_vis_dsp",
        "--conf_aware_ranking",
    ]
    if not part2_scene:
        init_cmd.append("--infer_video")
    
    train_cmd = [
        "python", "train.py",
        "-s", input_dir,
        "-m", str(output_path),
        "--n_views", str(n_views),
        "--iterations", str(iterations),
        "--test_iterations", str(iterations),

        "--position_lr_init", "0.000016",
        "--position_lr_final", "0.00000016",
        "--feature_lr", "0.00025",
        "--scaling_lr", "0.0005",
        "--rotation_lr", "0.0001",
        
        "--pp_optimizer",
        "--optim_pose"
    ]
    if part2_scene:
        train_cmd.extend(["-r", "1"])
    
    render_cmd = [
        "python", "render.py",
        "-s", input_dir,
        "-m", str(output_path),
        "--n_views", str(n_views),
        "--iterations", str(iterations),
        "--infer_video"
    ]
    if part2_scene:
        render_cmd.extend(["-r", "1"])

    commands = [
        (init_cmd, 0.2, "Initialization"),
        (train_cmd, 0.4, "Training"),
        (render_cmd, 0.8, "Rendering")
    ]

    for cmd, prog, name in commands:
        progress(prog, f"Running {name}...")
        if not run_process(cmd):
            return None, f"Error in {name}"
        print(f"After {name}, contents of output dir:")
        for item in os.listdir(output_path):
            print(f"  - {item}")

    report = summarize_part2_sampling(source_info, n_views)
    if part2_scene:
        progress(0.9, "Running Part2 evaluation metrics...")
        eval_images = "images"
        sparse_eval = Path(input_dir) / f"sparse_{n_views}/1/images.txt"
        if not sparse_eval.exists():
            test_images = Path(input_dir) / "test_images"
            original_data_dir = resolve_original_data_dir(source_info)
            if test_images.is_dir() and original_data_dir and PART2_EVAL_SCRIPT.exists():
                prep_cmd = [
                    "python", str(PART2_EVAL_SCRIPT),
                    "--source-path", input_dir,
                    "--original-data-dir", original_data_dir,
                    "--n-views", str(n_views),
                    "--overwrite",
                ]
                if not run_process(prep_cmd):
                    return None, report + "\nError in Part2 eval preparation"
                eval_images = "test_images"
            else:
                return None, report + f"\nMissing eval sparse file: {sparse_eval}"

        eval_render_cmd = [
            "python", "render.py",
            "-s", input_dir,
            "-m", str(output_path),
            "-r", "1",
            "--images", eval_images,
            "--n_views", str(n_views),
            "--iterations", str(iterations),
            "--eval",
            "--optim_test_pose_iter", "500",
        ]
        if not run_process(eval_render_cmd):
            return None, report + "\nError in Part2 eval rendering"

        metrics_cmd = [
            "python", "metrics.py",
            "-s", input_dir,
            "-m", str(output_path),
            "--n_views", str(n_views),
        ]
        if not run_process(metrics_cmd):
            return None, report + "\nError in Part2 metrics"
        report += "\n" + load_metric_summary(output_path)

    video_path = output_path / f"interp/ours_{iterations}/interp_{n_views}_view.mp4"
    if video_path.exists():
        print(f"Found video at: {video_path}")
        return str(video_path), report or f"Video: {video_path}"
    else:
        print("Video not found, searching for alternatives...")
        for mp4_file in output_path.rglob("*.mp4"):
            print(f"Found video at: {mp4_file}")
            return str(mp4_file), report or f"Video: {mp4_file}"
    return None, report + "\nVideo not found"

with gr.Blocks() as demo:
    gr.Markdown("# InstantSplat Demo")
    with gr.Row():
        with gr.Column():
            input_dir = gr.Textbox(label="Input Directory")
            output_dir = gr.Textbox(label="Output Directory")
            n_views = gr.Number(value=0, precision=0, minimum=0, label="Number of Views (0 = auto from source_info.txt)")
            iterations = gr.Slider(minimum=1000, maximum=30000, value=1000, step=1000, label="Training Iterations")
            process_btn = gr.Button("Process Scene")
        with gr.Column():
            output_video = gr.Video(label="Output Video")
            metrics_output = gr.Textbox(label="Part2 Metrics / Run Summary", lines=12)
    
    process_btn.click(fn=process_scene, inputs=[input_dir, output_dir, n_views, iterations], outputs=[output_video, metrics_output])

if __name__ == "__main__":
    # Gradio 默认只允许返回 CWD 或系统临时目录下的文件；输出常在 Part2_Scripts/InstantSplat_outputs，否则会 InvalidPathError
    demo.launch(allowed_paths=[str(PROJECT_ROOT)])
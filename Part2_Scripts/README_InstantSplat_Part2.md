# Part2：InstantSplat 数据准备与 Gradio 使用说明

本文说明 **原始数据应如何摆放**、如何运行 **`prepare_instantsplat_part2_data.sh`**、如何运行 **`instantsplat_gradio.py`**，以及 **Gradio 脚本应放在哪里**。

---

## 1. 原始数据目录结构（`dataset_root`）

`prepare_instantsplat_part2_data.sh` 里每一项 **`IMAGE_DIRS`** 必须指向 **「数据集根目录下的 `images` 文件夹」**，即：

```text
<dataset_root>/                    # = dirname(你的 images 路径)，例如 .../DL3DV-2
├── images/                        # IMAGE_DIRS 指向这一层（不是 dataset_root 本身）
│   ├── 000001.png                 # 平铺：脚本只扫描该目录的直接子文件，不递归子文件夹
│   ├── 000002.jpg
│   └── ...
├── sparse/                        # 与 images 同级；用于符号链接到准备好的场景目录
│   └── 0/
│       └── images.bin             # 必须存在，脚本才会创建场景里的 sparse → 指向此处
│       # 以及 COLMAP 常规文件（cameras.bin、points3D.bin 等），视你的数据而定
└── cameras.json                   # 本脚本不读取；供后续 prepare_instantsplat_part2_eval.py / 评测使用（RegGS 风格位姿）
```

### 对 `images/` 的要求

- **至少 3 张**图像，否则内嵌 Python 会报错。
- **后缀**：`.png`、`.jpg`、`.jpeg`、`.bmp`、`.tiff`（大小写不敏感）。
- **排序**：按文件名中**第一个数字**排序；无数字的文件会排在很后面。

### 对 `sparse/` 的要求

- 若 **`sparse/0/images.bin` 不存在**：脚本只打印 **WARNING**，**不会**在场景目录创建 `sparse` 链接；后续 **`metrics.py` 的 ATE** 等可能不可用（`run_instantsplat_part2_batch.sh` 也会检查该文件）。

### 自定义数据集

在 `prepare_instantsplat_part2_data.sh` 中把 **`SCENE_NAMES`**、**`IMAGE_DIRS`**、**`P2_STRIDES`** 改成**一一对应**的三组（场景名、该场景的 `images` 绝对路径、抽帧 stride）。若使用 `run_instantsplat_part2_batch.sh` 做评测，还需在其中的 **`original_data_dir_for_scene`** 里为新场景名配置 **`cameras.json` 所在的数据集根目录**。

---

## 2. 运行 `prepare_instantsplat_part2_data.sh`

### 默认行为

- **`PROJECT_DIR`**：默认 `/root/autodl-tmp/AIAA3201_3DGS_Project`（可通过环境变量覆盖）。
- **输出根目录 `ASSET_ROOT`**：默认  
  **`$PROJECT_DIR/Part2_Scripts/InstantSplat_assets/part2`**
- 每个场景输出到：**`$ASSET_ROOT/<场景名>/`**，例如 `.../part2/dl3dv_2/`，内含：
  - `images/`、`test_images/`（默认符号链接到原始文件；`COPY_IMAGES=1` 则复制）
  - `test_files.txt`、`train_sampled_files.txt`、`source_info.txt`
  - `sparse` → 指向 `<dataset_root>/sparse`（满足 `images.bin` 条件时）

### 命令示例

```bash
cd /root/autodl-tmp/AIAA3201_3DGS_Project/Part2_Scripts
bash prepare_instantsplat_part2_data.sh
```

### 常用环境变量

| 变量 | 含义 |
|------|------|
| `PROJECT_DIR` | 课程项目根目录 |
| `ASSET_ROOT` | 准备好的场景父目录（默认 `Part2_Scripts/InstantSplat_assets/part2`） |
| `COPY_IMAGES=1` | 复制图像而非符号链接 |

示例：把场景输出改到 InstantSplat 自带的 `assets/part2` 下：

```bash
ASSET_ROOT=/root/autodl-tmp/AIAA3201_3DGS_Project/third_party/InstantSplat/assets/part2 \
  bash prepare_instantsplat_part2_data.sh
```

完成后可按脚本提示继续运行 **`run_instantsplat_part2_batch.sh`**，或使用 Gradio（见下）。

---

## 3. `instantsplat_gradio.py` 要不要放在 `third_party/InstantSplat` 里？

**不强制，但推荐在仓库里保留 `third_party/InstantSplat/instantsplat_gradio.py` 这一份作为主入口。**

仓库里可能有两份同名文件：

| 路径 | 说明 |
|------|------|
| **`third_party/InstantSplat/instantsplat_gradio.py`** | `PROJECT_ROOT` 指向课程项目根目录（`parents[2]`），便于解析 **`Part2_Scripts/prepare_instantsplat_part2_eval.py`** 与 `allowed_paths`。**推荐日常使用。** |
| **`Part2_Scripts/instantsplat_gradio.py`** | 与上者逻辑基本一致，`PROJECT_ROOT` 为 `parents[1]`（项目根）。便于和 Part2 脚本放在同一目录维护。 |

**关键约束：运行 Gradio 时，进程的当前工作目录（cwd）必须是 InstantSplat 源码根目录**，因为子进程执行的是相对路径命令（例如 `python init_geo.py`、`python train.py`），这些文件位于 **`third_party/InstantSplat/`** 下。

因此无论脚本文件本身放在哪，启动时都应：

```bash
cd /root/autodl-tmp/AIAA3201_3DGS_Project/third_party/InstantSplat
conda activate instantsplat   # 或你实际使用的环境
python instantsplat_gradio.py
```

若脚本在 **`Part2_Scripts/`**，仍须 **先 `cd` 到 `third_party/InstantSplat`** 再执行：

```bash
cd /root/autodl-tmp/AIAA3201_3DGS_Project/third_party/InstantSplat
python ../../Part2_Scripts/instantsplat_gradio.py
```

（路径按你机器上的项目根调整。）

### Gradio 里如何填路径

- **Input Directory**：填 **`prepare_instantsplat_part2_data.sh` 生成后的场景目录**，例如  
  `.../Part2_Scripts/InstantSplat_assets/part2/re10k_1`  
  或你通过 `ASSET_ROOT` 指定的目录下的 `<场景名>`。该目录下须有 **`images/`** 与 Part2 用的 **`source_info.txt`**（脚本已写好 `split_method` 等）。
- **Output Directory**：任意可写路径即可，例如  
  `.../Part2_Scripts/InstantSplat_outputs/re10k_1/9_views`；**Number of Views 填 0** 时会从 `source_info.txt` 的 `sampled_count` 自动读取。

### 依赖与检查

- 已安装 InstantSplat 依赖（含 `gradio`、`torch` 等），且 **`mast3r/checkpoints/`** 下有所需 MASt3R 权重（见 InstantSplat 官方 README）。
- 若遇 **Gradio 无法显示输出视频**（`InvalidPathError`），请使用**已带 `allowed_paths` 的 `demo.launch()`** 的版本，或将输出目录放在项目根下（与 `launch` 中 `allowed_paths` 一致）。

---

## 4. 与批量脚本的关系（可选）

- **`run_instantsplat_part2_batch.sh`**：在 InstantSplat 目录下依次调用 `init_geo.py`、`train.py`、`render.py`，可选评测；**输入场景**默认来自 **`InstantSplat_assets/part2`**，**输出**默认在 **`Part2_Scripts/InstantSplat_outputs/<场景名>/<N>_views/`**。
- **`prepare_instantsplat_part2_eval.py`**：为测试视角生成 `sparse_<N>/1/cameras.txt` 等；Gradio 在 Part2 流程中若缺少该目录会自动调用（需 **`test_files.txt`**、`test_images/`、原始 **`cameras.json`**）。

更细的流程区别可参考脚本内注释或课程说明。

---

## 5. 快速检查清单

1. [ ] `<dataset_root>/images/` 平铺、≥3 张、后缀合法  
2. [ ] `<dataset_root>/sparse/0/images.bin` 存在（需要 ATE / 链 sparse 时）  
3. [ ] `<dataset_root>/cameras.json` 存在（需要 Part2 自动评测 / `prepare_instantsplat_part2_eval.py` 时）  
4. [ ] 已修改 `prepare_instantsplat_part2_data.sh` 中 **`IMAGE_DIRS`** 等指向你的磁盘路径  
5. [ ] 运行 Gradio 前 **`cd` 到 `third_party/InstantSplat`**

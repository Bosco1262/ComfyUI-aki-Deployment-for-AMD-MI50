# 秋叶整合包 ComfyUI + AMD MI50（ZLUDA / gfx906 / Windows）

[English](README.md) | [简体中文](README.zh-CN.md)

在 Windows 上，通过 [ZLUDA](https://github.com/vosen/ZLUDA) 让 **秋叶 ComfyUI 整合包**（使用**绘世启动器**）跑在 **AMD Instinct MI50**（刷入 **Radeon VII PRO vBIOS**）上。

本仓库记录了一套**已完整跑通**的方案：实际执行的命令、一个小型兼容补丁（[`ComfyUI/zluda_patch.py`](ComfyUI/zluda_patch.py)）、每一步为什么需要、验证方法以及故障排查表。所有结论均来自单台机器上的实测（命令行输出、日志、文件哈希），不是猜测。

## 结论速览

要让这套整合包在 ZLUDA 上跑通，需要且只需要四件事：

1. **把 torch 换成 cu118 版本。** ZLUDA 只提供 CUDA 11 的库，用 CUDA 12+ 编译的 PyTorch 一定不行。
2. **关掉 ComfyUI 在非 N 卡上自动开启的 `cudaMallocAsync`**：在启动参数里加 `--disable-cuda-malloc`。
3. **放入补丁 `ComfyUI\zluda_patch.py`，并在 `ComfyUI\main.py` 顶部挂一小段引导代码。** 补丁做 5 件事：
   1. 预加载 ZLUDA 的 DLL，
   2. 让 rocBLAS 指向 gfx906 的 Tensile 内核，
   3. 关闭 cuDNN，
   4. 注意力只保留 math SDP，
   5. 绕开“带 bias 的线性层”所走的 cuBLASLt 融合 GEMM。
4. **启动器的精度设置保持 FP16 或 Auto 都行。** 精度不是根因，属于可选优化。

## 实测环境

### 硬件 / 驱动 / 系统

| 项目                       | 实测值                                                                                                                                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPU                        | **AMD Radeon Pro VII**（MI50 刷 Radeon VII PRO vBIOS），架构 **gfx906**，**16 GB HBM2**                                                                                                                  |
| ComfyUI 识别的显存         | `Total VRAM 16368 MB`                                                                                                                                                                                    |
| 系统内存                   | **16310 MB（16 GB）**，偏紧：不要用大量 offload 的选项（见[故障排查](#故障排查)）                                                                                                                        |
| 操作系统                   | Windows 10（10.0.19045）                                                                                                                                                                                 |
| AMD 驱动 / 运行时          | `HIP Driver: 50731541` / `Runtime Version: 50731541`，即 **HIP SDK 5.7.1**                                                                                                                               |
| HIP SDK 安装路径           | `C:\Program Files\AMD\ROCm\5.7`（环境变量 `HIP_PATH` / `HIP_PATH_57` / `HIP_PATH_64`）                                                                                                                   |
| rocBLAS Tensile 内核目录   | `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library`（587 个文件，**含 gfx906 内核**：`TensileLibrary_lazy_gfx906.dat`、`TensileLibrary_Type_*_gfx906.co/.dat`、`Kernels.so-000-gfx906-xnack-.hsaco`）    |
| ZLUDA                      | 绘世启动器自带的 `zluda.extpack`，内含 5 个 DLL：`cublas64_11.dll`、`cusparse64_11.dll`、`nvcuda.dll`、`nvml.dll`、`nvrtc64_112_0.dll`（2024-04 构建，ROCm 5.x 时代），所以**只能配 CUDA 11.x 的 torch** |
| 启动器附带的 gfx906 内核包 | `.launcher\tensile-gfx906.<hash>.extpack`（内含 `TensileLibrary_*_gfx906.*`）                                                                                                                            |

### 软件

| 项目                      | 值                                                                                                                                                                                                                                                             |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 整合包根目录              | `ComfyUI-aki-v3.2\`，即包含 `绘世启动器.exe`、`python\`、`ComfyUI\` 的文件夹                                                                                                                                                                                   |
| WebUI 目录（`.app_path`） | `ComfyUI`                                                                                                                                                                                                                                                      |
| Python                    | **3.13.11**（`python\python.exe`，pip 25.3）                                                                                                                                                                                                                   |
| 绘世启动器                | 启动器 2.9.2.419（外壳 `绘世启动器.exe`，bootstrap 1.11.0.11）                                                                                                                                                                                                 |
| ComfyUI                   | `dec5d9450a5290bcf63430409ea41018e67f41c3`（2026-08-05）                                                                                                                                                                                                       |
| **改造后 torch**          | **torch 2.7.1+cu118 / torchvision 0.22.1+cu118 / torchaudio 2.7.1+cu118**                                                                                                                                                                                      |
| 原装 torch（供参考）      | torch 2.13.0+cu130 / torchvision 0.28.0+cu130 / torchaudio 2.11.0+cu130                                                                                                                                                                                        |
| 已卸载                    | `sageattention 2.2.0+cu130torch2.9.0andhigher.post4`、`triton-windows 3.5.1.post24`（版本锁写在包名里，pip 不会自动发现冲突）                                                                                                                                  |
| 其他关键包                | numpy 2.3.5、comfy_kitchen 0.2.26、comfy_aimdo 0.4.11、gguf 0.17.1、spandrel 0.4.1、transformers 4.57.6、accelerate 1.12.0、sam_2 1.0（声明 `torch>=2.5.1`，所以**不能**降到 torch 2.5 以下）                                                                  |
| 自定义节点                | ComfyUI-Manager、Impact-Pack、GGUF、RMBG、controlnet_aux、IPAdapter_plus、LTXVideo、WanVideoWrapper、VideoHelperSuite、UltimateSDUpscale、AlekPet（其中 `comfyui_controlnet_aux` 的 requirements 只写裸 `torch` / `torchvision`，装它不会把 torch 版本扯回去） |

### 为什么 torch 必须是 cu118

- ZLUDA 只提供 CUDA 11.x 的库名（`cublas64_11.dll`、`nvrtc64_112_0.dll` 等）。cu12x 的 torch 会去找 `cublas64_12.dll`、`cudnn64_9.dll`，这些是 NVIDIA 专有库。
- Python 3.13 在 cu118 线上只有 torch **2.6.0 / 2.7.0 / 2.7.1** 三档可选（2.5.1 及更早没有 cp313 轮子）。选 **2.7.1** 是因为 ComfyUI 的 `extended_fp16_support()` 需要 torch ≥ 2.7，而 `sam_2` 需要 ≥ 2.5.1。

## 安装步骤

以下路径均相对于启动器的**整合包根目录**（包含 `绘世启动器.exe`、`python\`、`ComfyUI\` 的文件夹）。启动器的命令提示符使用整合包自带的 Python，所以直接输入 `pip` / `python` 即可。

### 第 1 步：把 torch 换成 cu118

1. 双击运行绘世启动器（`绘世启动器.exe`），点击左侧“**高级选项**”，再点击右上角“**启动命令提示符**”。
2. 在打开的命令提示符中，依次输入：

```bat
pip freeze > env_before_cu118.txt

pip uninstall -y sageattention triton-windows

pip install --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple "torch==2.7.1+cu118" "torchvision==0.22.1+cu118" "torchaudio==2.7.1+cu118"

python -c "import torch;print(torch.__version__, torch.version.cuda)"
```

| 命令                | 作用                                |
| ------------------- | ----------------------------------- |
| `pip freeze > ...`  | 先备份当前的包列表                  |
| `pip uninstall ...` | 卸载按 cu130 / torch ≥ 2.9 编译的包 |
| `pip install ...`   | 安装 cu118 三件套                   |
| `python -c ...`     | 验证，应输出 `2.7.1+cu118 11.8`     |

> 提示：如果访问包索引需要代理，可在 `pip install` 命令后追加 `--proxy http://<主机>:<端口>`。

### 第 2 步：放置补丁并挂到 `main.py`

1. 将本仓库下的所有文件复制到整合包根目录，使补丁最终位于 `ComfyUI\zluda_patch.py`。真正需要的只有 [`ComfyUI/zluda_patch.py`](ComfyUI/zluda_patch.py)，`README*.md` 与 `LICENSE` 不是必需的。
2. 在 `ComfyUI\main.py` 的**第一行之前**插入下面这段（测试所用的 ComfyUI 版本中，第一行是 `import comfy.options`）：

```python
# ==== ZLUDA 兼容补丁（手工添加，可安全删除） ====
try:
    import os as _zluda_os
    import sys as _zluda_sys

    _zluda_sys.path.insert(0, _zluda_os.path.dirname(_zluda_os.path.abspath(__file__)))
    import zluda_patch  # noqa: F401
except Exception as _zluda_e:
    print("[ZLUDA] Patch FAILED:", repr(_zluda_e))
# ==== ZLUDA 兼容补丁结束 ====
```

> ⚠️ 必须加在**第一行之前**（在 `import comfy.options` 之前），因为补丁要在 torch 被导入之前完成 DLL 预加载。
>
> ⚠️ 如果启动器以后更新或重置 ComfyUI，会覆盖 `main.py`，把这段加回来即可。`zluda_patch.py` 也可能被清掉，建议留一份备份（比如本仓库）。

### 第 3 步：启动参数

1. 正常启动绘世启动器，在“**设置**”中把“**配置模型**”切换至“**专家**”。
2. 在“**高级选项 → 其他设置 → 自定义参数**”中添加：

```
--disable-cuda-malloc
```

原因：`ComfyUI\cuda_malloc.py` 会在非 N 卡上自动开启 `cudaMallocAsync`（`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`），而 ZLUDA 未必实现该分配器。`--disable-cuda-malloc`（定义在 `comfy/cli_args.py`）会强制关闭它。

**不要使用**的参数：`--use-sage-attention`（包已卸载）、`--use-flash-attention`（未安装）、`--enable-triton-backend`（ZLUDA 下无用）、`--fast` / `--fast cublas_ops`（这条路会直接用 cuBLASLt，必崩）、`--cuda-malloc`。

精度设置（启动器 → 计算精度设置）：`Upcast Attention` 关；Precision / Text Encoder / UNet / VAE 设 **FP16 或 Auto 均可**，根因不是精度。

### 第 4 步（可选）：用真实模型试跑，以 Anima Base 1.0 为例

这是“分离式”模型，必须用三个加载器 + 官方模板：

| 文件                          | 放置目录                                                  |
| ----------------------------- | --------------------------------------------------------- |
| `anima-base-v1.0.safetensors` | `ComfyUI\models\diffusion_models\`（`models\unet\` 等价） |
| `qwen_3_06b_base.safetensors` | `ComfyUI\models\text_encoders\`                           |
| `qwen_image_vae.safetensors`  | `ComfyUI\models\vae\`                                     |

工作流：直接用整合包自带的官方模板 `ComfyUI\blueprints\Text to Image (Anima Base 1.0).json`（拖进网页，或菜单 Workflow → Templates 搜 `Anima`）。模板节点为 `Load Diffusion Model` + `Load CLIP`（type 保持 `stable_diffusion`，ComfyUI 会按文件内容自动识别 Qwen3 编码器）+ `Load VAE`；KSampler 默认 steps 30 / cfg 4 / `er_sde` / `simple`，尺寸组为 1MP（约 1024×1024）。

**常见错误：** 如果用 `Load Checkpoint` 直接加载 `anima-base-v1.0.safetensors`，会报 `ERROR: clip input is invalid: None`，因为该文件**不含文本编码器**（`comfy\sd.py` 的 `load_checkpoint_guess_config()` 找不到 CLIP 就返回 `clip=None`，`comfy\nodes.py:74` 抛错）。

## 补丁原理

[`ComfyUI/zluda_patch.py`](ComfyUI/zluda_patch.py) 按顺序做 5 件事，缺一不可：

1. **预加载 ZLUDA 的 DLL**（`nvcuda.dll`、`nvml.dll`、`cublas64_11.dll`、`cusparse64_11.dll`、`nvrtc64_112_0.dll`），并把 ZLUDA 目录与 `%HIP_PATH%\bin` 加进 `PATH` / `os.add_dll_directory`。
   _为什么：_ ZLUDA 的 `nvcuda.dll` 必须**先于 torch 的 CUDA 初始化**被加载，否则 torch 会报 `Found no NVIDIA driver on your system`。启动器本来自己在做这件事（每次运行把 DLL 解压到 `%TEMP%\<GUID>`），但它的注入链路在我们的场景下不稳定，所以补丁自己扫描 `%TEMP%`、`%TMP%`、`%LOCALAPPDATA%\Temp`（以及整合包根目录下的 `zluda\`）找到那个目录并预加载。
2. **设置 `ROCBLAS_TENSILE_LIBPATH`** 为 `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library`。
   _为什么：_ ZLUDA 的 cuBLAS 把矩阵乘转给 **rocBLAS**，rocBLAS 需要**对应架构（gfx906）的 Tensile 内核**。启动器把该变量指向 `%TEMP%` 下某个没有这些内核的目录，结果就是 `rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906` 并 `Fatal Python error: Aborted`。
3. **`torch.backends.cudnn.enabled = False`**
   _为什么：_ ZLUDA 没有 cuDNN；不关就会 `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`（VAE 的卷积会最先中招）。关闭后卷积走可用的回退路径（已实测）。
4. **只保留 math SDP 注意力后端**（关闭 flash 与 mem-efficient）。
   _为什么：_ ZLUDA 不支持 flash attention 的内核；绘世启动器自己的 ZLUDA 兼容补丁也是这么做的。
5. **把“带 bias 的线性层”拆成 `matmul` + 加法**（并对 `torch.addmm` 做同样处理）。
   _为什么：_ PyTorch 对 `F.linear(x, w, b)` / `addmm` 使用**只有 cuBLASLt 实现的 bias 融合 GEMM**（`gemm_and_bias`）。ZLUDA **完全没有 cuBLASLt 实现**（5 个 DLL 里没有任何 `cublasLt*` 导出），于是 torch 会去用 `torch\lib\cublasLt64_11.dll`（NVIDIA 正版，544 MB），必然返回 `CUBLAS_STATUS_NOT_SUPPORTED`。实测：无 bias 的 `mm` / `bmm` / `baddbmm` 全部正常，**只有带 bias 的 linear / addmm 会崩**，与精度（fp16 / bf16 / fp32）无关。

## 验证清单

### 1. 启动日志

应出现下面两行（`<TEMP>` 指你的临时文件夹，通常是 `%LOCALAPPDATA%\Temp`）：

```
[ZLUDA] Preload: zluda_dir=<TEMP>\<GUID> | loaded=5/5
[ZLUDA] OK torch=2.7.1+cu118 | Tensile=C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library | cudnn=False | sdp=math | linear-patch[F.linear=_linear]
```

同样的内容也会追加写入整合包根目录下的 `_zluda_status.txt`（以及临时文件夹里的 `zluda_status.txt`）。

### 2. 一键自检

不启动界面。在启动器的命令提示符里进入整合包的 `ComfyUI` 文件夹后运行：

```bat
cd ComfyUI

python -c "import zluda_patch, torch, torch.nn.functional as F; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('patched:', F.linear.__name__, torch.addmm.__name__, torch.backends.cudnn.enabled); x=torch.randn(2,128,512,device='cuda',dtype=torch.float16); w=torch.randn(1024,512,device='cuda',dtype=torch.float16); b=torch.randn(1024,device='cuda',dtype=torch.float16); print('linear:', float(F.linear(x,w,b).sum()))"
```

期望输出：

```
cuda: True
device: AMD Radeon Pro VII
patched: _linear _addmm False
linear: <某个数字>
```

自检需要能找到 ZLUDA 目录：要么是之前启动器运行时留在临时文件夹里的 `<GUID>` 目录，要么是整合包根目录下的 `zluda\` 文件夹（用 7-Zip 解压启动器的 `.launcher\zluda.<hash>.extpack` 即可得到）。

### 3. ComfyUI 启动日志里的对照项

| 行                                                                                 | 期望值                                                  |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `pytorch version:`                                                                 | `2.7.1+cu118`                                           |
| `Total VRAM ...`                                                                   | 测试所用显卡为 `16368 MB`                               |
| `VAE load device: ... dtype:`                                                      | `torch.float16`（或 bfloat16，取决于精度设置）          |
| `model weight dtype ...`                                                           | `torch.float16` / `torch.bfloat16`                      |
| `Unsupported Pytorch detected. DynamicVRAM requires Pytorch version 2.8 or later.` | **正常提示**，ComfyUI 会退回旧版 ModelPatcher，不是错误 |

## 故障排查

| 症状（日志关键字）                                                                                            | 真正原因                                                                                                                                           | 处理                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `ValueError: Model "..." is corrupt or invalid... Please download the model again`                            | **文件不存在或路径错**，不是文件损坏。`comfy\utils.py`（测试版本的 122-146 行）把 `safe_open()` 的**任何**异常（含 FileNotFoundError）都报成“损坏” | 检查路径与文件名；不要盲目重新下载                                                                                        |
| `rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906` + `Fatal Python error: Aborted` | `ROCBLAS_TENSILE_LIBPATH` 指向的目录没有 gfx906 内核（常见于启动器的 `%TEMP%\<GUID>` 目录被清理或轮换）                                            | 补丁第 2 条会把它指到 HIP SDK 的 `bin\rocblas\library`                                                                    |
| `CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmulAlgoGetHeuristic(...)`                    | 带 bias 的线性层 / `addmm` 走了 cuBLASLt 融合路径                                                                                                  | 补丁第 5 条；先用[一键自检](#2-一键自检)确认 `F.linear` 已是 `_linear`                                                    |
| `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`                                                                    | cuDNN 未关闭                                                                                                                                       | 补丁第 3 条                                                                                                               |
| `RuntimeError: Found no NVIDIA driver on your system`                                                         | ZLUDA DLL **没有在 `import torch` 之前**被预加载（例如手工提前 import torch，或启动器注入失效）                                                    | 补丁第 1 条；确认启动日志有 `loaded=5/5`                                                                                  |
| `ERROR: clip input is invalid: None`                                                                          | 用 `Load Checkpoint` 加载了**不含文本编码器**的模型（如 anima-base）                                                                               | 改用 `Load Diffusion Model` + `Load CLIP` + `Load VAE`，或直接用官方模板                                                  |
| 显存不足 / OOM                                                                                                | 16 GB 显存跑大模型                                                                                                                                 | 降分辨率、`--fp8_e4m3fn-unet`、`--reserve-vram 1`。系统内存只有 16 GB 时**不要**用 `--lowvram` / `--novram`，会被换页拖死 |
| 在启动器里点“安装 PyTorch / 安装依赖”后一切复原                                                               | 启动器按内置清单重装（它给 ComfyUI 核心钉的 Python 是 3.11.6，cu118 最高只到 2.3.1，在 3.13 上装不上）                                             | 不要在启动器里点“安装依赖 / 切换 Torch 版本 / 重装环境”                                                                   |

### 回退到原装 torch

在启动器的命令提示符里：

```bat
pip install --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" "torchaudio==2.11.0+cu130"

pip install "triton-windows==3.5.1.post24"

pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
```

回退后 ZLUDA 就用不了了，`main.py` 也不该再调用 `zluda_patch.py`（删掉引导代码即可）。

## 已知限制与后续可优化项

1. **`torch.backends.cuda.preferred_blas_library(...)` 在 Windows 上无效**（torch 自己会警告 `experimental feature. It is not supported on Windows`）。别指望用它来切 BLAS 后端，只能像补丁那样在 Python 层绕。
2. **不要改名或删除 `torch\lib\cublasLt64_11.dll`。** `torch_cuda.dll` 静态引用了它，改名会让 `import torch` 直接失败（已实测确认）。
3. **精度设置：** 现在 FP16 / Auto 都能跑。若想用模型原生 bf16，把 UNet Precision 改回 `Auto` 即可。
4. **性能：** 目前注意力是 math SDP（ZLUDA 的限制），显存与速度都不会最优；可尝试 `--use-split-cross-attention`，或降低分辨率。
5. **补丁的扩展方向：** 若以后在别的算子上再遇到 `CUBLAS_STATUS_NOT_SUPPORTED`，可在 `patch_torch()` 里继续绕（例如把 `F.conv2d` 换成 `unfold + matmul` 的等价实现，已实测可用）。
6. **防止启动器覆盖：** 启动器更新 ComfyUI 会重置 `main.py`，需要重新插入引导代码；`zluda_patch.py` 也可能被清掉，建议留备份。
7. **诊断资料：** 启动器（疑难解答页面）生成的 `Diagnostics-*.log` 是环境快照加控制台缓冲区，**会包含多次运行的历史输出**。判断某行日志是否来自新进程，要看 `_zluda_status.txt` 的时间戳。

## 仓库结构

```
.
├── ComfyUI/
│   └── zluda_patch.py   # 兼容补丁（复制到整合包根目录下的 ComfyUI\）
├── LICENSE
├── README.md
└── README.zh-CN.md
```

部署后的整合包中涉及的文件（除特别注明外，均相对于整合包根目录）：

| 路径                                                            | 说明                                                |
| --------------------------------------------------------------- | --------------------------------------------------- |
| `python\python.exe`                                             | Python 3.13.11                                      |
| `ComfyUI\main.py`                                               | **在文件最顶部插入了引导代码**                      |
| `ComfyUI\zluda_patch.py`                                        | 本方案新增的补丁（来自本仓库）                      |
| `_zluda_status.txt`                                             | 补丁自检输出（每次启动追加一行）                    |
| `.launcher\zluda.<hash>.extpack`                                | ZLUDA 的 5 个 DLL（可用 7-Zip 解压）                |
| `.launcher\tensile-gfx906.<hash>.extpack`                       | gfx906 的 rocBLAS Tensile 内核（备用来源）          |
| `.launcher\preference.json`                                     | 启动器偏好（含代理地址、自定义参数等）              |
| `ComfyUI\blueprints\Text to Image (Anima Base 1.0).json`        | Anima 官方模板                                      |
| `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library`（绝对路径） | 补丁所指向的 gfx906 Tensile 内核目录                |
| `%TEMP%\<GUID>\`（绝对路径，每次运行新建）                      | 启动器解压 ZLUDA DLL 的位置，补丁会自动扫描并预加载 |

若更换整合包版本或 HIP SDK 版本，请重新核对[实测环境](#实测环境)中的版本匹配关系。

## 许可证

本项目采用 [MIT License](LICENSE)。

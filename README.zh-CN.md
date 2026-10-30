# 秋叶整合包 ComfyUI + AMD MI50（ZLUDA / gfx906 / Windows）

[English](README.md) | [简体中文](README.zh-CN.md)

在 Windows 上，通过 [ZLUDA](https://github.com/vosen/ZLUDA) 让 **秋叶 ComfyUI 整合包**（使用**绘世启动器**）跑在 **AMD Instinct MI50**（刷入 **Radeon VII PRO vBIOS**）上。

本仓库记录了一套**已完整跑通**的方案：实际执行的命令、一个小型兼容补丁（[`ComfyUI\zluda_patch.py`](ComfyUI/zluda_patch.py)）、每一步为什么需要、验证方法以及故障排查表。所有结论均来自单台机器上的实测（命令行输出、日志、文件哈希），不是猜测。

## 结论速览

要让这套整合包在 ZLUDA 上跑通，需要且只需要三件事：

1. **把 torch 换成 cu118 版本。** ZLUDA 只提供 CUDA 11 的库，用 CUDA 12+ 编译的 PyTorch 一定不行。
2. **放入补丁 `ComfyUI\zluda_patch.py`，并在 `ComfyUI\main.py` 顶部挂一小段引导代码。** 补丁按 PART 0–3 组织，共十个单元：`PRELOAD`、`TORCH`、`SDPA`、`LINEAR`、`CLI_ARGS`、`MODEL_MANAGEMENT`、`ONNX`、`SUPPORTED_MODELS`、`OPS`、`MODEL_DETECTION`。前四个覆盖跑不起来的故障（ZLUDA DLL / Tensile、cuDNN + SDP、注意力包装、cuBLASLt 线性层 + INT8）；后六个覆盖启动参数、精度、ONNX Runtime、LTXAV、可选算子和额外层。详见[补丁原理](#补丁原理)。
3. **四行精度可以全留“由 ComfyUI 决定”（想要最快就把三行模型精度钉成 FP16——只对支持 fp16 的模型）**：gfx906 上 bf16 只有 fp16 的 0.55 倍速，fp32 则权重显存翻倍。补丁的自动策略会在模型支持 fp16 时选 fp16，放得下 fp32 时选 fp32，否则 bf16；VAE 跟随自己的 `working_dtypes`（有 fp16 就 fp16，否则 bf16；Flux 保持 bf16）。对不列出 fp16 的 VAE 强行钉 FP16 才可能溢出成 NaN、解码出全黑图。见[精度与性能](#精度与性能实测)。

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

1. 将本仓库下的所有文件复制到整合包根目录，使补丁最终位于 `ComfyUI\zluda_patch.py`。运行时只有 [`ComfyUI\zluda_patch.py`](ComfyUI/zluda_patch.py) 会被加载；`zluda_dtype_check.py` 是可选的[离线自检](#2-一键自检)，`README*.md` 与 `LICENSE` 不是必需的。
2. 在 `ComfyUI\main.py` 的**第一行之前**插入下面这段（测试所用的 ComfyUI 版本中，第一行是 `import comfy.options`）：

```python
# ==== ZLUDA compatibility patch (added manually, safe to remove) ====
# ==== ZLUDA 兼容补丁（手工添加，可安全删除） ====
try:
    import os as _zluda_os
    import sys as _zluda_sys

    _zluda_sys.path.insert(0, _zluda_os.path.dirname(_zluda_os.path.abspath(__file__)))
    import zluda_patch  # noqa: F401
except Exception as _zluda_e:
    # This block has to work before the patch is importable, so it guards its own output:
    # a console code page that cannot encode Chinese drops that line only, never the English one.
    #
    # 这段代码在补丁能导入之前就得工作，所以自己保证写日志不出错：
    # 控制台代码页可能编不了中文，编不了就只丢中文那一行，英文行不受影响。
    def _zluda_print(text):
        try:
            print(text)
        except Exception:
            pass

    _zluda_print("[ZLUDA] Patch FAILED: %r" % (_zluda_e,))
    _zluda_print("[ZLUDA] 说明: 引导代码本身出错：检查它是否插在 main.py 第一行之前、zluda_patch.py 是否在 ComfyUI\\ 下。")
# ==== End of ZLUDA compatibility patch ====
# ==== ZLUDA 兼容补丁结束 ====
```

> ⚠️ 必须加在**第一行之前**（在 `import comfy.options` 之前），因为补丁要在 torch 被导入之前完成 DLL 预加载。
>
> ⚠️ 如果启动器以后更新或重置 ComfyUI，会覆盖 `main.py`，把这段加回来即可。`zluda_patch.py` 也可能被清掉，建议留一份备份（比如本仓库）。

### 第 3 步：启动器选项

下面四项按启动器里的顺序排列。

#### 1. Cross-Attention 优化方案

| 选项                          | 结论                                                                 |
| ----------------------------- | -------------------------------------------------------------------- |
| 尝试使用 xFormers（推荐）     | 不可用：整合包没有装 xFormers。                                      |
| 使用 Sage Attention           | 不可用：SageAttention 已卸载。                                       |
| 使用缩放点积（SDP）方案       | 能用，但补丁会把它强制成 math SDP，比 Split 慢。                     |
| 使用 Sub-Quadratic 方案       | 能用，但比 Split 慢 2.71 倍（实测见[精度与性能](#精度与性能实测)）。 |
| **使用旧版（Split）优化方案** | **选这个**，ZLUDA 下能用的选项里最快。                               |

选择“**使用旧版（Split）优化方案**”。**不要**再往自定义参数里加 `--use-split-cross-attention`；下拉框才是正确入口。Split 与 SubQuad 的实测数字见[精度与性能](#精度与性能实测)。

#### 2. 计算精度设置

| 面板项                                           | 设为                | 说明                                                    |
| ------------------------------------------------ | ------------------- | ------------------------------------------------------- |
| 使用向上采样法提高 XAttn 精度 (upcast-attention) | **保持关**          | 开了会把注意力升到 fp32，这卡上更慢                     |
| 通常模型精度                                     | **由 ComfyUI 决定** | 推荐；FP16 是支持 fp16 的模型的最高速覆盖               |
| 文本编码器 (Text Encoder) 模型精度               | **由 ComfyUI 决定** | 默认本来就是 fp16，钉 FP16 属于保险                     |
| U-Net 模型精度                                   | **由 ComfyUI 决定** | 策略：支持 fp16 用 fp16，放得下 fp32 用 fp32，否则 bf16 |
| **VAE 模型精度**                                 | **由 ComfyUI 决定** | 跟随 VAE 的 `working_dtypes`：有 fp16 就 fp16，否则 bf16。Flux `[bf16, fp32]` 保持 bf16；强行 FP16 才可能溢出 |

四行都留在“由 ComfyUI 决定”时，补丁会按模型选：模型列表里有 fp16 就用 fp16；`[bfloat16, float32]` 模型（Lumina2、Z-Image 等，它们把 fp16 排除在外，因为 fp16 会出 NaN）在 fp32 权重放得下时用 fp32，否则 bf16；VAE 跟随自己的列表：有 fp16 就 fp16，否则 bf16（Flux 保持 bf16）。三行模型精度钉 FP16 仍是最快选项，但只对确实支持 fp16 的模型。

⚠️ **VAE 这一行信任 VAE 的 `working_dtypes`。** 列表里有 fp16（例如 `[bf16, fp16, fp32]`）就用 fp16；Flux 的 `[bf16, fp32]` 没有 fp16，所以保持 bf16。对不列出 fp16 的 VAE 强行 **半精度 (FP16)** 会绕过列表，可能溢出成 NaN/全黑（实测见[精度与性能](#精度与性能实测)）。设 `ZLUDA_VAE_DTYPE=comfy` 会把这一行交回 ComfyUI 原顺序；`ZLUDA_ALLOW_BF16=1` 则同时作用于 VAE 和 UNet 两侧。如果出黑图并伴随 `RuntimeWarning: invalid value encountered in cast` 且没有 `FAILED`，先查这一行；如果是 `comfy_ops: FAILED`，那是 `SDPA` 单元。

**兜底**：如果以后启动器不再转发这些行，就把 `--fp16-unet`（U-Net）和 `--fp16-text-enc`（文本编码器）写进“**高级选项 → 其他设置 → 自定义参数**”。**不要**给不列出 fp16 的 VAE（Flux）加 `--fp16-vae`，那正是上面的溢出。

⚠️ **必须避开的坑**：显式的 `--bf16-unet` / `--bf16-text-enc` 在自动选择精度**之前**被检查，所以只要这两行还是 BF16，它就会压过一切；别设 BF16。显式 `--bf16-vae` 是安全的，但 VAE 列表里已有 bf16 时没必要设。

✅ **启动器确实会转发这些行。** 把 U-Net 模型精度设成“半精度 (FP16)”时，启动日志会出现 `model weight dtype torch.float16`，控制台回显的命令行里会含 `--fp16-unet`。自动路径不可能是来源：对 `[bfloat16, float32]` 模型，策略只会返回 fp32/bf16，不会返回 fp16；`PRIORITIZE_FP16` 也关着（`Fast` 没开）。启动器二进制里只有它自己的选项名，所以这条链路只能实测。

#### 3. 预览图生成模式

| 选项         | 建议                                                                      |
| ------------ | ------------------------------------------------------------------------- |
| **关闭预览** | **生产 / 长时间跑图选这个。** 不做预览解码，不占额外显存。                |
| 自动         | **在 ComfyUI 0.30.2 里等于 Latent2RGB**；便宜，不加载额外模型，默认即可。 |
| Latent2RGB   | 与“自动”相同；便宜，不加载额外模型。                                      |
| TAESD        | 预览更清晰，但要加载小模型、每次预览多一次解码；有余量再用。              |

#### 4. 自定义参数

如需调整自定义参数，请打开“**设置**”，将“**配置模式**”切换至“**专家**”后在“**高级选项**”里的最下方“**其他设置**”就可以看到“**自定义参数**”了。

**不需要在自定义参数里写 `--disable-cuda-malloc`**：`CLI_ARGS` 单元会自己设 `args.disable_cuda_malloc`，以及 `disable_pinned_memory`、`disable_async_offload`、`reserve_vram`，所以“自定义参数”里什么都不用写。

**不要加** `--use-sage-attention`、`--use-flash-attention`、`--enable-triton-backend`、`--fast` / `--fast cublas_ops`（这条路会直接用 cuBLASLt，必崩）、`--cuda-malloc`。

### 第 4 步（可选）：用真实模型试跑，以 Anima Base 1.0 为例

这是“分离式”模型，必须用三个加载器 + 官方模板：

| 文件                          | 放置目录                                                  |
| ----------------------------- | --------------------------------------------------------- |
| `anima-base-v1.0.safetensors` | `ComfyUI\models\diffusion_models\`（`models\unet\` 等价） |
| `qwen_3_06b_base.safetensors` | `ComfyUI\models\text_encoders\`                           |
| `qwen_image_vae.safetensors`  | `ComfyUI\models\vae\`                                     |

工作流：直接用整合包自带的官方模板 `ComfyUI\blueprints\Text to Image (Anima Base 1.0).json`（拖进网页，或菜单 Workflow → Templates 搜 `Anima`）。模板节点为 `Load Diffusion Model` + `Load CLIP`（type 保持 `stable_diffusion`，ComfyUI 会按文件内容自动识别 Qwen3 编码器）+ `Load VAE`；KSampler 默认 steps 30 / cfg 4 / `er_sde` / `simple`，尺寸组为 1MP（约 1024×1024）。

**常见错误**：如果用 `Load Checkpoint` 直接加载 `anima-base-v1.0.safetensors`，会报 `ERROR: clip input is invalid: None`，因为该文件**不含文本编码器**（`comfy/sd.py` 的 `load_checkpoint_guess_config()` 找不到 CLIP 就返回 `clip=None`，`comfy/nodes.py:74` 抛错）。

## 补丁原理

[`ComfyUI\zluda_patch.py`](ComfyUI/zluda_patch.py) 按 PART 0–3 组织：**PART 0** 契约（路径与开关）、**PART 1** 工具（报告、环境变量解析、一套通用导入钩子的 loader/finder）、**PART 2** 十个补丁单元、**PART 3** 装配（`MODULE_PATCHES`、`TORCH_STEPS`、`_finish_torch()`、`_bootstrap()`）。下面十节按文件顺序排列，每节一个单元，统一用四段式：**现象**（用户看到什么）、**原因**（为什么会这样）、**修法**（补丁怎么做）、**代价**（有没有副作用）。代码里的单元头多一段 **失败**（`Failure`），写的是会出现在状态行里的字符串。

| #   | 单元               |
| --- | ------------------ |
| 1   | `PRELOAD`          |
| 2   | `TORCH`            |
| 3   | `SDPA`             |
| 4   | `LINEAR`           |
| 5   | `CLI_ARGS`         |
| 6   | `MODEL_MANAGEMENT` |
| 7   | `ONNX`             |
| 8   | `SUPPORTED_MODELS` |
| 9   | `OPS`              |
| 10  | `MODEL_DETECTION`  |

前四个单元覆盖“跑不起来”的故障；后六个覆盖启动参数、精度、ONNX Runtime、LTXAV、可选算子和额外层。

### 1. `PRELOAD`

**作用：** 在导入 torch 之前加载 ZLUDA 的 DLL，并把 rocBLAS 指向 gfx906 的 Tensile 内核。

- **现象**：`RuntimeError: Found no NVIDIA driver on your system`；或 `rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906`，随后 `Fatal Python error: Aborted`。

- **原因**：ZLUDA 的 `nvcuda.dll` 必须**先于 torch 的 CUDA 初始化**被加载；启动器本来自己在做这件事（每次运行把 DLL 解压到 `%TEMP%\<GUID>`），但它的注入链路在我们的场景下不稳定。ZLUDA 的 cuBLAS 再把矩阵乘转给 **rocBLAS**，rocBLAS 需要**对应架构（gfx906）的 Tensile 内核**；启动器把 `ROCBLAS_TENSILE_LIBPATH` 指向 `%TEMP%` 下某个没有这些内核的目录。

- **修法**：补丁自己扫描 `%TEMP%`、`%TMP%`、`%LOCALAPPDATA%\Temp`（以及整合包根目录下的 `zluda\`）找到含 `cublas64_11.dll` 的目录；把 ZLUDA 目录与 `%HIP_PATH%\bin` 加进 `PATH` / `os.add_dll_directory`；预加载 `nvcuda.dll`、`nvml.dll`、`cublas64_11.dll`、`cusparse64_11.dll`、`nvrtc64_112_0.dll`；并把 `ROCBLAS_TENSILE_LIBPATH` 设为 `<HIP SDK>\bin\rocblas\library`，优先读 `HIP_PATH`，取不到时回退到实测环境的 `C:\Program Files\AMD\ROCm\5.7`。

- **代价**：没有。只是启动时多扫几个临时目录。

### 2. `TORCH`

**作用：** 在 torch 层关闭 cuDNN，并只保留 math SDP 后端。

- **现象**：`cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`，VAE 的卷积会最先中招；或注意力落到 ZLUDA 上不存在的 flash / mem-efficient 内核。

- **原因**：ZLUDA 没有 cuDNN，也没有 flash 内核；绘世启动器自己的 ZLUDA 兼容补丁也是这么做的。

- **修法**：`torch.backends.cudnn.enabled = False`；`enable_flash_sdp(False)`、`enable_mem_efficient_sdp(False)`、`enable_math_sdp(True)`。

- **代价**：没有。FLASH / cuDNN 本来就不存在，调用一直回退到 math；`SDPA` 单元负责处理 `set_priority=True` 把 FLASH / cuDNN 又顶回前面的情况。

### 3. `SDPA`

**作用：** 把 `F.sdpa` 和 `comfy.ops.sdpa` 包成 math-only，让所有注意力调用都走 math SDP 后端。

- **现象**：要么直接崩：

  ```
  RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
  RuntimeError: cutlassF: no kernel found to launch!
  ```

  要么更糟：工作流跑完不报错，但保存出来是**纯黑图**，`SaveImage` 打印
  `RuntimeWarning: invalid value encountered in cast`（说明解码结果里是 NaN）。

- **原因**：`comfy/ops.py` 自己装了一个 SDPA 包装器，用
  `sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True)`，列表是
  `[FLASH, CUDNN, EFFICIENT, MATH]`。`set_priority=True` 会**无视** `TORCH` 单元设的
  `enable_flash_sdp(False)`，而 ZLUDA 上 FLASH / cuDNN 都不存在。

  有三处会走到它：文本编码器（`attention_pytorch`）、VAE
  （`comfy/ldm/modules/diffusionmodules/model.py` 直接调用）、主模型（split 注意力在没有
  xformers 时回退）。所以先是文本编码器坏，接着 VAE 坏，主模型则在回退时坏；不抛异常时
  就是 NaN → 黑图。

  实测（batch×heads = 16，T = 512，D = 128）：

  | 调用路径                                               | fp16     | bf16     | fp32     |
  | ------------------------------------------------------ | -------- | -------- | -------- |
  | 裸 `F.scaled_dot_product_attention`                    | OK       | OK       | OK       |
  | `comfy.ops.scaled_dot_product_attention`（优先级列表） | **失败** | **失败** | **失败** |
  | 只给 math（`SDPBackend.MATH`）                         | OK       | OK       | OK       |

- **修法**：`comfy/ops.py` 自己装了一个 SDPA 包装器，带
  `SDPA_BACKEND_PRIORITY = [FLASH_ATTENTION, CUDNN_ATTENTION, EFFICIENT_ATTENTION, MATH]` 和
  `set_priority=True`（0.30.2 里是 56-93 行），所以 `TORCH` 单元的 `enable_*` 开关会被无视、每次调用
  都按这个列表走。只补 `F` 没用（`comfy.ops` 在导入时就把真实函数绑定了）；补丁把 `F` 和
  `comfy.ops` 自己的引用都包住，于是所有调用都在 `sdpa_kernel([MATH])` 里跑。
  `comfy.ops` 在引导代码之后才导入，所以用导入钩子补。
  那三个 `can_use_flash_attention` / `can_use_cudnn_attention` /
  `can_use_efficient_attention` 探测只在**有它们的构建上**才强制为 `False`——0.30.2 一个都没有
  （实测：这些检查在 `torch.backends.cuda` 里，由 `comfy/ops.py:86-88` 调用，真正生效的是上面
  那份优先级列表），所以在这版上那段是空操作，真正干活的是包装器。

  现场踩坑：第一版在 `comfy.ops` 自己的 `exec_module` 里又 `import comfy.ops`，
  会抛 `AttributeError: cannot access submodule 'ops' of module 'comfy'`；结果
  `comfy.ops` 那份没被补上，图一直是黑的。现在 `comfy.ops` 有自己的一行
  `comfy_ops:` 状态，失败会显示成 `comfy_ops: FAILED ...`，不会再藏在 torch 行里；
  但仍然要 grep 整个状态文件的 `FAILED`，不要只看那行。

- **代价**：没有。FLASH / cuDNN 本来就不存在，调用一直回退到 math。S=4096、
  batch×heads=16、D=128 时：`attention_pytorch` 39.1 ms、math SDP 39.1 ms、split 26.7 ms。
  这是正确性修复；想要快就换 split（见[注意力后端](#精度与性能实测)）。

### 4. `LINEAR`

**作用：** 把带 bias 的 `F.linear` / `addmm` 拆成 `matmul + add`，并在 `F.linear` 之前反量化 INT8 tensorwise 权重。

- **现象**：`CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul ...`；INT8 tensorwise 模型是同一个调用，`abType 3 cType 10 computeType 72`（INT8 × INT8 → INT32）。

- **原因**：PyTorch 对 `F.linear(x, w, b)` / `addmm` 使用**只有 cuBLASLt 实现的 bias 融合 GEMM**（`gemm_and_bias`）。ZLUDA **完全没有 cuBLASLt 实现**（5 个 DLL 里没有任何 `cublasLt*` 导出），于是 torch 会去用 `torch\lib\cublasLt64_11.dll`（NVIDIA 正版，544 MB），必然返回 `CUBLAS_STATUS_NOT_SUPPORTED`。实测：无 bias 的 `mm` / `bmm` / `baddbmm` 全部正常，**只有带 bias 的 linear / addmm 会崩**，与精度（fp16 / bf16 / fp32）无关。INT8 tensorwise 是同一件事再往下一层：`comfy_kitchen` 的 `TensorWiseINT8Layout` 最后走到 `torch._int_mm`，也就是 cuBLASLt 的 INT8 GEMM，ZLUDA 没有实现。**计算精度不是问题**：`int8_tensorwise` 会设 `quantize_input: False`，权重以 `QuantizedTensor` 原样交给 `F.linear`，从来没被反量化，`--fp16-unet` 救不了。

- **修法**：带 bias 的路径拆成 `matmul` + 加法；`F.linear` 还会把 `QuantizedTensor` 权重反量化成输入 dtype；`torch.addmm` 只做 bias 拆分。反量化结果可以按权重对象缓存，开关是 `ZLUDA_INT8_DEQUANT_CACHE=1`（默认关闭）。

  看 safetensors 头里的 `comfy_quant` JSON 判断模型是否受影响，而不是文件名：

  | `comfy_quant`                        | 受影响？                        |
  | ------------------------------------ | ------------------------------- |
  | `{"format": "int8_tensorwise", ...}` | **是**，会走 `torch._int_mm`    |
  | `{"format": "convrot_w4a4", ...}`    | 否，用 comfy_kitchen 自己的内核 |
  | 无（普通 fp16 / bf16 / fp8）         | 否                              |

  文件名带 `int8_convrot` 也可能写着 `format: int8_tensorwise`；`convrot` 标记本身不会让它绕开 `torch._int_mm`。

  在 `z_image_turbo_int8_convrot` 的 11520×3840 qkv 层上实测（m=90，fp16 输入）：

  | 做法                 | 每次调用 |
  | -------------------- | -------- |
  | 不处理               | 崩       |
  | 每次反量化（默认）   | 24.4 ms  |
  | 反量化一次并缓存     | 2.0 ms   |
  | 同尺寸原生 fp16 GEMM | 1.34 ms  |

  缓存**默认关闭**，因为原始 INT8 权重仍在，再存一份完整 fp16：6.14 B 模型是 5.72 GB INT8 + 11.44 GB fp16 = 17.16 GB，约为 fp16 模型的 1.5 倍（3 倍 INT8），16 GB 卡放不下。显存有余时设 `ZLUDA_INT8_DEQUANT_CACHE=1`。缓存用 `weakref.WeakKeyDictionary` 做键，模型释放时一起释放。

- **代价**：能用但**不快**。优先级：真正的 `convrot_w4a4` 模型 > `fp8` 模型（它的 handler 失败时本来就会退化成反量化）> `bf16`（注意 16 GB 系统内存）。

### 5. `CLI_ARGS`

**作用：** 在 `comfy.cli_args` 解析完成后覆盖启动参数。

- **现象**：`Enabled pinned memory ...` 出现在启动日志里（说明启动参数没生效）。

- **原因**：启动器把选项存在 `.launcher\preference.json`：重置就没了，绕过 `main.py` 的入口更是完全拿不到。

- **修法**：`comfy/cli_args.py` 在导入时解析，之后 `cuda_malloc.py`、`comfy/model_management.py`、`comfy/ldm/modules/attention.py` 才读 `args`，所以补丁钩住这次导入，在对象刚构造完时改一次：

  | 参数                      | 为什么                                                                 |
  | ------------------------- | ---------------------------------------------------------------------- |
  | `--disable-pinned-memory` | Windows 上会锁最多 40% 内存；这台只有 16 GB，pinned 缓冲也是常见崩溃源 |
  | `--disable-async-offload` | torch 2.7 走旧版 ModelPatcher，异步卸载没有收益                        |
  | `--disable-cuda-malloc`   | ZLUDA 没有 cudaMallocAsync；补丁自己会设 `args.disable_cuda_malloc`    |
  | `--reserve-vram 1.5`      | 给系统和其它程序留出显存，也给 VAE 解码的峰值留出余量                  |

  启动器**显式**设的 `--reserve-vram` 优先，不会被覆盖。开关：`ZLUDA_NO_ARG_OVERRIDES=1` 关掉整组，`ZLUDA_RESERVE_VRAM=<GB>` 改预留值，`ZLUDA_FORCE_FP16=1`（会顺带覆盖 fp8 模型的标志，所以默认关），`ZLUDA_SPLIT_ATTENTION=1`。

  **为什么是 1.5 GB（实测）。** 预留就是“模型不许占用的那一部分”，它直接换来 UNet 能常驻多少：

  | 预留       | VAE 解码                                  | Z-Image 的 UNet                         | 说明                                             |
  | ---------- | ----------------------------------------- | --------------------------------------- | ------------------------------------------------ |
  | 0.9 GB     | 退回分块解码                              | 完整加载                                | 解码失败那次分配让驱动的剩余显存读数坏到进程结束 |
  | 2.5 GB     | 正常                                      | **部分卸载：每步从内存流式 201–712 MB** | 解码稳了，但牺牲了常驻                           |
  | **1.5 GB** | 正常（VAE 现在是 bf16，峰值只有当初一半） | 完整加载                                | 实测出来的中间值                                 |

  验证方法：用出问题的那套工作流跑一次，启动日志里 UNet 应该是 `loaded completely`（不是 `loaded partially`），并且不再出现 `Ran out of memory when regular VAE decoding`。

- **代价**：没有。

### 6. `MODEL_MANAGEMENT`

**作用：** 选择 UNet 的计算精度：支持 fp16 就用 fp16，放得下 fp32 就用 fp32，否则 bf16。VAE 跟随自己的 `working_dtypes`：有 fp16 就 fp16，否则 bf16。

- **现象**：bf16 排第一的模型（Anima）即使支持 fp16，日志里还是 `model weight dtype torch.bfloat16`；或者 `[bfloat16, float32]` 模型在放不下时仍被推到 `torch.float32`（OOM）。

- **原因**：`unet_dtype()` 按 `supported_inference_dtypes` 的**模型声明顺序**循环，而 ZLUDA 上报 (8, 8)，`should_use_bf16()` 在问设备之前就返回 `True`。所以 bf16 排第一的模型（Anima Base 1.0 是 `[bfloat16, float16, float32]`）即使支持 fp16 也会选 bf16；`[bfloat16, float32]` 模型会选 bf16，而 bf16 只有 fp16 的 0.55 倍速。全局禁令能救 Anima，却会把 `[bfloat16, float32]` 模型推到 fp32，权重显存翻倍。

- **修法**：补丁包住 `comfy.model_management.unet_dtype()`。显式 `--fp16-unet` / `--bf16-unet` / `--fp32-unet` 仍然优先；否则模型列表里有 fp16 就返回 fp16；只有 bf16 + fp32 时，如果 `model_params * 4` 放得进权重预算就返回 fp32，否则 bf16。预算是 `maximum_vram_for_weights()`（总显存 × 0.88 − 0.8 GiB − 预留）再乘 `ZLUDA_FP32_MARGIN`（默认 0.9）。本机是 11.77 GiB × 0.9 = **10.6 GiB**，也就是只有约 **2.8B 参数**以内的模型能吃到 fp32。这个余量不是装饰：fp32 下激活也是 4 字节，而预算只算权重。预留显存每多 1 GB，上限就降 0.9 GiB——设 2.5 GB 时是 9.7 GiB（2.6B），设 0.9 GB 时是 11.1 GiB（3.0B）。

  补丁还包住 `unet_manual_cast()`。ComfyUI 会把 fp8 或未知权重存储的 manual cast 定成 bf16，而 bf16 在这张卡上只有 0.55 倍速；当模型列表里有 fp16、设备也支持 fp16 时，补丁改成返回 fp16。bf16 权重在上游返回 `None`，这里不动；显式钉了 `--bf16-unet` 的也不动。

  VAE 单独处理：`vae_dtype()` 信任 VAE 的 `working_dtypes`：有 fp16 就 fp16，否则 bf16；显式 `--fp16-vae` / `--bf16-vae` / `--fp32-vae` 仍然优先。Flux 的 `[bf16, fp32]` 保持 bf16。

  开关：`ZLUDA_UNET_DTYPE` 和 `ZLUDA_VAE_DTYPE`（默认都是 `auto`）各把**一侧**交回 ComfyUI 原顺序；`ZLUDA_ALLOW_BF16=1` 是粗粒度别名，把两侧都设成 `comfy`（`[bf16, fp16, ...]` 就回到 bf16）。显式写了的一侧优先于别名，所以 `ZLUDA_ALLOW_BF16=1 ZLUDA_UNET_DTYPE=auto` 会保留 UNet 策略、只放掉 VAE。策略自身出意外时，这一次调用会交回 ComfyUI 并记成 `FAILED`，补丁不会把模型加载搞崩。

  **日志：** 每次加载模型打印一行 `[ZLUDA] model_management: unet ...`，每次加载 VAE 打印一行 `... vae ...`。行里写明 dtype 和原因（`policy:` 表示上面这套策略，`comfy order:` 表示交回 ComfyUI 自己的函数），策略改判上游结果时再追加 `comfy would pick ...`；检查点的存储 dtype 不是浮点时，原因前面会标 `weights uint8` / `weights unknown`（ComfyUI 自己那行 `model weight dtype` 说的是**推理** dtype，不是存储）。manual cast 被改判时另打一行 `... manual cast float16 (policy: upstream bf16, ...)`。启动那一行还会列出启动器钉住的精度标志（`flags=...`）：钉住的行会让策略让位，所以 `flags=fp16_unet` 表示策略装上了，但 UNet 这一行没有走它。启动器的**通用精度 = 半精度 (FP16)** 也会落到那里：ComfyUI 会把 `--force-fp16` 变成 `args.fp16_unet = True`，所以想让策略自己选，那一行必须留在“由 ComfyUI 决定”。

- **代价**：默认没有。`[bfloat16, float32]` 模型放得下 fp32 时会用 2 倍权重显存，换来接近 fp16 的速度；放不下就留在 bf16。VAE 的 fp16/bf16 都是 2 字节，不增加显存。

### 7. `ONNX`

**作用：** 从 ONNX Runtime 中去掉 CUDA / TensorRT 执行提供程序。

- **现象**：RMBG / controlnet_aux / Impact 报 onnxruntime provider 错误（provider 加载失败或在假驱动里崩）。

- **原因**：`controlnet_aux`（DW Pose、线稿、深度）、RMBG、Impact 的检测器都用 ONNX Runtime 跑小模型，它的默认 provider 列表把 `CUDAExecutionProvider` 排第一。装的是 CUDA 12 构建的 `onnxruntime-gpu`，而 ZLUDA 只冒充 CUDA 11，所以这个 provider 要么加载失败，要么在假驱动里崩。

- **修法**：补丁把 `CUDAExecutionProvider` / `TensorrtExecutionProvider` 从 `get_available_providers()` 里过滤掉（模块级函数和 session 真正读的 pybind 副本都改），并把仍然请求 CUDA 的 session 改成 `CPUExecutionProvider`。开关：`ZLUDA_KEEP_ONNX_CUDA=1` 保持原样。

- **代价**：没有。这些小模型本来就小，CPU 跑感觉不到差别，但能避免崩溃。

### 8. `SUPPORTED_MODELS`

**作用：** 修正 LTXAV 的 `memory_usage_factor`。

- **现象**：LTXAV 在 16 GB 卡上 OOM（ComfyUI 以为它几乎不吃显存）。

- **原因**：上游把 LTXAV 的 `memory_usage_factor` 留在 `0.077`，`model_base.py` 用它乘激活显存估算。父类 `LTXV` 用的是 `(cross_attention_dim / 2048) * 5.5`。

- **修法**：补丁把两个数字都从父类上读（`memory_usage_factor` = 5.5、默认 `cross_attention_dim` = 2048），再对 LTXAV 乘 1.5，这样两边的公式不会漂移（例如 `cross_attention_dim` 4096 → 16.5、2048 → 8.25）。

- **代价**：ComfyUI 会更早 offload，但不会爆显存。

### 9. `OPS`

**作用：** 给 `torch.topk`、`torch.stft` / `torch.istft`、`torch.jit.script` 提供可选的 CPU 兜底。

- **现象**：Zluda 仓库无条件替换 `torch.topk`、`torch.stft` / `torch.istft`、`torch.jit.script` 时，每次 `topk` 都变成 D2H + topk + H2D，而 torchvision 的 NMS 内部就用 `topk`，检测节点会明显变慢。

- **原因**：这套环境里没有证据表明需要这些兜底。

- **修法**：默认关闭，按需单独开：`ZLUDA_TOPK_CPU=1`、`ZLUDA_STFT_CPU=1`（音频模型）、`ZLUDA_NO_JITSCRIPT=1`（带 TorchScript 模型的节点包）。

- **代价**：开了就会慢，见上面每次 `topk` 的往返代价。

### 10. `MODEL_DETECTION`

**作用：** 从检查点数 Anima / Cosmos-Predict2 的层数。

- **现象**：出图**全黑**、没有任何报错，日志里有一行 `unet unexpected: [...]` 列出超出 ComfyUI 所建层数的 `blocks.N...`，以及出图保存路径上的 `RuntimeWarning: invalid value encountered in cast`。

- **原因**：ComfyUI 0.30.2 只按 `model_channels` 推 Anima / Cosmos-Predict2 检查点的层数——2048 → 28 层，5120 → 36 层（`comfy/model_detection.py:832-837`）。社区微调里存在层数更多的版本：本机实测 `silvermoonmixAnimaEvolved_v2329BTurbo.safetensors` 是一个 **40 层、BF16** 的 Anima 微调（928 个张量，`net.blocks.0 … net.blocks.39`，外加一个可学习的 `net.pos_embedder`）。ComfyUI 建出 28 层模型，于是 blocks 28-39（240 个键）和 pos embedder 被静默丢掉，采样出的 latent 是 NaN——全黑就是 NaN 经 `np.clip(...).astype(np.uint8)` 的结果。

- **修法**：补丁自己用 `count_blocks(state_dict, key_prefix + "blocks.{}.")` 数检查点里的层数，只要 `anima` / `cosmos_predict2` 的层数与它不同就覆盖 `num_blocks`，并打印 `model_detection: anima num_blocks 28 -> 40 (counted in the checkpoint)`。上游在 `725e6ec60`（“Support anima tunes with extra blocks”，PR #15555）修了同一件事，所以已经会数层数的 ComfyUI 上这是空操作。开关：`ZLUDA_ANIMA_EXTRA_BLOCKS=0`（默认开启，见补丁开头的开关块）。

- **代价**：对层数与表一致的检查点没有代价；数出来的值就是检查点本身的内容。那 3 个 `pos_embedder.*` 键仍然会是 unexpected，与上游修复的行为一致——建出来的模型用 `rope3d` 位置编码，可学习的 pos embedder 不属于它。

`SDPA`、`CLI_ARGS`、`MODEL_MANAGEMENT`、`ONNX`、`SUPPORTED_MODELS` 和 `MODEL_DETECTION`
是模块级单元：它们在 `MODULE_PATCHES`（模块名 → 单元）里注册，由同一个 `sys.meta_path`
钩子在模块执行完的瞬间应用。钩子在 `import torch` **之前**安装，所以 torch 晚到补丁时
`comfy.ops` 仍然会被覆盖。`OPS` 由 `_finish_torch()` 在直接导入和回退两条路径上都补一次。
文件末尾的 `_bootstrap()` 是唯一在导入时执行的代码，上面全是定义。

## 精度与性能（实测）

以下数字均为**测试机实测**（MI50 刷 Radeon VII PRO vBIOS / gfx906 / HIP SDK 5.7.1 / torch 2.7.1+cu118 / ZLUDA）。

### 1. 为什么 FP16 最快

ZLUDA 在这台机器上冒充 NVIDIA 显卡，这会让 ComfyUI 里一条**专门为 gfx906 准备的排除规则失效**：

- ZLUDA 下 `torch.version.hip` 是 `None`、`torch.version.cuda` 是 `11.8`，所以 `is_amd()` 返回 False（它要求 `torch.version.hip` 有值）。
- 于是 `should_use_bf16()` 里“RDNA2 及更老不支持 bf16”那段判断被跳过，而那份架构列表里**明确写着 `gfx906`**。
- 同时 ZLUDA 上报的计算能力是 **(8, 8)**、`torch.cuda.is_bf16_supported()` 返回 True，于是 `should_use_bf16()` 最终返回 True。

| 精度 | 实测        | 相对 fp16 |
| ---- | ----------- | --------- |
| fp16 | 9.54 TFLOPS | 1.00×     |
| bf16 | 5.26 TFLOPS | 0.55×     |
| fp32 | 9.10 TFLOPS | 0.95×     |

（4096×4096 矩阵乘。）

结论：**fp16 最快；bf16 只有 0.55 倍；fp32 约 0.95 倍，但权重显存翻倍。** 第 3 步的 Auto 策略据此选择：支持 fp16 就用 fp16，放得下 fp32 才用 fp32，否则 bf16。

### 2. VAE 这一行为什么特殊

第 3 步把 VAE 留在“**由 ComfyUI 决定**”，也就是信任 VAE 的 `working_dtypes`：有 fp16 就 fp16，否则 bf16。

Flux 的 `ae.safetensors` 是 `[bf16, fp32]`，故意排除了 fp16。对它强行“**半精度 (FP16)**”会绕过列表：5 位指数装不下激活值，溢出成 NaN，出图全黑。现场验证：只把这一行从“半精度 (FP16)”改回“由 ComfyUI 决定”，出图立刻恢复，其它设置都没动。

对于列表里有 fp16 的 VAE（例如 `[bf16, fp16, fp32]`），Auto 会用 fp16。请在目标分辨率测试：`working_dtypes` 是架构级的，不是分辨率级的。fp16 和 bf16 都是 2 字节，不增加显存；gfx906 上 fp16 更快。

`ZLUDA_VAE_DTYPE=comfy`（或 `ZLUDA_ALLOW_BF16=1`，两边一起改）会把 VAE 交回 ComfyUI 原顺序，`[bf16, fp16, ...]` 的 VAE 会回到 bf16。

### 3. 注意力后端性能：Split vs SubQuad

SubQuad 慢的原因：它内部的 `kv_chunk_size` 默认是 `√T`。1024×1024 出图时自注意力是 T=4096，于是它把一次大矩阵乘切成 **64 个小块**，每块都要单独做 baddbmm + softmax；Split 只做一次大 baddbmm。

| 实现                       | 耗时     | 相对  |
| -------------------------- | -------- | ----- |
| Split 式（1 次大 baddbmm） | 23.34 ms | 1.00× |
| SubQuad 式（64 个小块）    | 63.22 ms | 2.71× |

（batch×heads=16、T=4096、D=128、fp16。这是简化复现；真实实现每块还要做掩码和在线 softmax 重缩放，所以实际差距更大。）

用日志确认当前是哪一个：

- `Using sub quadratic optimization for attention` → SubQuad（慢）
- `Using split optimization for attention` → Split
- `Using pytorch attention` → PyTorch SDP

### 4. 量化模型与精度标志的关系

| 类型                        | 权重存储        | 权重显存（vs fp16）     | 计算 / 额外显存                    | 建议                                         |
| --------------------------- | --------------- | ----------------------- | ---------------------------------- | -------------------------------------------- |
| GGUF Q4 / Q5 / Q6 / Q8      | 量化块          | 约 0.3–0.5×             | 每层反量化到 fp16；临时一层缓冲    | 最省，由 ComfyUI 决定（支持 fp16 才钉 FP16） |
| fp8 存储                    | 1 B/参数        | 约 0.5×                 | fp8 或反量化到 fp16；缓冲小        | 好，由 ComfyUI 决定（支持 fp16 才钉 FP16）   |
| `int8_tensorwise`，不缓存   | 1 B/参数        | 约 0.5×                 | 每次调用反量化；临时 fp16 副本；慢 | 没得选再用                                   |
| `int8_tensorwise`，缓存     | 1 B + fp16      | 约 1.5× fp16（3× INT8） | 一次反量化后常驻                   | 16 GB 通常放不下                             |
| `convrot_w4a4`              | 约 4.5 bit/参数 | 约 0.3×                 | comfy_kitchen kernel，不反量化     | INT8 系里最好                                |
| torchao / bitsandbytes INT8 | 各自格式        | 取决于实现              | 自己的 kernel                      | 一般不受影响                                 |
| fp16                        | 2 B/参数        | 1×                      | 原生                               | 基准                                         |
| fp32                        | 4 B/参数        | 2×                      | 原生                               | 16 GB 上避免                                 |

实测的 6.14 B 模型：INT8 5.72 GB、fp16 11.44 GB、fp32 22.88 GB；`int8_tensorwise` 开缓存是 5.72 + 11.44 = 17.16 GB，约 1.5× fp16 / 3× INT8，所以缓存默认关闭（`ZLUDA_INT8_DEQUANT_CACHE=1` 只在显存有余时用）。

- GGUF 的机制：`cast_bias_weight()` 里计算 dtype 取输入激活的 dtype（也就是模型推理 dtype），再交给 `dequantize_tensor()` 反量化；所以“由 ComfyUI 决定”/半精度 (FP16) 会让量化模型反量化到 fp16 而不是 bf16。权重仍以量化形式保存，不会变糊也不会变慢。
- 要单独覆盖某个 GGUF 模型，用 `UnetLoaderGGUFAdvanced` 节点的 `dequant_dtype` / `patch_dtype` 下拉（`default / target / float32 / float16 / bfloat16`），不影响全局。
- fp8 存储的模型属于数值路径变化，建议同一工作流跑两遍对比再定。
- `comfy_kitchen int8_tensorwise` 就是 `LINEAR` 单元处理的那一类；torchao / bitsandbytes INT8 用自己的 kernel，不受 `--fp16-unet` 影响。

## 验证清单

### 1. 启动日志

应出现下面这几行（`<TEMP>` 指你的临时文件夹，通常是 `%LOCALAPPDATA%\Temp`）：

```
[ZLUDA] switches: anima_extra_blocks=True | reserve_vram=1.5 | fp32_margin=0.9 | force_fp16=False | split_attention=False | allow_bf16=False | unet_dtype=auto | vae_dtype=auto | arg_overrides=True | onnx_cuda=False | int8_cache=False | topk_cpu=False | stft_cpu=False | no_jitscript=False
[ZLUDA] Preload: zluda_dir=<TEMP>\<GUID> | loaded=5/5
[ZLUDA] OK torch=2.7.1+cu118 | Tensile=C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library | cudnn=False | sdp=math | sdpa=math-only | linear-patch[F.linear=_linear]
[ZLUDA] ops: default
[ZLUDA] hooks: comfy.cli_args, comfy.model_management, comfy.model_detection, comfy.supported_models, comfy.ops, onnxruntime
[ZLUDA] args: disable_pinned_memory=True, disable_async_offload=True, disable_cuda_malloc=True, reserve_vram=1.5
[ZLUDA] model_management: unet=policy | vae=fp16-priority | flags=none
[ZLUDA] comfy_ops: math-only
[ZLUDA] supported_models: LTXAV memory_usage_factor patched
[ZLUDA] model_detection: extra-block tunes load whole
[ZLUDA] onnxruntime: CUDA/TensorRT providers filtered
```

`model_management:` 还会在每次加载模型、每次加载 VAE 时各打印一行决策记录。模型和 VAE 都声明
`[bfloat16, float16, float32]` 时（Anima + Wan）：

```
[ZLUDA] model_management: unet float16 (policy: model lists fp16; comfy would pick bfloat16)
[ZLUDA] model_management: vae float16 (policy: fp16 in [bfloat16, float16, float32]; comfy would pick bfloat16)
```

`[bfloat16, float32]` 模型会打印决策用的那对数，例如
`unet bfloat16 (policy: fp32 29.8 GiB > 10.6 GiB)` 或
`unet float32 (policy: fp32 7.5 GiB <= 10.6 GiB)`（所需权重 vs 留余量后的预算）。
调用方没有报告单一存储 dtype、或者报的是量化 dtype 时，原因前面会标出来：
`unet bfloat16 (policy: weights uint8, fp32 22.4 GiB > 10.6 GiB)`。
启动器把 U-Net 行钉成 FP16 时，这一行会变成 `... (comfy order: --fp16-unet)`：显式标志优先，
策略不参与。VAE 同理（`--fp16-vae`、`no fp16 in [...]`、`list [float32]`）。
在启动器里跑时，钉住的精度行会替换掉 `flags=none`，例如 `flags=fp16_unet,fp16_text_enc`。

manual cast 被改判时，每次加载模型会另打一行：

```
[ZLUDA] model_management: manual cast float16 (policy: upstream bf16, weights float8_e4m3fn)
```

那里的 `weights unknown` 表示加载器没有报告存储 dtype（量化检查点）；`weights float8_e4m3fn` /
`weights uint8` 则是它报告的存储类型。

启动行和每次决策行都会追加写入整合包根目录下的 `_zluda_status.txt`（以及临时文件夹里的 `zluda_status.txt`）。

> **输出约定：** 状态行一律是纯英文——方便 grep、不受控制台代码页影响，本文档也是逐字引用的。
> 需要人来处理的提示会在**下一行**单独输出 `[ZLUDA] 说明: ...` 中文，而不是拼在状态行后面。
> 中文行在个别控制台里可能显示成乱码，但英文行不受影响；`_zluda_status.txt` 是 UTF-8，
> 永远是最可靠的查看位置。决策行（`unet ...` / `vae ...`）是记录而不是告警，不会跟 `说明` 行。

`switches:` 最先打印，此时补丁还没开始干活。`ops:` 紧跟 torch 行；`hooks:` 在它后面，
在第一次导入 `comfy.*` 之前。`args:` 在 `comfy.cli_args` 解析时出现；
`model_management:`、`comfy_ops:`、`supported_models:` 在启动过程中出现（之后每次加载
模型 / VAE，`model_management:` 还会再出现一行决策记录）；
`model_detection:` 在模块被补时出现，之后只有真的检测到层数不同的检查点才再出现；
`onnxruntime:` 只有真的有人导入 ONNX Runtime 时才出现（即 RMBG / controlnet_aux /
Impact 节点运行的时候，不是启动时）。`ops: default` 表示没有开启任何可选算子兜底，
这是预期状态；走回退路径时这一行会在 `Late torch=` 之后照样输出（`OPS` 在回退路径下
也补上了）。

**torch 行看起来健康并不代表全对**：`comfy.ops` 那一半有自己的一行 `comfy_ops:`，
它可以是 `FAILED` 而 torch 行仍然显示 `sdpa=math-only`。一定要把整个状态文件 grep
一遍 `FAILED`。

如果日志里出现下面这些行，说明补丁没有完全生效（`failed=` 后面最多列 3 个失败项）。
这些行后面都会紧跟一行 `说明: ...` 中文提示；中文显示成乱码时看英文行即可。

| 日志行                                                           | 含义                                                                                                                                                          | 处理                                                                                                                                                                       |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[ZLUDA] Preload: No ZLUDA dir found (nvcuda.dll not preloaded)` | 没找到含 ZLUDA DLL 的目录，`nvcuda.dll` 未预加载                                                                                                              | 先让启动器正常跑一次 ComfyUI（它会把 DLL 解到 `%TEMP%\<GUID>`）；或把 `.launcher\zluda.<hash>.extpack` 用 7-Zip 解压到整合包根目录的 `zluda\`                              |
| `[ZLUDA] Preload: zluda_dir=... \| loaded=3/5 \| failed=...`     | 部分 DLL 加载失败                                                                                                                                             | 看 `failed=` 列出的项：`xxx.dll:missing` 是文件缺失；`WinError 193` 通常是拿到了占位文件或 HTML，而不是真正的 DLL                                                          |
| `[ZLUDA] Direct import failed: ...`                              | 补丁在导入 torch 时抛了异常                                                                                                                                   | 后面应紧跟 `Fallback hook installed`；若没有，说明连回退钩子都没装上                                                                                                       |
| `[ZLUDA] Fallback hook installed`                                | 已安装导入钩子，torch 加载完成后会补打补丁                                                                                                                    | 下一次 `import torch` 之后应出现 `Late torch=...`                                                                                                                          |
| `[ZLUDA] Late torch=...`                                         | 回退路径补打补丁成功                                                                                                                                          | 确认行内 `Tensile=`、`cudnn=False`、`sdp=math`、`linear-patch[F.linear=_linear]` 都在                                                                                      |
| 状态行里的 `sdpa=missing`                                        | 这个 torch 没有 `F.scaled_dot_product_attention`，math-only 没包上，注意力会退回会崩的 FLASH / cuDNN 优先级列表                                               | 确认 torch 版本（需要 2.0 以上）；这一项出现在 `OK torch=` 状态行里，不在单独一行                                                                                          |
| `[ZLUDA] comfy_ops: FAILED ...`                                  | `comfy.ops` 的 math-only 包装失败，注意力会退回会崩的优先级列表                                                                                               | 确认 `comfy.ops` 是否被别的插件提前导入，或它本身没有 `scaled_dot_product_attention`（comfy 太旧）；这一行出现时 torch 行仍可能显示 `sdpa=math-only`，一定要 grep `FAILED` |
| `[ZLUDA] <key>: FAILED ...`                                      | 某个模块的导入钩子执行失败，该模块的补丁没生效。`<key>` 是 `args`、`model_management`、`model_detection`、`supported_models`、`comfy_ops`、`onnxruntime` 之一 | 看行里的 key；`comfy_ops` 之外的模块失败通常只影响对应功能                                                                                                                 |
| `[ZLUDA] Patch FAILED: ...`                                      | `main.py` 顶部的引导代码本身出错                                                                                                                              | 检查引导代码是否插在 `main.py` 第一行之前、`zluda_patch.py` 是否在 `ComfyUI\` 下                                                                                           |

### 2. 一键自检

最省事的是仓库里附带的离线脚本。在整合包根目录运行：

```bat
python zluda_dtype_check.py
```

期望输出 `39 passed, 0 failed`。它会读取 `ComfyUI\zluda_patch.py`，但**不执行**末尾的引导代码
（不预加载 DLL、不导入 torch、不写状态文件），用一个假的 `comfy.model_management` 和假的 torch
驱动 `_apply_model_management()`，逐条断言每个策略分支返回的 dtype 以及它打印的状态文本：UNet
六条、VAE 四条、manual cast 修正、两个分侧开关、别名，以及 fail-open 路径。把另一份补丁的路径作为
第一个参数传进去，就可以检查别处的副本。

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

另外五条检查覆盖模块级单元，都不需要启动界面：

```bat
python -c "import zluda_patch, comfy.ops; print('sdpa:', comfy.ops.scaled_dot_product_attention.__name__)"
```

期望 `sdpa: _math_only`。

```bat
python -c "import zluda_patch, comfy.options; comfy.options.enable_args_parsing(); import comfy.cli_args as c; print('args:', c.args.disable_pinned_memory, c.args.disable_async_offload, c.args.disable_cuda_malloc, c.args.reserve_vram)"
```

期望 `args: True True True 1.5`。

```bat
python -c "import zluda_patch, torch, comfy.model_management as mm; dev = mm.get_torch_device(); print('vae:', mm.vae_dtype(dev, [torch.bfloat16, torch.float32])); print('unet(fp16,bf16,fp32):', mm.unet_dtype(dev, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]))"
```

期望 `vae: torch.bfloat16`、`unet(fp16,bf16,fp32): torch.float16`——UNet 策略在模型支持 fp16 时选 fp16；VAE 这一项用 `[bf16, fp32]` 检查，所以按列表返回 bf16（列表里有 fp16 的 VAE 会返回 fp16）。两次调用还会在 stderr 各打一行决策记录，例如 `[ZLUDA] model_management: unet float16 (policy: model lists fp16; comfy would pick bfloat16)`。

```bat
python -c "import zluda_patch, torch, comfy.model_management as mm; dev = mm.get_torch_device(); print('small:', mm.unet_dtype(dev, 100_000_000, supported_dtypes=[torch.bfloat16, torch.float32])); print('large:', mm.unet_dtype(dev, 8_000_000_000, supported_dtypes=[torch.bfloat16, torch.float32]))"
```

期望 `small: torch.float32`、`large: torch.bfloat16`——这条命令才会走到预算分支（上一条的 `model_params` 默认 0，永远进不去）。stderr 会把决策用的数字写出来，例如 16 GB 卡 + `--reserve-vram 1.5` 下的 `fp32 0.4 GiB <= 10.6 GiB` 和 `fp32 29.8 GiB > 10.6 GiB`。

```bat
python -c "import zluda_patch, onnxruntime as ort; print('ort:', ort.get_available_providers())"
```

期望 `ort: ['CPUExecutionProvider']`。如果仍然列出 `CUDAExecutionProvider`，说明钩子没跑起来：
命令里第一个导入的必须是 `zluda_patch`。

### 3. ComfyUI 启动日志里的对照项

| 行                                                                                         | 期望值                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pytorch version:`                                                                         | `2.7.1+cu118`                                                                                                                                                           |
| `Total VRAM ...`                                                                           | 测试所用显卡为 `16368 MB`                                                                                                                                               |
| `VAE load device: ..., offload device: ..., dtype:`                                        | `cuda:0` / `cpu` / VAE 的 dtype。VAE 行设成**由 ComfyUI 决定**时跟随 VAE 列表：Flux `[bf16, fp32]` → **bf16**；列表里有 fp16 的 VAE → **fp16**。出现 `float32` 说明 VAE 只允许 fp32 或设了 `--fp32-vae`。紧邻的 `[ZLUDA] model_management: vae ...` 决策行会写出同一个 dtype 和选择原因 |
| `model weight dtype ...`                                                                   | 与紧邻的 `[ZLUDA] model_management: unet ...` 决策行一致；那一行还写明是补丁策略还是 ComfyUI 原顺序（`policy:` / `comfy order:`）。`[bfloat16, float32]` 模型只有在决策行给出 `fp32 ... <= ...` 的预算通过时才是 `torch.float32` |
| `Using ... attention`                                                                      | `split` 或 `pytorch`，**不要** `sub quadratic`                                                                                                                          |
| `Enabled pinned memory ...`                                                                | **不应出现**；出现说明 `CLI_ARGS` 单元没生效                                                                                                                            |
| `[ZLUDA] onnxruntime: forcing CPUExecutionProvider`                                        | 跑 ONNX 节点时出现（后面紧跟一行 `说明: ...` 中文）；后面不应再有 `Failed to load ...onnxruntime_providers_cuda.dll`                                                    |
| `Unsupported Pytorch detected. DynamicVRAM support requires Pytorch version 2.8 or later.` | **正常提示**，ComfyUI 会退回旧版 ModelPatcher，不是错误                                                                                                                 |

> 读 `VAE load device` 这行时只看**第一个字段**：它是 VAE 的运行设备，出现 `cpu` 才代表“使用 CPU 运行 VAE”被打开；`offload device: cpu` 是默认的备用位置（只有加 `--gpu-only` 才会变成 `cuda:0`），与 CPU VAE 无关。

## 故障排查

| 症状（日志关键字）                                                                                            | 真正原因                                                                                                                                                             | 处理                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError: Model "..." is corrupt or invalid... Please download the model again`                            | **文件不存在或路径错**，不是文件损坏。`comfy/utils.py`（测试版本的 122-146 行）把 `safe_open()` 的**任何**异常（含 FileNotFoundError）都报成“损坏”                   | 检查路径与文件名；不要盲目重新下载                                                                                                                   |
| `rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906` + `Fatal Python error: Aborted` | `ROCBLAS_TENSILE_LIBPATH` 指向的目录没有 gfx906 内核（常见于启动器的 `%TEMP%\<GUID>` 目录被清理或轮换）                                                              | `PRELOAD` 会把它指到 HIP SDK 的 `bin\rocblas\library`                                                                                                |
| `CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmulAlgoGetHeuristic(...)`                    | 带 bias 的线性层 / `addmm` 走了 cuBLASLt 融合路径                                                                                                                    | `LINEAR`；先用[一键自检](#2-一键自检)确认 `F.linear` 已是 `_linear`                                                                                  |
| `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`                                                                    | cuDNN 未关闭，**或者** `comfy.ops` 的注意力优先级把 FLASH / cuDNN 又顶回前面（`SDPA` 单元）                                                                          | `TORCH`；若仍出现，确认状态行有 `sdpa=math-only` 并 grep 状态文件里的 `FAILED`                                                                       |
| 黑图、没有报错、日志里有 `unet unexpected: ['blocks.N...']`                                                   | 检查点的层数比 ComfyUI 为它建的层数多，多出来的层被静默丢掉，采样出的 latent 是 NaN                                                                                  | `MODEL_DETECTION` 会数检查点里的层数，`model_detection: ... num_blocks 28 -> 40` 就是它生效的标志。没有这个补丁的 ComfyUI 请升级（上游 `725e6ec60`） |
| `RuntimeError: Found no NVIDIA driver on your system`                                                         | ZLUDA DLL **没有在 `import torch` 之前**被预加载（例如手工提前 import torch，或启动器注入失效）                                                                      | `PRELOAD`；确认启动日志有 `loaded=5/5`                                                                                                               |
| `ERROR: clip input is invalid: None`                                                                          | 用 `Load Checkpoint` 加载了**不含文本编码器**的模型（如 anima-base）                                                                                                 | 改用 `Load Diffusion Model` + `Load CLIP` + `Load VAE`，或直接用官方模板                                                                             |
| 显存不足 / OOM                                                                                                | 16 GB 显存跑大模型                                                                                                                                                   | 降分辨率、`--fp8_e4m3fn-unet`，或把 `--reserve-vram` 从 1.5 提到 2.5。系统内存只有 16 GB 时**不要**用 `--lowvram` / `--novram`，会被换页拖死         |
| 下一次生图打印 `0.00 MB usable`、三个模型全部 `loaded partially ... offloaded`                                | 上一次生图撞上 OOM，失败的那次分配之后驱动的剩余显存读数一直停在 0。实测：把所有东西都释放、再调 `empty_cache`，`mem_get_info` 仍返回 0，而此时 12 GB 的分配照样成功 | **重启 ComfyUI。** 进程内没有任何办法恢复这个读数；之后每次生图都从系统内存流式跑，会明显变慢                                                        |
| 在启动器里点“安装 PyTorch / 安装依赖”后一切复原                                                               | 启动器按内置清单重装（它给 ComfyUI 核心钉的 Python 是 3.11.6，cu118 最高只到 2.3.1，在 3.13 上装不上）                                                               | 不要在启动器里点“安装依赖 / 切换 Torch 版本 / 重装环境”                                                                                              |
| `CUBLAS_STATUS_NOT_SUPPORTED ... abType 3 cType 10 computeType 72`                                            | 跑了 `int8_tensorwise` 模型，走到了 `torch._int_mm`（`LINEAR` 单元）                                                                                                 | 改用 `convrot_w4a4` / `fp8` / `bf16` 模型，或接受每次调用反量化                                                                                      |
| 出图全黑、状态行健康、`_math_only` 已生效                                                                     | VAE 行被钉成 FP16，而该 VAE 的 `working_dtypes` 不含 fp16（Flux），解码器溢出成 NaN                                                                                  | 把启动器“VAE 模型精度”改回“由 ComfyUI 决定”                                                                                                          |
| `[ZLUDA] model_management: unet ...（comfy order: --fp16-unet）`                                              | 启动器的精度行被钉住（启动行的 `flags=` 会列出来），显式标志优先，策略不参与——这是设计行为                                                                          | 想让策略自己选，就把那一行改回“由 ComfyUI 决定”                                                                                                      |
| `[ZLUDA] model_management: unet bfloat16 (policy: fp32 ... > ...)`                                            | `[bfloat16, float32]` 模型在权重预算（扣掉 `--reserve-vram` 和 10% 余量）里放不下 fp32，补丁为了常驻选择 bf16（fp16 的 0.55 倍速）                                   | 接受 bf16、调低 `ZLUDA_RESERVE_VRAM`，或强制 `--fp32-unet` 并观察是否部分卸载；列表里有 fp16 的模型不会走这条分支                                     |
| RMBG / controlnet_aux / Impact 报 onnxruntime provider 错误                                                   | CUDA 12 的 `onnxruntime-gpu` provider 在 ZLUDA 下（`ONNX` 单元）                                                                                                     | 确认状态行有 `onnxruntime: CUDA/TensorRT providers filtered`                                                                                         |

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
3. **精度设置：** 四行都可以留“由 ComfyUI 决定”：补丁会在模型支持 fp16 时选 fp16，放得下 fp32 时选 fp32，否则 bf16；VAE 跟随自己的 `working_dtypes`（有 fp16 就 fp16，否则 bf16；Flux 保持 bf16）。对确实支持 fp16 的模型，把模型行钉成 **半精度 (FP16)** 仍是最快选项，但那会让策略让位（`flags=fp16_unet`，决策行变成 `comfy order:`）；想让自动规则（含 fp32 预算）生效，就把三行模型精度留在“由 ComfyUI 决定”。`ZLUDA_UNET_DTYPE=comfy` / `ZLUDA_VAE_DTYPE=comfy` 可以只把一侧交回去，另一侧不受影响。
4. **性能：** 注意力后端与精度都会显著影响速度，见[精度与性能](#精度与性能实测)；此外可尝试降低分辨率。
5. **补丁的扩展方向：** 若以后在别的算子上再遇到 `CUBLAS_STATUS_NOT_SUPPORTED`，可在对应的 PART 2 单元里继续绕（`TORCH_STEPS` 管 torch 级补丁，`LINEAR` 管矩阵乘形状的；例如把 `F.conv2d` 换成 `unfold + matmul` 的等价实现，已实测可用）。
6. **防止启动器覆盖：** 启动器更新 ComfyUI 会重置 `main.py`，需要重新插入引导代码；`zluda_patch.py` 也可能被清掉，建议留备份。
7. **诊断资料：** 启动器（**疑难解答**页面）生成的 `Diagnostics-*.log` 是环境快照加控制台缓冲区，**会包含多次运行的历史输出**。判断某行日志是否来自新进程，要看 `_zluda_status.txt` 的时间戳。
8. **启动器只负责解包 ZLUDA 的 DLL。** 补丁自己会预加载它们，只要在它之前没有任何东西
   导入过 torch，这就够了。补丁以前恰好违反了这条规则：`comfy_kitchen` 的类名在导入时就被解析，
   而 `comfy_kitchen` 会导入 torch，于是那个 torch 绑定不到 CUDA 驱动，整个进程报
   `Found no NVIDIA driver on your system`（实测：用预加载的 DLL 调 `cuInit(0)` 返回 0、
   `cuDeviceGetCount` 报 1 个设备，而同一进程里 `torch.cuda.is_available()` 是 `False`）。
   现在这些名字改成首次使用时才解析（`LINEAR` 单元），所以裸跑 `python main.py`、IDE 调试和
   [一键自检](#2-一键自检)都能看到显卡。
9. **INT8 tensorwise 模型能用但不快。** `LINEAR` 单元的反量化是每次调用 24.4 ms，而原生 fp16 GEMM
   是 1.34 ms。这张卡上优先用 `convrot_w4a4`、`fp8` 或 `bf16` 模型。
10. **ONNX Runtime 一律走 CPU。** 这是故意的（`ONNX` 单元）：装的是 CUDA 12 构建的
   `onnxruntime-gpu`。如果以后真的换上匹配的构建并想试验，用 `ZLUDA_KEEP_ONNX_CUDA=1`
   保持 provider 列表原样。
11. **KL 类 VAE 的显存估算在 ZLUDA 下只有真实值的 1/2.73。** `comfy/sd.py` 只在 `is_amd()`
   为真时才把 VAE 显存估算乘 `VAE_KL_MEM_RATIO = 2.73`；这个判断读的是 `torch.version.hip`，
   而 ZLUDA 下它是 `None`，所以估算停在 1.0，AutoencoderKL 解码只按真实峰值的约 37% 规划。
   补丁设的 `--reserve-vram 1.5` 就是拿来吸收这个差额的：低于约 1.2 GB，分块解码的回退会回来。
   这是 ComfyUI 自身的行为，补丁从外面修不了。
12. **manual cast 修正盖不到 bf16 权重。** bf16 权重在上游返回 `None`（“不需要转换”），随后
   `SeedVR2.set_inference_dtype()` 自己把 manual cast 设成 bf16，那个模型仍然按 bf16（0.55 倍）
   计算。包装只改“上游已经定成 bf16”的情况（fp8 / 未知存储）。手头没有 SeedVR2 检查点可测，
   所以这条路径保持原样。
13. **模型的 dtype 列表优先于文件格式。** GGUF 等量化构建的 dtype 仍然来自
   `supported_inference_dtypes`：`ZImage` 是 `[bfloat16, float32]`，所以
   `z-image-turbo-Q6_K.gguf` 一样跑 bf16（0.55 倍），日志同样是
   `fp32 22.4 GiB > 10.6 GiB`。只有声明了 fp16 的模型（Anima、Wan、SeedVR2 等）走快路径。

## 环境变量开关

首次发布之后新增的行为都能在不改补丁逻辑的前提下关掉或调整。**它们全部集中在补丁的 PART 0**，每个开关一行、默认值写在旁边，每次启动会以一行 `switches: ...` 状态行打印当前状态：

| 变量                         | 默认   | 作用                                                                                                     |
| ---------------------------- | ------ | -------------------------------------------------------------------------------------------------------- |
| `ZLUDA_NO_ARG_OVERRIDES=1`   | 关     | 不再覆盖启动参数，交回启动器                                                                             |
| `ZLUDA_RESERVE_VRAM=<GB>`    | `1.5`  | 改预留显存；显式 `--reserve-vram` 仍然优先                                                               |
| `ZLUDA_FORCE_FP16=1`         | 关     | 强制 UNet 和文本编码器走 fp16（也会覆盖 fp8 存储模型的标志）                                             |
| `ZLUDA_SPLIT_ATTENTION=1`    | 关     | 用 split 注意力替代 pytorch/math SDP                                                                     |
| `ZLUDA_ALLOW_BF16=1`         | 关     | 粗粒度别名：两侧 dtype 策略都回 ComfyUI 原顺序（状态：`unet=comfy-order \| vae=comfy-order`）；显式写的一侧优先 |
| `ZLUDA_UNET_DTYPE=comfy`     | `auto` | 只把 UNet 的 dtype 策略（含 manual cast 修正）交回 ComfyUI 原顺序（状态：`unet=comfy-order`）             |
| `ZLUDA_VAE_DTYPE=comfy`      | `auto` | 只把 VAE 的列表偏好交回 ComfyUI 原顺序（状态：`vae=comfy-order`）                                         |
| `ZLUDA_FP32_MARGIN=<0..1>`   | `0.9`  | fp32 权重预算的安全余量；越小越保守，超出 (0, 1] 时回退到 0.9                                             |
| `ZLUDA_KEEP_ONNX_CUDA=1`     | 关     | 不动 ONNX Runtime 的 provider 列表                                                                       |
| `ZLUDA_INT8_DEQUANT_CACHE=1` | 关     | 缓存反量化后的 INT8 权重（原始 INT8 + fp16，约 1.5× fp16 / 3× INT8，快 12 倍）                           |
| `ZLUDA_TOPK_CPU=1`           | 关     | 让 `torch.topk` 走 CPU                                                                                   |
| `ZLUDA_STFT_CPU=1`           | 关     | 让 `torch.stft` / `torch.istft` 走 CPU（音频模型）                                                       |
| `ZLUDA_ANIMA_EXTRA_BLOCKS=0` | **开** | 停止从检查点数 Anima / Cosmos-Predict2 的层数（`MODEL_DETECTION`）；因为修复默认开启，所以这个开关是反的 |
| `ZLUDA_NO_JITSCRIPT=1`       | 关     | 把 `torch.jit.script` 变成空操作                                                                         |

> `ZLUDA_SPLIT_ATTENTION=1` 是**强制** split；**关掉并不强制 math**——它只是不碰启动器的
> “**Cross-Attention 优化方案**”。ComfyUI 日志里的 `Using split optimization for attention`
> 反映的是启动器的选择，不是这个开关。

## 仓库结构

```
.
├── ComfyUI/
│   └── zluda_patch.py         # 兼容补丁（复制到整合包根目录下的 ComfyUI\）
├── zluda_dtype_check.py       # 精度策略的离线自检脚本（复制到整合包根目录）
├── LICENSE
├── README.md
└── README.zh-CN.md
```

部署后的整合包中涉及的文件（除特别注明外，均相对于整合包根目录）：

| 路径                                                     | 说明                                                                                              |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `python\python.exe`                                      | Python 3.13.11                                                                                    |
| `ComfyUI\main.py`                                        | **在文件最顶部插入了引导代码**                                                                    |
| `ComfyUI\zluda_patch.py`                                 | 本方案新增的补丁（来自本仓库）                                                                    |
| `_zluda_status.txt`                                      | 补丁的状态记录：每个事件一行（启动行，外加每次 dtype 决策一行）                                   |
| `.launcher\zluda.<hash>.extpack`                         | ZLUDA 的 5 个 DLL（可用 7-Zip 解压）                                                              |
| `.launcher\tensile-gfx906.<hash>.extpack`                | gfx906 的 rocBLAS Tensile 内核（备用来源）                                                        |
| `.launcher\preference.json`                              | 启动器偏好（含代理地址、自定义参数等）                                                            |
| `ComfyUI\blueprints\Text to Image (Anima Base 1.0).json` | Anima 官方模板                                                                                    |
| `%HIP_PATH%\bin\rocblas\library`（绝对路径）             | 补丁所指向的 gfx906 Tensile 内核目录（`HIP_PATH` 取不到时回退到 `C:\Program Files\AMD\ROCm\5.7`） |
| `%TEMP%\<GUID>\`（绝对路径，每次运行新建）               | 启动器解压 ZLUDA DLL 的位置，补丁会自动扫描并预加载                                               |

若更换整合包版本或 HIP SDK 版本，请重新核对[实测环境](#实测环境)中的版本匹配关系。

## 许可证

本项目采用 [MIT License](LICENSE)。

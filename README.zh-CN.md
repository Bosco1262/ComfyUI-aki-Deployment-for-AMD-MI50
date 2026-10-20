# 秋叶整合包 ComfyUI + AMD MI50（ZLUDA / gfx906 / Windows）

[English](README.md) | [简体中文](README.zh-CN.md)

在 Windows 上，通过 [ZLUDA](https://github.com/vosen/ZLUDA) 让 **秋叶 ComfyUI 整合包**（使用**绘世启动器**）跑在 **AMD Instinct MI50**（刷入 **Radeon VII PRO vBIOS**）上。

本仓库记录了一套**已完整跑通**的方案：实际执行的命令、一个小型兼容补丁（[`ComfyUI\zluda_patch.py`](ComfyUI/zluda_patch.py)）、每一步为什么需要、验证方法以及故障排查表。所有结论均来自单台机器上的实测（命令行输出、日志、文件哈希），不是猜测。

## 结论速览

要让这套整合包在 ZLUDA 上跑通，需要且只需要三件事：

1. **把 torch 换成 cu118 版本。** ZLUDA 只提供 CUDA 11 的库，用 CUDA 12+ 编译的 PyTorch 一定不行。
2. **放入补丁 `ComfyUI\zluda_patch.py`，并在 `ComfyUI\main.py` 顶部挂一小段引导代码。** 补丁一共做 12 件事：前 5 件缺一不可（预加载 ZLUDA DLL、让 rocBLAS 指向 gfx906 Tensile 内核、关闭 cuDNN、注意力只留 math SDP、绕开 cuBLASLt 的 bias 融合 GEMM），后 7 件负责消掉其余会崩和会变慢的路径（`comfy.ops` 的 SDPA 优先级、INT8 tensorwise 权重、启动参数、bf16、ONNX Runtime、LTXAV 显存估算、可选算子兜底）。详见[补丁原理](#补丁原理)。
   `cudaMallocAsync` 不需要再手动加参数了，补丁自己会关，`--disable-cuda-malloc` 现在可选。
3. **把精度钉成 FP16，但 VAE 那一行除外**：gfx906 上 bf16 只有 fp16 的 0.55 倍速，而把 VAE 钉成 FP16 会溢出成 NaN、解码出全黑图。见[精度与性能](#精度与性能实测)。

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

1. 将本仓库下的所有文件复制到整合包根目录，使补丁最终位于 `ComfyUI\zluda_patch.py`。真正需要的只有 [`ComfyUI\zluda_patch.py`](ComfyUI/zluda_patch.py)，`README*.md` 与 `LICENSE` 不是必需的。
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

### 第 3 步：启动参数

1. 正常启动绘世启动器，在“**设置**”中把“**配置模型**”切换至“**专家**”。
2. 在“**高级选项 → 其他设置 → 自定义参数**”中添加：

```
--disable-cuda-malloc
```

原因：`ComfyUI\cuda_malloc.py` 会在非 N 卡上自动开启 `cudaMallocAsync`（`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`），而 ZLUDA 未必实现该分配器。`--disable-cuda-malloc`（定义在 `comfy/cli_args.py`）会强制关闭它。

有了补丁 8，这个参数已经**可选**：补丁自己会设 `args.disable_cuda_malloc`、
`args.disable_pinned_memory`、`args.disable_async_offload` 和 `args.reserve_vram`，
所以即使 `preference.json` 被重置、或从别的入口启动，配置依然生效。保留这个参数无害。

**不要使用**的参数：`--use-sage-attention`（包已卸载）、`--use-flash-attention`（未安装）、`--enable-triton-backend`（ZLUDA 下无用）、`--fast` / `--fast cublas_ops`（这条路会直接用 cuBLASLt，必崩）、`--cuda-malloc`。

精度设置见[精度与性能](#精度与性能实测)：一句话，**三行模型精度钉 FP16（现在留“由 ComfyUI 决定”也安全），VAE 那一行保持“由 ComfyUI 决定”，任何地方都不要出现 BF16**。

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

[`ComfyUI\zluda_patch.py`](ComfyUI/zluda_patch.py) 一共做 12 件事。第 1–5 件缺一不可；第 6–12 件负责消掉剩下的崩溃点和慢路径。每条都按同一套写法：**现象**（用户看到什么）、**原因**（为什么会这样）、**修法**（补丁怎么做）、**代价**（有没有副作用）。

### 1. 预加载 ZLUDA 的 DLL

- **现象**：`RuntimeError: Found no NVIDIA driver on your system`。

- **原因**：ZLUDA 的 `nvcuda.dll` 必须**先于 torch 的 CUDA 初始化**被加载。启动器本来自己在做这件事（每次运行把 DLL 解压到 `%TEMP%\<GUID>`），但它的注入链路在我们的场景下不稳定。

- **修法**：补丁自己扫描 `%TEMP%`、`%TMP%`、`%LOCALAPPDATA%\Temp`（以及整合包根目录下的 `zluda\`）找到那个目录，把 ZLUDA 目录与 `%HIP_PATH%\bin` 加进 `PATH` / `os.add_dll_directory`，再预加载 `nvcuda.dll`、`nvml.dll`、`cublas64_11.dll`、`cusparse64_11.dll`、`nvrtc64_112_0.dll`。

- **代价**：没有。只是启动时多扫几个临时目录。

### 2. 设置 `ROCBLAS_TENSILE_LIBPATH`

- **现象**：`rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906`，随后 `Fatal Python error: Aborted`。

- **原因**：ZLUDA 的 cuBLAS 把矩阵乘转给 **rocBLAS**，rocBLAS 需要**对应架构（gfx906）的 Tensile 内核**；启动器把该变量指向 `%TEMP%` 下某个没有这些内核的目录。

- **修法**：补丁把 `ROCBLAS_TENSILE_LIBPATH` 设为 `<HIP SDK>\bin\rocblas\library`，优先读环境变量 `HIP_PATH`，取不到时回退到实测环境的 `C:\Program Files\AMD\ROCm\5.7`。

- **代价**：没有。

### 3. 关掉 cuDNN

- **现象**：`cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`，VAE 的卷积会最先中招。

- **原因**：ZLUDA 没有 cuDNN。

- **修法**：`torch.backends.cudnn.enabled = False`；关闭后卷积走可用的回退路径（已实测）。

- **代价**：没有。

### 4. 注意力只保留 math SDP

- **现象**：注意力会落到 ZLUDA 上不存在的 flash / mem-efficient 内核（和第 6 条是同一个故障的两层）。

- **原因**：ZLUDA 不支持 flash attention 的内核；绘世启动器自己的 ZLUDA 兼容补丁也是这么做的。

- **修法**：关闭 flash 与 mem-efficient，只保留 math SDP 后端。

- **代价**：没有。FLASH / cuDNN 本来就不存在，调用一直回退到 math，实测数字见第 6 条。

### 5. 拆开带 bias 的线性层

- **现象**：`CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul ...`。

- **原因**：PyTorch 对 `F.linear(x, w, b)` / `addmm` 使用**只有 cuBLASLt 实现的 bias 融合 GEMM**（`gemm_and_bias`）。ZLUDA **完全没有 cuBLASLt 实现**（5 个 DLL 里没有任何 `cublasLt*` 导出），于是 torch 会去用 `torch\lib\cublasLt64_11.dll`（NVIDIA 正版，544 MB），必然返回 `CUBLAS_STATUS_NOT_SUPPORTED`。实测：无 bias 的 `mm` / `bmm` / `baddbmm` 全部正常，**只有带 bias 的 linear / addmm 会崩**，与精度（fp16 / bf16 / fp32）无关。

- **修法**：把“带 bias 的线性层”拆成 `matmul` + 加法，并对 `torch.addmm` 做同样处理。

- **代价**：没有。

### 6. 所有注意力都走 math SDP

- **现象**：要么直接崩：

  ```
  RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
  RuntimeError: cutlassF: no kernel found to launch!
  ```

  要么更糟：工作流跑完不报错，但保存出来是**纯黑图**，`SaveImage` 打印
  `RuntimeWarning: invalid value encountered in cast`（说明解码结果里是 NaN）。

- **原因**：`comfy/ops.py` 自己装了一个 SDPA 包装器，用
  `sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True)`，列表是
  `[FLASH, CUDNN, EFFICIENT, MATH]`。`set_priority=True` 会**无视**第 4 条设的
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

- **修法**：`comfy/ops.py` 在导入时就把 `torch.nn.functional.scaled_dot_product_attention`
  绑成了自己的引用，所以只补 `F` 没用（实测 `comfy.ops uses patched fn -> False`）。补丁
  同时包住 `F` 和 `comfy.ops` 里的那份引用，让每次调用都在 `sdpa_kernel([MATH])` 里执行；
  并把 `can_use_flash_attention` / `can_use_cudnn_attention` /
  `can_use_efficient_attention` 三个探测强制返回 `False`（它们在这里会误报 FLASH 可用，
  从而跳过真正能用的 `repeat_kv` 回退）。`comfy.ops` 在引导代码之后才导入，所以用导入
  钩子补。

  现场踩坑：第一版在 `comfy.ops` 自己的 `exec_module` 里又 `import comfy.ops`，
  会抛 `AttributeError: cannot access submodule 'ops' of module 'comfy'`；而**启动状态行
  仍然显示** `sdpa=math-only | ops-hook=installed`（这些字是在装包装器时写进去的），只有
  后面那行才暴露失败。结果 `comfy.ops` 那份没被补上，图一直是黑的。所以**不要只看那一行，
  整个状态文件都要 grep `FAILED`**。

- **代价**：没有。FLASH / cuDNN 本来就不存在，调用一直回退到 math。S=4096、
  batch×heads=16、D=128 时：`attention_pytorch` 39.1 ms、math SDP 39.1 ms、split 26.7 ms。
  这是正确性修复；想要快就换 split（见[注意力后端](#3-注意力后端把默认的-subquad-换成-split)）。

### 7. INT8 tensorwise 回退

- **现象**：

  ```
  RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul ...
  abType 3 cType 10 computeType 72 scaleType 10
  ```

  `abType 3` + `computeType 72` = INT8 × INT8 → INT32 的 GEMM。

- **原因**：`comfy_kitchen` 的 `TensorWiseINT8Layout` 最后走到 `torch._int_mm`，也就是
  cuBLASLt 的 INT8 GEMM，ZLUDA 没有实现。**计算精度不是问题**：`int8_tensorwise` 会设
  `quantize_input: False`，权重以 `QuantizedTensor` 原样交给 `F.linear`，从来没被反量化，
  `--fp16-unet` 救不了。

  看 safetensors 头里的 `comfy_quant` JSON 判断模型是否受影响，而不是文件名：

  | `comfy_quant`                        | 受影响？                        |
  | ------------------------------------ | ------------------------------- |
  | `{"format": "int8_tensorwise", ...}` | **是**，会走 `torch._int_mm`    |
  | `{"format": "convrot_w4a4", ...}`    | 否，用 comfy_kitchen 自己的内核 |
  | 无（普通 fp16 / bf16 / fp8）         | 否                              |

  文件名带 `int8_convrot` 也可能写着 `format: int8_tensorwise`；`convrot` 标记本身不会让它
  绕开 `torch._int_mm`。

- **修法**：`F.linear` / `torch.addmm` 在矩阵乘之前，把 `QuantizedTensor` 权重反量化成
  输入 dtype。在 `z_image_turbo_int8_convrot` 的 11520×3840 qkv 层上实测（m=90，fp16 输入）：

  | 做法                 | 每次调用 |
  | -------------------- | -------- |
  | 不处理               | 崩       |
  | 每次反量化（默认）   | 24.4 ms  |
  | 反量化一次并缓存     | 2.0 ms   |
  | 同尺寸原生 fp16 GEMM | 1.34 ms  |

  缓存**默认关闭**，因为权重占用会翻倍：6.14 B 模型从 5.72 GB INT8 变成 11.44 GB fp16，
  而模型加文本编码器已经占了 16 GB 卡里的 11.26 GB。显存有余时设
  `ZLUDA_INT8_DEQUANT_CACHE=1`。缓存用 `weakref.WeakKeyDictionary` 做键，模型释放时一起
  释放。

- **代价**：能用但**不快**。优先级：真正的 `convrot_w4a4` 模型 > `fp8` 模型（它的 handler
  失败时本来就会退化成反量化）> `bf16`（注意 16 GB 系统内存）。

### 8. 启动参数由补丁接管

- **现象**：`Enabled pinned memory ...` 出现在启动日志里（说明启动参数没生效）。

- **原因**：启动器把选项存在 `.launcher\preference.json`：重置就没了，绕过 `main.py` 的入口更是完全拿不到。

- **修法**：`comfy/cli_args.py` 在导入时解析，之后 `cuda_malloc.py`、`comfy/model_management.py`、`comfy/ldm/modules/attention.py` 才读 `args`，所以补丁钩住这次导入，在对象刚构造完时改一次：

  | 参数                      | 为什么                                                                 |
  | ------------------------- | ---------------------------------------------------------------------- |
  | `--disable-pinned-memory` | Windows 上会锁最多 40% 内存；这台只有 16 GB，pinned 缓冲也是常见崩溃源 |
  | `--disable-async-offload` | torch 2.7 走旧版 ModelPatcher，异步卸载没有收益                        |
  | `--disable-cuda-malloc`   | ZLUDA 没有 cudaMallocAsync；命令行上已经不用再加                       |
  | `--reserve-vram 0.9`      | 给系统和其它程序留出显存                                               |

  启动器**显式**设的 `--reserve-vram` 优先，不会被覆盖。开关：`ZLUDA_NO_ARG_OVERRIDES=1` 关掉整组，`ZLUDA_RESERVE_VRAM=<GB>` 改预留值，`ZLUDA_FORCE_FP16=1`（会顺带覆盖 fp8 模型的标志，所以默认关），`ZLUDA_SPLIT_ATTENTION=1`。

- **代价**：没有。

### 9. 不让它选 bf16

- **现象**：`model weight dtype torch.bfloat16`，模型直接跑慢一半。

- **原因**：ZLUDA 上报算力 (8, 8)，`should_use_bf16()` 只要 `props.major >= 8` 就返回 `True`，根本没问设备。gfx906 实测 fp16 9.54 TFLOPS、bf16 5.26。`supported_inference_dtypes` 里 bf16 排第一的模型（Anima）会直接跑慢一半。

- **修法**：补丁把 `comfy.model_management.should_use_bf16` 换成 `lambda *a, **k: False`；所有调用方用的都是这个属性，包括 `model_management` 内部的裸名调用。**显式**的 `--bf16-unet` 仍然优先：`unet_dtype()` 先检查参数标志，再去看模型支持的 dtype 列表，所以启动器的精度行**不能**设成 BF16（见“启动器计算精度设置怎么设”一节）。开关：`ZLUDA_ALLOW_BF16=1` 恢复原行为。

- **代价**：没有。

### 10. ONNX Runtime 走 CPU

- **现象**：RMBG / controlnet_aux / Impact 报 onnxruntime provider 错误（provider 加载失败或在假驱动里崩）。

- **原因**：`controlnet_aux`（DW Pose、线稿、深度）、RMBG、Impact 的检测器都用 ONNX Runtime 跑小模型，它的默认 provider 列表把 `CUDAExecutionProvider` 排第一。装的是 CUDA 12 构建的 `onnxruntime-gpu`，而 ZLUDA 只冒充 CUDA 11，所以这个 provider 要么加载失败，要么在假驱动里崩。

- **修法**：补丁把 `CUDAExecutionProvider` / `TensorrtExecutionProvider` 从 `get_available_providers()` 里过滤掉（模块级函数和 session 真正读的 pybind 副本都改），并把仍然请求 CUDA 的 session 改成 `CPUExecutionProvider`。开关：`ZLUDA_KEEP_ONNX_CUDA=1` 保持原样。

- **代价**：没有。这些小模型本来就小，CPU 跑感觉不到差别，但能避免崩溃。

### 11. LTXAV 显存估算

- **现象**：LTXAV 在 16 GB 卡上 OOM（ComfyUI 以为它几乎不吃显存）。

- **原因**：上游把 LTXAV 的 `memory_usage_factor` 留在 `0.077`，`model_base.py` 用它乘激活显存估算。父类 `LTXV` 用的是 `(cross_attention_dim / 2048) * 5.5`。

- **修法**：补丁把同样公式用到 LTXAV 上再乘 1.5（4096 → 16.5）。

- **代价**：ComfyUI 会更早 offload，但不会爆显存。

### 12. 可选算子兜底

- **现象**：Zluda 仓库无条件替换 `torch.topk`、`torch.stft` / `torch.istft`、`torch.jit.script` 时，每次 `topk` 都变成 D2H + topk + H2D，而 torchvision 的 NMS 内部就用 `topk`，检测节点会明显变慢。

- **原因**：这套环境里没有证据表明需要这些兜底。

- **修法**：默认关闭，按需单独开：`ZLUDA_TOPK_CPU=1`、`ZLUDA_STFT_CPU=1`（音频模型）、`ZLUDA_NO_JITSCRIPT=1`（带 TorchScript 模型的节点包）。

- **代价**：开了就会慢，见上面每次 `topk` 的往返代价。

第 6、8–11 条靠 `sys.meta_path` 导入钩子生效，因为 `comfy.*` 是在 `main.py` 引导代码之后
才导入的；钩子在模块执行完的瞬间运行，不会重复导入。第 12 条（可选算子兜底）在 torch 加载
完成后补上，两条路径都会补：直接导入成功后补一次，走回退加载器时也补一次。

## 精度与性能（实测）

以下数字均为**测试机实测**（MI50 刷 Radeon VII PRO vBIOS / gfx906 / HIP SDK 5.7.1 / torch 2.7.1+cu118 / ZLUDA）。

### 1. 为什么必须钉 FP16，而不是 BF16

ZLUDA 在这台机器上冒充 NVIDIA 显卡，这会让 ComfyUI 里一条**专门为 gfx906 准备的排除规则失效**：

- ZLUDA 下 `torch.version.hip` 是 `None`、`torch.version.cuda` 是 `11.8`，所以 `is_amd()` 返回 False（它要求 `torch.version.hip` 有值）。
- 于是 `should_use_bf16()` 里“RDNA2 及更老不支持 bf16”那段判断被跳过，而那份架构列表里**明确写着 `gfx906`**。
- 同时 ZLUDA 上报的计算能力是 **(8, 8)**、`torch.cuda.is_bf16_supported()` 返回 True，于是 `should_use_bf16()` 最终返回 True。
- 模型如果把 bf16 排在 `supported_inference_dtypes` 第一位（例如 Anima Base 1.0 是 `[bfloat16, float16, float32]`），`unet_dtype()` 的第一轮循环就直接返回 bf16。

| 精度 | 实测        | 相对 fp16 |
| ---- | ----------- | --------- |
| fp16 | 9.54 TFLOPS | 1.00×     |
| bf16 | 5.26 TFLOPS | 0.55×     |
| fp32 | 9.10 TFLOPS | 0.95×     |

（4096×4096 矩阵乘。）

结论：**钉 FP16**。不要选 FP32——速度和 fp16 基本持平，但显存占用翻倍，16 GB 卡上是负收益。

自检：启动日志里应为 `model weight dtype torch.float16, manual cast: None`；出现 `torch.bfloat16` 就说明还在走慢路径。

### 2. 启动器“计算精度设置”怎么设

启动器“高级选项 → 计算精度设置”有四行加一个开关，建议：

| 面板项                                           | 设为                           | 说明                                      |
| ------------------------------------------------ | ------------------------------ | ----------------------------------------- |
| 通常模型精度                                     | **FP16**                       |                                           |
| 文本编码器 (Text Encoder) 模型精度               | **FP16**                       | 默认本来就是 fp16，设成 FP16 属于保险     |
| U-Net 模型精度                                   | **FP16**                       | 直接对应上面那条 bf16 问题                |
| **VAE 模型精度**                                 | **由 ComfyUI 决定（或 BF16）** | **FP16 会溢出成 NaN、解码出全黑图**，见下 |
| 使用向上采样法提高 XAttn 精度 (upcast-attention) | **保持关**                     | 开了会把注意力升到 fp32，这卡上更慢       |

有了补丁 9，三行模型精度留“由 ComfyUI 决定”也是安全的（不会再自动选 bf16）；保持 FP16
仍然推荐，但已经属于保险而不是必需。

⚠️ **VAE 这一行是“全部钉 FP16”的唯一例外。** ComfyUI 对 `--fp16-vae` 自己的说明就写着
“might cause black images”，这里是在真机上确认的那一例。Z-Image-Turbo 的 VAE 是
`ae.safetensors`。`vae_dtype()` 先看 `args.fp16_vae` / `args.bf16_vae` / `args.fp32_vae`
这三个显式标志，之后才遍历 VAE 自己在 `comfy/sd.py` 里声明的 `working_dtypes` 列表，取设备
能接受的第一个。留“由 ComfyUI 决定”时这个 VAE 解析到 bf16；钉成 FP16 后，解码器在 fp16
下运行，而 fp16 只有 5 位指数，装不下这个 VAE 的激活值——溢出成 NaN，出图全黑（bf16 和
fp32 一样是 8 位指数，不会溢出）。现场验证：只把这一行从 FP16 改回“由 ComfyUI 决定”，
出图立刻恢复，其它设置（Q6_K GGUF 模型、FP8 文本编码器、补丁 6 和 7）都没动。

怎么和补丁 6 区分：两者都会出黑图并伴随 `RuntimeWarning: invalid value encountered in
cast`。如果状态文件里**没有** `FAILED`、且
`comfy.ops.scaled_dot_product_attention.__name__` 已经是 `_math_only_sdpa`，那就是 VAE
这一行的问题；如果状态文件里有 `math-only SDPA FAILED`，那才是补丁 6。

⚠️ **一个未证实的环节**：启动器文件里能找到它的精度选项名（`launchargs.precision.fp16/bf16/fp32/...`）和“由 ComfyUI 决定”（`launchargs.followcomfyui`），但**找不到** ComfyUI 的原生参数串（`--fp16-unet` 这类），所以面板有没有真的把精度传给 ComfyUI，我们无法从文件上证实。

**验证方法**：把面板设成 FP16 启动一次，看启动器控制台回显的完整命令行里有没有 `--fp16-unet`，再看 ComfyUI 日志的 `model weight dtype` 是不是 `torch.float16`。

**兜底**：如果不确定，直接在“高级选项 → 其他设置 → 自定义参数”里写 ComfyUI 的原生参数 `--fp16-unet`（对应面板的 U-Net 一行）、`--fp16-text-enc`；这条路已被 `--disable-cuda-malloc` 证明能到 ComfyUI。**不要**在这里加 `--fp16-vae`，那正是上面说的 VAE 溢出。面板与自定义参数设成同一精度不冲突。

⚠️ **必须避开的坑**：显式的 `--bf16-unet` / `--bf16-text-enc` / `--bf16-vae` 在自动选择精度**之前**被检查，所以只要面板某一行还是 BF16，它就会压过一切——包括补丁 9（它只拦住“自动选 bf16”）。三行模型精度设成 FP16 或“由 ComfyUI 决定”，别设 BF16。

### 3. 注意力后端：把默认的 SubQuad 换成 Split

没装 xformers / flash / sage 时，不加参数会用 SubQuad，ComfyUI 自己会打印一句提示：

```
Using sub quadratic optimization for attention, if you have memory or speed issues try using: --use-split-cross-attention
```

SubQuad 慢的原因：它内部的 `kv_chunk_size` 默认是 `√T`。1024×1024 出图时自注意力是 T=4096，于是它把一次大矩阵乘切成 **64 个小块**，每块都要单独做 baddbmm + softmax；Split 只做一次大 baddbmm。

| 实现                       | 耗时     | 相对  |
| -------------------------- | -------- | ----- |
| Split 式（1 次大 baddbmm） | 23.34 ms | 1.00× |
| SubQuad 式（64 个小块）    | 63.22 ms | 2.71× |

（batch×heads=16、T=4096、D=128、fp16。这是简化复现；真实实现每块还要做掩码和在线 softmax 重缩放，所以实际差距更大。）

**“默认是 SubQuad”有一个前提**：ZLUDA 下 `is_nvidia()` 返回 True，所以**不加任何注意力参数时** ComfyUI 会自动改用 `attention_pytorch`（PyTorch SDP），并不是 SubQuad；只有启动器显式传了 `--use-quad-cross-attention` 时才会落到 SubQuad。请用日志确认当前是哪一个：

- `Using sub quadratic optimization for attention` → SubQuad（慢）
- `Using split optimization for attention` → Split
- `Using pytorch attention` → PyTorch SDP

**建议**：在“高级选项 → 其他设置 → 自定义参数”里加 `--use-split-cross-attention`；如果启动器 Cross Attention 项设成了 SubQuad，请改成 Split 或“由 ComfyUI 决定”。

### 4. 量化模型与精度标志的关系

| 模型类型                          | 钉 FP16 的影响      | 说明                                                                  |
| --------------------------------- | ------------------- | --------------------------------------------------------------------- |
| GGUF（Q4 / Q5 / Q8）              | **有益无害**        | 反量化的计算 dtype 跟随推理 dtype，钉 FP16 就反量化到 fp16，避开 bf16 |
| fp8 存储的模型                    | 会**覆盖** fp8 标志 | `--fp16-unet` 在 `unet_dtype()` 里的检查顺序排在 fp8 之前             |
| INT8（torchao / bitsandbytes 类） | 一般不受影响        | 核心没有 int8-unet 参数，这类量化由节点自己管 dtype                   |

- GGUF 的机制：`cast_bias_weight()` 里计算 dtype 取输入激活的 dtype（也就是模型推理 dtype），再交给 `dequantize_tensor()` 反量化；所以钉 FP16 会让量化模型反量化到 fp16 而不是 bf16。权重仍以量化形式保存，不会变糊也不会变慢。
- 要单独覆盖某个 GGUF 模型，用 `UnetLoaderGGUFAdvanced` 节点的 `dequant_dtype` / `patch_dtype` 下拉（`default / target / float32 / float16 / bfloat16`），不影响全局。
- fp8 存储的模型属于数值路径变化，建议同一工作流跑两遍对比再定。

## 验证清单

### 1. 启动日志

应出现下面这几行（`<TEMP>` 指你的临时文件夹，通常是 `%LOCALAPPDATA%\Temp`）：

```
[ZLUDA] Preload: zluda_dir=<TEMP>\<GUID> | loaded=5/5
[ZLUDA] OK torch=2.7.1+cu118 | Tensile=C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library | cudnn=False | sdp=math | sdpa=math-only | linear-patch[F.linear=_linear] | ops-hook=installed
[ZLUDA] hooks: comfy.cli_args, comfy.model_management, comfy.supported_models, onnxruntime
[ZLUDA] ops: default
[ZLUDA] args: disable_pinned_memory=True, disable_async_offload=True, disable_cuda_malloc=True, reserve_vram=0.9
[ZLUDA] model_management: bf16_off=True | vae_fp32=False
[ZLUDA] supported_models: LTXAV memory_usage_factor patched
[ZLUDA] onnxruntime: CUDA/TensorRT providers filtered
```

同样的内容也会追加写入整合包根目录下的 `_zluda_status.txt`（以及临时文件夹里的 `zluda_status.txt`）。

> **输出约定：** 状态行一律是纯英文——方便 grep、不受控制台代码页影响，本文档也是逐字引用的。
> 需要人来处理的提示会在**下一行**单独输出 `[ZLUDA] 说明: ...` 中文，而不是拼在状态行后面。
> 中文行在个别控制台里可能显示成乱码，但英文行不受影响；`_zluda_status.txt` 是 UTF-8，
> 永远是最可靠的查看位置。

`hooks:` 只出现一次，在第一次导入 `comfy.*` 之前；`args:` 在 `comfy.cli_args` 解析时出现；
`model_management:` 和 `supported_models:` 在启动过程中出现；`onnxruntime:` 只有真的有人
导入 ONNX Runtime 时才出现（即 RMBG / controlnet_aux / Impact 节点运行的时候，不是启动时）。
`ops: default` 表示没有开启任何可选算子兜底，这是预期状态；走回退路径时这一行会在
`Late torch=` 之后照样输出（第 12 项在回退路径下也补上了）。

**状态行看起来健康并不代表全对**：补丁 6 曾经在状态行仍显示 `sdpa=math-only | ops-hook=installed`
的情况下失败过。一定要把整个状态文件 grep 一遍 `FAILED`。

如果日志里出现下面这些行，说明补丁没有完全生效（`failed=` 后面最多列 3 个失败项）。
这些行后面都会紧跟一行 `说明: ...` 中文提示；中文显示成乱码时看英文行即可。

| 日志行                                                                                 | 含义                                             | 处理                                                                                                                                          |
| -------------------------------------------------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `[ZLUDA] Preload: No ZLUDA dir found (nvcuda.dll not preloaded)` | 没找到含 ZLUDA DLL 的目录，`nvcuda.dll` 未预加载 | 先让启动器正常跑一次 ComfyUI（它会把 DLL 解到 `%TEMP%\<GUID>`）；或把 `.launcher\zluda.<hash>.extpack` 用 7-Zip 解压到整合包根目录的 `zluda\` |
| `[ZLUDA] Preload: zluda_dir=... \| loaded=3/5 \| failed=...`                           | 部分 DLL 加载失败                                | 看 `failed=` 列出的项：`xxx.dll:missing` 是文件缺失；`WinError 193` 通常是拿到了占位文件或 HTML，而不是真正的 DLL                             |
| `[ZLUDA] Direct import failed: ...`                                                    | 补丁在导入 torch 时抛了异常                      | 后面应紧跟 `Fallback hook installed`；若没有，说明连回退钩子都没装上                                                                          |
| `[ZLUDA] Fallback hook installed`                                                      | 已安装导入钩子，torch 加载完成后会补打补丁       | 下一次 `import torch` 之后应出现 `Late torch=...`                                                                                             |
| `[ZLUDA] Late torch=...`                                                               | 回退路径补打补丁成功                             | 确认行内 `cudnn=False \| sdp=math \| linear-patch[F.linear=_linear]` 都在                                                                     |
| 状态行里的 `sdpa=missing`                                                                | 这个 torch 没有 `F.scaled_dot_product_attention`，math-only 没包上，注意力会退回会崩的 FLASH / cuDNN 优先级列表 | 确认 torch 版本（需要 2.0 以上）；这一项出现在 `OK torch=` 状态行里，不在单独一行 |
| `[ZLUDA] math-only SDPA FAILED ...`                                                    | `comfy.ops` 的 math-only 包装失败，注意力会退回会崩的优先级列表 | 确认 `comfy.ops` 是否被别的插件提前导入，或它本身没有 `scaled_dot_product_attention`（comfy 太旧）；这一行出现时状态行仍可能显示 `sdpa=math-only`，一定要 grep `FAILED`                                       |
| `[ZLUDA] hook FAILED ...`                                                              | 某个模块的导入钩子执行失败，该模块的补丁没生效     | 看行里的模块名；`comfy.ops` 之外的模块失败通常只影响对应功能                                                                                  |
| `[ZLUDA] Patch FAILED: ...`                                                            | `main.py` 顶部的引导代码本身出错                 | 检查引导代码是否插在 `main.py` 第一行之前、`zluda_patch.py` 是否在 `ComfyUI\` 下                                                              |

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

另外三条检查覆盖新增的补丁，都不需要启动界面：

```bat
python -c "import zluda_patch, comfy.ops; print('sdpa:', comfy.ops.scaled_dot_product_attention.__name__)"
```

期望 `sdpa: _math_only_sdpa`。

```bat
python -c "import zluda_patch, comfy.options; comfy.options.enable_args_parsing(); import comfy.cli_args as c; print('args:', c.args.disable_pinned_memory, c.args.disable_async_offload, c.args.disable_cuda_malloc, c.args.reserve_vram)"
```

期望 `args: True True True 0.9`。

```bat
python -c "import zluda_patch, onnxruntime as ort; print('ort:', ort.get_available_providers())"
```

期望 `ort: ['CPUExecutionProvider']`。如果仍然列出 `CUDAExecutionProvider`，说明钩子没跑起来：
命令里第一个导入的必须是 `zluda_patch`。

### 3. ComfyUI 启动日志里的对照项

| 行                                                                                         | 期望值                                                                                             |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `pytorch version:`                                                                         | `2.7.1+cu118`                                                                                      |
| `Total VRAM ...`                                                                           | 测试所用显卡为 `16368 MB`                                                                          |
| `VAE load device: ..., offload device: ..., dtype:`                                        | `cuda:0` / `cpu` / VAE 的 dtype。VAE 行设成**由 ComfyUI 决定**时这台机器上是 bf16，不要强行钉 fp16 |
| `model weight dtype ...`                                                                   | `torch.float16`（**不是** `bfloat16`）                                                             |
| `Using ... attention`                                                                      | `split` 或 `pytorch`，**不要** `sub quadratic`                                                     |
| `Enabled pinned memory ...`                                                                | **不应出现**；出现说明补丁 8 没生效                                                                |
| `[ZLUDA] ONNX Runtime: forcing CPUExecutionProvider`                                       | 跑 ONNX 节点时出现（后面紧跟一行 `说明: ...` 中文）；后面不应再有 `Failed to load ...onnxruntime_providers_cuda.dll`                |
| `Unsupported Pytorch detected. DynamicVRAM support requires Pytorch version 2.8 or later.` | **正常提示**，ComfyUI 会退回旧版 ModelPatcher，不是错误                                            |

> 读 `VAE load device` 这行时只看**第一个字段**：它是 VAE 的运行设备，出现 `cpu` 才代表“使用 CPU 运行 VAE”被打开；`offload device: cpu` 是默认的备用位置（只有加 `--gpu-only` 才会变成 `cuda:0`），与 CPU VAE 无关。

## 故障排查

| 症状（日志关键字）                                                                                            | 真正原因                                                                                                                                           | 处理                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `ValueError: Model "..." is corrupt or invalid... Please download the model again`                            | **文件不存在或路径错**，不是文件损坏。`comfy/utils.py`（测试版本的 122-146 行）把 `safe_open()` 的**任何**异常（含 FileNotFoundError）都报成“损坏” | 检查路径与文件名；不要盲目重新下载                                                                                        |
| `rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906` + `Fatal Python error: Aborted` | `ROCBLAS_TENSILE_LIBPATH` 指向的目录没有 gfx906 内核（常见于启动器的 `%TEMP%\<GUID>` 目录被清理或轮换）                                            | 补丁第 2 条会把它指到 HIP SDK 的 `bin\rocblas\library`                                                                    |
| `CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmulAlgoGetHeuristic(...)`                    | 带 bias 的线性层 / `addmm` 走了 cuBLASLt 融合路径                                                                                                  | 补丁第 5 条；先用[一键自检](#2-一键自检)确认 `F.linear` 已是 `_linear`                                                    |
| `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`                                                                    | cuDNN 未关闭，**或者** `comfy.ops` 的注意力优先级把 FLASH / cuDNN 又顶回前面（补丁 6）                                                             | 补丁第 3 条；若仍出现，确认状态行有 `sdpa=math-only` 并 grep 状态文件里的 `FAILED`                                        |
| `RuntimeError: Found no NVIDIA driver on your system`                                                         | ZLUDA DLL **没有在 `import torch` 之前**被预加载（例如手工提前 import torch，或启动器注入失效）                                                    | 补丁第 1 条；确认启动日志有 `loaded=5/5`                                                                                  |
| `ERROR: clip input is invalid: None`                                                                          | 用 `Load Checkpoint` 加载了**不含文本编码器**的模型（如 anima-base）                                                                               | 改用 `Load Diffusion Model` + `Load CLIP` + `Load VAE`，或直接用官方模板                                                  |
| 显存不足 / OOM                                                                                                | 16 GB 显存跑大模型                                                                                                                                 | 降分辨率、`--fp8_e4m3fn-unet`、`--reserve-vram 1`。系统内存只有 16 GB 时**不要**用 `--lowvram` / `--novram`，会被换页拖死 |
| 在启动器里点“安装 PyTorch / 安装依赖”后一切复原                                                               | 启动器按内置清单重装（它给 ComfyUI 核心钉的 Python 是 3.11.6，cu118 最高只到 2.3.1，在 3.13 上装不上）                                             | 不要在启动器里点“安装依赖 / 切换 Torch 版本 / 重装环境”                                                                   |
| `CUBLAS_STATUS_NOT_SUPPORTED ... abType 3 cType 10 computeType 72`                                            | 跑了 `int8_tensorwise` 模型，走到了 `torch._int_mm`（补丁 7）                                                                                      | 改用 `convrot_w4a4` / `fp8` / `bf16` 模型，或接受每次调用反量化                                                           |
| 出图全黑、状态行健康、`_math_only_sdpa` 已生效                                                                | VAE 行被钉成 FP16，解码器溢出成 NaN                                                                                                                | 把启动器“VAE 模型精度”改回“由 ComfyUI 决定”                                                                           |
| RMBG / controlnet_aux / Impact 报 onnxruntime provider 错误                                                   | CUDA 12 的 `onnxruntime-gpu` provider 在 ZLUDA 下（补丁 10）                                                                                       | 确认状态行有 `onnxruntime: CUDA/TensorRT providers filtered`                                                              |

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
3. **精度设置：** 有了补丁 9，三行模型精度留 `Auto` 也不会再自动选 bf16，可以放心；显式的 `--bf16-*` 仍然优先。**VAE 那一行保持 `Auto`**，原因见[精度与性能](#精度与性能实测)里的 VAE 溢出说明。
4. **性能：** 注意力后端与精度都会显著影响速度，见[精度与性能](#精度与性能实测)；此外可尝试降低分辨率。
5. **补丁的扩展方向：** 若以后在别的算子上再遇到 `CUBLAS_STATUS_NOT_SUPPORTED`，可在 `patch_torch()` 里继续绕（例如把 `F.conv2d` 换成 `unfold + matmul` 的等价实现，已实测可用）。
6. **防止启动器覆盖：** 启动器更新 ComfyUI 会重置 `main.py`，需要重新插入引导代码；`zluda_patch.py` 也可能被清掉，建议留备份。
7. **诊断资料：** 启动器（疑难解答页面）生成的 `Diagnostics-*.log` 是环境快照加控制台缓冲区，**会包含多次运行的历史输出**。判断某行日志是否来自新进程，要看 `_zluda_status.txt` 的时间戳。
8. **仍然必须从启动器启动。** 补丁会预加载 ZLUDA 的 DLL，但它没法让 torch 真的绑定过去：
   这台机器上实测，用预加载的 DLL 调 `cuInit(0)` 返回 0、`cuDeviceGetCount` 报 1 个设备，
   但同一进程里 `torch.cuda.is_available()` 仍然是 `False`。真正起作用的是启动器的注入。
   结果就是：`python main.py`、IDE 调试、任何直接 `import torch` 的脚本都看不到显卡。
9. **INT8 tensorwise 模型能用但不快。** 补丁 7 的反量化是每次调用 24.4 ms，而原生 fp16 GEMM
   是 1.34 ms。这张卡上优先用 `convrot_w4a4`、`fp8` 或 `bf16` 模型。
10. **ONNX Runtime 一律走 CPU。** 这是故意的（补丁 10）：装的是 CUDA 12 构建的
    `onnxruntime-gpu`。如果以后真的换上匹配的构建并想试验，用 `ZLUDA_KEEP_ONNX_CUDA=1`
    保持 provider 列表原样。

## 环境变量开关

首次发布之后新增的行为都可以用环境变量关掉或调整，不必改补丁。它们只在启动时读一次：

| 变量                         | 默认  | 作用                                                         |
| ---------------------------- | ----- | ------------------------------------------------------------ |
| `ZLUDA_NO_ARG_OVERRIDES=1`   | 关    | 不再覆盖启动参数，交回启动器                                 |
| `ZLUDA_RESERVE_VRAM=<GB>`    | `0.9` | 改预留显存；显式 `--reserve-vram` 仍然优先                   |
| `ZLUDA_FORCE_FP16=1`         | 关    | 强制 UNet 和文本编码器走 fp16（也会覆盖 fp8 存储模型的标志） |
| `ZLUDA_SPLIT_ATTENTION=1`    | 关    | 用 split 注意力替代 pytorch/math SDP                         |
| `ZLUDA_ALLOW_BF16=1`         | 关    | 让 `should_use_bf16()` 恢复正常判断                          |
| `ZLUDA_VAE_FP32=1`           | 关    | 模型允许时强制 VAE 用 fp32                                   |
| `ZLUDA_KEEP_ONNX_CUDA=1`     | 关    | 不动 ONNX Runtime 的 provider 列表                           |
| `ZLUDA_INT8_DEQUANT_CACHE=1` | 关    | 缓存反量化后的 INT8 权重（权重显存约翻倍，快 12 倍）         |
| `ZLUDA_TOPK_CPU=1`           | 关    | 让 `torch.topk` 走 CPU                                       |
| `ZLUDA_STFT_CPU=1`           | 关    | 让 `torch.stft` / `torch.istft` 走 CPU（音频模型）           |
| `ZLUDA_NO_JITSCRIPT=1`       | 关    | 把 `torch.jit.script` 变成空操作                             |

## 仓库结构

```
.
├── ComfyUI/
│   └── zluda_patch.py         # 兼容补丁（复制到整合包根目录下的 ComfyUI\）
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
| `_zluda_status.txt`                                      | 补丁自检输出（每次启动追加一行）                                                                  |
| `.launcher\zluda.<hash>.extpack`                         | ZLUDA 的 5 个 DLL（可用 7-Zip 解压）                                                              |
| `.launcher\tensile-gfx906.<hash>.extpack`                | gfx906 的 rocBLAS Tensile 内核（备用来源）                                                        |
| `.launcher\preference.json`                              | 启动器偏好（含代理地址、自定义参数等）                                                            |
| `ComfyUI\blueprints\Text to Image (Anima Base 1.0).json` | Anima 官方模板                                                                                    |
| `%HIP_PATH%\bin\rocblas\library`（绝对路径）             | 补丁所指向的 gfx906 Tensile 内核目录（`HIP_PATH` 取不到时回退到 `C:\Program Files\AMD\ROCm\5.7`） |
| `%TEMP%\<GUID>\`（绝对路径，每次运行新建）               | 启动器解压 ZLUDA DLL 的位置，补丁会自动扫描并预加载                                               |

若更换整合包版本或 HIP SDK 版本，请重新核对[实测环境](#实测环境)中的版本匹配关系。

## 许可证

本项目采用 [MIT License](LICENSE)。

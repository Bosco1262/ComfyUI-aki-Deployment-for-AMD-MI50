# ComfyUI Aki Bundle on AMD MI50 (ZLUDA, gfx906, Windows)

[English](README.md) | [简体中文](README.zh-CN.md)

Run the **ComfyUI "Aki" all-in-one bundle** (秋叶整合包, started with the **绘世 / Hui Shi launcher**) on an **AMD Instinct MI50** (flashed with the **Radeon VII PRO vBIOS**) through [ZLUDA](https://github.com/vosen/ZLUDA) on Windows.

This repository documents a setup that was **verified end to end**: the exact commands, a small compatibility patch ([`ComfyUI\zluda_patch.py`](ComfyUI/zluda_patch.py)), why each part is needed, how to verify it, and a troubleshooting table. All conclusions come from hands-on testing on a single machine (command output, logs, file hashes), not from guesses.

## TL;DR

Three things are needed to run this bundle on ZLUDA:

1. **Replace torch with the cu118 build.** ZLUDA only provides CUDA 11 libraries, so a PyTorch built for CUDA 12+ cannot work.
2. **Put the patch `ComfyUI\zluda_patch.py` in place and hook it into the top of `ComfyUI\main.py`** with a short bootstrap block. The patch does twelve things: five without which nothing runs (preload the ZLUDA DLLs, point rocBLAS at the gfx906 Tensile kernels, disable cuDNN, keep only math SDP attention, bypass the cuBLASLt bias-fused GEMM), and seven that remove the crashes and slow paths left over (SDPA priority through `comfy.ops`, INT8 tensorwise weights, launch arguments, bf16, ONNX Runtime, the LTXAV estimate, optional operator fallbacks). See [How the patch works](#how-the-patch-works).
   `cudaMallocAsync` no longer needs a launch argument: the patch turns it off itself, so `--disable-cuda-malloc` is optional now.
3. **Pin the precision to FP16 except the VAE row**: on gfx906 bf16 runs at only 0.55x the fp16 speed, while a VAE forced to FP16 overflows to NaN and decodes to a black image. See [Precision and performance](#precision-and-performance-measured).

## Tested environment

### Hardware, driver, OS

| Item                                         | Value                                                                                                                                                                                                                       |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPU                                          | **AMD Radeon Pro VII** (MI50 flashed with the Radeon VII PRO vBIOS), architecture **gfx906**, **16 GB HBM2**                                                                                                                |
| VRAM reported by ComfyUI                     | `Total VRAM 16368 MB`                                                                                                                                                                                                       |
| System RAM                                   | **16310 MB (16 GB)**, which is tight: avoid options that offload heavily (see [Troubleshooting](#troubleshooting))                                                                                                          |
| OS                                           | Windows 10 (10.0.19045)                                                                                                                                                                                                     |
| AMD driver / runtime                         | `HIP Driver: 50731541` / `Runtime Version: 50731541`, i.e. **HIP SDK 5.7.1**                                                                                                                                                |
| HIP SDK install path                         | `C:\Program Files\AMD\ROCm\5.7` (environment variables `HIP_PATH`, `HIP_PATH_57`, `HIP_PATH_64`)                                                                                                                            |
| rocBLAS Tensile kernel directory             | `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library` (587 files, **including the gfx906 kernels**: `TensileLibrary_lazy_gfx906.dat`, `TensileLibrary_Type_*_gfx906.co/.dat`, `Kernels.so-000-gfx906-xnack-.hsaco`)           |
| ZLUDA                                        | `zluda.extpack` shipped with the launcher, 5 DLLs: `cublas64_11.dll`, `cusparse64_11.dll`, `nvcuda.dll`, `nvml.dll`, `nvrtc64_112_0.dll` (built 2024-04, ROCm 5.x era), so it **only works with torch built for CUDA 11.x** |
| gfx906 kernel pack shipped with the launcher | `.launcher\tensile-gfx906.<hash>.extpack` (contains `TensileLibrary_*_gfx906.*`)                                                                                                                                            |

### Software

| Item                          | Value                                                                                                                                                                                                                                                                                              |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Package root                  | `ComfyUI-aki-v3.2\`, the folder that contains `绘世启动器.exe`, `python\` and `ComfyUI\`                                                                                                                                                                                                           |
| WebUI directory (`.app_path`) | `ComfyUI`                                                                                                                                                                                                                                                                                          |
| Python                        | **3.13.11** (`python\python.exe`, pip 25.3)                                                                                                                                                                                                                                                        |
| Launcher                      | 绘世启动器 2.9.2.419 (`绘世启动器.exe`, bootstrap 1.11.0.11)                                                                                                                                                                                                                                       |
| ComfyUI                       | `dec5d9450a5290bcf63430409ea41018e67f41c3` (2026-08-05)                                                                                                                                                                                                                                            |
| **torch after this setup**    | **torch 2.7.1+cu118 / torchvision 0.22.1+cu118 / torchaudio 2.7.1+cu118**                                                                                                                                                                                                                          |
| Stock torch (for reference)   | torch 2.13.0+cu130 / torchvision 0.28.0+cu130 / torchaudio 2.11.0+cu130                                                                                                                                                                                                                            |
| Uninstalled                   | `sageattention 2.2.0+cu130torch2.9.0andhigher.post4`, `triton-windows 3.5.1.post24` (the version lock is part of the package name, so pip does not notice the conflict)                                                                                                                            |
| Other key packages            | numpy 2.3.5, comfy_kitchen 0.2.26, comfy_aimdo 0.4.11, gguf 0.17.1, spandrel 0.4.1, transformers 4.57.6, accelerate 1.12.0, sam_2 1.0 (declares `torch>=2.5.1`, so torch must **not** go below 2.5)                                                                                                |
| Custom nodes                  | ComfyUI-Manager, Impact-Pack, GGUF, RMBG, controlnet_aux, IPAdapter_plus, LTXVideo, WanVideoWrapper, VideoHelperSuite, UltimateSDUpscale, AlekPet (the requirements of `comfyui_controlnet_aux` list a bare `torch` / `torchvision`, so installing it does not pull torch back to another version) |

### Why torch must be cu118

- ZLUDA only provides CUDA 11.x library names (`cublas64_11.dll`, `nvrtc64_112_0.dll`, ...). A cu12x torch looks for `cublas64_12.dll` and `cudnn64_9.dll` instead, which are NVIDIA proprietary libraries.
- On Python 3.13 the cu118 index only offers torch **2.6.0 / 2.7.0 / 2.7.1** (2.5.1 and earlier have no cp313 wheels). **2.7.1** was chosen because ComfyUI's `extended_fp16_support()` needs torch 2.7 or newer, and `sam_2` needs 2.5.1 or newer.

## Setup

All paths below are relative to the launcher **package root** (the folder that contains `绘世启动器.exe`, `python\` and `ComfyUI\`). The launcher's command prompt uses the package's bundled Python, so a plain `pip` / `python` refers to it.

### Step 1: Replace torch with the cu118 build

1. Double-click the launcher (`绘世启动器.exe`). In the left sidebar click **Advanced options (高级选项)**, then click **Launch command prompt (启动命令提示符)** at the top right.
2. In the command prompt that opens, enter the following commands one by one:

```bat
pip freeze > env_before_cu118.txt

pip uninstall -y sageattention triton-windows

pip install --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple "torch==2.7.1+cu118" "torchvision==0.22.1+cu118" "torchaudio==2.7.1+cu118"

python -c "import torch;print(torch.__version__, torch.version.cuda)"
```

| Command             | Purpose                                                     |
| ------------------- | ----------------------------------------------------------- |
| `pip freeze > ...`  | Back up the current package list first                      |
| `pip uninstall ...` | Uninstall the packages built for cu130 / torch 2.9 or newer |
| `pip install ...`   | Install the cu118 trio                                      |
| `python -c ...`     | Verify; it should print `2.7.1+cu118 11.8`                  |

> Tip: if you need a proxy to reach the package indexes, append `--proxy http://<host>:<port>` to the `pip install` command.

### Step 2: Install the patch and hook it into `main.py`

1. Copy all files of this repository into the package root, so that the patch ends up at `ComfyUI\zluda_patch.py`. Only [`ComfyUI\zluda_patch.py`](ComfyUI/zluda_patch.py) matters; `README*.md` and `LICENSE` are not needed.
2. Insert the following block **before the first line** of `ComfyUI\main.py` (in the tested ComfyUI version the first line is `import comfy.options`):

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

> ⚠️ It must go **before the first line** (before `import comfy.options`): the patch has to preload the ZLUDA DLLs before torch is imported.
>
> ⚠️ If the launcher updates or resets ComfyUI, `main.py` is overwritten: insert the block again. `zluda_patch.py` may also be wiped, so keep a copy (this repository).

### Step 3: Launch arguments

1. Start the launcher normally. In **Settings (设置)**, switch the **configuration model (配置模型)** to **Expert (专家)**.
2. Go to **Advanced options (高级选项) → Other settings (其他设置) → Custom arguments (自定义参数)** and add:

```
--disable-cuda-malloc
```

Why: `ComfyUI\cuda_malloc.py` automatically enables `cudaMallocAsync` (`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`) on non-NVIDIA cards, and ZLUDA does not necessarily implement that allocator. `--disable-cuda-malloc` (defined in `comfy/cli_args.py`) forces it off.

With patch 8 in place this argument is **optional**: the patch sets
`args.disable_cuda_malloc`, `args.disable_pinned_memory`, `args.disable_async_offload` and
`args.reserve_vram` itself, so the bundle keeps working even when `preference.json` is
reset or ComfyUI is started through some other entry point. Keeping the argument is
harmless.

**Do not use** these arguments: `--use-sage-attention` (the package is uninstalled), `--use-flash-attention` (not installed), `--enable-triton-backend` (useless under ZLUDA), `--fast` / `--fast cublas_ops` (this path calls cuBLASLt directly and always crashes), `--cuda-malloc`.

Precision settings are covered in [Precision and performance](#precision-and-performance-measured). In one line: **pin the three model rows to FP16 (Auto is safe now), keep the VAE row at Auto, and never let BF16 appear anywhere.**

### Step 4 (optional): Try a real model, for example Anima Base 1.0

This is a "split" model: it needs three loaders and the official template.

| File                          | Where to put it                                                   |
| ----------------------------- | ----------------------------------------------------------------- |
| `anima-base-v1.0.safetensors` | `ComfyUI\models\diffusion_models\` (`models\unet\` is equivalent) |
| `qwen_3_06b_base.safetensors` | `ComfyUI\models\text_encoders\`                                   |
| `qwen_image_vae.safetensors`  | `ComfyUI\models\vae\`                                             |

Workflow: use the template bundled with the package, `ComfyUI\blueprints\Text to Image (Anima Base 1.0).json` (drag it into the web page, or use Workflow → Templates and search for `Anima`). It uses `Load Diffusion Model` + `Load CLIP` (keep the type at `stable_diffusion`; ComfyUI recognizes the Qwen3 text encoder from the file content) + `Load VAE`. KSampler defaults: steps 30, cfg 4, `er_sde`, `simple`; the size group is 1 MP (about 1024×1024).

**Common mistake:** loading `anima-base-v1.0.safetensors` with `Load Checkpoint` fails with `ERROR: clip input is invalid: None`, because that file **contains no text encoder** (`load_checkpoint_guess_config()` in `comfy/sd.py` returns `clip=None`, and `comfy/nodes.py:74` raises).

## How the patch works

[`ComfyUI\zluda_patch.py`](ComfyUI/zluda_patch.py) does twelve things. Items 1-5 are the ones without which nothing runs; items 6-12 remove the crashes and slow paths that remain after that. Every item uses the same four labels: **Symptom** (what the user sees), **Reason** (why it happens), **Fix** (what the patch does) and **Cost** (whether there is a downside).

### 1. Preload the ZLUDA DLLs

- **Symptom:** `RuntimeError: Found no NVIDIA driver on your system`.

- **Reason:** ZLUDA's `nvcuda.dll` must be loaded **before** torch initializes CUDA. The launcher normally does this itself (it unpacks the DLLs into `%TEMP%\<GUID>` on every run), but its injection was not reliable in this setup.

- **Fix:** the patch scans `%TEMP%`, `%TMP%` and `%LOCALAPPDATA%\Temp` for that folder (and `zluda\` in the package root), adds the ZLUDA folder and `%HIP_PATH%\bin` to `PATH` / `os.add_dll_directory`, and preloads `nvcuda.dll`, `nvml.dll`, `cublas64_11.dll`, `cusparse64_11.dll`, `nvrtc64_112_0.dll`.

- **Cost:** none. It only scans a few temp folders at startup.

### 2. Set `ROCBLAS_TENSILE_LIBPATH`

- **Symptom:** `rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906`, followed by `Fatal Python error: Aborted`.

- **Reason:** ZLUDA's cuBLAS forwards matrix multiplications to **rocBLAS**, which needs the Tensile kernels of the **matching architecture (gfx906)**; the launcher points this variable at a folder under `%TEMP%` that does not contain them.

- **Fix:** the patch sets `ROCBLAS_TENSILE_LIBPATH` to `<HIP SDK>\bin\rocblas\library`, reading the `HIP_PATH` environment variable first and falling back to `C:\Program Files\AMD\ROCm\5.7` from the tested setup.

- **Cost:** none.

### 3. Turn cuDNN off

- **Symptom:** `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`, with the VAE's convolutions hit first.

- **Reason:** ZLUDA has no cuDNN.

- **Fix:** `torch.backends.cudnn.enabled = False`; with cuDNN off, convolutions use a fallback path that works (verified).

- **Cost:** none.

### 4. Keep only the math SDP attention backend

- **Symptom:** attention lands on the flash / mem-efficient kernels, which do not exist under ZLUDA (the same failure as item 6, one layer down).

- **Reason:** ZLUDA cannot run the flash attention kernels; the launcher's own ZLUDA compatibility patch does the same.

- **Fix:** flash and mem-efficient are turned off, leaving only the math SDP backend.

- **Cost:** none. FLASH and cuDNN never existed here, so the call always fell back to math; see item 6 for the measured numbers.

### 5. Split the linear layer with a bias

- **Symptom:** `CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul ...`.

- **Reason:** PyTorch runs `F.linear(x, w, b)` / `addmm` through a **bias-fused GEMM that only exists in cuBLASLt** (`gemm_and_bias`). ZLUDA has **no cuBLASLt implementation at all** (none of its 5 DLLs exports a `cublasLt*` symbol), so torch falls back to NVIDIA's genuine `torch\lib\cublasLt64_11.dll` (544 MB), which necessarily returns `CUBLAS_STATUS_NOT_SUPPORTED`. Measured: `mm` / `bmm` / `baddbmm` without bias all work; **only linear / addmm with a bias crash**, regardless of precision (fp16 / bf16 / fp32).

- **Fix:** "linear layer with bias" is split into `matmul` + add, and `torch.addmm` gets the same treatment.

- **Cost:** none.

### 6. Math-only SDPA for every attention call

- **Symptom:** Either a hard crash

  ```
  RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
  RuntimeError: cutlassF: no kernel found to launch!
  ```

  or, worse, a workflow that finishes without an error and saves a **solid black image**:
  `SaveImage` prints `RuntimeWarning: invalid value encountered in cast`, which means the
  decoded image contains NaN.

- **Reason:** `comfy/ops.py` installs its own SDPA wrapper and calls
  `sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True)` with
  `[FLASH_ATTENTION, CUDNN_ATTENTION, EFFICIENT_ATTENTION, MATH]`. `set_priority=True`
  **overrides** the `enable_flash_sdp(False)` from item 4, so PyTorch walks that list no
  matter what the flags say, and neither FLASH nor cuDNN exists under ZLUDA.

  `comfy.ops.scaled_dot_product_attention` is reached from three places: the text encoder
  (`attention_pytorch`), the VAE (`comfy/ldm/modules/diffusionmodules/model.py` calls it
  directly) and the main model (`attention_split` falls back to it when xformers is
  unavailable). So the text encoder breaks first, the VAE next, and the image comes out black
  whenever the call does not raise.

  Measured (batch×heads = 16, T = 512, D = 128):

  | Call path                                                | fp16     | bf16     | fp32     |
  | -------------------------------------------------------- | -------- | -------- | -------- |
  | bare `F.scaled_dot_product_attention`                    | OK       | OK       | OK       |
  | `comfy.ops.scaled_dot_product_attention` (priority list) | **FAIL** | **FAIL** | **FAIL** |
  | math only (`SDPBackend.MATH`)                            | OK       | OK       | OK       |

- **Fix:** `comfy/ops.py` binds `torch.nn.functional.scaled_dot_product_attention` at import
  time, so patching `F` alone does nothing (`comfy.ops uses patched fn -> False`).
  The patch wraps both the `F` function and `comfy.ops`'s own reference so every call runs
  inside `sdpa_kernel([MATH])`, and forces the `can_use_flash_attention` /
  `can_use_cudnn_attention` / `can_use_efficient_attention` probes to `False` (they report
  FLASH as usable here, which would skip the `repeat_kv` fallback that actually works).
  `comfy.ops` is imported after the bootstrap block, so the patch hooks its import.

  Field gotcha: the first version re-imported `comfy.ops` inside its own `exec_module`,
  which raises `AttributeError: cannot access submodule 'ops' of module 'comfy' (most likely
  due to a circular import)`. The startup line **still reported**
  `sdpa=math-only | ops-hook=installed`, because those strings are appended when the wrappers
  are installed; only a later line revealed the failure. The `comfy.ops` half stayed
  unpatched and the images stayed black. Always grep the whole status file for `FAILED`
  instead of trusting the healthy-looking line.

- **Cost:** none. FLASH and cuDNN never existed here, so the call always fell back to math.
  At S = 4096, batch×heads = 16, D = 128: `attention_pytorch` 39.1 ms, math SDP 39.1 ms,
  split 26.7 ms. This is a correctness fix; use split attention if you want speed, see
  [Attention backend](#3-attention-backend-replace-the-default-subquad-with-split).

### 7. INT8 tensorwise fallback

- **Symptom:**

  ```
  RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul ...
  abType 3 cType 10 computeType 72 scaleType 10
  ```

  `abType 3` with `computeType 72` is an INT8 × INT8 → INT32 GEMM.

- **Reason:** `comfy_kitchen`'s `TensorWiseINT8Layout` handler ends in `torch._int_mm`, a
  cuBLASLt INT8 GEMM that ZLUDA does not implement. The compute dtype is **not** the
  problem: `int8_tensorwise` sets `quantize_input: False`, so the weight reaches `F.linear`
  as a `QuantizedTensor` and is never dequantized — `--fp16-unet` cannot help.

  Read the `comfy_quant` JSON blobs in the safetensors header, not the file name, to tell
  whether a model is affected:

  | `comfy_quant`                        | Affected?                              |
  | ------------------------------------ | -------------------------------------- |
  | `{"format": "int8_tensorwise", ...}` | **Yes**, it reaches `torch._int_mm`    |
  | `{"format": "convrot_w4a4", ...}`    | No, it uses comfy_kitchen's own kernel |
  | none (plain fp16 / bf16 / fp8)       | No                                     |

  A file named `..._int8_convrot` can still carry `format: int8_tensorwise`; the `convrot`
  flag alone does not move it off the `torch._int_mm` path.

- **Fix:** `F.linear` / `torch.addmm` dequantize a `QuantizedTensor` weight to the input
  dtype before the matmul. Measured on the 11520×3840 qkv layer of
  `z_image_turbo_int8_convrot` (m = 90, fp16 input):

  | Approach                        | Per call |
  | ------------------------------- | -------- |
  | no handling                     | crash    |
  | dequantize every call (default) | 24.4 ms  |
  | dequantize once + cache         | 2.0 ms   |
  | native fp16 GEMM, same shape    | 1.34 ms  |

  The cache is **off by default** because it doubles the weight footprint: 5.72 GB of INT8
  weights become 11.44 GB of fp16 for a 6.14 B model, and the model plus its text encoder
  already uses 11.26 GB of the 16 GB card. Set `ZLUDA_INT8_DEQUANT_CACHE=1` when there is
  spare VRAM. The cache is keyed with `weakref.WeakKeyDictionary`, so it is released
  together with the model.

- **Cost:** Usable, not fast. Prefer, in order: a real `convrot_w4a4` model
  (comfy_kitchen's own kernel), an `fp8` model (its handler already falls back to
  dequantization on failure), then `bf16` — but watch the 16 GB of system RAM with large
  bf16 models.

### 8. Launch arguments the patch owns

- **Symptom:** `Enabled pinned memory ...` shows up in the startup log (the launch arguments did not take effect).

- **Reason:** the launcher keeps its options in `.launcher\preference.json`: a launcher reset drops them, and any entry point that skips `main.py` never sets them at all.

- **Fix:** `comfy/cli_args.py` parses at import time and every consumer (`cuda_malloc.py`, `comfy/model_management.py`, `comfy/ldm/modules/attention.py`) reads `args` afterwards, so the patch hooks that import and overrides the object once, right after it is built:

  | Argument                  | Why                                                                                                        |
  | ------------------------- | ---------------------------------------------------------------------------------------------------------- |
  | `--disable-pinned-memory` | Windows pins up to 40% of RAM; this machine has 16 GB, and pinned buffers are a frequent source of crashes |
  | `--disable-async-offload` | torch 2.7 falls back to the legacy ModelPatcher, so async offload buys nothing                             |
  | `--disable-cuda-malloc`   | ZLUDA has no cudaMallocAsync; `--disable-cuda-malloc` is no longer needed on the command line              |
  | `--reserve-vram 0.9`      | keeps the OS and other programs from fighting the card                                                     |

  An explicit `--reserve-vram` from the launcher wins, so a value you set by hand is not overwritten. Switches: `ZLUDA_NO_ARG_OVERRIDES=1` disables the whole group, `ZLUDA_RESERVE_VRAM=<GB>` changes the reserve, `ZLUDA_FORCE_FP16=1` also forces fp16 for fp8-stored models (off by default for that reason), `ZLUDA_SPLIT_ATTENTION=1` selects split attention.

- **Cost:** none.

### 9. bf16 stays off

- **Symptom:** `model weight dtype torch.bfloat16`, and the model runs at half speed.

- **Reason:** ZLUDA reports compute capability (8, 8), and `should_use_bf16()` returns `True` as soon as `props.major >= 8`, before it asks the device anything. Measured on gfx906: fp16 9.54 TFLOPS, bf16 5.26. Models that list `bfloat16` first in `supported_inference_dtypes` (Anima) therefore run at half speed.

- **Fix:** the patch replaces `comfy.model_management.should_use_bf16` with `lambda *a, **k: False`; every caller uses that attribute, including the bare-name calls inside `model_management` itself. An **explicit** `--bf16-unet` still wins: `unet_dtype()` checks the flag before it looks at the model's dtype list, which is why the launcher's precision rows must not be set to BF16 (see [Precision and performance](#2-how-to-set-the-launchers-compute-precision-settings)). Switch: `ZLUDA_ALLOW_BF16=1` restores the old behaviour.

- **Cost:** none.

### 10. ONNX Runtime on the CPU

- **Symptom:** onnxruntime provider errors from RMBG / controlnet_aux / Impact (the provider fails to load or crashes inside the fake driver).

- **Reason:** `controlnet_aux` (DW Pose, lineart, depth), RMBG and Impact's detectors run their models through ONNX Runtime, whose default provider list puts `CUDAExecutionProvider` first. The installed `onnxruntime-gpu` is a CUDA 12 build while ZLUDA only impersonates CUDA 11.

- **Fix:** the patch filters `CUDAExecutionProvider` / `TensorrtExecutionProvider` out of `get_available_providers()` (both the module-level function and the pybind copy the session path reads) and rewrites any session that still asks for CUDA to `CPUExecutionProvider`. Switch: `ZLUDA_KEEP_ONNX_CUDA=1` leaves ONNX Runtime untouched.

- **Cost:** none. Those models are small, so the CPU cost is not noticeable; the alternative is a crash.

### 11. LTXAV memory estimate

- **Symptom:** LTXAV runs out of VRAM on a 16 GB card (ComfyUI plans as if it needs almost nothing).

- **Reason:** upstream leaves LTXAV's `memory_usage_factor` at `0.077`, and `model_base.py` multiplies the activation estimate with it. The parent `LTXV` class uses `(cross_attention_dim / 2048) * 5.5`.

- **Fix:** the patch applies that formula to LTXAV with a 1.5x margin (4096 → 16.5).

- **Cost:** ComfyUI may offload earlier, but it does not OOM.

### 12. Optional operator fallbacks

- **Symptom:** the Zluda fork replaces `torch.topk`, `torch.stft` / `torch.istft` and `torch.jit.script` unconditionally, so every `topk` becomes a D2H + `topk` + H2D round trip and torchvision's NMS calls `topk` internally: detection nodes slow down drastically.

- **Reason:** nothing in this setup has been shown to need those fallbacks.

- **Fix:** they are off by default and enabled one at a time: `ZLUDA_TOPK_CPU=1`, `ZLUDA_STFT_CPU=1` (audio models), `ZLUDA_NO_JITSCRIPT=1` (node packs that ship TorchScript models).

- **Cost:** enabling one makes it slower, see the round-trip cost of `topk` above.

Items 6 and 8-11 are applied through a `sys.meta_path` import hook, because `comfy.*` is
imported after `main.py`'s bootstrap block; each hook runs right after its module
executes, so nothing is imported twice. Item 12 (the optional operator fallbacks) is applied
once torch has loaded, on both paths: right after the direct import, or from the fallback
loader when the direct import failed.

## Precision and performance (measured)

All numbers below are **measured on the test machine** (MI50 flashed with the Radeon VII PRO vBIOS / gfx906 / HIP SDK 5.7.1 / torch 2.7.1+cu118 / ZLUDA).

### 1. Why FP16 must be pinned, not BF16

ZLUDA impersonates an NVIDIA card on this machine, which defeats a rule in ComfyUI that exists **specifically for gfx906**:

- Under ZLUDA `torch.version.hip` is `None` and `torch.version.cuda` is `11.8`, so `is_amd()` returns False (it requires `torch.version.hip`).
- Therefore the "RDNA2 and older don't support bf16" branch in `should_use_bf16()` is skipped, even though that architecture list **explicitly contains `gfx906`**.
- At the same time ZLUDA reports compute capability **(8, 8)** and `torch.cuda.is_bf16_supported()` returns True, so `should_use_bf16()` ends up returning True.
- If a model lists bf16 first in `supported_inference_dtypes` (Anima Base 1.0 is `[bfloat16, float16, float32]`), the first iteration of `unet_dtype()`'s loop returns bf16 directly.

| Precision | Measured    | Relative to fp16 |
| --------- | ----------- | ---------------- |
| fp16      | 9.54 TFLOPS | 1.00×            |
| bf16      | 5.26 TFLOPS | 0.55×            |
| fp32      | 9.10 TFLOPS | 0.95×            |

(4096×4096 matrix multiplication.)

Conclusion: **pin FP16**. Do not choose FP32: it is about as fast as fp16 but doubles the memory footprint, which is a net loss on a 16 GB card.

Self-check: the startup log should show `model weight dtype torch.float16, manual cast: None`; `torch.bfloat16` means the slow path is still active.

### 2. How to set the launcher's "compute precision settings"

The launcher's **Advanced options → compute precision settings** has four rows plus one toggle:

| Row                                       | Set to             | Notes                                                               |
| ----------------------------------------- | ------------------ | ------------------------------------------------------------------- |
| General model precision                   | **FP16**           |                                                                     |
| Text encoder model precision              | **FP16**           | The default is already fp16; setting FP16 is just insurance         |
| U-Net model precision                     | **FP16**           | Directly addresses the bf16 issue above                             |
| **VAE model precision**                   | **Auto (or BF16)** | **FP16 overflows to NaN and decodes a black image**, see below      |
| Upcast XAttn precision (upcast-attention) | **Leave off**      | Enabling it upcasts attention to fp32, which is slower on this card |

With patch 9 in place, "let ComfyUI decide" is safe on the three model rows: bf16 is no
longer selected automatically. Keeping them at FP16 is still the recommended setting, but
it is now insurance rather than a requirement.

⚠️ **The VAE row is the one exception to "pin everything to FP16".** ComfyUI's own
description of `--fp16-vae` already says it "might cause black images"; this is that case
confirmed on real hardware. For Z-Image-Turbo the VAE is `ae.safetensors`. `vae_dtype()`
checks `args.fp16_vae` / `args.bf16_vae` / `args.fp32_vae` first, and only then walks the
VAE's own `working_dtypes` list (declared in `comfy/sd.py`) to pick the first entry the
device accepts. Left at "let ComfyUI decide" that resolves to bf16 for this VAE; pinned to
FP16, the decoder runs in fp16, whose 5-bit exponent cannot hold that VAE's activations —
they overflow to NaN and the image comes out black (bf16 has fp32's 8-bit exponent and does
not overflow). Verified in the field: switching only this row from FP16 to Auto made
generation work, with everything else unchanged.

How to tell this apart from patch 6: both produce a black image plus
`RuntimeWarning: invalid value encountered in cast`. If the status file has **no**
`FAILED`, and `comfy.ops.scaled_dot_product_attention.__name__` is already
`_math_only_sdpa`, it is the VAE row. If the status file contains `math-only SDPA FAILED`,
it is patch 6.

⚠️ **One unverified link**: the launcher binaries contain its precision option names (`launchargs.precision.fp16/bf16/fp32/...`) and "let ComfyUI decide" (`launchargs.followcomfyui`), but **no ComfyUI-native flag strings** (such as `--fp16-unet`), so whether the panel actually forwards the precision to ComfyUI cannot be confirmed from the files.

**How to verify:** set the panel to FP16 and start once; check whether the command line echoed in the launcher console contains `--fp16-unet`, and whether the ComfyUI log shows `model weight dtype` as `torch.float16`.

**Fallback:** if unsure, write ComfyUI's native flags in **Advanced options → Other settings → Custom arguments**: `--fp16-unet` (the U-Net row) and `--fp16-text-enc`. That path is already proven to reach ComfyUI by `--disable-cuda-malloc`. Do **not** add `--fp16-vae` there — that is exactly the VAE overflow described above. Setting the panel and the custom arguments to the same precision does not conflict.

⚠️ **A pitfall to avoid**: an explicit `--bf16-unet` / `--bf16-text-enc` / `--bf16-vae` is
checked **before** the automatic dtype selection, so any row still set to BF16 wins over
everything — including patch 9, which only prevents _automatic_ bf16 selection. Set the
three model rows to FP16 (or Auto), never to BF16.

### 3. Attention backend: replace the default SubQuad with Split

With no xformers / flash / sage installed, SubQuad is used without arguments, and ComfyUI prints a hint itself:

```
Using sub quadratic optimization for attention, if you have memory or speed issues try using: --use-split-cross-attention
```

Why SubQuad is slow: its internal `kv_chunk_size` defaults to `√T`. For a 1024×1024 image the self-attention is T=4096, so it splits one large matrix multiplication into **64 small chunks**, each doing its own baddbmm + softmax; Split does a single large baddbmm.

| Implementation                 | Time     | Relative |
| ------------------------------ | -------- | -------- |
| Split-like (one large baddbmm) | 23.34 ms | 1.00×    |
| SubQuad-like (64 chunks)       | 63.22 ms | 2.71×    |

(batch×heads=16, T=4096, D=128, fp16. This is a simplified reproduction; the real implementation also does masking and online softmax rescaling per chunk, so the real gap is larger.)

**"The default is SubQuad" comes with a condition**: under ZLUDA `is_nvidia()` returns True, so with **no attention arguments at all** ComfyUI switches to `attention_pytorch` (PyTorch SDP), not SubQuad; SubQuad is only reached when the launcher explicitly passes `--use-quad-cross-attention`. Check the log for which one is active:

- `Using sub quadratic optimization for attention` → SubQuad (slow)
- `Using split optimization for attention` → Split
- `Using pytorch attention` → PyTorch SDP

**Recommendation:** add `--use-split-cross-attention` to **Advanced options → Other settings → Custom arguments**. If the launcher's Cross Attention row is set to SubQuad, change it to Split or "let ComfyUI decide".

### 4. Quantized models and precision flags

| Model type                          | Effect of pinning FP16      | Notes                                                                                                             |
| ----------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| GGUF (Q4 / Q5 / Q8)                 | **Beneficial, no downside** | The dequantization compute dtype follows the inference dtype, so pinning FP16 dequantizes to fp16 and avoids bf16 |
| fp8-stored models                   | **Overrides** the fp8 flags | `--fp16-unet` is checked before fp8 in `unet_dtype()`                                                             |
| INT8 (torchao / bitsandbytes style) | Generally unaffected        | Core has no int8-unet flag; such quantization manages its own dtype                                               |

- GGUF mechanism: in `cast_bias_weight()` the compute dtype follows the input activation dtype (i.e. the model inference dtype), which is then handed to `dequantize_tensor()`. So pinning FP16 makes quantized models dequantize to fp16 instead of bf16. The weights stay quantized in storage, so nothing becomes blurry or slower.
- To override a single GGUF model, use the `dequant_dtype` / `patch_dtype` dropdowns of the `UnetLoaderGGUFAdvanced` node (`default / target / float32 / float16 / bfloat16`); this does not affect the global setting.
- fp8-stored models change the numeric path, so run the same workflow twice and compare before deciding.

## Verification

### 1. Startup log

These lines should appear (`<TEMP>` is your temp folder, usually `%LOCALAPPDATA%\Temp`):

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

The same lines are appended to `_zluda_status.txt` in the package root (and to `zluda_status.txt` in the temp folder).

> **Output convention:** status lines are plain English — they are greppable, they survive any
> console code page, and this document quotes them verbatim. A message that needs a human decision is
> followed by its own `[ZLUDA] 说明: ...` line in Chinese instead of being appended to the status line.
> The Chinese line may show up as mojibake in some consoles; the English line is unaffected, and
> `_zluda_status.txt` is UTF-8 and always the most reliable place to look.

`hooks:` appears once, before the first `comfy.*` import. `args:` appears when
`comfy.cli_args` is parsed, `model_management:` and `supported_models:` during startup,
and `onnxruntime:` only when something imports ONNX Runtime (an RMBG / controlnet_aux /
Impact node, not at startup). `ops: default` means no optional operator fallback is
enabled, which is the intended state; on the fallback hook path it is printed after
`Late torch=` as well (item 12 is applied there too).

The status line looking healthy is **not** proof that everything worked: patch 6 has been
seen failing while the line still said `sdpa=math-only | ops-hook=installed`. Always grep
the whole status file for `FAILED`.

If any of the lines below appear, the patch is not fully active (`failed=` lists at most 3 entries).
Each of them is followed by its own `说明: ...` Chinese line.

| Log line                                                                               | Meaning                                                                    | Fix                                                                                                                                                                   |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[ZLUDA] Preload: No ZLUDA dir found (nvcuda.dll not preloaded)` | No folder with the ZLUDA DLLs was found, so `nvcuda.dll` was not preloaded | Let the launcher run ComfyUI once (it unpacks the DLLs into `%TEMP%\<GUID>`), or unpack `.launcher\zluda.<hash>.extpack` into `zluda\` in the package root with 7-Zip |
| `[ZLUDA] Preload: zluda_dir=... \| loaded=3/5 \| failed=...`                           | Some DLLs failed to load                                                   | Look at what `failed=` lists: `xxx.dll:missing` means the file is absent; `WinError 193` usually means a placeholder or HTML file instead of a real DLL               |
| `[ZLUDA] Direct import failed: ...`                                                    | The patch raised while importing torch                                     | `Fallback hook installed` should follow; if it does not, even the fallback hook failed to install                                                                     |
| `[ZLUDA] Fallback hook installed`                                                      | The import hook is installed and will patch torch once it loads            | `Late torch=...` should appear after the next `import torch`                                                                                                          |
| `[ZLUDA] Late torch=...`                                                               | The fallback path patched torch successfully                               | Confirm the line contains `cudnn=False \| sdp=math \| linear-patch[F.linear=_linear]`                                                                                 |
| `sdpa=missing` in the status line                                                       | This torch has no `F.scaled_dot_product_attention`, so math-only was not wrapped and attention falls back to the crashing FLASH / cuDNN priority list | Check the torch version (2.0 or newer is required); this appears inside the `OK torch=` status line, not on its own line |
| `[ZLUDA] math-only SDPA FAILED ...`                                                    | The `comfy.ops` math-only wrapper failed, so attention falls back to the priority list that crashes | Check whether another extension imported `comfy.ops` first, or whether this `comfy` has no `scaled_dot_product_attention` at all (too old); the status line can still say `sdpa=math-only`, so grep for `FAILED` |
| `[ZLUDA] hook FAILED ...`                                                              | A module import hook raised, so that module is unpatched                    | Look at the module name in the line; a failure outside `comfy.ops` usually only affects that feature                                                                     |
| `[ZLUDA] Patch FAILED: ...`                                                            | The bootstrap block at the top of `main.py` itself raised                  | Check that the block sits before the first line of `main.py` and that `zluda_patch.py` is inside `ComfyUI\`                                                           |

### 2. One-shot self-check (does not start the UI)

In the launcher's command prompt, go to the `ComfyUI` folder of the package and run:

```bat
cd ComfyUI

python -c "import zluda_patch, torch, torch.nn.functional as F; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('patched:', F.linear.__name__, torch.addmm.__name__, torch.backends.cudnn.enabled); x=torch.randn(2,128,512,device='cuda',dtype=torch.float16); w=torch.randn(1024,512,device='cuda',dtype=torch.float16); b=torch.randn(1024,device='cuda',dtype=torch.float16); print('linear:', float(F.linear(x,w,b).sum()))"
```

Expected output:

```
cuda: True
device: AMD Radeon Pro VII
patched: _linear _addmm False
linear: <some number>
```

The check needs a ZLUDA folder it can find: a `<GUID>` folder that a previous launcher run left in your temp folder, or a `zluda\` folder in the package root (unpack the launcher's `.launcher\zluda.<hash>.extpack` there with 7-Zip).

Three more checks cover the newer patches. None of them needs the UI running:

```bat
python -c "import zluda_patch, comfy.ops; print('sdpa:', comfy.ops.scaled_dot_product_attention.__name__)"
```

Expected `sdpa: _math_only_sdpa`.

```bat
python -c "import zluda_patch, comfy.options; comfy.options.enable_args_parsing(); import comfy.cli_args as c; print('args:', c.args.disable_pinned_memory, c.args.disable_async_offload, c.args.disable_cuda_malloc, c.args.reserve_vram)"
```

Expected `args: True True True 0.9`.

```bat
python -c "import zluda_patch, onnxruntime as ort; print('ort:', ort.get_available_providers())"
```

Expected `ort: ['CPUExecutionProvider']`. If `CUDAExecutionProvider` is still listed, the
hook did not run: the first import in the command has to be `zluda_patch`.

### 3. Reference lines in the ComfyUI startup log

| Line                                                                                       | Expected                                                                                                          |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `pytorch version:`                                                                         | `2.7.1+cu118`                                                                                                     |
| `Total VRAM ...`                                                                           | `16368 MB` on the tested card                                                                                     |
| `VAE load device: ..., offload device: ..., dtype:`                                        | `cuda:0` / `cpu` / the VAE dtype. With the VAE row at **Auto** this is bf16 on this card; do not force it to fp16 |
| `model weight dtype ...`                                                                   | `torch.float16` (**not** `bfloat16`)                                                                              |
| `Using ... attention`                                                                      | `split` or `pytorch`, **not** `sub quadratic`                                                                     |
| `Enabled pinned memory ...`                                                                | **must not appear**; if it does, patch 8 did not run                                                              |
| `[ZLUDA] ONNX Runtime: forcing CPUExecutionProvider`                                       | appears when an ONNX node runs, followed by its own `说明: ...` Chinese line; no `Failed to load ...onnxruntime_providers_cuda.dll` follows                     |
| `Unsupported Pytorch detected. DynamicVRAM support requires Pytorch version 2.8 or later.` | **Normal notice**: ComfyUI falls back to the legacy ModelPatcher; this is not an error                            |

> When reading the `VAE load device` line, look at the **first field** only: it is the run device of the VAE, and `cpu` there means "run the VAE on the CPU" is enabled. `offload device: cpu` is the default standby location (it only becomes `cuda:0` with `--gpu-only`) and is unrelated to CPU VAE.

## Troubleshooting

| Symptom (log keyword)                                                                                         | Real cause                                                                                                                                                                                             | Fix                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError: Model "..." is corrupt or invalid... Please download the model again`                            | The file is **missing or the path is wrong**, not corrupt. `comfy/utils.py` (lines 122-146 at the tested commit) reports _any_ exception from `safe_open()`, including FileNotFoundError, as "corrupt" | Check the path and file name; do not re-download blindly                                                                                                                     |
| `rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906` + `Fatal Python error: Aborted` | `ROCBLAS_TENSILE_LIBPATH` points to a directory without the gfx906 kernels (typically the launcher's `%TEMP%\<GUID>` folder was cleaned or rotated)                                                    | Patch item 2 points it at the HIP SDK's `bin\rocblas\library`                                                                                                                |
| `CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmulAlgoGetHeuristic(...)`                    | A linear layer / `addmm` with a bias took the cuBLASLt fused path                                                                                                                                      | Patch item 5; run the [self-check](#2-one-shot-self-check-does-not-start-the-ui) to confirm `F.linear` is `_linear`                                                          |
| `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`                                                                    | cuDNN is not disabled, **or** the `comfy.ops` attention priority list forces FLASH / cuDNN anyway (patch 6)                                                                                            | Patch item 3; if it persists, check `sdpa=math-only` in the status line and grep that file for `FAILED`                                                                      |
| `RuntimeError: Found no NVIDIA driver on your system`                                                         | The ZLUDA DLLs were **not preloaded before `import torch`** (for example torch was imported manually first, or the launcher's injection failed)                                                        | Patch item 1; make sure the startup log shows `loaded=5/5`                                                                                                                   |
| `ERROR: clip input is invalid: None`                                                                          | `Load Checkpoint` was used on a model that has **no text encoder** (such as anima-base)                                                                                                                | Use `Load Diffusion Model` + `Load CLIP` + `Load VAE`, or the official template                                                                                              |
| Out of VRAM / OOM                                                                                             | A large model on a 16 GB card                                                                                                                                                                          | Lower the resolution, use `--fp8_e4m3fn-unet` or `--reserve-vram 1`. **Avoid** `--lowvram` / `--novram` if the system has only 16 GB of RAM: swapping will stall the machine |
| Everything is reverted after "Install PyTorch / Install dependencies" in the launcher                         | The launcher reinstalls from its built-in manifest (it pins Python 3.11.6 for the ComfyUI core, and cu118 tops out at 2.3.1 there, which cannot be installed on Python 3.13)                           | Do not click "Install dependencies / Switch Torch version / Reinstall environment" in the launcher                                                                           |
| `CUBLAS_STATUS_NOT_SUPPORTED ... abType 3 cType 10 computeType 72`                                            | An `int8_tensorwise` model reached `torch._int_mm` (patch 7)                                                                                                                                           | Use a `convrot_w4a4` / `fp8` / `bf16` model, or accept dequantize-per-call                                                                                                   |
| Black image, healthy status line, `_math_only_sdpa` already active                                            | The VAE row is pinned to FP16 and the decoder overflows to NaN                                                                                                                                         | Set the launcher VAE precision row to Auto                                                                                                                                   |
| `onnxruntime` provider errors from RMBG / controlnet_aux / Impact                                             | CUDA 12 `onnxruntime-gpu` provider under ZLUDA (patch 10)                                                                                                                                              | Confirm the status line says `onnxruntime: CUDA/TensorRT providers filtered`                                                                                                 |

### Reverting to the stock torch

In the launcher's command prompt:

```bat
pip install --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" "torchaudio==2.11.0+cu130"

pip install "triton-windows==3.5.1.post24"

pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
```

After reverting, ZLUDA can no longer be used, and `main.py` should no longer call `zluda_patch.py` (remove the bootstrap block).

## Known limitations and ideas

1. **`torch.backends.cuda.preferred_blas_library(...)` has no effect on Windows** (torch itself warns `experimental feature. It is not supported on Windows`). Do not expect to switch the BLAS backend that way; the only option is to work around it at the Python level, as the patch does.
2. **Do not rename or delete `torch\lib\cublasLt64_11.dll`.** `torch_cuda.dll` references it statically, so renaming it makes `import torch` fail (confirmed by testing).
3. **Precision:** with patch 9 in place `Auto` no longer selects bf16 on the three model rows, so it is a safe setting; an explicit `--bf16-unet` / `--bf16-text-enc` flag still wins. Leave the **VAE** row at Auto — see the VAE overflow note in [Precision and performance](#precision-and-performance-measured).
4. **Performance:** both the attention backend and the precision have a large effect on speed; see [Precision and performance](#precision-and-performance-measured). Lowering the resolution also helps.
5. **Extending the patch:** if another operator fails with `CUBLAS_STATUS_NOT_SUPPORTED`, add a workaround in `patch_torch()` (for example replace `F.conv2d` with an equivalent `unfold + matmul`, which was tested to work).
6. **Launcher overwrites:** a launcher update of ComfyUI resets `main.py`, so re-insert the bootstrap block; `zluda_patch.py` may be wiped as well, so keep a backup.
7. **Diagnostics:** the launcher's `Diagnostics-*.log` (from its troubleshooting page, 疑难解答) is an environment snapshot plus the console buffer, and **contains the output of several runs**. To tell whether a log line comes from a new process, check the timestamps in `_zluda_status.txt`.
8. **The launcher is still required.** The patch preloads the ZLUDA DLLs, but it cannot make
   torch bind to them: measured on this machine, `cuInit(0)` returns 0 and
   `cuDeviceGetCount` reports one device through the preloaded DLLs, while
   `torch.cuda.is_available()` is still `False` in the same process. Only the launcher's
   injection does that. Consequences: `python main.py`, IDE debugging and any script that
   imports torch directly do not see the GPU.
9. **INT8 tensorwise models are usable, not fast.** Patch 7 dequantizes them, which costs
   24.4 ms per qkv call against 1.34 ms for a native fp16 GEMM. Prefer `convrot_w4a4`,
   `fp8` or `bf16` models on this card.
10. **ONNX Runtime always runs on the CPU.** That is intentional (patch 10): the installed
    `onnxruntime-gpu` is a CUDA 12 build. If you ever install a matching build and want to
    experiment, `ZLUDA_KEEP_ONNX_CUDA=1` leaves the providers alone.

## Environment switches

Every behaviour added after the original release can be turned off or tuned without
editing the patch. They are read once, at startup:

| Variable                     | Default | Effect                                                                      |
| ---------------------------- | ------- | --------------------------------------------------------------------------- |
| `ZLUDA_NO_ARG_OVERRIDES=1`   | off     | stop overriding launch arguments; the launcher owns them again              |
| `ZLUDA_RESERVE_VRAM=<GB>`    | `0.9`   | change the reserved VRAM; an explicit `--reserve-vram` still wins           |
| `ZLUDA_FORCE_FP16=1`         | off     | force fp16 for the UNet and text encoder (also overrides fp8-stored models) |
| `ZLUDA_SPLIT_ATTENTION=1`    | off     | use split attention instead of pytorch/math SDP                             |
| `ZLUDA_ALLOW_BF16=1`         | off     | let `should_use_bf16()` report normally again                               |
| `ZLUDA_VAE_FP32=1`           | off     | force the VAE to fp32 when the model allows it                              |
| `ZLUDA_KEEP_ONNX_CUDA=1`     | off     | leave ONNX Runtime's provider list untouched                                |
| `ZLUDA_INT8_DEQUANT_CACHE=1` | off     | cache dequantized INT8 weights (about 2x weight VRAM, 12x faster)           |
| `ZLUDA_TOPK_CPU=1`           | off     | run `torch.topk` on the CPU                                                 |
| `ZLUDA_STFT_CPU=1`           | off     | run `torch.stft` / `torch.istft` on the CPU (audio models)                  |
| `ZLUDA_NO_JITSCRIPT=1`       | off     | make `torch.jit.script` a no-op                                             |

## Repository layout

```
.
├── ComfyUI/
│   └── zluda_patch.py         # the compatibility patch (copy to ComfyUI\ in the package root)
├── LICENSE
├── README.md
└── README.zh-CN.md
```

Files involved in a deployed package (relative to the package root unless noted):

| Path                                                     | Description                                                                                                                       |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `python\python.exe`                                      | Python 3.13.11                                                                                                                    |
| `ComfyUI\main.py`                                        | **Bootstrap block inserted at the very top**                                                                                      |
| `ComfyUI\zluda_patch.py`                                 | The patch added by this setup (from this repository)                                                                              |
| `_zluda_status.txt`                                      | Self-check output of the patch (one line appended per start)                                                                      |
| `.launcher\zluda.<hash>.extpack`                         | The 5 ZLUDA DLLs (can be unpacked with 7-Zip)                                                                                     |
| `.launcher\tensile-gfx906.<hash>.extpack`                | rocBLAS Tensile kernels for gfx906 (alternative source)                                                                           |
| `.launcher\preference.json`                              | Launcher preferences (including proxy address and custom arguments)                                                               |
| `ComfyUI\blueprints\Text to Image (Anima Base 1.0).json` | Official Anima template                                                                                                           |
| `%HIP_PATH%\bin\rocblas\library` (absolute)              | gfx906 Tensile kernel directory that the patch points to (falls back to `C:\Program Files\AMD\ROCm\5.7` when `HIP_PATH` is unset) |
| `%TEMP%\<GUID>\` (absolute, new on every run)            | Where the launcher unpacks the ZLUDA DLLs; the patch scans and preloads from here                                                 |

If you use a different bundle version or HIP SDK version, re-check the version matching described in [Tested environment](#tested-environment).

## License

This project is licensed under the [MIT License](LICENSE).

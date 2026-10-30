# ComfyUI Aki Bundle on AMD MI50 (ZLUDA, gfx906, Windows)

[English](README.md) | [简体中文](README.zh-CN.md)

Run the **ComfyUI "Aki" all-in-one bundle** (秋叶整合包, started with the **绘世 / Hui Shi launcher**) on an **AMD Instinct MI50** (flashed with the **Radeon VII PRO vBIOS**) through [ZLUDA](https://github.com/vosen/ZLUDA) on Windows.

This repository documents a setup that was **verified end to end**: the exact commands, a small compatibility patch ([`ComfyUI\zluda_patch.py`](ComfyUI/zluda_patch.py)), why each part is needed, how to verify it, and a troubleshooting table. All conclusions come from hands-on testing on a single machine (command output, logs, file hashes), not from guesses.

## TL;DR

Three things are needed to run this bundle on ZLUDA:

1. **Replace torch with the cu118 build.** ZLUDA only provides CUDA 11 libraries, so a PyTorch built for CUDA 12+ cannot work.
2. **Put the patch `ComfyUI\zluda_patch.py` in place and hook it into the top of `ComfyUI\main.py`** with a short bootstrap block. The patch is organized as PART 0-3 with ten units: `PRELOAD`, `TORCH`, `SDPA`, `LINEAR`, `CLI_ARGS`, `MODEL_MANAGEMENT`, `ONNX`, `SUPPORTED_MODELS`, `OPS` and `MODEL_DETECTION`. The first four cover the failures that stop a run (ZLUDA DLLs / Tensile, cuDNN + SDP, the attention wrapper, cuBLASLt linear + INT8); the rest cover launch arguments, precision, ONNX Runtime, the LTXAV estimate, optional operator fallbacks and extra-block Anima tunes. See [How the patch works](#how-the-patch-works).
3. **Leave the four precision rows on **Let ComfyUI decide** (or pin the three model rows to FP16 for maximum speed, only for models that support fp16)**: on gfx906 bf16 runs at only 0.55x the fp16 speed, and fp32 doubles the weight memory. The patch's Auto policy picks fp16 when the model supports it, fp32 when it fits, otherwise bf16; the VAE follows its own `working_dtypes` (fp16 if listed, otherwise bf16; Flux stays bf16). Forcing Half Precision (FP16) on a VAE that does not list it can overflow to NaN and decode a black image. See [Precision and performance](#precision-and-performance-measured).

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

1. Double-click the launcher (`绘世启动器.exe`). In the left sidebar click **Advanced**, then click **Command Prompt** at the top right.
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

1. Copy all files of this repository into the package root, so that the patch ends up at `ComfyUI\zluda_patch.py`. Only [`ComfyUI\zluda_patch.py`](ComfyUI/zluda_patch.py) runs; `zluda_dtype_check.py` is the optional [offline self-check](#2-one-shot-self-check-does-not-start-the-ui), and `README*.md` / `LICENSE` are not needed.
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

### Step 3: Launcher options

The four settings below follow the launcher's order.

#### 1. Cross-Attention Optimization

| Option                                     | Verdict                                                                                                            |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Try xFormers (Recommended)                 | Not usable: xFormers is not installed in this bundle.                                                              |
| Use Sage Attention                         | Not usable: SageAttention is uninstalled.                                                                          |
| Use Scaled Dot-Product (SDP) Scheme        | Works, but the patch forces the math SDP backend; slower than Split.                                               |
| Use Sub-Quadratic Scheme                   | Works, but 2.71x slower than Split (measured in [Precision and performance](#precision-and-performance-measured)). |
| **Use Legacy (Split) Optimization Scheme** | **Use this.** Fastest of the options that work under ZLUDA.                                                        |

Select **Use Legacy (Split) Optimization Scheme**. Do **not** add `--use-split-cross-attention`; the dropdown is the supported path. The Split vs SubQuad numbers are in [Precision and performance](#precision-and-performance-measured).

#### 2. Precision Settings

| Row                                                                | Set to                 | Notes                                                                                      |
| ------------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------------------------------------ |
| Use Upcast Attention to Improve XAttn Precision (upcast-attention) | **Off**                | Enabling it upcasts attention to fp32, which is slower on this card                        |
| General Precision                                                  | **Let ComfyUI decide** | Recommended; Half Precision (FP16) is the max-speed override for models that support it    |
| Text Encoder Precision                                             | **Let ComfyUI decide** | The default is already fp16; setting Half Precision (FP16) is just insurance               |
| U-Net Precision                                                    | **Let ComfyUI decide** | The policy: fp16 if supported, fp32 if it fits, otherwise bf16                             |
| **VAE Precision**                                                  | **Let ComfyUI decide** | Follows the VAE's `working_dtypes`: fp16 if listed, else bf16. Flux `[bf16, fp32]` stays bf16; forcing Half Precision (FP16) on it can overflow |

With all four rows at **Let ComfyUI decide**, the patch picks the dtype per model: fp16 when the model's list contains it; for `[bfloat16, float32]` models (Lumina2, Z-Image, which leave fp16 out because it produces NaNs) fp32 when the fp32 weights fit the budget, otherwise bf16. The VAE follows its own list: fp16 if listed, otherwise bf16 (Flux stays bf16). Pinning the three model rows to **Half Precision (FP16)** is still the fastest option, but only for models that actually support fp16.

⚠️ **The VAE row trusts the VAE's `working_dtypes`.** A VAE that lists fp16 (e.g., `[bf16, fp16, fp32]`) uses fp16; Flux's `[bf16, fp32]` has no fp16, so it stays bf16. Forcing **Half Precision (FP16)** on a VAE that does not list it bypasses the list and can overflow to NaN/black (the measured case in [Precision and performance](#precision-and-performance-measured)). `ZLUDA_VAE_DTYPE=comfy` puts this row back on ComfyUI's original order, and `ZLUDA_ALLOW_BF16=1` does the same for both the VAE and the UNet policy. If you get a black image plus `RuntimeWarning: invalid value encountered in cast` with no `FAILED`, check this row; `comfy_ops: FAILED` instead points to the `SDPA` unit.

**Fallback:** if a future launcher stops forwarding the rows, write `--fp16-unet` (U-Net) and `--fp16-text-enc` (text encoder) under **Advanced → Other Settings → Custom Arguments**. Do **not** add `--fp16-vae` for a VAE that does not list fp16 (Flux); that is the overflow above.

⚠️ **Pitfall:** an explicit `--bf16-unet` / `--bf16-text-enc` wins over the automatic selection; don't set the three model rows to BF16. An explicit `--bf16-vae` is safe but unnecessary when the VAE already lists bf16.

✅ **The launcher does forward these rows.** Setting U-Net Precision to **Half Precision (FP16)** produces `model weight dtype torch.float16` and `--fp16-unet` on the echoed command line. The automatic path cannot be the source: for a `[bfloat16, float32]` model the policy returns fp32/bf16, never fp16; `PRIORITIZE_FP16` is off because `Fast` is off. (The launcher binary contains only its own option names, so this had to be measured.)

#### 3. Preview Image Generation Method

| Option              | Recommendation                                                                                            |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| **Disable Preview** | **Production / long runs.** No preview decode, no extra VRAM.                                             |
| Auto                | **= Latent2RGB in ComfyUI 0.30.2**; cheap, no extra model. Good default.                                  |
| Latent2RGB          | Same as Auto; cheap, no extra model.                                                                      |
| TAESD               | Better previews, but it loads a small model and runs a decode per preview; use only if you have headroom. |

#### 4. Custom Arguments

To change custom arguments, open **Settings**, switch **Configuration Mode** to **专家**, then look at the bottom of **Advanced → Other Settings**; **Custom Arguments** is there.

**You do not need to add `--disable-cuda-malloc`**: the `CLI_ARGS` unit sets `args.disable_cuda_malloc` itself, together with `disable_pinned_memory`, `disable_async_offload` and `reserve_vram`. There is nothing to write in **Custom Arguments**.

**Do not add** `--use-sage-attention`, `--use-flash-attention`, `--enable-triton-backend`, `--fast` / `--fast cublas_ops` (this path calls cuBLASLt directly and always crashes), or `--cuda-malloc`.

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

[`ComfyUI\zluda_patch.py`](ComfyUI/zluda_patch.py) is organized as PART 0-3: **PART 0** contract (paths and switches), **PART 1** toolkit (reporting, env parsing, one generic import-hook loader/finder), **PART 2** the ten patch units, and **PART 3** assembly (`MODULE_PATCHES`, `TORCH_STEPS`, `_finish_torch()`, `_bootstrap()`). The ten sections below follow the file order; each section is one unit and uses the same four labels: **Symptom** (what the user sees), **Reason** (why it happens), **Fix** (what the patch does) and **Cost** (whether there is a downside). The unit's own header in the code adds a fifth label, **Failure**, for the status string.

| #   | Unit               |
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

The first four units cover the failures that stop a run; the remaining six cover launch arguments, precision, ONNX Runtime, the LTXAV estimate, optional operator fallbacks and extra-block tunes.

### 1. `PRELOAD`

**What it does:** loads the ZLUDA DLLs and points rocBLAS at the gfx906 Tensile kernels, both before torch is imported.

- **Symptom:** `RuntimeError: Found no NVIDIA driver on your system`, or `rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906` followed by `Fatal Python error: Aborted`.

- **Reason:** ZLUDA's `nvcuda.dll` must be loaded **before** torch initializes CUDA; the launcher normally does this itself (it unpacks the DLLs into `%TEMP%\<GUID>` on every run), but its injection was not reliable in this setup. ZLUDA's cuBLAS then forwards matrix multiplications to **rocBLAS**, which needs the Tensile kernels of the **matching architecture (gfx906)**; the launcher points `ROCBLAS_TENSILE_LIBPATH` at a `%TEMP%` folder that does not contain them.

- **Fix:** the patch scans `%TEMP%`, `%TMP%` and `%LOCALAPPDATA%\Temp` (and `zluda\` in the package root) for a folder containing `cublas64_11.dll`; adds the ZLUDA folder and `%HIP_PATH%\bin` to `PATH` / `os.add_dll_directory`; preloads `nvcuda.dll`, `nvml.dll`, `cublas64_11.dll`, `cusparse64_11.dll`, `nvrtc64_112_0.dll`; and sets `ROCBLAS_TENSILE_LIBPATH` to `<HIP SDK>\bin\rocblas\library`, reading `HIP_PATH` first and falling back to `C:\Program Files\AMD\ROCm\5.7` from the tested setup.

- **Cost:** none. It only scans a few temp folders at startup.

### 2. `TORCH`

**What it does:** turns cuDNN off and leaves only the math SDP backend at the torch level.

- **Symptom:** `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`, with the VAE's convolutions hit first; or attention landing on flash / mem-efficient kernels that do not exist under ZLUDA.

- **Reason:** ZLUDA has no cuDNN and no flash kernels; the launcher's own ZLUDA compatibility patch does the same.

- **Fix:** `torch.backends.cudnn.enabled = False`; `enable_flash_sdp(False)`, `enable_mem_efficient_sdp(False)`, `enable_math_sdp(True)`.

- **Cost:** none. FLASH and cuDNN never existed here, so the call always fell back to math; the `SDPA` unit handles the `set_priority=True` path that can put FLASH / cuDNN back in front.

### 3. `SDPA`

**What it does:** wraps `F.sdpa` and `comfy.ops.sdpa` so every attention call runs on the math SDP backend.

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
  **overrides** the `enable_flash_sdp(False)` from the `TORCH` unit, so PyTorch walks that list no
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

- **Fix:** `comfy/ops.py` installs its own wrapper around
  `torch.nn.functional.scaled_dot_product_attention` with
  `SDPA_BACKEND_PRIORITY = [FLASH_ATTENTION, CUDNN_ATTENTION, EFFICIENT_ATTENTION, MATH]` and
  `set_priority=True` (lines 56-93 in 0.30.2), so the `enable_*` flags from the `TORCH` unit are
  ignored and every call walks that list. Patching `F` alone does nothing:
  `comfy.ops` binds the real function at import time. The patch wraps both the
  `F` function and `comfy.ops`'s own reference, so every call runs inside
  `sdpa_kernel([MATH])`. `comfy.ops` is imported after the bootstrap block, so the patch hooks
  its import. It also forces the `can_use_flash_attention` / `can_use_cudnn_attention` /
  `can_use_efficient_attention` probes to `False` **when a build has them** — ComfyUI 0.30.2 has
  none of the three (verified: the checks live in `torch.backends.cuda`, called from
  `comfy/ops.py:86-88`, and the priority list above is what actually runs), so on this version
  that part is a no-op and the wrapper is what does the work.

  Field gotcha: the first version re-imported `comfy.ops` inside its own `exec_module`,
  which raises `AttributeError: cannot access submodule 'ops' of module 'comfy' (most likely
due to a circular import)`. The `comfy.ops` half stayed unpatched and the images stayed
  black. The current layout gives `comfy.ops` its own `comfy_ops:` status line, so a failure
  shows up as `comfy_ops: FAILED ...` instead of hiding in the torch line; still grep the
  whole status file for `FAILED` rather than trusting a healthy-looking line.

- **Cost:** none. FLASH and cuDNN never existed here, so the call always fell back to math.
  At S = 4096, batch×heads = 16, D = 128: `attention_pytorch` 39.1 ms, math SDP 39.1 ms,
  split 26.7 ms. This is a correctness fix; use split attention if you want speed, see
  [Attention backend](#precision-and-performance-measured).

### 4. `LINEAR`

**What it does:** splits biased `F.linear` / `addmm` into `matmul + add`, and dequantizes INT8 tensorwise weights before `F.linear`.

- **Symptom:** `CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul ...`; for an INT8 tensorwise model the same call carries `abType 3 cType 10 computeType 72` (INT8 × INT8 → INT32).

- **Reason:** PyTorch runs `F.linear(x, w, b)` / `addmm` through a **bias-fused GEMM that only exists in cuBLASLt** (`gemm_and_bias`). ZLUDA has **no cuBLASLt implementation at all** (none of its 5 DLLs exports a `cublasLt*` symbol), so torch falls back to NVIDIA's genuine `torch\lib\cublasLt64_11.dll` (544 MB), which necessarily returns `CUBLAS_STATUS_NOT_SUPPORTED`. Measured: `mm` / `bmm` / `baddbmm` without bias all work; **only linear / addmm with a bias crash**, regardless of precision (fp16 / bf16 / fp32). INT8 tensorwise is the same story one layer down: `comfy_kitchen`'s `TensorWiseINT8Layout` handler ends in `torch._int_mm`, a cuBLASLt INT8 GEMM that ZLUDA does not implement. The compute dtype is **not** the problem: `int8_tensorwise` sets `quantize_input: False`, so the weight reaches `F.linear` as a `QuantizedTensor` and is never dequantized — `--fp16-unet` cannot help.

- **Fix:** the biased path is split into `matmul` + add; `F.linear` also dequantizes a `QuantizedTensor` weight to the input dtype. `torch.addmm` gets the bias split only. The dequantized weight can be cached per weight object with `ZLUDA_INT8_DEQUANT_CACHE=1` (off by default).

  Read the `comfy_quant` JSON blobs in the safetensors header, not the file name, to tell whether a model is affected:

  | `comfy_quant`                        | Affected?                              |
  | ------------------------------------ | -------------------------------------- |
  | `{"format": "int8_tensorwise", ...}` | **Yes**, it reaches `torch._int_mm`    |
  | `{"format": "convrot_w4a4", ...}`    | No, it uses comfy_kitchen's own kernel |
  | none (plain fp16 / bf16 / fp8)       | No                                     |

  A file named `..._int8_convrot` can still carry `format: int8_tensorwise`; the `convrot` flag alone does not move it off the `torch._int_mm` path.

  Measured on the 11520×3840 qkv layer of `z_image_turbo_int8_convrot` (m = 90, fp16 input):

  | Approach                        | Per call |
  | ------------------------------- | -------- |
  | no handling                     | crash    |
  | dequantize every call (default) | 24.4 ms  |
  | dequantize once + cache         | 2.0 ms   |
  | native fp16 GEMM, same shape    | 1.34 ms  |

  The cache is **off by default** because it keeps the original INT8 weights and adds a full fp16 copy: for a 6.14 B model, 5.72 GB of INT8 plus 11.44 GB of fp16 is 17.16 GB total, about 1.5x the fp16 model (3x the INT8 weights), which does not fit on a 16 GB card. Set `ZLUDA_INT8_DEQUANT_CACHE=1` only when there is spare VRAM. The cache is keyed with `weakref.WeakKeyDictionary`, so it is released together with the model.

- **Cost:** usable, not fast. Prefer, in order: a real `convrot_w4a4` model (comfy_kitchen's own kernel), an `fp8` model (its handler already falls back to dequantization on failure), then `bf16` — but watch the 16 GB of system RAM with large bf16 models.

### 5. `CLI_ARGS`

**What it does:** overrides the launch arguments right after `comfy.cli_args` parses.

- **Symptom:** `Enabled pinned memory ...` shows up in the startup log (the launch arguments did not take effect).

- **Reason:** the launcher keeps its options in `.launcher\preference.json`: a launcher reset drops them, and any entry point that skips `main.py` never sets them at all.

- **Fix:** `comfy/cli_args.py` parses at import time and every consumer (`cuda_malloc.py`, `comfy/model_management.py`, `comfy/ldm/modules/attention.py`) reads `args` afterwards, so the patch hooks that import and overrides the object once, right after it is built:

  | Argument                  | Why                                                                                                        |
  | ------------------------- | ---------------------------------------------------------------------------------------------------------- |
  | `--disable-pinned-memory` | Windows pins up to 40% of RAM; this machine has 16 GB, and pinned buffers are a frequent source of crashes |
  | `--disable-async-offload` | torch 2.7 falls back to the legacy ModelPatcher, so async offload buys nothing                             |
  | `--disable-cuda-malloc`   | ZLUDA has no cudaMallocAsync; the patch sets `args.disable_cuda_malloc` itself                             |
  | `--reserve-vram 1.5`      | keeps the OS and other programs from fighting the card, and leaves the VAE decode room for its peak        |

  An explicit `--reserve-vram` from the launcher wins, so a value you set by hand is not overwritten. Switches: `ZLUDA_NO_ARG_OVERRIDES=1` disables the whole group, `ZLUDA_RESERVE_VRAM=<GB>` changes the reserve, `ZLUDA_FORCE_FP16=1` also forces fp16 for fp8-stored models (off by default for that reason), `ZLUDA_SPLIT_ATTENTION=1` selects split attention.

  **Why 1.5 GB, measured.** The reserve is the part the models may not take, and it trades directly against how much of the UNet stays in VRAM:

  | Reserve    | VAE decode                                    | Z-Image UNet                                                        | Notes                                                                                               |
  | ---------- | --------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
  | 0.9 GB     | fell back to tiled decoding                   | fully loaded                                                        | the decode failed, and that failed allocation left the driver's free-VRAM report at 0 until restart |
  | 2.5 GB     | fine                                          | **partial offload: 201–712 MB streamed from system RAM every step** | safe but costs VRAM residency                                                                       |
  | **1.5 GB** | fine (the VAE is bf16 now, half its old peak) | fully loaded                                                        | the measured middle                                                                                 |

  Reproduce the check with the workflow in question: the startup log should show `loaded completely` for the UNet (not `loaded partially`), and no `Ran out of memory when regular VAE decoding`.

- **Cost:** none.

### 6. `MODEL_MANAGEMENT`

**What it does:** picks the UNet compute dtype: fp16 when the model supports it, fp32 when it fits, otherwise bf16. The VAE follows its own `working_dtypes`: fp16 if listed, otherwise bf16.

- **Symptom:** `model weight dtype torch.bfloat16` on a bf16-first model (Anima) even though fp16 works, or a `[bfloat16, float32]` model forced to `torch.float32` when it does not fit and OOMs.

- **Reason:** `unet_dtype()` walks `supported_inference_dtypes` **in the model's order**, and ZLUDA reports (8, 8), so `should_use_bf16()` is `True` before it asks the device. A model that lists bf16 first (Anima Base 1.0 is `[bfloat16, float16, float32]`) therefore picks bf16 even though fp16 is also supported; a `[bfloat16, float32]` model picks bf16, which is only 0.55x fp16 speed. A blanket ban fixes Anima but pushes `[bfloat16, float32]` models to fp32, which doubles the weight memory.

- **Fix:** the patch wraps `comfy.model_management.unet_dtype()`. Explicit `--fp16-unet` / `--bf16-unet` / `--fp32-unet` flags still win; otherwise, if the model's list contains fp16 it returns fp16; if the list is only bf16 + fp32 it returns fp32 when `model_params * 4` fits the weight budget, otherwise bf16. The budget is `maximum_vram_for_weights()` (total VRAM × 0.88 − 0.8 GiB − the reserve) multiplied by `ZLUDA_FP32_MARGIN` (default 0.9). On the tested card that is 11.77 GiB × 0.9 = **10.6 GiB**, so only models up to about **2.8 B parameters** can take fp32. The margin is not decoration: at fp32 the activations are 4 bytes as well, while the budget only counts weights. Each extra GB of `--reserve-vram` lowers the ceiling by 0.9 GiB — at 2.5 GB it is 9.7 GiB (2.6 B parameters), at 0.9 GB it is 11.1 GiB (3.0 B).

  It also wraps `unet_manual_cast()`. ComfyUI resolves the manual cast of a fp8 or unknown weight store to bf16, which is the 0.55x path on gfx906; when the model lists fp16 and the device reports fp16, the patch returns fp16 instead. bf16 weights return `None` upstream and are untouched, and a pinned `--bf16-unet` is left alone.

  The VAE is handled separately: `vae_dtype()` trusts the VAE's `working_dtypes`: fp16 if listed, otherwise bf16; explicit `--fp16-vae` / `--bf16-vae` / `--fp32-vae` still win. Flux's `[bf16, fp32]` stays bf16.

  Switches: `ZLUDA_UNET_DTYPE` and `ZLUDA_VAE_DTYPE` (`auto` by default) send **one side** back to ComfyUI's order; `ZLUDA_ALLOW_BF16=1` is the coarse alias that sets both sides to `comfy`, so `[bf16, fp16, ...]` goes back to bf16. An explicit side wins over the alias, so `ZLUDA_ALLOW_BF16=1 ZLUDA_UNET_DTYPE=auto` keeps the UNet policy and drops only the VAE one. If the policy itself raises, that one call is handed back to ComfyUI and reported as `FAILED`, so the patch cannot break a model load.

  **Log:** every model load prints one `[ZLUDA] model_management: unet ...` line and every VAE load one `... vae ...` line. The line names the dtype and the reason (`policy:` for the rules above, `comfy order:` when ComfyUI's own function decided), appends `comfy would pick ...` when the policy changed the answer, and prefixes `weights uint8` / `weights unknown` when the checkpoint's storage dtype is not a float (ComfyUI's own `model weight dtype` line is the **inference** dtype, not the storage). A manual-cast correction prints its own `... manual cast float16 (policy: upstream bf16, ...)` line. The startup line also lists the precision flags the launcher pinned (`flags=...`): a pinned row makes the policy step aside, so `flags=fp16_unet` means the policy is installed but was not consulted for the UNet row. A launcher **General Precision = Half Precision (FP16)** also lands there: ComfyUI turns `--force-fp16` into `args.fp16_unet = True`, so that row has to stay on **Let ComfyUI decide** if you want the policy to choose.

- **Cost:** none by default. When a `[bfloat16, float32]` model fits in fp32, it uses 2x the weight memory of bf16 — that is the trade-off for its 0.95x speed; otherwise it stays on bf16. VAE fp16 and bf16 are both 2 bytes, so the VAE choice costs no memory.

### 7. `ONNX`

**What it does:** removes the CUDA / TensorRT execution providers from ONNX Runtime.

- **Symptom:** onnxruntime provider errors from RMBG / controlnet_aux / Impact (the provider fails to load or crashes inside the fake driver).

- **Reason:** `controlnet_aux` (DW Pose, lineart, depth), RMBG and Impact's detectors run their models through ONNX Runtime, whose default provider list puts `CUDAExecutionProvider` first. The installed `onnxruntime-gpu` is a CUDA 12 build while ZLUDA only impersonates CUDA 11.

- **Fix:** the patch filters `CUDAExecutionProvider` / `TensorrtExecutionProvider` out of `get_available_providers()` (both the module-level function and the pybind copy the session path reads) and rewrites any session that still asks for CUDA to `CPUExecutionProvider`. Switch: `ZLUDA_KEEP_ONNX_CUDA=1` leaves ONNX Runtime untouched.

- **Cost:** none. Those models are small, so the CPU cost is not noticeable; the alternative is a crash.

### 8. `SUPPORTED_MODELS`

**What it does:** fixes LTXAV's `memory_usage_factor`.

- **Symptom:** LTXAV runs out of VRAM on a 16 GB card (ComfyUI plans as if it needs almost nothing).

- **Reason:** upstream leaves LTXAV's `memory_usage_factor` at `0.077`, and `model_base.py` multiplies the activation estimate with it. The parent `LTXV` class uses `(cross_attention_dim / 2048) * 5.5`.

- **Fix:** the patch reads both numbers off the parent class (`memory_usage_factor` = 5.5, default `cross_attention_dim` = 2048) and applies the result to LTXAV with a 1.5x margin, so the two formulas cannot drift apart (e.g. `cross_attention_dim` 4096 → 16.5, 2048 → 8.25).

- **Cost:** ComfyUI may offload earlier, but it does not OOM.

### 9. `OPS`

**What it does:** optional CPU fallbacks for `torch.topk`, `torch.stft` / `torch.istft` and `torch.jit.script`.

- **Symptom:** the Zluda fork replaces `torch.topk`, `torch.stft` / `torch.istft` and `torch.jit.script` unconditionally, so every `topk` becomes a D2H + `topk` + H2D round trip and torchvision's NMS calls `topk` internally: detection nodes slow down drastically.

- **Reason:** nothing in this setup has been shown to need those fallbacks.

- **Fix:** they are off by default and enabled one at a time: `ZLUDA_TOPK_CPU=1`, `ZLUDA_STFT_CPU=1` (audio models), `ZLUDA_NO_JITSCRIPT=1` (node packs that ship TorchScript models).

- **Cost:** enabling one makes it slower, see the round-trip cost of `topk` above.

### 10. `MODEL_DETECTION`

**What it does:** counts Anima / Cosmos-Predict2 blocks from the checkpoint.

- **Symptom:** the image comes out **completely black** with no error, and the log carries one
  `unet unexpected: [...]` warning listing `blocks.N...` beyond the depth ComfyUI built, plus
  `RuntimeWarning: invalid value encountered in cast` from the image save path.

- **Reason:** ComfyUI 0.30.2 derives the block count of Anima / Cosmos-Predict2 checkpoints from
  `model_channels` alone — 2048 → 28 blocks, 5120 → 36 (`comfy/model_detection.py:832-837`). Community tunes
  exist with more blocks than that: measured on this machine,
  `silvermoonmixAnimaEvolved_v2329BTurbo.safetensors` is a **40-block, BF16** Anima tune (928 tensors,
  `net.blocks.0 … net.blocks.39`, plus a learnable `net.pos_embedder`). ComfyUI built a 28-block model, so blocks
  28-39 (240 keys) and the pos embedder were dropped silently and the sampled latents came out NaN — the black image
  is the `np.clip(...).astype(np.uint8)` of NaN.

- **Fix:** the patch counts the blocks in the checkpoint itself with
  `count_blocks(state_dict, key_prefix + "blocks.{}.")` and overrides `num_blocks` for `anima` /
  `cosmos_predict2` whenever the count differs, reporting
  `model_detection: anima num_blocks 28 -> 40 (counted in the checkpoint)`. Upstream fixed the same thing in `725e6ec60`
  ("Support anima tunes with extra blocks", PR #15555), so on a ComfyUI that already counts this is a no-op.
  Switch: `ZLUDA_ANIMA_EXTRA_BLOCKS=0` (on by default, see the switch block at the top).

- **Cost:** none for checkpoints whose depth matches the table; the counted value is what the checkpoint itself
  contains. The 3 `pos_embedder.*` keys stay unexpected exactly as they do with the upstream fix — the built model uses
  `rope3d` positional encoding, so a learnable pos embedder is not part of it.

`SDPA`, `CLI_ARGS`, `MODEL_MANAGEMENT`, `ONNX`, `SUPPORTED_MODELS` and `MODEL_DETECTION` are
module-level units: they are registered in `MODULE_PATCHES` (module name → unit) and applied
through a single `sys.meta_path` hook right after the module executes. The hooks are installed
**before** `import torch`, so the fallback path where `torch` loads late still covers
`comfy.ops`. `OPS` is applied from `_finish_torch()`, on both the direct and the late path.
The bootstrap at the bottom of the file is the only code that runs at import time; everything
above it is definitions.

## Precision and performance (measured)

All numbers below are **measured on the test machine** (MI50 flashed with the Radeon VII PRO vBIOS / gfx906 / HIP SDK 5.7.1 / torch 2.7.1+cu118 / ZLUDA).

### 1. Why FP16 is the fastest choice

ZLUDA impersonates an NVIDIA card on this machine, which defeats a rule in ComfyUI that exists **specifically for gfx906**:

- Under ZLUDA `torch.version.hip` is `None` and `torch.version.cuda` is `11.8`, so `is_amd()` returns False (it requires `torch.version.hip`).
- Therefore the "RDNA2 and older don't support bf16" branch in `should_use_bf16()` is skipped, even though that architecture list **explicitly contains `gfx906`**.
- At the same time ZLUDA reports compute capability **(8, 8)** and `torch.cuda.is_bf16_supported()` returns True, so `should_use_bf16()` ends up returning True.

| Precision | Measured    | Relative to fp16 |
| --------- | ----------- | ---------------- |
| fp16      | 9.54 TFLOPS | 1.00×            |
| bf16      | 5.26 TFLOPS | 0.55×            |
| fp32      | 9.10 TFLOPS | 0.95×            |

(4096×4096 matrix multiplication.)

Conclusion: **fp16 is the fastest; bf16 is 0.55x; fp32 is 0.95x but needs 2x the weight memory.** Step 3's Auto policy uses this: fp16 where supported, fp32 only when it fits, otherwise bf16.

### 2. Why the VAE row is special

Step 3 keeps the VAE on **Let ComfyUI decide**, which trusts the VAE's `working_dtypes`: fp16 if listed, otherwise bf16.

Flux's `ae.safetensors` is `[bf16, fp32]` — it deliberately excludes fp16. Forcing **Half Precision (FP16)** on it bypasses the list: the decoder's 5-bit exponent cannot hold the VAE's activations, they overflow to NaN, and the image comes out black. Field check: switching only this row from **Half Precision (FP16)** to **Let ComfyUI decide** made generation work, with everything else unchanged.

For a VAE that lists fp16 (for example `[bf16, fp16, fp32]`), Auto uses fp16. Test at the target resolution: `working_dtypes` is architecture-level, not resolution-level. fp16 and bf16 are both 2 bytes, so there is no memory difference; fp16 is faster on gfx906.

`ZLUDA_VAE_DTYPE=comfy` (or `ZLUDA_ALLOW_BF16=1`, which does it for both sides) puts the VAE back on ComfyUI's original order, so a `[bf16, fp16, ...]` VAE goes back to bf16.

### 3. Attention backend performance: Split vs SubQuad

Why SubQuad is slow: its internal `kv_chunk_size` defaults to `√T`. For a 1024×1024 image the self-attention is T=4096, so it splits one large matrix multiplication into **64 small chunks**, each doing its own baddbmm + softmax; Split does a single large baddbmm.

| Implementation                 | Time     | Relative |
| ------------------------------ | -------- | -------- |
| Split-like (one large baddbmm) | 23.34 ms | 1.00×    |
| SubQuad-like (64 chunks)       | 63.22 ms | 2.71×    |

(batch×heads=16, T=4096, D=128, fp16. This is a simplified reproduction; the real implementation also does masking and online softmax rescaling per chunk, so the real gap is larger.)

Check the log for which mode is active:

- `Using sub quadratic optimization for attention` → SubQuad (slow)
- `Using split optimization for attention` → Split
- `Using pytorch attention` → PyTorch SDP

### 4. Quantized models and precision flags

| Model type                  | Weight storage   | Weight VRAM vs fp16  | Compute / extra VRAM                                      | Recommendation                                                               |
| --------------------------- | ---------------- | -------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------- |
| GGUF Q4 / Q5 / Q6 / Q8      | quantized blocks | ~0.3–0.5x            | dequantized to fp16 per layer; one transient layer buffer | most VRAM-efficient; Let ComfyUI decide (Half Precision FP16 only if listed) |
| fp8-stored                  | 1 byte/param     | ~0.5x                | fp8 or dequantized to fp16; small buffers                 | good; Let ComfyUI decide (Half Precision FP16 only if listed)                |
| `int8_tensorwise`, no cache | 1 byte/param     | ~0.5x                | dequantized per call; transient fp16 copy; slow           | only if no alternative                                                       |
| `int8_tensorwise`, cache    | 1 byte + fp16    | ~1.5x fp16 (3x INT8) | dequantized once, then persistent                         | usually does not fit 16 GB                                                   |
| `convrot_w4a4`              | ~4.5 bit/param   | ~0.3x                | comfy_kitchen kernel, no dequantization                   | best of the INT8-family                                                      |
| torchao / bitsandbytes INT8 | own format       | depends              | own kernels                                               | generally unaffected                                                         |
| fp16                        | 2 bytes/param    | 1x                   | native                                                    | baseline                                                                     |
| fp32                        | 4 bytes/param    | 2x                   | native                                                    | avoid on 16 GB                                                               |

For the tested 6.14 B model: INT8 5.72 GB, fp16 11.44 GB, fp32 22.88 GB; `int8_tensorwise` with the cache is 5.72 + 11.44 = 17.16 GB, i.e. ~1.5x fp16 / 3x INT8, so the cache is off by default (`ZLUDA_INT8_DEQUANT_CACHE=1` only with spare VRAM).

- GGUF mechanism: in `cast_bias_weight()` the compute dtype follows the input activation dtype (i.e. the model inference dtype), which is then handed to `dequantize_tensor()`. So pinning FP16 makes quantized models dequantize to fp16 instead of bf16. The weights stay quantized in storage, so nothing becomes blurry or slower.
- To override a single GGUF model, use the `dequant_dtype` / `patch_dtype` dropdowns of the `UnetLoaderGGUFAdvanced` node (`default / target / float32 / float16 / bfloat16`); this does not affect the global setting.
- fp8-stored models change the numeric path, so run the same workflow twice and compare before deciding.
- `comfy_kitchen int8_tensorwise` is the case handled by the `LINEAR` unit; torchao / bitsandbytes INT8 uses its own kernels and is not affected by `--fp16-unet`.

## Verification

### 1. Startup log

These lines should appear (`<TEMP>` is your temp folder, usually `%LOCALAPPDATA%\Temp`):

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

`model_management:` also prints one decision line per model load and per VAE load. With both the
model and the VAE listing `[bfloat16, float16, float32]` (Anima + Wan):

```
[ZLUDA] model_management: unet float16 (policy: model lists fp16; comfy would pick bfloat16)
[ZLUDA] model_management: vae float16 (policy: fp16 in [bfloat16, float16, float32]; comfy would pick bfloat16)
```

A `[bfloat16, float32]` model instead prints the number pair the decision used, for example
`unet bfloat16 (policy: fp32 29.8 GiB > 10.6 GiB)` or
`unet float32 (policy: fp32 7.5 GiB <= 10.6 GiB)` (weights needed vs budget after the margin).
When the caller does not report a single storage dtype, or reports a quantized one, the reason says
so: `unet bfloat16 (policy: weights uint8, fp32 22.4 GiB > 10.6 GiB)`.
A launcher row pinned to FP16 turns the line into `... (comfy order: --fp16-unet)`: the explicit
flag wins and the policy is not consulted. The same two shapes cover the VAE (`--fp16-vae`,
`no fp16 in [...]`, `list [float32]`). On a launcher run the pinned rows replace `flags=none`,
e.g. `flags=fp16_unet,fp16_text_enc`.

A manual-cast correction prints its own line, once per model load:

```
[ZLUDA] model_management: manual cast float16 (policy: upstream bf16, weights float8_e4m3fn)
```

`weights unknown` there means the loader did not report a storage dtype (a quantized checkpoint);
`weights float8_e4m3fn` / `weights uint8` name what it did report.

The startup lines and every decision line are appended to `_zluda_status.txt` in the package root (and to `zluda_status.txt` in the temp folder).

> **Output convention:** status lines are plain English — they are greppable, they survive any
> console code page, and this document quotes them verbatim. A message that needs a human decision is
> followed by its own `[ZLUDA] 说明: ...` line in Chinese instead of being appended to the status line.
> The Chinese line may show up as mojibake in some consoles; the English line is unaffected, and
> `_zluda_status.txt` is UTF-8 and always the most reliable place to look. Decision lines
> (`unet ...` / `vae ...`) are records, not warnings, so they never carry a `说明` line.

`switches:` is printed first, before any patch work. `ops:` appears immediately after the
torch line; `hooks:` follows it, before the first `comfy.*` import. `args:` appears when
`comfy.cli_args` is parsed; `model_management:`, `comfy_ops:` and `supported_models:` during
startup (and `model_management:` again for every model / VAE load); `model_detection:` when the
module is patched, and again only if a checkpoint with a different block count is detected;
`onnxruntime:` only when something imports ONNX Runtime
(an RMBG / controlnet_aux / Impact node, not at startup). `ops: default` means no optional
operator fallback is enabled, which is the intended state; on the fallback hook path it is
printed after `Late torch=` as well (`OPS` is applied there too).

A healthy torch line is **not** proof that everything worked: the `comfy.ops` half has its
own `comfy_ops:` line, and it can say `FAILED` while the torch line still says
`sdpa=math-only`. Always grep the whole status file for `FAILED`.

If any of the lines below appear, the patch is not fully active (`failed=` lists at most 3 entries).
Each of them is followed by its own `说明: ...` Chinese line.

| Log line                                                         | Meaning                                                                                                                                                                   | Fix                                                                                                                                                                                                             |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[ZLUDA] Preload: No ZLUDA dir found (nvcuda.dll not preloaded)` | No folder with the ZLUDA DLLs was found, so `nvcuda.dll` was not preloaded                                                                                                | Let the launcher run ComfyUI once (it unpacks the DLLs into `%TEMP%\<GUID>`), or unpack `.launcher\zluda.<hash>.extpack` into `zluda\` in the package root with 7-Zip                                           |
| `[ZLUDA] Preload: zluda_dir=... \| loaded=3/5 \| failed=...`     | Some DLLs failed to load                                                                                                                                                  | Look at what `failed=` lists: `xxx.dll:missing` means the file is absent; `WinError 193` usually means a placeholder or HTML file instead of a real DLL                                                         |
| `[ZLUDA] Direct import failed: ...`                              | The patch raised while importing torch                                                                                                                                    | `Fallback hook installed` should follow; if it does not, even the fallback hook failed to install                                                                                                               |
| `[ZLUDA] Fallback hook installed`                                | The import hook is installed and will patch torch once it loads                                                                                                           | `Late torch=...` should appear after the next `import torch`                                                                                                                                                    |
| `[ZLUDA] Late torch=...`                                         | The fallback path patched torch successfully                                                                                                                              | Confirm the line contains `Tensile=`, `cudnn=False`, `sdp=math` and `linear-patch[F.linear=_linear]`                                                                                                            |
| `sdpa=missing` in the status line                                | This torch has no `F.scaled_dot_product_attention`, so math-only was not wrapped and attention falls back to the crashing FLASH / cuDNN priority list                     | Check the torch version (2.0 or newer is required); this appears inside the `OK torch=` status line, not on its own line                                                                                        |
| `[ZLUDA] comfy_ops: FAILED ...`                                  | The `comfy.ops` math-only wrapper failed, so attention falls back to the priority list that crashes                                                                       | Check whether another extension imported `comfy.ops` first, or whether this `comfy` has no `scaled_dot_product_attention` at all (too old); the torch line can still say `sdpa=math-only`, so grep for `FAILED` |
| `[ZLUDA] <key>: FAILED ...`                                      | A module import hook raised, so that module is unpatched. `<key>` is one of `args`, `model_management`, `model_detection`, `supported_models`, `comfy_ops`, `onnxruntime` | Look at the key; a failure outside `comfy_ops` usually only affects that feature                                                                                                                                |
| `[ZLUDA] Patch FAILED: ...`                                      | The bootstrap block at the top of `main.py` itself raised                                                                                                                 | Check that the block sits before the first line of `main.py` and that `zluda_patch.py` is inside `ComfyUI\`                                                                                                     |

### 2. One-shot self-check (does not start the UI)

The quickest check is the offline script in this repository. From the package root:

```bat
python zluda_dtype_check.py
```

Expected `39 passed, 0 failed`. It loads `ComfyUI\zluda_patch.py` **without** running its bootstrap
(no DLL preload, no torch, no status file), drives `_apply_model_management()` with a fake
`comfy.model_management` and a fake torch, and asserts the dtype of every policy branch plus the
status text it prints: six UNet branches, four VAE branches, the manual-cast correction, both
switches, the alias, and the fail-open path. Pass another path as the first argument to check a copy
somewhere else.

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

Five more checks cover the module-level units. None of them needs the UI running:

```bat
python -c "import zluda_patch, comfy.ops; print('sdpa:', comfy.ops.scaled_dot_product_attention.__name__)"
```

Expected `sdpa: _math_only`.

```bat
python -c "import zluda_patch, comfy.options; comfy.options.enable_args_parsing(); import comfy.cli_args as c; print('args:', c.args.disable_pinned_memory, c.args.disable_async_offload, c.args.disable_cuda_malloc, c.args.reserve_vram)"
```

Expected `args: True True True 1.5`.

```bat
python -c "import zluda_patch, torch, comfy.model_management as mm; dev = mm.get_torch_device(); print('vae:', mm.vae_dtype(dev, [torch.bfloat16, torch.float32])); print('unet(fp16,bf16,fp32):', mm.unet_dtype(dev, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]))"
```

Expected `vae: torch.bfloat16` and `unet(fp16,bf16,fp32): torch.float16`: the UNet policy prefers fp16 when the model supports it; the VAE check uses `[bf16, fp32]`, so it follows the list and returns bf16 (a VAE that lists fp16 returns fp16). Each call also prints its decision on stderr, e.g. `[ZLUDA] model_management: unet float16 (policy: model lists fp16; comfy would pick bfloat16)`.

```bat
python -c "import zluda_patch, torch, comfy.model_management as mm; dev = mm.get_torch_device(); print('small:', mm.unet_dtype(dev, 100_000_000, supported_dtypes=[torch.bfloat16, torch.float32])); print('large:', mm.unet_dtype(dev, 8_000_000_000, supported_dtypes=[torch.bfloat16, torch.float32]))"
```

Expected `small: torch.float32` and `large: torch.bfloat16`: this exercises the budget branch that the first command cannot reach (`model_params` defaults to 0 there). The stderr lines carry the numbers, e.g. `fp32 0.4 GiB <= 10.6 GiB` and `fp32 29.8 GiB > 10.6 GiB` on a 16 GB card with `--reserve-vram 1.5`.

```bat
python -c "import zluda_patch, onnxruntime as ort; print('ort:', ort.get_available_providers())"
```

Expected `ort: ['CPUExecutionProvider']`. If `CUDAExecutionProvider` is still listed, the
hook did not run: the first import in the command has to be `zluda_patch`.

### 3. Reference lines in the ComfyUI startup log

| Line                                                                                       | Expected                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pytorch version:`                                                                         | `2.7.1+cu118`                                                                                                                                                                          |
| `Total VRAM ...`                                                                           | `16368 MB` on the tested card                                                                                                                                                          |
| `VAE load device: ..., offload device: ..., dtype:`                                        | `cuda:0` / `cpu` / the VAE dtype. With the VAE row at **Let ComfyUI decide** this follows the VAE list: Flux `[bf16, fp32]` → **bf16**; a VAE that lists fp16 → **fp16**. `float32` means the VAE only allows fp32 or `--fp32-vae` was set. The preceding `[ZLUDA] model_management: vae ...` line names the same dtype and the rule that chose it |
| `model weight dtype ...`                                                                   | Same dtype as the preceding `[ZLUDA] model_management: unet ...` decision line; that line also says whether the patch policy or ComfyUI's own order decided (`policy:` vs `comfy order:`). A `[bfloat16, float32]` model shows `torch.float32` only when the logged `fp32 ... <= ...` budget check passed |
| `Using ... attention`                                                                      | `split` or `pytorch`, **not** `sub quadratic`                                                                                                                                          |
| `Enabled pinned memory ...`                                                                | **must not appear**; if it does, the `CLI_ARGS` unit did not run                                                                                                                       |
| `[ZLUDA] onnxruntime: forcing CPUExecutionProvider`                                        | appears when an ONNX node runs, followed by its own `说明: ...` Chinese line; no `Failed to load ...onnxruntime_providers_cuda.dll` follows                                            |
| `Unsupported Pytorch detected. DynamicVRAM support requires Pytorch version 2.8 or later.` | **Normal notice**: ComfyUI falls back to the legacy ModelPatcher; this is not an error                                                                                                 |

> When reading the `VAE load device` line, look at the **first field** only: it is the run device of the VAE, and `cpu` there means "run the VAE on the CPU" is enabled. `offload device: cpu` is the default standby location (it only becomes `cuda:0` with `--gpu-only`) and is unrelated to CPU VAE.

## Troubleshooting

| Symptom (log keyword)                                                                                         | Real cause                                                                                                                                                                                                                                            | Fix                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError: Model "..." is corrupt or invalid... Please download the model again`                            | The file is **missing or the path is wrong**, not corrupt. `comfy/utils.py` (lines 122-146 at the tested commit) reports _any_ exception from `safe_open()`, including FileNotFoundError, as "corrupt"                                                | Check the path and file name; do not re-download blindly                                                                                                                                     |
| `rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906` + `Fatal Python error: Aborted` | `ROCBLAS_TENSILE_LIBPATH` points to a directory without the gfx906 kernels (typically the launcher's `%TEMP%\<GUID>` folder was cleaned or rotated)                                                                                                   | `PRELOAD` points it at the HIP SDK's `bin\rocblas\library`                                                                                                                                   |
| `CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmulAlgoGetHeuristic(...)`                    | A linear layer / `addmm` with a bias took the cuBLASLt fused path                                                                                                                                                                                     | `LINEAR`; run the [self-check](#2-one-shot-self-check-does-not-start-the-ui) to confirm `F.linear` is `_linear`                                                                              |
| `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`                                                                    | cuDNN is not disabled, **or** the `comfy.ops` attention priority list forces FLASH / cuDNN anyway (`SDPA` unit)                                                                                                                                       | `TORCH`; if it persists, check `sdpa=math-only` in the status line and grep that file for `FAILED`                                                                                           |
| Black image, no error, log has `unet unexpected: ['blocks.N...']`                                             | The checkpoint has more blocks than the depth ComfyUI builds for it, so the surplus blocks are dropped silently and the latents come out NaN                                                                                                          | `MODEL_DETECTION` counts the blocks in the checkpoint; `model_detection: ... num_blocks 28 -> 40` confirms it. On a ComfyUI without the patch, upgrade it (upstream `725e6ec60`)             |
| `RuntimeError: Found no NVIDIA driver on your system`                                                         | The ZLUDA DLLs were **not preloaded before `import torch`** (for example torch was imported manually first, or the launcher's injection failed)                                                                                                       | `PRELOAD`; make sure the startup log shows `loaded=5/5`                                                                                                                                      |
| `ERROR: clip input is invalid: None`                                                                          | `Load Checkpoint` was used on a model that has **no text encoder** (such as anima-base)                                                                                                                                                               | Use `Load Diffusion Model` + `Load CLIP` + `Load VAE`, or the official template                                                                                                              |
| Out of VRAM / OOM                                                                                             | A large model on a 16 GB card                                                                                                                                                                                                                         | Lower the resolution, use `--fp8_e4m3fn-unet` or raise `--reserve-vram` (1.5 → 2.5). **Avoid** `--lowvram` / `--novram` if the system has only 16 GB of RAM: swapping will stall the machine |
| `0.00 MB usable`, every model `loaded partially ... offloaded` in the **next** generation                     | The previous generation hit an OOM, and after a failed allocation the driver's free-VRAM report stays at 0. Measured: `mem_get_info` returns 0 even after everything is released and `empty_cache` is called, while a 12 GB allocation still succeeds | **Restart ComfyUI.** Nothing in the process recovers the report; all later generations are streamed from system RAM and are much slower                                                      |
| Everything is reverted after "Install PyTorch / Install dependencies" in the launcher                         | The launcher reinstalls from its built-in manifest (it pins Python 3.11.6 for the ComfyUI core, and cu118 tops out at 2.3.1 there, which cannot be installed on Python 3.13)                                                                          | Do not click "Install dependencies / Switch Torch version / Reinstall environment" in the launcher                                                                                           |
| `CUBLAS_STATUS_NOT_SUPPORTED ... abType 3 cType 10 computeType 72`                                            | An `int8_tensorwise` model reached `torch._int_mm` (`LINEAR` unit)                                                                                                                                                                                    | Use a `convrot_w4a4` / `fp8` / `bf16` model, or accept dequantize-per-call                                                                                                                   |
| Black image, healthy status line, `_math_only` already active                                                 | The VAE row is pinned to Half Precision (FP16) on a VAE that does not list fp16 (Flux), so the decoder overflows to NaN                                                                                                                                | Set the launcher VAE precision row to **Let ComfyUI decide**                                                                                                                                 |
| `[ZLUDA] model_management: unet ... (comfy order: --fp16-unet)`                                               | A launcher precision row is pinned (the startup line lists it under `flags=`), so the explicit flag wins and the dtype policy is not consulted — by design                                                                                            | Set that launcher row back to **Let ComfyUI decide** if you want the policy to choose                                                                                                        |
| `[ZLUDA] model_management: unet bfloat16 (policy: fp32 ... > ...)`                                           | A `[bfloat16, float32]` model does not fit fp32 in the weight budget (after `--reserve-vram` and the 10% margin), so the patch keeps bf16 (0.55x fp16 speed) to stay resident                                                                          | Accept bf16, lower `ZLUDA_RESERVE_VRAM`, or force `--fp32-unet` and watch for partial offload; a model that lists fp16 never takes this branch                                               |
| `onnxruntime` provider errors from RMBG / controlnet_aux / Impact                                             | CUDA 12 `onnxruntime-gpu` provider under ZLUDA (`ONNX` unit)                                                                                                                                                                                          | Confirm the status line says `onnxruntime: CUDA/TensorRT providers filtered`                                                                                                                 |

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
3. **Precision:** the four rows can stay on **Let ComfyUI decide**: the patch picks fp16 when the model supports it, fp32 when it fits, else bf16; the VAE follows its `working_dtypes` (fp16 if listed, else bf16; Flux stays bf16). Pinning a model row to **Half Precision (FP16)** is still the fastest choice for models that list fp16, but it makes the policy step aside (`flags=fp16_unet`, decision lines become `comfy order:`); keep the three model rows on **Let ComfyUI decide** if you want the automatic rules — including the fp32 budget — to apply. `ZLUDA_UNET_DTYPE=comfy` / `ZLUDA_VAE_DTYPE=comfy` hand one side back without touching the other.
4. **Performance:** both the attention backend and the precision have a large effect on speed; see [Precision and performance](#precision-and-performance-measured). Lowering the resolution also helps.
5. **Extending the patch:** if another operator fails with `CUBLAS_STATUS_NOT_SUPPORTED`, add a workaround in the matching PART 2 unit (`TORCH_STEPS` for torch-level patches, `LINEAR` for matmul-shaped ones; for example replace `F.conv2d` with an equivalent `unfold + matmul`, which was tested to work).
6. **Launcher overwrites:** a launcher update of ComfyUI resets `main.py`, so re-insert the bootstrap block; `zluda_patch.py` may be wiped as well, so keep a backup.
7. **Diagnostics:** the launcher's `Diagnostics-*.log` (from its **Watson** page) is an environment snapshot plus the console buffer, and **contains the output of several runs**. To tell whether a log line comes from a new process, check the timestamps in `_zluda_status.txt`.
8. **The launcher is required only to unpack the ZLUDA DLLs.** The patch preloads them itself,
   and that is enough as long as nothing imported torch before it. This patch used to break that
   rule: the `comfy_kitchen` class names were resolved at import, `comfy_kitchen` imports torch,
   and the resulting torch bound to no CUDA driver, so the whole process reported `Found no NVIDIA
driver on your system` (measured: `cuInit(0)` returned 0 and `cuDeviceGetCount` reported one
   device through the preloaded DLLs, while `torch.cuda.is_available()` was `False` in the same
   process). The names are resolved on first use now (`LINEAR` unit), so bare `python main.py`, IDE
   debugging and the [self-check](#2-one-shot-self-check-does-not-start-the-ui) see the GPU.
9. **INT8 tensorwise models are usable, not fast.** The `LINEAR` unit dequantizes them, which costs
   24.4 ms per qkv call against 1.34 ms for a native fp16 GEMM. Prefer `convrot_w4a4`,
   `fp8` or `bf16` models on this card.
10. **ONNX Runtime always runs on the CPU.** That is intentional (`ONNX` unit): the installed
    `onnxruntime-gpu` is a CUDA 12 build. If you ever install a matching build and want to
    experiment, `ZLUDA_KEEP_ONNX_CUDA=1` leaves the providers alone.
11. **KL-style VAEs are under-planned by 2.73x under ZLUDA.** `comfy/sd.py` multiplies the VAE
    memory estimate by `VAE_KL_MEM_RATIO = 2.73` only when `is_amd()` is true; that check reads
    `torch.version.hip`, which is `None` under ZLUDA, so the estimate stays at 1.0 and an
    AutoencoderKL decode is planned at about 37% of its real peak. The `--reserve-vram 1.5` the
    patch sets is what absorbs the difference: below roughly 1.2 GB the tiled-decode fallback is
    back. This is measured behaviour of ComfyUI, not something the patch can fix from the outside.
12. **The manual-cast correction does not cover bf16 checkpoints.** For bf16 weights ComfyUI
    returns `None` ("no cast needed"), and `SeedVR2.set_inference_dtype()` then sets the manual
    cast to bf16 itself, so that model still computes in bf16 (0.55x). The wrapper only changes a
    cast ComfyUI already resolved to bf16 (fp8 / unknown stores). No SeedVR2 checkpoint was
    available to measure a targeted fix, so that path is left alone.
13. **A model's dtype list wins over the file format.** GGUF and other quantized builds still take
    their dtype from `supported_inference_dtypes`: `ZImage` is `[bfloat16, float32]`, so
    `z-image-turbo-Q6_K.gguf` runs bf16 (0.55x) and prints `fp32 22.4 GiB > 10.6 GiB`, exactly like
    the fp16 safetensors build. Only models that list fp16 (Anima, Wan, SeedVR2, ...) get the fast
    path.

## Environment switches

Every behaviour added after the original release can be turned off or tuned without
editing the patch's logic. All of them are collected in **PART 0** of the patch, one line per
switch with its default written next to it, and the state is printed once per launch as one
`switches: ...` status line:

| Variable                     | Default | Effect                                                                                                                                             |
| ---------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ZLUDA_NO_ARG_OVERRIDES=1`   | off     | stop overriding launch arguments; the launcher owns them again                                                                                     |
| `ZLUDA_RESERVE_VRAM=<GB>`    | `1.5`   | change the reserved VRAM; an explicit `--reserve-vram` still wins                                                                                  |
| `ZLUDA_FORCE_FP16=1`         | off     | force fp16 for the UNet and text encoder (also overrides fp8-stored models)                                                                        |
| `ZLUDA_SPLIT_ATTENTION=1`    | off     | use split attention instead of pytorch/math SDP                                                                                                    |
| `ZLUDA_ALLOW_BF16=1`         | off     | coarse alias: send **both** dtype policies back to ComfyUI's order (status: `unet=comfy-order \| vae=comfy-order`); an explicit side wins          |
| `ZLUDA_UNET_DTYPE=comfy`     | `auto`  | send only the UNet dtype policy (and its manual-cast correction) back to ComfyUI's order (status: `unet=comfy-order`)                              |
| `ZLUDA_VAE_DTYPE=comfy`      | `auto`  | send only the VAE list preference back to ComfyUI's order (status: `vae=comfy-order`)                                                              |
| `ZLUDA_FP32_MARGIN=<0..1>`   | `0.9`   | safety margin on the fp32 weight budget; lower is stricter, a value outside (0, 1] falls back to 0.9                                               |
| `ZLUDA_KEEP_ONNX_CUDA=1`     | off     | leave ONNX Runtime's provider list untouched                                                                                                       |
| `ZLUDA_INT8_DEQUANT_CACHE=1` | off     | cache dequantized INT8 weights (keeps INT8 + adds fp16: ~1.5x fp16 / 3x INT8, 12x faster)                                                          |
| `ZLUDA_TOPK_CPU=1`           | off     | run `torch.topk` on the CPU                                                                                                                        |
| `ZLUDA_STFT_CPU=1`           | off     | run `torch.stft` / `torch.istft` on the CPU (audio models)                                                                                         |
| `ZLUDA_ANIMA_EXTRA_BLOCKS=0` | **on**  | stop counting Anima / Cosmos-Predict2 blocks from the checkpoint (`MODEL_DETECTION` unit); the switch is inverted because the fix is on by default |
| `ZLUDA_NO_JITSCRIPT=1`       | off     | make `torch.jit.script` a no-op                                                                                                                    |

> `ZLUDA_SPLIT_ATTENTION=1` **forces** split; **off does not force math** — it leaves the
> launcher's **Cross-Attention Optimization** alone. The `Using split optimization for attention` line in
> the ComfyUI log reflects the launcher's choice, not this switch.

## Repository layout

```
.
├── ComfyUI/
│   └── zluda_patch.py         # the compatibility patch (copy to ComfyUI\ in the package root)
├── zluda_dtype_check.py       # offline self-check for the dtype policy (copy to the package root)
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
| `_zluda_status.txt`                                      | Status record of the patch: one line per event (startup lines, plus one per dtype decision)                                        |
| `.launcher\zluda.<hash>.extpack`                         | The 5 ZLUDA DLLs (can be unpacked with 7-Zip)                                                                                     |
| `.launcher\tensile-gfx906.<hash>.extpack`                | rocBLAS Tensile kernels for gfx906 (alternative source)                                                                           |
| `.launcher\preference.json`                              | Launcher preferences (including proxy address and custom arguments)                                                               |
| `ComfyUI\blueprints\Text to Image (Anima Base 1.0).json` | Official Anima template                                                                                                           |
| `%HIP_PATH%\bin\rocblas\library` (absolute)              | gfx906 Tensile kernel directory that the patch points to (falls back to `C:\Program Files\AMD\ROCm\5.7` when `HIP_PATH` is unset) |
| `%TEMP%\<GUID>\` (absolute, new on every run)            | Where the launcher unpacks the ZLUDA DLLs; the patch scans and preloads from here                                                 |

If you use a different bundle version or HIP SDK version, re-check the version matching described in [Tested environment](#tested-environment).

## License

This project is licensed under the [MIT License](LICENSE).

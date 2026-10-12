# ComfyUI Aki Bundle on AMD MI50 (ZLUDA, gfx906, Windows)

[English](README.md) | [简体中文](README.zh-CN.md)

Run the **ComfyUI "Aki" all-in-one bundle** (秋叶整合包, started with the **绘世 / Hui Shi launcher**) on an **AMD Instinct MI50** (flashed with the **Radeon VII PRO vBIOS**) through [ZLUDA](https://github.com/vosen/ZLUDA) on Windows.

This repository documents a setup that was **verified end to end**: the exact commands, a small compatibility patch ([`ComfyUI/zluda_patch.py`](ComfyUI/zluda_patch.py)), why each part is needed, how to verify it, and a troubleshooting table. All conclusions come from hands-on testing on a single machine (command output, logs, file hashes), not from guesses.

## TL;DR

Four things are needed to run this bundle on ZLUDA:

1. **Replace torch with the cu118 build.** ZLUDA only provides CUDA 11 libraries, so a PyTorch built for CUDA 12+ cannot work.
2. **Turn off the `cudaMallocAsync` allocator** that ComfyUI enables automatically on non-NVIDIA cards: add `--disable-cuda-malloc` to the launch arguments.
3. **Put the patch `ComfyUI\zluda_patch.py` in place and hook it into the top of `ComfyUI\main.py`** with a short bootstrap block. The patch does five things:
   1. preloads the ZLUDA DLLs,
   2. points rocBLAS at the gfx906 Tensile kernels,
   3. disables cuDNN,
   4. keeps only the math SDP attention backend,
   5. bypasses the cuBLASLt bias-fused GEMM used by linear layers with a bias.
4. **Precision settings** in the launcher can stay at FP16 or Auto. Precision is not the root cause; changing it is an optional tweak.

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

1. Copy all files of this repository into the package root, so that the patch ends up at `ComfyUI\zluda_patch.py`. Only [`ComfyUI/zluda_patch.py`](ComfyUI/zluda_patch.py) matters; `README*.md` and `LICENSE` are not needed.
2. Insert the following block **before the first line** of `ComfyUI\main.py` (in the tested ComfyUI version the first line is `import comfy.options`):

```python
# ==== ZLUDA compatibility patch (added manually, safe to remove) ====
try:
    import os as _zluda_os
    import sys as _zluda_sys

    _zluda_sys.path.insert(0, _zluda_os.path.dirname(_zluda_os.path.abspath(__file__)))
    import zluda_patch  # noqa: F401
except Exception as _zluda_e:
    print("[ZLUDA] Patch FAILED:", repr(_zluda_e))
# ==== End of ZLUDA compatibility patch ====
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

**Do not use** these arguments: `--use-sage-attention` (the package is uninstalled), `--use-flash-attention` (not installed), `--enable-triton-backend` (useless under ZLUDA), `--fast` / `--fast cublas_ops` (this path calls cuBLASLt directly and always crashes), `--cuda-malloc`.

Precision (launcher → compute precision settings, 计算精度设置): set `Upcast Attention` to off; set Precision / Text Encoder / UNet / VAE to **FP16 or Auto**. Precision is not the root cause.

### Step 4 (optional): Try a real model, for example Anima Base 1.0

This is a "split" model: it needs three loaders and the official template.

| File                          | Where to put it                                                   |
| ----------------------------- | ----------------------------------------------------------------- |
| `anima-base-v1.0.safetensors` | `ComfyUI\models\diffusion_models\` (`models\unet\` is equivalent) |
| `qwen_3_06b_base.safetensors` | `ComfyUI\models\text_encoders\`                                   |
| `qwen_image_vae.safetensors`  | `ComfyUI\models\vae\`                                             |

Workflow: use the template bundled with the package, `ComfyUI\blueprints\Text to Image (Anima Base 1.0).json` (drag it into the web page, or use Workflow → Templates and search for `Anima`). It uses `Load Diffusion Model` + `Load CLIP` (keep the type at `stable_diffusion`; ComfyUI recognizes the Qwen3 text encoder from the file content) + `Load VAE`. KSampler defaults: steps 30, cfg 4, `er_sde`, `simple`; the size group is 1 MP (about 1024×1024).

**Common mistake:** loading `anima-base-v1.0.safetensors` with `Load Checkpoint` fails with `ERROR: clip input is invalid: None`, because that file **contains no text encoder** (`load_checkpoint_guess_config()` in `comfy\sd.py` returns `clip=None`, and `comfy\nodes.py:74` raises).

## How the patch works

[`ComfyUI/zluda_patch.py`](ComfyUI/zluda_patch.py) does five things in order, and none of them can be skipped:

1. **Preload the ZLUDA DLLs** (`nvcuda.dll`, `nvml.dll`, `cublas64_11.dll`, `cusparse64_11.dll`, `nvrtc64_112_0.dll`) and add the ZLUDA folder and `%HIP_PATH%\bin` to `PATH` / `os.add_dll_directory`.
   _Why:_ ZLUDA's `nvcuda.dll` must be loaded **before** torch initializes CUDA, otherwise torch reports `Found no NVIDIA driver on your system`. The launcher normally does this itself (it unpacks the DLLs into `%TEMP%\<GUID>` on every run), but its injection was not reliable in this setup. So the patch scans `%TEMP%`, `%TMP%` and `%LOCALAPPDATA%\Temp` for that folder (and `zluda\` in the package root) and preloads from it.
2. **Set `ROCBLAS_TENSILE_LIBPATH`** to `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library`.
   _Why:_ ZLUDA's cuBLAS forwards matrix multiplications to **rocBLAS**, which needs the Tensile kernels of the **matching architecture (gfx906)**. The launcher points this variable at a folder under `%TEMP%` that does not contain them, which ends in `rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906` and `Fatal Python error: Aborted`.
3. **`torch.backends.cudnn.enabled = False`**
   _Why:_ ZLUDA has no cuDNN. Left enabled it fails with `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR` (the VAE's convolutions are hit first). With cuDNN off, convolutions use a fallback path that works (verified).
4. **Keep only the math SDP attention backend** (flash and mem-efficient are turned off).
   _Why:_ ZLUDA cannot run the flash attention kernels; the launcher's own ZLUDA compatibility patch does the same.
5. **Split "linear layer with bias" into `matmul` + add** (and do the same for `torch.addmm`).
   _Why:_ PyTorch runs `F.linear(x, w, b)` / `addmm` through a **bias-fused GEMM that only exists in cuBLASLt** (`gemm_and_bias`). ZLUDA has **no cuBLASLt implementation at all** (none of its 5 DLLs exports a `cublasLt*` symbol), so torch falls back to NVIDIA's genuine `torch\lib\cublasLt64_11.dll` (544 MB), which necessarily returns `CUBLAS_STATUS_NOT_SUPPORTED`. Measured: `mm` / `bmm` / `baddbmm` without bias all work; **only linear / addmm with a bias crash**, regardless of precision (fp16 / bf16 / fp32).

## Verification

### 1. Startup log

These lines should appear (`<TEMP>` is your temp folder, usually `%LOCALAPPDATA%\Temp`):

```
[ZLUDA] Preload: zluda_dir=<TEMP>\<GUID> | loaded=5/5
[ZLUDA] OK torch=2.7.1+cu118 | Tensile=C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library | cudnn=False | sdp=math | linear-patch[F.linear=_linear]
```

The same lines are appended to `_zluda_status.txt` in the package root (and to `zluda_status.txt` in the temp folder).

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

### 3. Reference lines in the ComfyUI startup log

| Line                                                                               | Expected                                                                               |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `pytorch version:`                                                                 | `2.7.1+cu118`                                                                          |
| `Total VRAM ...`                                                                   | `16368 MB` on the tested card                                                          |
| `VAE load device: ... dtype:`                                                      | `torch.float16` (or bfloat16, depending on the precision settings)                     |
| `model weight dtype ...`                                                           | `torch.float16` / `torch.bfloat16`                                                     |
| `Unsupported Pytorch detected. DynamicVRAM requires Pytorch version 2.8 or later.` | **Normal notice**: ComfyUI falls back to the legacy ModelPatcher; this is not an error |

## Troubleshooting

| Symptom (log keyword)                                                                                         | Real cause                                                                                                                                                                                             | Fix                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError: Model "..." is corrupt or invalid... Please download the model again`                            | The file is **missing or the path is wrong**, not corrupt. `comfy\utils.py` (lines 122-146 at the tested commit) reports _any_ exception from `safe_open()`, including FileNotFoundError, as "corrupt" | Check the path and file name; do not re-download blindly                                                                                                                     |
| `rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU arch : gfx906` + `Fatal Python error: Aborted` | `ROCBLAS_TENSILE_LIBPATH` points to a directory without the gfx906 kernels (typically the launcher's `%TEMP%\<GUID>` folder was cleaned or rotated)                                                    | Patch item 2 points it at the HIP SDK's `bin\rocblas\library`                                                                                                                |
| `CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmulAlgoGetHeuristic(...)`                    | A linear layer / `addmm` with a bias took the cuBLASLt fused path                                                                                                                                      | Patch item 5; run the [self-check](#2-one-shot-self-check-does-not-start-the-ui) to confirm `F.linear` is `_linear`                                                          |
| `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`                                                                    | cuDNN is not disabled                                                                                                                                                                                  | Patch item 3                                                                                                                                                                 |
| `RuntimeError: Found no NVIDIA driver on your system`                                                         | The ZLUDA DLLs were **not preloaded before `import torch`** (for example torch was imported manually first, or the launcher's injection failed)                                                        | Patch item 1; make sure the startup log shows `loaded=5/5`                                                                                                                   |
| `ERROR: clip input is invalid: None`                                                                          | `Load Checkpoint` was used on a model that has **no text encoder** (such as anima-base)                                                                                                                | Use `Load Diffusion Model` + `Load CLIP` + `Load VAE`, or the official template                                                                                              |
| Out of VRAM / OOM                                                                                             | A large model on a 16 GB card                                                                                                                                                                          | Lower the resolution, use `--fp8_e4m3fn-unet` or `--reserve-vram 1`. **Avoid** `--lowvram` / `--novram` if the system has only 16 GB of RAM: swapping will stall the machine |
| Everything is reverted after "Install PyTorch / Install dependencies" in the launcher                         | The launcher reinstalls from its built-in manifest (it pins Python 3.11.6 for the ComfyUI core, and cu118 tops out at 2.3.1 there, which cannot be installed on Python 3.13)                           | Do not click "Install dependencies / Switch Torch version / Reinstall environment" in the launcher                                                                           |

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
3. **Precision:** FP16 and Auto both work. To use a model's native bf16, set the UNet precision back to `Auto`.
4. **Performance:** attention currently runs as math SDP (a ZLUDA limitation), so neither VRAM use nor speed is optimal. You can try `--use-split-cross-attention` or a lower resolution.
5. **Extending the patch:** if another operator fails with `CUBLAS_STATUS_NOT_SUPPORTED`, add a workaround in `patch_torch()` (for example replace `F.conv2d` with an equivalent `unfold + matmul`, which was tested to work).
6. **Launcher overwrites:** a launcher update of ComfyUI resets `main.py`, so re-insert the bootstrap block; `zluda_patch.py` may be wiped as well, so keep a backup.
7. **Diagnostics:** the launcher's `Diagnostics-*.log` (from its troubleshooting page, 疑难解答) is an environment snapshot plus the console buffer, and **contains the output of several runs**. To tell whether a log line comes from a new process, check the timestamps in `_zluda_status.txt`.

## Repository layout

```
.
├── ComfyUI/
│   └── zluda_patch.py   # the compatibility patch (copy to ComfyUI\ in the package root)
├── LICENSE
├── README.md
└── README.zh-CN.md
```

Files involved in a deployed package (relative to the package root unless noted):

| Path                                                           | Description                                                                       |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `python\python.exe`                                            | Python 3.13.11                                                                    |
| `ComfyUI\main.py`                                              | **Bootstrap block inserted at the very top**                                      |
| `ComfyUI\zluda_patch.py`                                       | The patch added by this setup (from this repository)                              |
| `_zluda_status.txt`                                            | Self-check output of the patch (one line appended per start)                      |
| `.launcher\zluda.<hash>.extpack`                               | The 5 ZLUDA DLLs (can be unpacked with 7-Zip)                                     |
| `.launcher\tensile-gfx906.<hash>.extpack`                      | rocBLAS Tensile kernels for gfx906 (alternative source)                           |
| `.launcher\preference.json`                                    | Launcher preferences (including proxy address and custom arguments)               |
| `ComfyUI\blueprints\Text to Image (Anima Base 1.0).json`       | Official Anima template                                                           |
| `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library` (absolute) | gfx906 Tensile kernel directory that the patch points to                          |
| `%TEMP%\<GUID>\` (absolute, new on every run)                  | Where the launcher unpacks the ZLUDA DLLs; the patch scans and preloads from here |

If you use a different bundle version or HIP SDK version, re-check the version matching described in [Tested environment](#tested-environment).

## License

This project is licensed under the [MIT License](LICENSE).

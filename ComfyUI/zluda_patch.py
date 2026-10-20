"""
ZLUDA compatibility patch (MI50 / gfx906 + ZLUDA + ComfyUI, added manually).

What it does (fully self-contained, does not rely on the launcher's injection mechanism):
  1) Preloads the ZLUDA DLLs itself (nvcuda / nvml / cublas64_11 / cusparse64_11 / nvrtc64_112_0).
     This must happen before `import torch`, otherwise torch reports "no NVIDIA driver found".
  2) Points ROCBLAS_TENSILE_LIBPATH at a directory that holds the gfx906 Tensile kernels.
  3) Disables cuDNN, which ZLUDA cannot use.
  4) Keeps only the math SDP attention backend.
  5) Splits "linear layer with bias" / addmm into matmul + add, which bypasses the
     bias-fused GEMM of cuBLASLt.
  6) Forces ComfyUI's SDPA wrapper onto the math backend. `comfy/ops.py` passes
     `set_priority=True`, which overrides `enable_flash_sdp(False)` from step 4 and puts
     FLASH / cuDNN first; neither exists under ZLUDA, so attention dies with
     `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR` (fp16/bf16) or
     `cutlassF: no kernel found to launch!` (fp32). When it does not raise, it
     returns NaN and the image comes out black.
  7) Makes `F.linear` usable with INT8 tensorwise weights (dequantize + cache).
     Without it `torch._int_mm` reaches cuBLASLt INT8, which ZLUDA does not implement:
     `CUBLAS_STATUS_NOT_SUPPORTED ... abType 3 computeType 72`.
  8) Overrides the launch arguments the moment `comfy.cli_args` is parsed: disables
     pinned memory / async offload / cudaMallocAsync and reserves 0.9 GB of VRAM, so the
     setup no longer depends on .launcher/preference.json staying intact.
  9) Keeps bf16 out of the compute dtype. ZLUDA reports compute capability (8, 8), so
     ComfyUI's should_use_bf16() returns True before it asks the device anything, and
     gfx906 then runs bf16 at about half the fp16 speed.
 10) Filters the CUDA / TensorRT execution providers out of ONNX Runtime: the installed
     onnxruntime-gpu is a CUDA 12 build and cannot work through ZLUDA.
 11) Fixes LTXAV's memory_usage_factor, which upstream leaves at 0.077 and which makes
     ComfyUI under-estimate activation memory.
 12) Optional operator fallbacks (topk / stft / jit.script), off by default.

Environment switches: ZLUDA_NO_ARG_OVERRIDES, ZLUDA_RESERVE_VRAM, ZLUDA_FORCE_FP16,
ZLUDA_SPLIT_ATTENTION, ZLUDA_ALLOW_BF16, ZLUDA_VAE_FP32, ZLUDA_KEEP_ONNX_CUDA,
ZLUDA_TOPK_CPU, ZLUDA_STFT_CPU, ZLUDA_NO_JITSCRIPT.

Output convention: status lines are plain English, because they are greppable and survive any
console code page; a message that needs a human decision is followed by its own Chinese line
prefixed with "说明: " instead of being appended to the status line.

When it is no longer needed: delete this file and remove the bootstrap block at the top of main.py.


ZLUDA 兼容补丁（MI50 / gfx906 + ZLUDA + ComfyUI，手工添加）。

作用（全部自给自足，不依赖绘世启动器的注入机制）：
  1) 自己预加载 ZLUDA 的 DLL（nvcuda / nvml / cublas64_11 / cusparse64_11 / nvrtc64_112_0）
     —— 必须发生在 import torch 之前，否则 torch 会报"找不到 NVIDIA 驱动"
  2) 把 ROCBLAS_TENSILE_LIBPATH 指向含 gfx906 Tensile 内核的目录
  3) 关闭 ZLUDA 无法使用的 cuDNN
  4) 注意力只保留 math SDP
  5) 把"带 bias 的线性层 / addmm"拆成 matmul + 加法，绕开 cuBLASLt 的 bias 融合 GEMM
  6) 强制 ComfyUI 的 SDPA 包装器走 math 后端。comfy/ops.py 传了 set_priority=True，
     会覆盖第 4 步的 enable_flash_sdp(False)，把 FLASH / cuDNN 排到最前；这两个在 ZLUDA 上
     都不存在，于是注意力会以 cuDNN error: CUDNN_STATUS_INTERNAL_ERROR（fp16/bf16）或
     cutlassF: no kernel found to launch!（fp32）失败；不报错时会返回 NaN，表现为全黑图。
  7) 让 F.linear 能处理 INT8 tensorwise 权重（反量化 + 缓存）。不处理时 torch._int_mm 会走到
     cuBLASLt INT8，而 ZLUDA 没有实现：CUBLAS_STATUS_NOT_SUPPORTED ... abType 3 computeType 72。
  8) 在 comfy.cli_args 解析完成的瞬间覆盖启动参数：关闭 pinned memory / async offload /
     cudaMallocAsync，并预留 0.9 GB 显存。这样就不再依赖 .launcher/preference.json 不被重置。
  9) 把 bf16 挡在计算精度之外。ZLUDA 把算力报成 (8, 8)，should_use_bf16() 在问设备之前
     就已经返回 True，而 gfx906 上 bf16 只有 fp16 的一半速度。
 10) 过滤掉 ONNX Runtime 的 CUDA / TensorRT provider：装的是 CUDA 12 构建的
     onnxruntime-gpu，在 ZLUDA 上不可能工作。
 11) 修正 LTXAV 的 memory_usage_factor。上游留在 0.077，导致 ComfyUI 低估激活显存。
 12) 可选的算子兜底（topk / stft / jit.script），默认关闭。

环境变量开关：ZLUDA_NO_ARG_OVERRIDES、ZLUDA_RESERVE_VRAM、ZLUDA_FORCE_FP16、
ZLUDA_SPLIT_ATTENTION、ZLUDA_ALLOW_BF16、ZLUDA_VAE_FP32、ZLUDA_KEEP_ONNX_CUDA、
ZLUDA_TOPK_CPU、ZLUDA_STFT_CPU、ZLUDA_NO_JITSCRIPT。

输出约定：状态行是纯英文，方便 grep，也不受控制台代码页影响；需要人来处理的提示会在下一行
单独输出 `说明: ...` 中文，而不是拼在状态行后面。

不需要时：删除本文件，并删掉 main.py 顶部那段引导代码即可。
"""

import ctypes
import glob
import importlib.abc
import importlib.util
import os
import sys
import tempfile
import time
import weakref

# Package root = the folder above the ComfyUI folder that contains this file.
# 整合包根目录 = 本文件所在的 ComfyUI 文件夹的上一级。
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HIP SDK root: prefer HIP_PATH, fall back to the default install path of the tested setup.
# HIP SDK 根目录：优先读 HIP_PATH，取不到时回退到实测环境的默认安装路径。
HIP_ROOT = os.environ.get("HIP_PATH") or r"C:\Program Files\AMD\ROCm\5.7"

# rocBLAS Tensile kernel directory; it must contain the gfx906 files.
# rocBLAS Tensile 内核目录，必须包含 gfx906 的文件。
TENSILE_DIR = os.path.join(HIP_ROOT, "bin", "rocblas", "library")

# Optional custom ZLUDA directory: if you extracted zluda.extpack to a fixed folder, put its path here.
# 自定义 ZLUDA 目录（可选）：如果你把 zluda.extpack 解压到了固定目录，把路径填进来。
EXTRA_ZLUDA_DIRS = [os.path.join(ROOT_DIR, "zluda")]
ZLUDA_DLLS = ("nvcuda.dll", "nvml.dll", "cublas64_11.dll", "cusparse64_11.dll", "nvrtc64_112_0.dll")
# Status files: one in the package root, one in the system temp directory.
# 状态文件：一份在整合包根目录，一份在系统临时目录。
REPORTS = [
    os.path.join(ROOT_DIR, "_zluda_status.txt"),
    os.path.join(tempfile.gettempdir(), "zluda_status.txt"),
]


# Write to a stream without ever losing the whole line to an encoding error.
# 向流写入，且绝不因为编码问题把整行丢掉。
def _write_stream(stream, text):
    try:
        # A non-UTF-8 console or redirected log raises UnicodeEncodeError on the whole
        # line, so replace the characters it cannot represent first.
        #
        # 非 UTF-8 的控制台或重定向日志会对整行抛 UnicodeEncodeError，
        # 所以先把编码不了的字符替换掉。
        encoding = getattr(stream, "encoding", None) or "utf-8"
        try:
            text = text.encode(encoding, "replace").decode(encoding, "replace")
        except (LookupError, UnicodeError):
            text = text.encode("ascii", "replace").decode("ascii")
        stream.write(text)
        stream.flush()
    except Exception:
        pass


# Print to stderr and append to every status file; write failures are ignored on purpose.
# 输出到 stderr 并追加写入所有状态文件；写入失败会被有意忽略。
def _report(text):
    line = "%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), text)
    _write_stream(sys.stderr, "[ZLUDA] " + text + "\n")
    for path in REPORTS:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass


# Chinese follow-up line for a message a human has to act on. Status lines stay plain
# English: they are greppable, they survive any console code page, and both READMEs quote
# them verbatim. The translation gets its own line instead of being appended to them.
#
# 需要人来处理的提示另起一行输出中文。状态行保持纯英文：方便 grep、不受控制台代码页
# 影响，两个 README 也是逐字引用的。译文单独一行，而不是拼在状态行后面。
def _hint(zh):
    _report("说明: " + zh)


# A status line with a FAILED fragment means one sub-patch did not take effect, so it gets a
# Chinese follow-up line like every other message a human has to act on.
#
# 状态行里带 FAILED 片段就说明某个子补丁没生效，所以和别的需要人处理的提示一样补一行中文。
def _hint_if_failed(msgs):
    if any("FAILED" in msg for msg in msgs):
        _hint(
            "上面有 FAILED 项，说明对应的子补丁没生效。状态行看起来健康并不代表全对，"
            "请把整个 _zluda_status.txt grep 一遍 FAILED。"
        )


# Point rocBLAS at the gfx906 kernels and clear TORCH_BLAS_PREFER_CUBLASLT.
# 让 rocBLAS 使用 gfx906 内核，并清除 TORCH_BLAS_PREFER_CUBLASLT。
def apply_env():
    os.environ["ROCBLAS_TENSILE_LIBPATH"] = TENSILE_DIR
    os.environ.pop("TORCH_BLAS_PREFER_CUBLASLT", None)


# Load the ZLUDA DLLs into the process before `import torch` (equivalent to the launcher's zluda_companion).
# 在 import torch 之前把 ZLUDA 的 DLL 加载进进程（等价于启动器的 zluda_companion）。
def preload_zluda():
    # The launcher unpacks the ZLUDA DLLs into a fresh <GUID> folder under the temp directory
    # on every run, so scan the temp directories for folders that contain cublas64_11.dll.
    #
    # 启动器每次运行都会把 ZLUDA 的 DLL 解压到临时目录下新的 <GUID> 文件夹，
    # 因此扫描各临时目录，找出包含 cublas64_11.dll 的文件夹。
    roots = []
    for key in ("TEMP", "TMP"):
        value = os.environ.get(key)
        if value:
            roots.append(value)
    roots.append(tempfile.gettempdir())
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        roots.append(os.path.join(local_appdata, "Temp"))

    candidates = []
    scanned = set()
    for root in roots:
        try:
            root = os.path.abspath(root)
        except Exception:
            continue
        if root in scanned or not os.path.isdir(root):
            continue
        scanned.add(root)
        for d in glob.glob(os.path.join(root, "*")):
            if os.path.isfile(os.path.join(d, "cublas64_11.dll")):
                try:
                    candidates.append((os.path.getmtime(d), d))
                except OSError:
                    pass
    for d in EXTRA_ZLUDA_DIRS:
        if os.path.isfile(os.path.join(d, "cublas64_11.dll")):
            try:
                candidates.append((os.path.getmtime(d), d))
            except OSError:
                pass

    if not candidates:
        return (
            "No ZLUDA dir found (nvcuda.dll not preloaded)",
            "没找到含 ZLUDA DLL 的目录，nvcuda.dll 未预加载。先让启动器正常跑一次 ComfyUI"
            "（它会把 DLL 解到 %TEMP%\\<GUID>），或把 .launcher\\zluda.<hash>.extpack "
            "用 7-Zip 解压到整合包根目录的 zluda\\",
        )

    # Newest folder first.
    # 最新的文件夹优先。
    candidates.sort(reverse=True)
    hip_bin = os.path.join(HIP_ROOT, "bin")

    # Try each candidate until both nvcuda.dll and cublas64_11.dll load.
    # 依次尝试每个候选目录，直到 nvcuda.dll 与 cublas64_11.dll 都加载成功。
    best = None
    for _, zd in candidates:
        search = [p for p in (zd, hip_bin) if p and os.path.isdir(p)]
        os.environ["PATH"] = os.pathsep.join(search + [os.environ.get("PATH", "")])
        for p in search:
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
        loaded, failed = [], []
        for name in ZLUDA_DLLS:
            p = os.path.join(zd, name)
            if not os.path.isfile(p):
                failed.append(name + ":missing")
                continue
            try:
                ctypes.WinDLL(p)
                loaded.append(name)
            except OSError as e:
                failed.append("%s:%s" % (name, e))
        if "nvcuda.dll" in loaded and "cublas64_11.dll" in loaded:
            best = (zd, loaded, failed)
            break
        best = best or (zd, loaded, failed)

    # The candidate list is non-empty here, so the loop above always assigned `best`.
    # 走到这里候选列表必不为空，因此上面的循环必然已经给 `best` 赋过值。
    zd, loaded, failed = best
    text = "zluda_dir=%s | loaded=%d/%d%s" % (
        zd,
        len(loaded),
        len(ZLUDA_DLLS),
        (" | failed=" + ";".join(failed[:3])) if failed else "",
    )
    if failed:
        return text, (
            "部分 DLL 加载失败：failed= 里的 xxx.dll:missing 是文件缺失；"
            "WinError 193 通常是拿到了占位文件或 HTML，而不是真正的 DLL。"
        )
    return text, None


# =============================================================================
# 1) comfy.ops SDPA backend priority override
# 1) 覆盖 comfy.ops 的 SDPA 后端优先级
# =============================================================================
# comfy/ops.py installs this priority list with `set_priority=True`:
#
#     SDPA_BACKEND_PRIORITY = [FLASH_ATTENTION, CUDNN_ATTENTION, EFFICIENT_ATTENTION, MATH]
#     with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True):
#         return torch.nn.functional.scaled_dot_product_attention(...)
#
# `set_priority=True` makes PyTorch walk that order regardless of the enable_* flags
# that patch_torch() sets. FLASH and cuDNN do not exist under ZLUDA, so every
# attention call fails. Measured on gfx906 (1x16x512x128, fp16):
#
#     bare F.sdpa                     fp16 OK    bf16 OK    fp32 OK
#     comfy.ops.sdpa (priority list)   fp16 FAIL  bf16 FAIL  fp32 FAIL
#       -> "cuDNN error: CUDNN_STATUS_INTERNAL_ERROR" / "cutlassF: no kernel found"
#     math only                       fp16 OK    bf16 OK    fp32 OK
#
# The text encoder reaches attention_pytorch through optimized_attention_for_device
# (small_input=True), the VAE reaches comfy.ops.scaled_dot_product_attention
# directly, and the main model reaches it whenever split attention falls back. So this
# breaks the whole graph, and when it does not raise it returns NaN -> black images.
#
# comfy/ops.py binds the real function at import time, so patching
# torch.nn.functional alone is not enough: comfy.ops must be patched as well.
#
#
# comfy/ops.py 用 `set_priority=True` 安装了下面这个优先级列表：
#
#     SDPA_BACKEND_PRIORITY = [FLASH_ATTENTION, CUDNN_ATTENTION, EFFICIENT_ATTENTION, MATH]
#     with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True):
#         return torch.nn.functional.scaled_dot_product_attention(...)
#
# `set_priority=True` 会让 PyTorch 无视 patch_torch() 设置的 enable_* 开关、直接按这个
# 顺序逐个尝试。FLASH 与 cuDNN 在 ZLUDA 上都不存在，因此每次注意力调用都会失败。
# 在 gfx906 上实测（1x16x512x128，fp16）：
#
#     裸调 F.sdpa                     fp16 OK    bf16 OK    fp32 OK
#     comfy.ops.sdpa（优先级列表）    fp16 FAIL  bf16 FAIL  fp32 FAIL
#       -> "cuDNN error: CUDNN_STATUS_INTERNAL_ERROR" / "cutlassF: no kernel found"
#     仅 math                         fp16 OK    bf16 OK    fp32 OK
#
# 文本编码器经 optimized_attention_for_device（small_input=True）走到 attention_pytorch，
# VAE 直接走 comfy.ops.scaled_dot_product_attention，主模型在 split attention 回退时
# 也会走到这里。所以整张图都会坏；不抛异常的时候它返回 NaN，出来的就是全黑图。
#
# comfy/ops.py 在导入时就把真实函数绑定了，所以只补 torch.nn.functional 不够，
# 必须同时补 comfy.ops。set_priority=True 会无视 patch_torch() 里的 enable_* 开关。
# -----------------------------------------------------------------------------

_MATH_ONLY_SDPA_FLAG = "_zluda_math_only_sdpa"


def _wrap_attr(module, name):
    target = getattr(module, name, None)
    if target is None or getattr(target, _MATH_ONLY_SDPA_FLAG, False):
        return False

    def _math_only_sdpa(q, k, v, *args, **kwargs):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH], set_priority=True):
            return target(q, k, v, *args, **kwargs)

    setattr(_math_only_sdpa, _MATH_ONLY_SDPA_FLAG, True)
    setattr(module, name, _math_only_sdpa)
    return True


def _install_math_only_sdpa(comfy_ops):
    # comfy_ops is the module object that was just executed. It must NOT be imported
    # again here: while exec_module runs, comfy.ops is not in sys.modules yet, so
    # `import comfy.ops` raises "cannot access submodule 'ops' of module 'comfy'
    # (most likely due to a circular import)". Observed in the field on 2026-10-14:
    # the patch reported sdpa=math-only but logged
    #   math-only SDPA FAILED AttributeError("cannot access submodule 'ops' ...")
    # and the images stayed black.
    #
    # comfy_ops 是刚刚执行完的模块对象。这里绝对不能再 import 一次：exec_module 执行期间
    # comfy.ops 还没进 sys.modules，`import comfy.ops` 会报
    # "cannot access submodule 'ops' of module 'comfy' (most likely due to a circular import)"。
    # 2026-10-14 实测踩到：补丁报告 sdpa=math-only，但日志里是
    #   math-only SDPA FAILED AttributeError("cannot access submodule 'ops' ...")
    # 于是图仍然是全黑。
    if comfy_ops is None:
        return False
    wrapped = _wrap_attr(comfy_ops, "scaled_dot_product_attention")
    # The can_use_* probes are only used for the GQA fast path; they report FLASH
    # as usable here, which would skip the repeat_kv fallback that actually works.
    #
    # can_use_* 探测只用于 GQA 快路径；在这里会误报 FLASH 可用，从而跳过真正能用的
    # repeat_kv 回退，所以要一并关掉。
    for probe in ("can_use_flash_attention", "can_use_cudnn_attention", "can_use_efficient_attention"):
        if hasattr(comfy_ops, probe):
            setattr(comfy_ops, probe, lambda *a, **k: False)
    # True when the wrapper is in place now, or was already installed by an earlier
    # call: `_wrap_attr` returns False in both cases it can hit, and only a missing
    # attribute means math-only is not in place.
    #
    # 包装器是新装的、或者之前就装好了都算 True：`_wrap_attr` 返回 False 的两种情况
    # 里，只有属性不存在才代表 math-only 没装上。
    return wrapped or getattr(comfy_ops, "scaled_dot_product_attention", None) is not None


# comfy.ops is imported after the bootstrap block runs, so hook the import and
# patch it the moment it loads.
#
# comfy.ops 在引导代码之后才被导入，所以挂钩导入，在它加载的瞬间就补上。
class _ComfyOpsLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        # Only the SDPA wrapper needs comfy.ops itself. The INT8 fallback lives on
        # F.linear, and comfy.ops.MixedPrecisionOps.Linear._forward calls F.linear,
        # so wrapping F.linear in patch_torch() already covers it.
        #
        # 只有 SDPA 包装器需要 comfy.ops 本体。INT8 回退挂在 F.linear 上，而
        # comfy.ops.MixedPrecisionOps.Linear._forward 调用的就是 F.linear，
        # 所以 patch_torch() 里包住 F.linear 已经覆盖到了。
        try:
            if not _install_math_only_sdpa(module):
                # Same greppable prefix as the exception path below: there was no
                # scaled_dot_product_attention to wrap, so math-only is not in place.
                #
                # 和下面异常分支用同一个可 grep 的前缀：没有
                # scaled_dot_product_attention 可包，math-only 没装上。
                _report("math-only SDPA FAILED comfy.ops.scaled_dot_product_attention is missing")
                _hint(
                    "comfy.ops 里没有 scaled_dot_product_attention，math-only 没装上："
                    "注意力会退回会崩的 FLASH / cuDNN 优先级列表。"
                )
        except Exception as e:
            _report("math-only SDPA FAILED %r" % (e,))
            _hint(
                "comfy.ops 的 math-only SDPA 包装失败：注意力会退回会崩的 FLASH / "
                "cuDNN 优先级列表，图会变黑。"
            )


class _ComfyOpsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "comfy.ops":
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = importlib.util.find_spec(fullname)
        if spec is not None and spec.loader is not None:
            spec.loader = _ComfyOpsLoader(spec.loader)
        return spec


# =============================================================================
# 2) INT8 tensorwise fallback
# 2) INT8 tensorwise 回退
# =============================================================================
# comfy_kitchen's TensorWiseINT8Layout handler ends in torch._int_mm, i.e. a
# cuBLASLt INT8 GEMM (abType 3 = int8, computeType 72 = CUBLAS_COMPUTE_32I).
# ZLUDA does not implement it, so any model carrying `int8_tensorwise` layers dies:
#
#     RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling
#     cublasLtMatmul ... abType 3 cType 10 computeType 72
#
# This is *not* about the compute dtype: --fp16-unet cannot help, because
# `int8_tensorwise` sets quantize_input=False, so cast_bias_weight hands the
# QuantizedTensor straight to F.linear and never dequantizes it.
#
# The dequantized result can be cached per weight object, but that doubles the
# weight footprint (measured: 5.72 GB INT8 -> 11.44 GB fp16 for a 6.14 B parameter
# model), which does not fit next to the text encoder on a 16 GB card. So the cache
# is off by default; set ZLUDA_INT8_DEQUANT_CACHE=1 to trade VRAM for speed.
# Measured on the 11520x3840 qkv layer of z_image_turbo_int8_convrot (m=90, fp16):
#
#     no cache    24.4 ms per call
#     with cache   2.0 ms per call   (native fp16 GEMM of the same shape: 1.34 ms)
#
# Either way this is the price of running an INT8 tensorwise model here; a
# convrot_w4a4 / fp8 / bf16 model avoids it entirely.
#
#
# comfy_kitchen 的 TensorWiseINT8Layout 处理函数最终会落到 torch._int_mm，也就是
# cuBLASLt 的 INT8 GEMM（abType 3 = int8，computeType 72 = CUBLAS_COMPUTE_32I）。
# ZLUDA 没有实现它，所以任何带 `int8_tensorwise` 层的模型都会挂掉：
#
#     RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling
#     cublasLtMatmul ... abType 3 cType 10 computeType 72
#
# 这跟计算精度无关：--fp16-unet 也救不了，因为 `int8_tensorwise` 设了
# quantize_input=False，cast_bias_weight 会把 QuantizedTensor 直接交给 F.linear，
# 从不反量化。
#
# 反量化结果可以按权重对象缓存，但权重占用会翻倍（实测 6.14B 参数模型：
# INT8 5.72 GB -> fp16 11.44 GB），16 GB 卡上放不下文本编码器和它并存。所以缓存
# 默认关闭；设 ZLUDA_INT8_DEQUANT_CACHE=1 用显存换速度。
# 在 z_image_turbo_int8_convrot 的 11520x3840 qkv 层上实测（m=90，fp16）：
#
#     不缓存    每次调用 24.4 ms
#     开缓存    每次调用 2.0 ms   （同形状的原生 fp16 GEMM：1.34 ms）
#
# 无论哪种，这都是在这台机器上跑 INT8 tensorwise 模型要付的代价；换成
# convrot_w4a4 / fp8 / bf16 模型就完全避开了。
# -----------------------------------------------------------------------------

_DEQUANT_CACHE = weakref.WeakKeyDictionary()
_DEQUANT_WARNED = set()
# Caching the dequantized weights is much faster but doubles the weight footprint
# (measured: 5.72 GB INT8 -> 11.44 GB fp16 for a 6.14 B parameter model), which
# does not fit next to the text encoder on a 16 GB card. Off by default; set
# ZLUDA_INT8_DEQUANT_CACHE=1 to trade VRAM for speed.
#
# 缓存反量化权重会快很多，但权重占用翻倍（实测 6.14B 参数模型：INT8 5.72 GB ->
# fp16 11.44 GB），在 16 GB 卡上放不下文本编码器。默认关闭；设
# ZLUDA_INT8_DEQUANT_CACHE=1 用显存换速度。
_DEQUANT_CACHE_ENABLED = os.environ.get("ZLUDA_INT8_DEQUANT_CACHE", "") not in ("", "0", "false", "False")


def _quant_weight_type_names():
    """Class names that count as a comfy_kitchen quantized weight.

    算作 comfy_kitchen 量化权重的类名。
    """
    names = {"QuantizedTensor"}
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        names.add(QuantizedTensor.__name__)
    except Exception:
        pass
    return names


_QUANT_NAMES = _quant_weight_type_names()


def _dequantize_cached(weight, dtype):
    # Keyed by the weight object, so the entry is dropped together with the model.
    # 以权重对象为键，因此缓存随模型一起释放。
    if not _DEQUANT_CACHE_ENABLED:
        if "warn" not in _DEQUANT_WARNED:
            _DEQUANT_WARNED.add("warn")
            _report(
                "INT8 tensorwise model: dequantizing weights on every forward pass. This works but "
                "is slower than a bf16 / fp8 / convrot_w4a4 model; set ZLUDA_INT8_DEQUANT_CACHE=1 "
                "to cache them if you have spare VRAM."
            )
            _hint(
                "INT8 tensorwise 模型：每次前向都会反量化权重；可用但比 bf16 / fp8 / "
                "convrot_w4a4 慢。显存充裕时设 ZLUDA_INT8_DEQUANT_CACHE=1 可缓存提速。"
            )
        return weight.dequantize().to(dtype=dtype)
    try:
        cached = _DEQUANT_CACHE.get(weight)
    except TypeError:
        # Not weak-referenceable: dequantize without caching.
        # 不可弱引用：直接反量化，不缓存。
        return weight.dequantize().to(dtype=dtype)
    if cached is not None and cached.dtype == dtype:
        return cached
    dequantized = weight.dequantize().to(dtype=dtype)
    try:
        _DEQUANT_CACHE[weight] = dequantized
    except TypeError:
        pass
    if "warn" not in _DEQUANT_WARNED:
        _DEQUANT_WARNED.add("warn")
        _report(
            "INT8 tensorwise model: weights are dequantized to %s and cached. This works but costs "
            "roughly twice the weight VRAM." % (dtype,)
        )
        _hint(
            "INT8 tensorwise 模型：权重会反量化成 %s 并缓存；可用，但权重显存占用约翻倍。"
            % (dtype,)
        )
    return dequantized


def patch_torch(torch):
    msgs = []

    # ZLUDA has no cuDNN; leaving it enabled fails with CUDNN_STATUS_INTERNAL_ERROR.
    # ZLUDA 没有 cuDNN；不关闭会报 CUDNN_STATUS_INTERNAL_ERROR。
    try:
        torch.backends.cudnn.enabled = False
        msgs.append("cudnn=%s" % torch.backends.cudnn.enabled)
    except Exception as e:
        msgs.append("cudnn FAILED %r" % (e,))

    # Keep only the math SDP backend (flash and mem-efficient attention are turned off).
    # 注意力只保留 math SDP（关闭 flash 与 mem-efficient）。
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        msgs.append("sdp=math")
    except Exception as e:
        msgs.append("sdp FAILED %r" % (e,))

    # The enable_* flags above are not enough on their own: comfy/ops.py puts
    # FLASH / cuDNN back in front through set_priority=True, so wrap the function
    # that everything actually calls.
    #
    # 上面的 enable_* 开关本身不够：comfy/ops.py 用 set_priority=True 又把 FLASH /
    # cuDNN 排到最前，所以要包住大家真正调用的那个函数。
    try:
        import torch.nn.functional as F

        if _wrap_attr(F, "scaled_dot_product_attention"):
            msgs.append("sdpa=math-only")
        elif getattr(F, "scaled_dot_product_attention", None) is None:
            # Nothing to wrap: this torch has no such function, and comfy/ops.py cannot call it either.
            # 没有可包的对象：这个 torch 没有这个函数，comfy/ops.py 也调不到它。
            msgs.append("sdpa=missing")
            _hint(
                "torch 里没有 F.scaled_dot_product_attention，注意力会退回会崩的 "
                "FLASH / cuDNN 优先级列表。"
            )
        else:
            # Already wrapped: math-only is already in place, this call changed nothing.
            # 已经被包装过：math-only 已经生效，这次调用没有改动什么。
            msgs.append("sdpa=math-only[already]")
    except Exception as e:
        msgs.append("sdpa FAILED %r" % (e,))

    # F.linear / addmm with a bias goes through the bias-fused GEMM of cuBLASLt, which ZLUDA does
    # not implement (CUBLAS_STATUS_NOT_SUPPORTED), so split it into matmul + add.
    # The same wrapper also has to keep comfy_kitchen's INT8 weights away from torch._int_mm.
    #
    # 带 bias 的 F.linear / addmm 会走 cuBLASLt 的 bias 融合 GEMM，而 ZLUDA 没有实现它
    # （CUBLAS_STATUS_NOT_SUPPORTED），所以把它拆成 matmul + 加法。
    # 同一个包装器还要把 comfy_kitchen 的 INT8 权重挡在 torch._int_mm 之外。
    try:
        import torch.nn.functional as F

        if getattr(F.linear, "__name__", "") != "_linear":
            _orig_linear = F.linear

            def _linear(input, weight, bias=None):
                if type(weight).__name__ in _QUANT_NAMES:
                    weight = _dequantize_cached(weight, input.dtype)
                if bias is not None and input.is_cuda and weight.dim() == 2:
                    return torch.matmul(input, weight.t()) + bias
                return _orig_linear(input, weight, bias)

            F.linear = _linear

        if getattr(torch.addmm, "__name__", "") != "_addmm":
            _orig_addmm = torch.addmm

            def _addmm(bias, m1, m2, *, beta=1, alpha=1, out=None):
                if out is None and beta == 1 and alpha == 1 and getattr(bias, "is_cuda", False):
                    return torch.matmul(m1, m2) + bias
                return _orig_addmm(bias, m1, m2, beta=beta, alpha=alpha, out=out)

            torch.addmm = _addmm

        import torch.nn.functional as _F

        msgs.append("linear-patch[F.linear=%s]" % _F.linear.__name__)
    except Exception as e:
        msgs.append("linear-patch FAILED %r" % (e,))

    # comfy.ops is imported later, so hook the import and patch it as soon as it loads.
    # comfy.ops 稍后才被导入，所以挂钩导入，在它加载时立刻补上。
    try:
        sys.meta_path.insert(0, _ComfyOpsFinder())
        msgs.append("ops-hook=installed")
    except Exception as e:
        msgs.append("ops-hook FAILED %r" % (e,))

    return msgs


# Environment variables and DLL preloading must be done before torch is imported.
# 环境变量设置与 DLL 预加载必须在导入 torch 之前完成。
apply_env()
_pre, _pre_zh = preload_zluda()
_report("Preload: " + _pre)
if _pre_zh:
    _hint(_pre_zh)

try:
    import torch

    _msgs = patch_torch(torch)
    _report(
        "OK torch=%s | Tensile=%s | %s"
        % (
            torch.__version__,
            os.environ.get("ROCBLAS_TENSILE_LIBPATH"),
            " | ".join(_msgs),
        )
    )
    _hint_if_failed(_msgs)
except Exception as _e:
    _report("Direct import failed: %r" % (_e,))
    _hint(
        "补丁在导入 torch 时抛了异常。后面应紧跟 Fallback hook installed；"
        "若没有，说明连回退钩子都没装上。"
    )

    # torch cannot be imported yet: install an import hook that patches it right after it loads.
    # 此时还无法导入 torch：安装导入钩子，在 torch 加载完成后立刻打补丁。
    class _Loader(importlib.abc.Loader):
        def __init__(self, loader):
            self.loader = loader

        def create_module(self, spec):
            return self.loader.create_module(spec)

        def exec_module(self, module):
            apply_env()
            self.loader.exec_module(module)
            _msgs = patch_torch(module)
            _report("Late torch=%s | %s" % (module.__version__, " | ".join(_msgs)))
            _hint_if_failed(_msgs)
            # The module tail skipped this call because torch was not in sys.modules
            # back then, so the fallback path has to do it here. Any failure is
            # reported instead of raised: raising would break `import torch` itself.
            #
            # 模块末尾当时因为 sys.modules 里没有 torch 而跳过了这一步，所以回退路径要在这里补上。
            # 失败只报告不抛出：抛出会把 `import torch` 本身搞挂。
            try:
                _fallbacks = _install_operator_fallbacks(module)
            except Exception as e:
                # No `ops:` line here: a half-applied list cannot be vouched for, and
                # the `ops FAILED` line above is the signal to grep for.
                #
                # 这里不输出 `ops:` 行：装到一半的结果无法保证准确，上面那行
                # `ops FAILED` 才是要 grep 的信号。
                _report("ops FAILED %r" % (e,))
                _hint(
                    "可选算子兜底没有装上：设了 ZLUDA_TOPK_CPU / ZLUDA_STFT_CPU / "
                    "ZLUDA_NO_JITSCRIPT 也不会生效。"
                )
            else:
                _report("ops: " + (", ".join(_fallbacks) if _fallbacks else "default"))

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "torch":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec and spec.loader:
                spec.loader = _Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())
    _report("Fallback hook installed")
    _hint(
        "已安装导入钩子，torch 加载完成后会补打补丁；下一次 import torch 之后应出现 "
        "Late torch= 和 ops: 两行。"
    )


# =============================================================================
# 8) Launch arguments the setup can no longer rely on
# 8) 不能再依赖的启动参数
# =============================================================================
# The launcher keeps its flags in .launcher/preference.json: a launcher reset drops
# them, and any entry point that does not go through main.py never sets them at all.
# comfy/cli_args.py parses at import time and everything else (cuda_malloc.py,
# comfy/model_management.py, comfy/ldm/modules/attention.py) reads `args` afterwards,
# so overriding the object once, right after it is built, is enough.
#
# 启动器把参数存在 .launcher/preference.json 里：重置就没了，绕过 main.py 的入口更是
# 完全拿不到。comfy/cli_args.py 在导入时解析，之后 cuda_malloc.py /
# comfy/model_management.py / comfy/ldm/modules/attention.py 才读取 args，
# 所以只要在它刚构造完之后改一次对象就够。

_ARGS_OVERRIDES = {}
if os.environ.get("ZLUDA_NO_ARG_OVERRIDES", "") not in ("", "0"):
    _report("args: overrides disabled by ZLUDA_NO_ARG_OVERRIDES")
else:
    # Windows pinned memory caps at 40% of RAM; this machine has 16 GB of RAM, and the
    # pinned buffers are also a frequent source of generation crashes.
    #
    # Windows 的固定内存上限是物理内存的 40%；这台机器有 16 GB 内存，而且固定缓冲区
    # 也是生成崩溃的常见来源。
    _ARGS_OVERRIDES["disable_pinned_memory"] = True
    # torch 2.7 falls back to the legacy ModelPatcher, so async offload buys nothing.
    # torch 2.7 会回退到旧的 ModelPatcher，所以异步 offload 没有收益。
    _ARGS_OVERRIDES["disable_async_offload"] = True
    # ZLUDA has no cudaMallocAsync; this replaces the launcher's --disable-cuda-malloc.
    # ZLUDA 没有 cudaMallocAsync；这里等价于启动器的 --disable-cuda-malloc。
    _ARGS_OVERRIDES["disable_cuda_malloc"] = True
    try:
        _RESERVE_VRAM = float(os.environ.get("ZLUDA_RESERVE_VRAM", "0.9"))
    except ValueError:
        _RESERVE_VRAM = 0.9
    if _RESERVE_VRAM > 0:
        _ARGS_OVERRIDES["reserve_vram"] = _RESERVE_VRAM
    # Opt-in: force fp16 compute. It also overrides the fp8 flags of fp8-stored models,
    # so it stays off by default; patch 9 already keeps bf16 out of the selection.
    #
    # 可选：强制 fp16 计算。它还会盖掉 fp8 存储模型自己的 fp8 标志，所以默认关闭；
    # 第 9 项补丁本来就已经把 bf16 挡在精度选择之外了。
    if os.environ.get("ZLUDA_FORCE_FP16", "") not in ("", "0"):
        _ARGS_OVERRIDES["fp16_unet"] = True
        _ARGS_OVERRIDES["fp16_text_enc"] = True
    # Opt-in: split attention instead of pytorch attention. Measure first: the math SDPA
    # path used today is not obviously slower than split.
    #
    # 可选：用 split attention 取代 pytorch attention。先实测：目前走的 math SDPA
    # 路径并不比 split 明显更慢。
    if os.environ.get("ZLUDA_SPLIT_ATTENTION", "") not in ("", "0"):
        _ARGS_OVERRIDES["use_split_cross_attention"] = True
        _ARGS_OVERRIDES["use_quad_cross_attention"] = False


def _patch_cli_args(module):
    args = getattr(module, "args", None)
    if args is None:
        _report("args: parsing deferred, overrides skipped")
        _hint(
            "comfy.cli_args 还没有 args 对象，启动参数覆盖被跳过：pinned memory / "
            "async offload / cudaMallocAsync 可能没有按补丁的默认值关掉。"
        )
        return
    applied = []
    for key, value in _ARGS_OVERRIDES.items():
        if key == "reserve_vram" and getattr(args, key, None) is not None:
            # An explicit --reserve-vram from the launcher wins.
            # 启动器显式给了 --reserve-vram 时以它为准。
            continue
        setattr(args, key, value)
        applied.append("%s=%s" % (key, value))
    _report("args: " + (", ".join(applied) if applied else "nothing to override"))


# =============================================================================
# 9) bf16 must never be selected on gfx906
# 9) gfx906 上绝不能选中 bf16
# =============================================================================
# ZLUDA reports compute capability (8, 8), and should_use_bf16() returns True as soon as
# props.major >= 8, before it ever asks the device whether bf16 is really supported.
# Measured on gfx906: fp16 9.54 TFLOPS, bf16 5.26 TFLOPS, while models that list
# bfloat16 first in supported_inference_dtypes (Anima is one) take the first branch.
# Every caller uses comfy.model_management.should_use_bf16, including the bare-name
# calls inside model_management itself, so replacing the attribute is enough.
#
# ZLUDA 把算力报成 (8, 8)，should_use_bf16() 在 props.major >= 8 就直接返回 True，
# 根本没问设备是否真的支持 bf16。gfx906 实测 fp16 9.54 TFLOPS、bf16 5.26 TFLOPS，
# 而 supported_inference_dtypes 里 bf16 排第一的模型（Anima 就是）会直接跑慢一半。
# 所有调用方用的都是 comfy.model_management.should_use_bf16（包括 model_management
# 内部的裸名调用），所以替换这个属性就够了。
def _patch_model_management(module):
    if getattr(module, "_zluda_patched", False):
        return
    allow_bf16 = os.environ.get("ZLUDA_ALLOW_BF16", "") not in ("", "0")
    if not allow_bf16:
        module.should_use_bf16 = lambda *a, **k: False
    # Opt-in: force the VAE to fp32 when it allows it (fp16 VAE can decode to black).
    # 可选：在 VAE 允许时强制用 fp32（fp16 VAE 可能解码出全黑图）。
    if os.environ.get("ZLUDA_VAE_FP32", "") not in ("", "0"):
        _original_vae_dtype = module.vae_dtype

        def _vae_dtype(device=None, allowed_dtypes=[]):
            if module.torch.float32 in allowed_dtypes:
                return module.torch.float32
            return _original_vae_dtype(device, allowed_dtypes)

        module.vae_dtype = _vae_dtype
    module._zluda_patched = True
    _report(
        "model_management: bf16_off=%s | vae_fp32=%s"
        % (not allow_bf16, os.environ.get("ZLUDA_VAE_FP32", "") not in ("", "0"))
    )


# =============================================================================
# 10) ONNX Runtime: no CUDA / TensorRT execution provider
# 10) ONNX Runtime：不要 CUDA / TensorRT 执行提供程序
# =============================================================================
# onnxruntime-gpu 1.23 is built for CUDA 12 and ZLUDA only provides CUDA 11 libraries,
# so the provider DLL either fails to load or crashes inside the fake CUDA driver.
# comfyui_controlnet_aux puts CUDAExecutionProvider first by default, and the RMBG /
# Impact nodes build their sessions from the same provider list.
#
# onnxruntime-gpu 1.23 是 CUDA 12 构建，ZLUDA 只提供 CUDA 11 的库：provider DLL
# 要么加载失败，要么在假 CUDA 驱动里崩。controlnet_aux 默认把 CUDA 排第一，
# RMBG / Impact 用的也是同一份 provider 列表。
def _patch_onnxruntime(module):
    if getattr(module, "_zluda_patched", False):
        return
    if os.environ.get("ZLUDA_KEEP_ONNX_CUDA", "") not in ("", "0"):
        module._zluda_patched = True
        _report("onnxruntime: left untouched (ZLUDA_KEEP_ONNX_CUDA)")
        return
    original = module.get_available_providers
    unavailable = ("CUDAExecutionProvider", "TensorrtExecutionProvider")

    def _filtered():
        return [p for p in original() if p not in unavailable]

    module.get_available_providers = _filtered
    try:
        # The session path reads this pybind copy, not the module-level function.
        # 创建 session 时读的是这个 pybind 副本，不是模块级函数。
        module.capi._pybind_state.get_available_providers = _filtered
    except Exception:
        pass

    _session = module.InferenceSession

    class _CpuInferenceSession(_session):
        def __init__(self, *a, providers=None, **kw):
            if providers:
                providers = [p for p in providers if p not in unavailable] or ["CPUExecutionProvider"]
                _write_stream(sys.stdout, "[ZLUDA] ONNX Runtime: forcing CPUExecutionProvider\n")
                _write_stream(
                    sys.stdout,
                    "[ZLUDA] 说明: ONNX Runtime 强制使用 CPUExecutionProvider"
                    "（CUDA 12 构建的 onnxruntime-gpu 在 ZLUDA 上不可用）。\n",
                )
            super().__init__(*a, providers=providers, **kw)

    module.InferenceSession = _CpuInferenceSession
    module._zluda_patched = True
    _report("onnxruntime: CUDA/TensorRT providers filtered")


# =============================================================================
# 11) LTXAV memory estimate
# 11) LTXAV 显存估算
# =============================================================================
# Upstream leaves memory_usage_factor at 0.077, which is the multiplier model_base.py
# applies to the activation estimate, so ComfyUI plans as if LTXAV needs almost nothing
# and then runs out of VRAM. The parent LTXV class uses
# (cross_attention_dim / 2048) * 5.5; LTXAV gets the same formula with a 1.5x margin.
#
# 上游把 memory_usage_factor 留在 0.077，model_base.py 用它乘激活显存估算，
# 于是 ComfyUI 以为 LTXAV 几乎不吃显存，然后在 16 GB 卡上 OOM。父类 LTXV 用的是
# (cross_attention_dim / 2048) * 5.5；LTXAV 采用同一公式，再加 1.5 倍余量。
def _patch_supported_models(module):
    ltxav = getattr(module, "LTXAV", None)
    if ltxav is None or getattr(ltxav, "_zluda_patched", False):
        return
    _original_init = ltxav.__init__

    def _init(self, unet_config):
        _original_init(self, unet_config)
        dim = unet_config.get("cross_attention_dim", 4096)
        self.memory_usage_factor = (dim / 2048) * 5.5 * 1.5

    ltxav.__init__ = _init
    ltxav._zluda_patched = True
    _report("supported_models: LTXAV memory_usage_factor patched")


# =============================================================================
# 12) Optional operator fallbacks (off by default)
# 12) 可选的算子兜底（默认关闭）
# =============================================================================
# ZLUDA does not implement every operator. The Zluda fork replaces topk / stft /
# jit.script unconditionally, which is expensive: every topk becomes a D2H + topk +
# H2D round trip, and torchvision NMS calls topk internally. Enable only the one that
# actually misbehaves on this machine.
#
# ZLUDA 并非实现了所有算子。Zluda 仓库是无条件替换 topk / stft / jit.script 的，
# 代价很大：每次 topk 都变成 D2H + topk + H2D，而 torchvision 的 NMS 内部就用 topk。
# 只开这台机器上真正出问题的那一个。
def _install_operator_fallbacks(torch_module):
    # Two entry points call this (the module tail and the fallback loader) and they are
    # mutually exclusive today, but stay idempotent so a future double call cannot wrap
    # topk twice, and so the second call still reports what was applied.
    #
    # Read through __dict__ on purpose: torch defines a module-level __getattr__, and a
    # missing attribute would go through it instead of returning the default.
    #
    #
    # 有两个入口会调用它（模块末尾与回退加载器），目前互斥；这里保持幂等：以后即使重复
    # 调用也不会把 topk 包两层，而且第二次调用仍能报出已经装了什么。
    #
    # 故意走 __dict__ 读：torch 定义了模块级 __getattr__，属性不存在时会先走它而不是
    # 直接返回默认值。
    if torch_module.__dict__.get("_zluda_ops_patched", False):
        return list(torch_module.__dict__.get("_zluda_ops_applied", ()))
    applied = []
    if os.environ.get("ZLUDA_TOPK_CPU", "") not in ("", "0"):
        _original_topk = torch_module.topk

        def _topk(input, *a, **k):
            device = input.device
            values, indices = _original_topk(input.cpu(), *a, **k)
            return torch_module.return_types.topk((values.to(device), indices.to(device)))

        torch_module.topk = _topk
        applied.append("topk")
    if os.environ.get("ZLUDA_STFT_CPU", "") not in ("", "0"):
        _original_stft = torch_module.stft
        _original_istft = torch_module.istft

        def _stft(input, window, *a, **k):
            return _original_stft(input=input.cpu(), window=window.cpu(), *a, **k).to(input.device)

        def _istft(input, window, *a, **k):
            return _original_istft(input=input.cpu(), window=window.cpu(), *a, **k).to(input.device)

        torch_module.stft = _stft
        torch_module.istft = _istft
        applied.append("stft")
    if os.environ.get("ZLUDA_NO_JITSCRIPT", "") not in ("", "0"):
        torch_module.jit.script = lambda f, *a, **k: f
        applied.append("jit.script")
    torch_module._zluda_ops_patched = True
    torch_module._zluda_ops_applied = tuple(applied)
    return applied


# =============================================================================
# 13) Import hooks: patch each module the moment it finishes executing
# 13) 导入钩子：每个模块执行完的瞬间就给它打补丁
# =============================================================================
# comfy.* is imported after the bootstrap block in main.py, so these modules do not
# exist yet when this file runs. Hook the import instead of importing them here: the
# latter would drag the whole ComfyUI import chain into a standalone
# `import zluda_patch` self-check.
#
# comfy.* 是在 main.py 的引导代码之后才导入的，所以这些模块此刻还不存在；
# 用导入钩子，而不是在这里 import：后者会把整条 ComfyUI 导入链拉进
# 独立的 `import zluda_patch` 自检里。
class _ModulePatchLoader(importlib.abc.Loader):
    def __init__(self, loader, fn):
        self.loader = loader
        self.fn = fn

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        try:
            self.fn(module)
        except Exception as e:
            _report("hook FAILED %s %r" % (module.__name__, e))
            _hint(
                "模块补丁钩子在 %s 上执行失败，这个模块的补丁没生效。" % module.__name__
            )


class _ModulePatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self, mapping):
        self.mapping = mapping
        self.busy = set()

    def find_spec(self, fullname, path=None, target=None):
        fn = self.mapping.get(fullname)
        if fn is None or fullname in self.busy:
            return None
        # importlib.util.find_spec() walks sys.meta_path again, so guard the recursion.
        # importlib.util.find_spec() 会再次遍历 sys.meta_path，用 busy 集合防递归。
        self.busy.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self.busy.discard(fullname)
        if spec is not None and spec.loader is not None:
            spec.loader = _ModulePatchLoader(spec.loader, fn)
        return spec


_MODULE_PATCHES = {
    "comfy.cli_args": _patch_cli_args,
    "comfy.model_management": _patch_model_management,
    "comfy.supported_models": _patch_supported_models,
    "onnxruntime": _patch_onnxruntime,
}

# Patch what is already imported, then hook everything that is not.
# 已经导入的直接补，其余挂到导入钩子上。
for _name, _fn in _MODULE_PATCHES.items():
    _loaded = sys.modules.get(_name)
    if _loaded is not None:
        try:
            _fn(_loaded)
        except Exception as _e:
            _report("hook FAILED %s %r" % (_name, _e))
            _hint(
                "补丁 %s 没有补上（该模块在导入补丁之前就已经加载）。" % _name
            )
sys.meta_path.insert(0, _ModulePatchFinder(_MODULE_PATCHES))
_report("hooks: " + ", ".join(_MODULE_PATCHES))

if "torch" in sys.modules:
    _fallbacks = _install_operator_fallbacks(sys.modules["torch"])
    _report("ops: " + (", ".join(_fallbacks) if _fallbacks else "default"))

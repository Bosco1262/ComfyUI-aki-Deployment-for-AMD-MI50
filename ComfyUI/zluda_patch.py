"""
ZLUDA compatibility patch for MI50 / gfx906 + ZLUDA + ComfyUI.

One file, standard library only. Import it before ComfyUI imports torch:

    import zluda_patch  # noqa: F401

Contract
  switches      ZLUDA_* environment variables, defaults in PART 0.
  status lines  ASCII English, "[ZLUDA] " on stderr, mirrored to the status
                files. A line a human must act on is followed by its own
                "说明: " line.
  failure token "FAILED", grep the status file for it.
  uninstall     delete this file and the import line in main.py.

Layout
  PART 0  contract    paths, switches
  PART 1  toolkit     reporting, env parsing, import hook loader and finder
  PART 2  patches     one unit per behaviour, stable IDs
  PART 3  assembly    registries, torch bring-up, bootstrap

Units
  PRELOAD, TORCH, SDPA, LINEAR, CLI_ARGS, MODEL_MANAGEMENT, ONNX,
  SUPPORTED_MODELS, OPS, MODEL_DETECTION


ZLUDA 兼容补丁（MI50 / gfx906 + ZLUDA + ComfyUI）。

单文件，只用标准库。必须在 ComfyUI 导入 torch 之前导入：

    import zluda_patch  # noqa: F401

契约
  开关      ZLUDA_* 环境变量，默认值在 PART 0。
  状态行    纯 ASCII 英文，走 stderr 并带 "[ZLUDA] " 前缀，同时镜像到状态
            文件。需要人处理的行，下面单独跟一行 "说明: "。
  失败标记  "FAILED"，在状态文件里 grep 它。
  卸载      删掉本文件，以及 main.py 里的那行 import。

布局
  PART 0  契约    路径、开关
  PART 1  工具    报告、环境变量解析、导入钩子的 loader 与 finder
  PART 2  补丁    一个行为一个单元，稳定 ID
  PART 3  装配    注册表、torch 引导、bootstrap

单元
  PRELOAD、TORCH、SDPA、LINEAR、CLI_ARGS、MODEL_MANAGEMENT、ONNX、
  SUPPORTED_MODELS、OPS、MODEL_DETECTION
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
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, NamedTuple, Optional

# ==================================================================================================
# PART 0 — CONTRACT
# PART 0 — 契约
# ==================================================================================================

# Package root = the folder above the ComfyUI folder that contains this file.
# 整合包根目录 = 本文件所在的 ComfyUI 文件夹的上一级。
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HIP SDK root: prefer HIP_PATH, fall back to the tested default.
# HIP SDK 根目录：优先读 HIP_PATH，取不到时回退到实测环境的默认路径。
HIP_ROOT = os.environ.get("HIP_PATH") or r"C:\Program Files\AMD\ROCm\5.7"

# rocBLAS Tensile kernel directory; it must contain the gfx906 files.
# rocBLAS Tensile 内核目录，必须包含 gfx906 的文件。
TENSILE_DIR = os.path.join(HIP_ROOT, "bin", "rocblas", "library")

# Optional custom ZLUDA directory, used when the extpack was extracted by hand.
# 可选的自定义 ZLUDA 目录，用于手工解压 extpack 的情况。
EXTRA_ZLUDA_DIRS = [os.path.join(ROOT_DIR, "zluda")]

# DLLs in load order: nvcuda first, then the libraries torch will ask for.
# 按加载顺序排列的 DLL：nvcuda 在前，然后是 torch 会用的库。
ZLUDA_DLLS = (
    "nvcuda.dll",
    "nvml.dll",
    "cublas64_11.dll",
    "cusparse64_11.dll",
    "nvrtc64_112_0.dll",
)

# Status files: one in the package root, one in the system temp directory.
# 状态文件：一份在整合包根目录，一份在系统临时目录。
REPORTS = [
    os.path.join(ROOT_DIR, "_zluda_status.txt"),
    os.path.join(tempfile.gettempdir(), "zluda_status.txt"),
]

# --------------------------------------------------------------------------------------------------
# SWITCHES — one canonical order, environment first
# SWITCHES — 唯一顺序，环境变量优先
# --------------------------------------------------------------------------------------------------
# Every switch is read once at import. The order below is also the order of the
# "switches:" status line, so the log and the source can be compared line by
# line.
#
# 每个开关只在导入时读一次。下面的顺序就是 "switches:" 状态行的顺序，日志和
# 源码可以逐行对照。


# Read a boolean switch: unset falls back to the default; "0", "false",
# "False" and "" are false.
#
# 读取布尔开关：未设置时用默认值；"0"、"false"、"False" 和 "" 视为假。
def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in ("", "0", "false", "False")


# Read a float switch; a malformed value falls back to the default.
# 读取浮点开关；值写错时回退到默认值。
def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Read a choice switch; unset or an unknown value falls back to the default.
# 读取枚举开关；未设置或写了不认识的值时回退到默认值。
def _env_choice(name, default, choices=("auto", "comfy")):
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value if value in choices else default


# Count Anima / Cosmos-Predict2 blocks from the checkpoint; "=0" turns it off.
# 从检查点数 Anima / Cosmos-Predict2 的层数；"=0" 关闭。
ZLUDA_ANIMA_EXTRA_BLOCKS = _env_bool("ZLUDA_ANIMA_EXTRA_BLOCKS", True)

# VRAM in GB reserved for the OS and other programs; 0 disables the reserve.
# 为系统和其它程序预留的显存，单位 GB；0 表示不预留。
ZLUDA_RESERVE_VRAM = _env_float("ZLUDA_RESERVE_VRAM", 1.5)

# Safety margin on the fp32 weight budget. At fp32 the activations double as
# well, and the budget only counts weights, so the decision keeps a margin.
# A value outside (0, 1] falls back to 0.9.
#
# fp32 权重预算的安全余量。fp32 下激活也会翻倍，而预算只算权重，所以判断要
# 留余量。取值超出 (0, 1] 时回退到 0.9。
ZLUDA_FP32_MARGIN = _env_float("ZLUDA_FP32_MARGIN", 0.9)
if not 0 < ZLUDA_FP32_MARGIN <= 1:
    ZLUDA_FP32_MARGIN = 0.9

# Force fp16 compute; also overrides the fp8 flags of fp8-stored models.
# 强制 fp16 计算；会顺带盖掉 fp8 存储模型自己的 fp8 标志。
ZLUDA_FORCE_FP16 = _env_bool("ZLUDA_FORCE_FP16")

# Use split attention instead of pytorch / math SDP.
# 用 split attention 取代 pytorch / math SDP。
ZLUDA_SPLIT_ATTENTION = _env_bool("ZLUDA_SPLIT_ATTENTION")

# Disable both dtype policies at once; each side then uses ComfyUI's own order.
# Kept as the coarse switch: this card has no native bf16, so "hand both back"
# is the one thing a user usually wants. The two switches below are the fine
# controls, one per side.
#
# 一次关掉两侧的 dtype 策略，两侧都回到 ComfyUI 自己的顺序。保留为粗粒度
# 总开关：这张卡没有原生 bf16，"两边都交回去"通常是用户唯一想要的；下面两个
# 才是分侧细调。
ZLUDA_ALLOW_BF16 = _env_bool("ZLUDA_ALLOW_BF16")

# Per-side policy: "auto" = this patch decides, "comfy" = upstream order.
# 分侧开关："auto" = 本补丁决定，"comfy" = 交回 ComfyUI 原顺序。
ZLUDA_UNET_DTYPE = _env_choice("ZLUDA_UNET_DTYPE", "auto")
ZLUDA_VAE_DTYPE = _env_choice("ZLUDA_VAE_DTYPE", "auto")

# The alias only fills the side that was not set explicitly, so
# "ZLUDA_ALLOW_BF16=1 ZLUDA_UNET_DTYPE=auto" keeps the UNet policy.
#
# 别名只填空缺的那一侧，所以 "ZLUDA_ALLOW_BF16=1 ZLUDA_UNET_DTYPE=auto"
# 仍会保留 UNet 策略。
if ZLUDA_ALLOW_BF16:
    if os.environ.get("ZLUDA_UNET_DTYPE") is None:
        ZLUDA_UNET_DTYPE = "comfy"
    if os.environ.get("ZLUDA_VAE_DTYPE") is None:
        ZLUDA_VAE_DTYPE = "comfy"

# Keep the launcher's own launch arguments; 1 = the launcher owns them again.
# 保留启动器自己的启动参数；1 = 改回由启动器说了算。
ZLUDA_NO_ARG_OVERRIDES = _env_bool("ZLUDA_NO_ARG_OVERRIDES")

# Leave ONNX Runtime's provider list untouched.
# 不动 ONNX Runtime 的 provider 列表。
ZLUDA_KEEP_ONNX_CUDA = _env_bool("ZLUDA_KEEP_ONNX_CUDA")

# Cache dequantized INT8 weights; keeps the INT8 weights and adds a full fp16
# copy, about 1.5x the fp16 model (3x the INT8 weights).
#
# 缓存 INT8 反量化权重；原始 INT8 仍在，再存一份完整 fp16，总权重约 1.5× fp16
# 模型（3× INT8 权重）。
ZLUDA_INT8_DEQUANT_CACHE = _env_bool("ZLUDA_INT8_DEQUANT_CACHE")

# Run torch.topk on the CPU; detection nodes get slow otherwise.
# 让 torch.topk 走 CPU；否则检测类节点会明显变慢。
ZLUDA_TOPK_CPU = _env_bool("ZLUDA_TOPK_CPU")

# Run torch.stft / torch.istft on the CPU; audio models need this.
# 让 torch.stft / torch.istft 走 CPU；音频模型需要。
ZLUDA_STFT_CPU = _env_bool("ZLUDA_STFT_CPU")

# Make torch.jit.script a no-op; node packs that ship TorchScript need this.
# 让 torch.jit.script 变成空操作；自带 TorchScript 的节点包需要。
ZLUDA_NO_JITSCRIPT = _env_bool("ZLUDA_NO_JITSCRIPT")

# ==================================================================================================
# PART 1 — TOOLKIT
# PART 1 — 工具
# ==================================================================================================

# Write to a stream without losing the whole line to an encoding error.
# 向流写入，且绝不因为编码问题把整行丢掉。
def _write_stream(stream, text):
    try:
        # A non-UTF-8 console or redirected log raises UnicodeEncodeError on the
        # whole line, so replace the characters it cannot represent first.
        #
        # 非 UTF-8 的控制台或重定向日志会对整行抛 UnicodeEncodeError，所以先
        # 把编码不了的字符替换掉。
        encoding = getattr(stream, "encoding", None) or "utf-8"
        try:
            text = text.encode(encoding, "replace").decode(encoding, "replace")
        except (LookupError, UnicodeError):
            text = text.encode("ascii", "replace").decode("ascii")
        stream.write(text)
        stream.flush()
    except Exception:
        pass


# Print to stderr and append to every status file; write failures are ignored.
# 输出到 stderr 并追加写入所有状态文件；写入失败会被有意忽略。
def _report(text):
    line = "%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), text)
    _write_stream(sys.stderr, "[ZLUDA] " + text + "\n")
    for path in REPORTS:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except Exception:
            pass


# Chinese follow-up line for a message a human must act on.
# 需要人处理时另起一行的中文提示。
def _hint(zh):
    _report("说明: " + zh)


_WARNED = set()


# Report a runtime event once per process; later calls are silent.
# 运行时事件每个进程只报一次，后续调用静默。
def _report_once(key, text, hint=""):
    if key in _WARNED:
        return
    _WARNED.add(key)
    _report(text)
    if hint:
        _hint(hint)


# What a unit reports: a status fragment plus an optional Chinese hint. An
# empty status keeps the unit out of the log.
#
# 补丁单元返回什么：一个状态片段，外加可选的中文提示。状态为空时该单元不写
# 日志。
class Outcome(NamedTuple):
    status: str
    hint: str = ""


# A module hook plus the metadata the framework needs to report it.
# 模块钩子，外加框架报告所需的信息。
@dataclass(frozen=True)
class ModulePatch:
    key: str
    module: str
    apply: Callable[[ModuleType], Outcome]
    failure_zh: str


# One import hook: run `before` before the module body and `after` after it.
# 一个导入钩子：模块体执行前跑 `before`，执行后跑 `after`。
@dataclass(frozen=True)
class Hook:
    after: Callable[[ModuleType], None]
    before: Optional[Callable[[], None]] = None


# Wrap a loader: run the hook around the wrapped module body.
# 包装 loader：在被包装模块的模块体前后执行钩子。
class _HookLoader(importlib.abc.Loader):
    def __init__(self, loader, hook):
        self.loader = loader
        self.hook = hook

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        if self.hook.before is not None:
            self.hook.before()
        self.loader.exec_module(module)
        self.hook.after(module)


# Intercept named imports, wrap their loaders, and never break the import.
# 拦截指定模块的导入，包装其 loader，并且绝不让钩子本身搞挂导入。
class _HookFinder(importlib.abc.MetaPathFinder):
    def __init__(self, hooks):
        self.hooks = dict(hooks)
        self.busy = set()

    def find_spec(self, fullname, path=None, target=None):
        hook = self.hooks.get(fullname)
        if hook is None or fullname in self.busy:
            return None
        # importlib.util.find_spec() walks sys.meta_path again, so guard the
        # recursion; a broken hook returns None instead of raising.
        #
        # importlib.util.find_spec() 会再次遍历 sys.meta_path，所以用 busy 防
        # 递归；钩子自身出错时返回 None，而不是抛异常。
        self.busy.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        except (ImportError, AttributeError, ValueError):
            return None
        finally:
            self.busy.discard(fullname)
        if spec is not None and spec.loader is not None:
            spec.loader = _HookLoader(spec.loader, hook)
        return spec


# ==================================================================================================
# PART 2 — PATCHES
# PART 2 — 补丁
# ==================================================================================================
# Each unit below declares what it changes, why, how it fails, and its switch.
# 下面每个单元都声明改什么、为什么、怎么失败，以及对应的开关。


# --------------------------------------------------------------------------------------------------
# PRELOAD — ZLUDA DLLs and the rocBLAS Tensile path
# PRELOAD — ZLUDA DLL 与 rocBLAS Tensile 路径
# --------------------------------------------------------------------------------------------------
# Purpose   set ROCBLAS_TENSILE_LIBPATH and load the ZLUDA DLLs before torch.
# Cause     torch binds CUDA at import time; without the DLLs it reports
#           "no NVIDIA driver found" and never sees the GPU.
# Measured  n/a.
# Failure   Preload: ... | failed=... plus a Chinese hint.
# Switch    none.
#
# 作用  设置 ROCBLAS_TENSILE_LIBPATH，并在 torch 之前加载 ZLUDA 的 DLL。
# 原因  torch 在导入时就绑定 CUDA；没有这些 DLL 会报
#       "no NVIDIA driver found"，完全看不到显卡。
# 实测  不适用。
# 失败  Preload: ... | failed=...，并跟一行中文提示。
# 开关  无。


# Point rocBLAS at the gfx906 kernels and clear TORCH_BLAS_PREFER_CUBLASLT.
# 让 rocBLAS 使用 gfx906 内核，并清除 TORCH_BLAS_PREFER_CUBLASLT。
def apply_env():
    os.environ["ROCBLAS_TENSILE_LIBPATH"] = TENSILE_DIR
    os.environ.pop("TORCH_BLAS_PREFER_CUBLASLT", None)


# Load the ZLUDA DLLs into the process before `import torch`.
# 在 `import torch` 之前把 ZLUDA 的 DLL 加载进进程。
def preload_zluda():
    # The launcher unpacks the DLLs into a fresh <GUID> folder on every run, so
    # scan the temp roots for folders that contain cublas64_11.dll.
    #
    # A directory listed in EXTRA_ZLUDA_DIRS is scanned as well, for a hand
    # extraction of the extpack.
    #
    #
    # 启动器每次运行都会把 DLL 解压到临时目录下新的 <GUID> 文件夹，因此扫描
    # 各临时目录，找出包含 cublas64_11.dll 的文件夹。
    #
    # EXTRA_ZLUDA_DIRS 里的目录也会被扫描，用于手工解压 extpack 的情况。
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
        # TEMP, TMP and gettempdir() often point at the same directory.
        # TEMP、TMP 和 gettempdir() 经常指向同一个目录。
        if root in scanned or not os.path.isdir(root):
            continue
        scanned.add(root)
        for folder in glob.glob(os.path.join(root, "*")):
            dll = os.path.join(folder, "cublas64_11.dll")
            if os.path.isfile(dll):
                try:
                    candidates.append((os.path.getmtime(folder), folder))
                except OSError:
                    pass
    for folder in EXTRA_ZLUDA_DIRS:
        if os.path.isfile(os.path.join(folder, "cublas64_11.dll")):
            try:
                candidates.append((os.path.getmtime(folder), folder))
            except OSError:
                pass

    if not candidates:
        return Outcome(
            "No ZLUDA dir found (nvcuda.dll not preloaded)",
            "没找到含 ZLUDA DLL 的目录，nvcuda.dll 未预加载。先让启动器正常跑一次 "
            "ComfyUI（它会把 DLL 解到 %TEMP%\\<GUID>），或把 "
            ".launcher\\zluda.<hash>.extpack 用 7-Zip 解压到整合包根目录的 zluda\\",
        )

    # Newest folder first.
    # 最新的文件夹优先。
    candidates.sort(reverse=True)
    hip_bin = os.path.join(HIP_ROOT, "bin")

    # Try each candidate until both nvcuda.dll and cublas64_11.dll load.
    # 依次尝试每个候选目录，直到 nvcuda.dll 与 cublas64_11.dll 都加载成功。
    best = None
    for _, folder in candidates:
        search = [p for p in (folder, hip_bin) if p and os.path.isdir(p)]
        os.environ["PATH"] = os.pathsep.join(search + [os.environ.get("PATH", "")])
        for p in search:
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
        loaded, failed = [], []
        for name in ZLUDA_DLLS:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                failed.append(name + ":missing")
                continue
            try:
                ctypes.WinDLL(path)
            except OSError as exc:
                failed.append("%s:%s" % (name, exc))
            else:
                loaded.append(name)
        if "nvcuda.dll" in loaded and "cublas64_11.dll" in loaded:
            best = (folder, loaded, failed)
            break
        best = best or (folder, loaded, failed)

    # The candidate list is non-empty here, so the loop always assigned `best`.
    # 走到这里候选列表必不为空，因此上面的循环必然已经给 `best` 赋过值。
    folder, loaded, failed = best
    status = "zluda_dir=%s | loaded=%d/%d%s" % (
        folder,
        len(loaded),
        len(ZLUDA_DLLS),
        (" | failed=" + ";".join(failed[:3])) if failed else "",
    )
    if failed:
        return Outcome(
            status,
            "部分 DLL 加载失败：failed= 里的 xxx.dll:missing 是文件缺失；"
            "WinError 193 通常是拿到了占位文件或 HTML，而不是真正的 DLL。",
        )
    return Outcome(status)


# --------------------------------------------------------------------------------------------------
# TORCH — cuDNN and the SDPA backend flags
# TORCH — cuDNN 与 SDPA 后端开关
# --------------------------------------------------------------------------------------------------
# Purpose   turn cuDNN off and leave only the math SDP backend.
# Cause     ZLUDA has no cuDNN and no flash kernels; the default flags route
#           attention into kernels that do not exist.
# Measured  n/a; both failures are immediate.
# Failure   cudnn FAILED / sdp FAILED.
# Switch    none.
#
# 作用  关闭 cuDNN，只留下 math SDP 后端。
# 原因  ZLUDA 没有 cuDNN，也没有 flash 内核；默认开关会把注意力送进不存在的
#       内核。
# 实测  不适用；两种失败都是立刻发生的。
# 失败  cudnn FAILED / sdp FAILED。
# 开关  无。
_HINT_CUDNN = "cuDNN 关闭失败：注意力会报 CUDNN_STATUS_INTERNAL_ERROR，出图会失败或全黑。"
_HINT_SDP = "SDPA 后端开关没设上：注意力可能走到 ZLUDA 上不存在的 FLASH / cuDNN。"


# Disable cuDNN; ZLUDA cannot use it.
# 关闭 cuDNN；ZLUDA 用不了它。
def _patch_cudnn(torch_module):
    torch_module.backends.cudnn.enabled = False
    return Outcome("cudnn=%s" % torch_module.backends.cudnn.enabled)


# Keep only the math SDP backend.
# 注意力只保留 math SDP。
def _patch_sdp(torch_module):
    torch_module.backends.cuda.enable_flash_sdp(False)
    torch_module.backends.cuda.enable_mem_efficient_sdp(False)
    torch_module.backends.cuda.enable_math_sdp(True)
    return Outcome("sdp=math")


# --------------------------------------------------------------------------------------------------
# SDPA — math-only attention for torch and comfy.ops
# SDPA — 给 torch 与 comfy.ops 装 math-only 注意力
# --------------------------------------------------------------------------------------------------
# Purpose   wrap F.sdpa and comfy.ops.sdpa so every call runs math-only.
# Cause     comfy/ops.py passes set_priority=True, which overrides the
#           enable_flash_sdp(False) flags and puts FLASH / cuDNN first.
# Measured  gfx906, 1x16x512x128 fp16: bare F.sdpa OK, comfy.ops FAIL,
#           math only OK; without the raise it returns NaN, a black image.
# Failure   sdpa=missing / sdpa FAILED / comfy_ops: FAILED ...
# Switch    none.
#
# 作用  把 F.sdpa 与 comfy.ops.sdpa 都包成 math-only。
# 原因  comfy/ops.py 传了 set_priority=True，会覆盖 enable_flash_sdp(False)，
#       把 FLASH / cuDNN 排到最前。
# 实测  gfx906，1x16x512x128 fp16：裸调 F.sdpa 正常，comfy.ops 失败，仅 math
#       正常；不抛异常时会返回 NaN，也就是全黑图。
# 失败  sdpa=missing / sdpa FAILED / comfy_ops: FAILED ...
# 开关  无。
_SDPA_WRAPPED = "_zluda_math_only_sdpa"
_HINT_SDPA = "SDPA 包装失败：注意力会退回会崩的 FLASH / cuDNN 优先级列表。"
_HINT_SDPA_MISSING = (
    "torch 里没有 F.scaled_dot_product_attention，注意力会退回会崩的 "
    "FLASH / cuDNN 优先级列表。"
)
_HINT_COMFY_SDPA = (
    "comfy.ops 里没有 scaled_dot_product_attention，math-only 没装上："
    "注意力会退回会崩的 FLASH / cuDNN 优先级列表。"
)


# Wrap `module.name` so every call runs inside sdpa_kernel([MATH]).
# 把 `module.name` 包起来，让每次调用都跑在 sdpa_kernel([MATH]) 里。
def _wrap_sdpa(module, name):
    target = getattr(module, name, None)
    if target is None or getattr(target, _SDPA_WRAPPED, False):
        return False

    def _math_only(q, k, v, *args, **kwargs):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH], set_priority=True):
            return target(q, k, v, *args, **kwargs)

    setattr(_math_only, _SDPA_WRAPPED, True)
    setattr(module, name, _math_only)
    return True


# Math-only for torch.nn.functional.scaled_dot_product_attention.
# 给 torch.nn.functional.scaled_dot_product_attention 装 math-only。
def _patch_torch_sdpa(torch_module):
    functional = torch_module.nn.functional
    if getattr(functional, "scaled_dot_product_attention", None) is None:
        return Outcome("sdpa=missing", _HINT_SDPA_MISSING)
    if _wrap_sdpa(functional, "scaled_dot_product_attention"):
        return Outcome("sdpa=math-only")
    return Outcome("sdpa=math-only[already]")


# Probe replacement for older comfy.ops builds that report FLASH as usable.
# 替换旧版 comfy.ops 的探测函数，那些版本会把 FLASH 误报成可用。
def _always_false(*args, **kwargs):
    return False


# Math-only for comfy.ops, plus the can_use_* probes older builds expose.
# 给 comfy.ops 装 math-only，并处理旧版才有的 can_use_* 探测。
def _apply_comfy_ops(module):
    # comfy_ops is the module object just executed. Do not `import comfy.ops`
    # here: during exec_module it is not in sys.modules yet, and the import
    # raises "cannot access submodule 'ops' ...".
    #
    # comfy_ops 是刚刚执行完的模块对象。这里不要再 `import comfy.ops`：
    # exec_module 执行期间它还没进 sys.modules，导入会报
    # "cannot access submodule 'ops' ..."。
    if getattr(module, "scaled_dot_product_attention", None) is None:
        return Outcome(
            "FAILED scaled_dot_product_attention is missing", _HINT_COMFY_SDPA
        )
    _wrap_sdpa(module, "scaled_dot_product_attention")
    for probe in (
        "can_use_flash_attention",
        "can_use_cudnn_attention",
        "can_use_efficient_attention",
    ):
        if hasattr(module, probe):
            setattr(module, probe, _always_false)
    return Outcome("math-only")


# --------------------------------------------------------------------------------------------------
# LINEAR — F.linear / addmm, bias split and INT8 tensorwise
# LINEAR — F.linear / addmm，bias 拆分与 INT8 tensorwise
# --------------------------------------------------------------------------------------------------
# Purpose   split biased linear / addmm into matmul + add, and dequantize INT8
#           tensorwise weights before F.linear.
# Cause     cuBLASLt has no bias-fused GEMM and no INT8 GEMM under ZLUDA
#           (CUBLAS_STATUS_NOT_SUPPORTED ... abType 3 computeType 72).
# Measured  11520x3840 qkv, m=90, fp16: no cache 24.4 ms, cached 2.0 ms,
#           native fp16 GEMM 1.34 ms.
# Failure   linear-patch FAILED; the first forward prints a Chinese hint.
# Switch    ZLUDA_INT8_DEQUANT_CACHE, off by default; total ~1.5x fp16 / 3x INT8.
#
# 作用  把带 bias 的 linear / addmm 拆成 matmul + add，并在 F.linear 之前
#       反量化 INT8 tensorwise 权重。
# 原因  ZLUDA 下 cuBLASLt 既没有 bias 融合 GEMM，也没有 INT8 GEMM
#       （CUBLAS_STATUS_NOT_SUPPORTED ... abType 3 computeType 72）。
# 实测  11520x3840 qkv，m=90，fp16：不缓存 24.4 ms，缓存 2.0 ms，同形状原生
#       fp16 GEMM 1.34 ms。
# 失败  linear-patch FAILED；首次前向会打一行中文提示。
# 开关  ZLUDA_INT8_DEQUANT_CACHE，默认关；总权重约 1.5× fp16 / 3× INT8。
_HINT_LINEAR = (
    "F.linear / addmm 的包装没装上：带 bias 的 GEMM 与 INT8 模型都会走到 "
    "ZLUDA 没实现的 cuBLASLt 路径。"
)
_HINT_INT8_UNCACHED = (
    "INT8 tensorwise 模型：每次前向都会反量化权重；可用但比 bf16 / fp8 / "
    "convrot_w4a4 慢。显存充裕时设 ZLUDA_INT8_DEQUANT_CACHE=1 可缓存提速。"
)
_HINT_INT8_CACHED = (
    "INT8 tensorwise 模型：权重会反量化成 %s 并缓存；原始 INT8 权重仍在，总权重"
    "显存约为 fp16 模型的 1.5 倍（约 3 倍 INT8），16 GB 卡上大模型通常放不下。"
)
_DEQUANT_CACHE = weakref.WeakKeyDictionary()
_QUANT_NAMES = None


# Class names that count as a comfy_kitchen quantized weight, resolved on first
# use rather than at import: comfy_kitchen imports torch, and a torch imported
# before preload_zluda() binds to no CUDA driver.
#
# 算作 comfy_kitchen 量化权重的类名，首次使用时才解析而不是在导入时：
# comfy_kitchen 会导入 torch，而在 preload_zluda() 之前导入的 torch 绑不到
# 任何 CUDA 驱动。
def _quant_weight_type_names():
    global _QUANT_NAMES
    if _QUANT_NAMES is None:
        names = {"QuantizedTensor"}
        try:
            from comfy_kitchen.tensor import QuantizedTensor

            names.add(QuantizedTensor.__name__)
        except Exception:
            pass
        _QUANT_NAMES = names
    return _QUANT_NAMES


# Dequantize an INT8 weight to `dtype`, optionally caching per weight object.
# 把 INT8 权重反量化成 `dtype`，可选地按权重对象缓存。
def _dequantize_cached(weight, dtype):
    if not ZLUDA_INT8_DEQUANT_CACHE:
        _report_once(
            "int8-uncached",
            "INT8 tensorwise model: dequantizing weights on every forward pass. "
            "This works but is slower than a bf16 / fp8 / convrot_w4a4 model; "
            "set ZLUDA_INT8_DEQUANT_CACHE=1 to cache them if you have spare VRAM.",
            _HINT_INT8_UNCACHED,
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
    _report_once(
        "int8-cached",
        "INT8 tensorwise model: weights are dequantized to %s and cached. "
        "The original INT8 weights stay, so total weight VRAM is about 1.5x "
        "the fp16 model (3x the INT8 weights)." % (dtype,),
        _HINT_INT8_CACHED % (dtype,),
    )
    return dequantized


# Bias-fused linear / addmm and INT8 tensorwise go through one wrapper.
# 带 bias 的 linear / addmm 与 INT8 tensorwise 共用一个包装器。
def _patch_torch_linear(torch_module):
    functional = torch_module.nn.functional

    if getattr(functional.linear, "__name__", "") != "_linear":
        _original_linear = functional.linear

        def _linear(input, weight, bias=None):
            if type(weight).__name__ in _quant_weight_type_names():
                weight = _dequantize_cached(weight, input.dtype)
            if bias is not None and input.is_cuda and weight.dim() == 2:
                return torch_module.matmul(input, weight.t()) + bias
            return _original_linear(input, weight, bias)

        functional.linear = _linear

    if getattr(torch_module.addmm, "__name__", "") != "_addmm":
        _original_addmm = torch_module.addmm

        # Parameter names must match torch's signature exactly:
        # addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None).
        #
        # 参数名必须和 torch 的签名完全一致：
        # addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None)。
        def _addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
            if (
                out is None
                and beta == 1
                and alpha == 1
                and getattr(input, "is_cuda", False)
            ):
                return torch_module.matmul(mat1, mat2) + input
            return _original_addmm(input, mat1, mat2, beta=beta, alpha=alpha, out=out)

        torch_module.addmm = _addmm

    return Outcome(
        "linear-patch[F.linear=%s]" % getattr(functional.linear, "__name__", "?")
    )


# --------------------------------------------------------------------------------------------------
# CLI_ARGS — launch arguments the setup can no longer rely on
# CLI_ARGS — 不能再依赖的启动参数
# --------------------------------------------------------------------------------------------------
# Purpose   override the launch arguments right after comfy.cli_args parses.
# Cause     the launcher keeps them in .launcher/preference.json; a reset drops
#           them, and other entry points never set them at all.
# Measured  n/a.
# Failure   args: FAILED ... / args: overrides disabled ...
# Switch    ZLUDA_NO_ARG_OVERRIDES, ZLUDA_RESERVE_VRAM, ZLUDA_FORCE_FP16,
#           ZLUDA_SPLIT_ATTENTION.
#
# 作用  在 comfy.cli_args 解析完成的瞬间覆盖启动参数。
# 原因  启动器把参数存在 .launcher/preference.json 里，重置就没了；绕过
#       main.py 的入口更是完全拿不到。
# 实测  不适用。
# 失败  args: FAILED ... / args: overrides disabled ...
# 开关  ZLUDA_NO_ARG_OVERRIDES、ZLUDA_RESERVE_VRAM、ZLUDA_FORCE_FP16、
#       ZLUDA_SPLIT_ATTENTION。
_HINT_ARGS = (
    "comfy.cli_args 的启动参数覆盖没生效：pinned memory / async offload / "
    "cudaMallocAsync 可能没有按补丁的默认值关掉。"
)
_HINT_ARGS_MISSING = (
    "comfy.cli_args 还没有 args 对象，启动参数覆盖被跳过：pinned memory / "
    "async offload / cudaMallocAsync 可能没有按补丁的默认值关掉。"
)


# Launch arguments the patch owns; an explicit --reserve-vram still wins.
# 补丁掌管的启动参数；启动器显式给的 --reserve-vram 仍然优先。
def _cli_overrides():
    # Pinned memory caps at 40% of RAM and crashes often; torch 2.7 falls back
    # to the legacy ModelPatcher, so async offload buys nothing; ZLUDA has no
    # cudaMallocAsync.
    #
    # 固定内存上限是物理内存的 40%，而且经常导致崩溃；torch 2.7 会回退到旧的
    # ModelPatcher，所以异步 offload 没有收益；ZLUDA 没有 cudaMallocAsync。
    overrides = {
        "disable_pinned_memory": True,
        "disable_async_offload": True,
        "disable_cuda_malloc": True,
    }
    # The reserve is the part the models may not take. 0.9 GB was too little,
    # 2.5 GB pushed the UNet into partial offload, and 1.5 GB keeps the VAE
    # decode on the fast path and the UNet whole.
    #
    # 预留就是模型不许占用的那一部分。0.9 GB 太少，2.5 GB 会把 UNet 挤成部分
    # 卸载；1.5 GB 既让 VAE 解码走快路径，又能让 UNet 完整加载。
    if ZLUDA_RESERVE_VRAM > 0:
        overrides["reserve_vram"] = ZLUDA_RESERVE_VRAM
    # Opt-in: force fp16 compute. It overrides the fp8 flags of fp8-stored
    # models, so it stays off by default.
    #
    # 可选：强制 fp16 计算。它会盖掉 fp8 存储模型的 fp8 标志，所以默认关闭。
    if ZLUDA_FORCE_FP16:
        overrides["fp16_unet"] = True
        overrides["fp16_text_enc"] = True
    # Opt-in: split attention instead of pytorch attention; measure first.
    # 可选：用 split attention 取代 pytorch attention；先实测。
    if ZLUDA_SPLIT_ATTENTION:
        overrides["use_split_cross_attention"] = True
        overrides["use_quad_cross_attention"] = False
    return overrides


# Apply the overrides to the args object comfy.cli_args just built.
# 把覆盖值写到 comfy.cli_args 刚构造好的 args 对象上。
def _apply_cli_args(module):
    if ZLUDA_NO_ARG_OVERRIDES:
        return Outcome("overrides disabled by ZLUDA_NO_ARG_OVERRIDES")
    args = getattr(module, "args", None)
    if args is None:
        return Outcome("FAILED parsing deferred, overrides skipped", _HINT_ARGS_MISSING)
    applied = []
    for key, value in _cli_overrides().items():
        if key == "reserve_vram" and getattr(args, key, None) is not None:
            # An explicit --reserve-vram from the launcher wins.
            # 启动器显式给了 --reserve-vram 时以它为准。
            continue
        setattr(args, key, value)
        applied.append("%s=%s" % (key, value))
    return Outcome(", ".join(applied) if applied else "nothing to override")


# --------------------------------------------------------------------------------------------------
# MODEL_MANAGEMENT — UNet dtype policy + VAE list preference
# MODEL_MANAGEMENT — UNet 精度策略 + VAE 列表偏好
# --------------------------------------------------------------------------------------------------
# Purpose   UNet: fp16 if supported, else fp32 if it fits, else bf16.
#           VAE: follow working_dtypes; fp16 if listed, else bf16.
# Cause     unet_dtype() follows the model's dtype order; a bf16-first model
#           (Anima) picks bf16 even when fp16 works. VAE lists differ by model.
# Measured  fp16 9.54 TFLOPS, bf16 5.26; fp32 is fp16-speed but 2x weight memory.
# Output    once per process:
#             model_management: unet=<policy|comfy-order|off> | vae=<...> | flags=<pins>
#           once per model / VAE load:
#             model_management: unet <dtype> (policy|comfy order: <why>)
#             model_management: vae <dtype> (policy|comfy order: <why>)
#             model_management: manual cast <dtype> (policy: ...)
#           "comfy would pick ..." is appended when the policy changes the answer;
#           a quantized weight store is named in the reason ("weights uint8").
# Failure   model_management: FAILED ... (that call is handed back to ComfyUI).
# Switch    ZLUDA_UNET_DTYPE, ZLUDA_VAE_DTYPE (auto|comfy, one side each);
#           ZLUDA_ALLOW_BF16=1 is the alias that sets both sides to comfy;
#           ZLUDA_FP32_MARGIN (fp32 budget margin, default 0.9).
#
# 作用  UNet：支持 fp16 就 fp16，放得下 fp32 就 fp32，否则 bf16。
#       VAE：跟随 working_dtypes，有 fp16 就 fp16，否则 bf16。
# 原因  unet_dtype() 按模型 dtype 顺序走；bf16 排第一的模型（Anima）即使支持
#       fp16 也会选 bf16。不同 VAE 的列表不同。
# 实测  fp16 9.54 TFLOPS，bf16 5.26；fp32 速度接近 fp16，但权重显存翻倍。
# 输出  每个进程一次：
#         model_management: unet=<policy|comfy-order|off> | vae=<...> | flags=<钉住的标志>
#       每次模型 / VAE 加载一行：
#         model_management: unet <dtype> (policy|comfy order: <原因>)
#         model_management: vae <dtype> (policy|comfy order: <原因>)
#         model_management: manual cast <dtype> (policy: ...)
#       策略改判上游结果时追加 "comfy would pick ..."；量化权重会在原因里标出
#       （"weights uint8"）。
# 失败  model_management: FAILED ...（这一次调用交回 ComfyUI）。
# 开关  ZLUDA_UNET_DTYPE、ZLUDA_VAE_DTYPE（auto|comfy，各管一侧）；
#       ZLUDA_ALLOW_BF16=1 是别名，把两侧都设成 comfy；
#       ZLUDA_FP32_MARGIN（fp32 预算余量，默认 0.9）。
_HINT_MM = "model_management 补丁没装上：无法按模型 / VAE 选择 dtype。"
_HINT_MM_NO_ARGS = (
    "没有拿到 comfy.cli_args.args，读不到精度标志，因此两侧 dtype 策略都没有装："
    "模型精度完全由 ComfyUI 自己决定。"
)
_HINT_MM_FALLBACK = (
    "精度策略在运行期出错，这一次已交回 ComfyUI 原顺序：模型还能加载，但 dtype "
    "不是按本机规则选的。"
)

# Precision flags a launcher may pin. They are printed once and make the
# policies step aside, so the record shows why a policy was not consulted.
#
# 启动器可能钉住的精度标志。启动时打印一次；这些标志会让策略让位，记录里才能
# 看出为什么没走策略。
_PRECISION_FLAGS = (
    "force_fp32",
    "fp32_unet", "fp64_unet", "bf16_unet", "fp16_unet",
    "fp8_e4m3fn_unet", "fp8_e5m2_unet", "fp8_e8m0fnu_unet",
    "fp16_text_enc", "bf16_text_enc", "fp32_text_enc",
    "fp16_vae", "bf16_vae", "fp32_vae",
)

# The subsets that hand a decision back to ComfyUI's own function.
# 把决定交回 ComfyUI 自己函数的那几组标志。
_UNET_FLAGS = (
    "force_fp32",
    "fp32_unet", "fp64_unet", "bf16_unet", "fp16_unet",
    "fp8_e4m3fn_unet", "fp8_e5m2_unet", "fp8_e8m0fnu_unet",
)
_VAE_FLAGS = ("fp16_vae", "bf16_vae", "fp32_vae")


# Flags that are set on the args object, in the canonical order above.
# args 对象上已设置的标志，顺序同上。
def _flag_list(args):
    if args is None:
        return None
    return [name for name in _PRECISION_FLAGS if getattr(args, name, False)]


# First flag from a group that is set, or None.
# 一组标志里第一个被设置的，取不到返回 None。
def _first_flag(args, names):
    if args is None:
        return None
    for name in names:
        if getattr(args, name, False):
            return name
    return None


# Short dtype name for status lines: torch.float16 -> float16.
# 状态行里的短 dtype 名：torch.float16 -> float16。
def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


# A dtype list for status lines: [float16, float32].
# 状态行里的 dtype 列表：[float16, float32]。
def _dtype_names(dtypes):
    return "[" + ", ".join(_dtype_name(dtype) for dtype in dtypes) + "]"


# Bytes to GiB, for the fp32 budget line.
# 字节换算成 GiB，用于 fp32 预算那一行。
def _gib(value):
    return value / (1024.0 ** 3)


# Weight storage name for status lines; None means the caller did not say
# (a quantized checkpoint, or a call that only asked for the inference dtype).
#
# 状态行里的权重存储名；None 表示调用方没说（量化检查点，或只问推理 dtype 的
# 调用）。
def _weights_name(weight_dtype):
    return "unknown" if weight_dtype is None else _dtype_name(weight_dtype)


# Prefix that names a non-float weight store in a policy reason, else "".
# 非浮点权重存储的前缀，浮点权重时返回空串。
def _weights_note(weight_dtype):
    if weight_dtype is None:
        return "weights unknown, "
    name = _dtype_name(weight_dtype)
    if name in ("float16", "bfloat16", "float32", "float64"):
        return ""
    return "weights %s, " % name


# Apply the UNet dtype policy and the VAE list preference, each on its own.
# 应用 UNet 的 dtype 策略和 VAE 的列表偏好，两侧各自独立。
def _apply_model_management(module):
    args = getattr(module, "args", None)
    unet_policy = ZLUDA_UNET_DTYPE == "auto"
    vae_policy = ZLUDA_VAE_DTYPE == "auto"

    # Without the parsed arguments no precision flag can be read, so neither
    # wrapper is installed and ComfyUI's own functions stay in charge.
    #
    # 拿不到解析后的参数就读不到任何精度标志，因此两侧包装都不装，交给
    # ComfyUI 自己的函数。
    if args is None:
        return Outcome("unet=off | vae=off | flags=unknown", _HINT_MM_NO_ARGS)

    flags = _flag_list(args)

    if unet_policy or vae_policy:
        torch = module.torch
        original_should_use_fp16 = module.should_use_fp16
        original_should_use_bf16 = module.should_use_bf16
        original_unet_dtype = module.unet_dtype
        original_vae_dtype = module.vae_dtype
        original_manual_cast = getattr(module, "unet_manual_cast", None)
        float8_types = getattr(module, "FLOAT8_TYPES", ())

        # Can fp32 weights fit the weight budget? The pair is logged, so the
        # record shows the numbers the decision used. The margin covers the
        # activations, which are also 4 bytes at fp32 while the budget (from
        # maximum_vram_for_weights) only counts weights.
        #
        # fp32 权重能否放进权重预算？这一对数会写进日志，记录里能看到决策用的
        # 数字。余量是为激活留的：fp32 下激活也是 4 字节，而预算
        # （maximum_vram_for_weights）只算权重。
        def _fp32_budget(device, model_params):
            if model_params is None or model_params <= 0:
                return None
            # Read at call time: another patch may replace it, and a late
            # failure must not turn into a model-load error.
            #
            # 调用时才取：别的补丁可能替换过它，晚到的异常也不能变成加载模型
            # 时的崩溃。
            budget_fn = getattr(module, "maximum_vram_for_weights", None)
            if budget_fn is None:
                return None
            try:
                budget = budget_fn(device)
            except Exception:
                return None
            return model_params * 4, budget * ZLUDA_FP32_MARGIN

        def _upstream_unet(device, model_params, supported_dtypes, weight_dtype):
            return original_unet_dtype(
                device, model_params, supported_dtypes, weight_dtype)

        # Log one policy decision; also say what ComfyUI's own order would pick
        # when the two disagree, so the record shows the policy's effect.
        #
        # 记录一次策略决策；两者不一致时并列写出 ComfyUI 原顺序会选什么，记录里
        # 才能看出策略的实际作用。
        def _report_unet(choice, reason, device, model_params,
                         supported_dtypes, weight_dtype):
            try:
                upstream = _upstream_unet(
                    device, model_params, supported_dtypes, weight_dtype)
            except Exception:
                upstream = None
            text = "model_management: unet %s (policy: %s%s" % (
                _dtype_name(choice), _weights_note(weight_dtype), reason)
            if upstream is not None and upstream != choice:
                text += "; comfy would pick %s" % _dtype_name(upstream)
            _report(text + ")")
            return choice

        # Log one decision ComfyUI made itself, with the reason it got it.
        # 记录一次由 ComfyUI 自己做的决定，并写下交回决定的原因。
        def _delegate_unet(reason, device, model_params,
                           supported_dtypes, weight_dtype):
            result = _upstream_unet(
                device, model_params, supported_dtypes, weight_dtype)
            _report("model_management: unet %s (comfy order: %s)"
                    % (_dtype_name(result), reason))
            return result

        def _unet_choice(device, model_params, supported_dtypes, weight_dtype):
            # Explicit flags, fp8 weights and fp16 weights keep ComfyUI's own
            # handling; the line records which one handed the decision back.
            #
            # 显式标志、fp8 权重和 fp16 权重交给 ComfyUI 自己处理；日志行记录是
            # 哪一条把决定交了回去。
            flag = _first_flag(args, _UNET_FLAGS)
            if flag is not None:
                return _delegate_unet(
                    "--" + flag.replace("_", "-"),
                    device, model_params, supported_dtypes, weight_dtype)
            if weight_dtype in float8_types:
                return _delegate_unet(
                    "fp8 weights %s" % _dtype_name(weight_dtype),
                    device, model_params, supported_dtypes, weight_dtype)
            if weight_dtype == torch.float16:
                return _delegate_unet(
                    "fp16 weights",
                    device, model_params, supported_dtypes, weight_dtype)

            # Prefer fp16 whenever the model declares it.
            # 模型声明支持 fp16 时优先 fp16。
            if (torch.float16 in supported_dtypes
                    and original_should_use_fp16(
                        device=device, model_params=model_params)):
                return _report_unet(
                    torch.float16, "model lists fp16", device, model_params,
                    supported_dtypes, weight_dtype)

            # bf16 + fp32 only: fp32 when the weights fit, otherwise bf16 when
            # the device can run it, otherwise ComfyUI's own answer. bf16 is
            # only the fallback, never a gate on the fp32 decision: gating it
            # would silently return fp32 for a model that does not fit.
            #
            # 只有 bf16 + fp32 时：权重放得下用 fp32；放不下再用 bf16，前提是
            # 设备能跑；否则交回 ComfyUI。bf16 只是兜底，不能当成 fp32 判断的
            # 闸门——那样会静默返回放不下的 fp32。
            if (torch.bfloat16 in supported_dtypes
                    and torch.float32 in supported_dtypes):
                budget = _fp32_budget(device, model_params)
                if budget is not None and budget[0] <= budget[1]:
                    return _report_unet(
                        torch.float32,
                        "fp32 %.1f GiB <= %.1f GiB" % (
                            _gib(budget[0]), _gib(budget[1])),
                        device, model_params, supported_dtypes, weight_dtype)
                if budget is None:
                    reason = "fp32 size unknown"
                else:
                    reason = "fp32 %.1f GiB > %.1f GiB" % (
                        _gib(budget[0]), _gib(budget[1]))
                if original_should_use_bf16(device, model_params=model_params):
                    return _report_unet(
                        torch.bfloat16, reason, device, model_params,
                        supported_dtypes, weight_dtype)
                return _delegate_unet(
                    reason + ", bf16 unusable",
                    device, model_params, supported_dtypes, weight_dtype)

            return _delegate_unet(
                "no policy branch",
                device, model_params, supported_dtypes, weight_dtype)

        # Never let the policy break a load: an unexpected error hands that one
        # call back to ComfyUI, which is what the model would get anyway.
        #
        # 策略绝不能让加载崩掉：意外出错就把这一次调用交回 ComfyUI，结果和
        # 没装补丁时一样。
        def _unet_dtype(device=None, model_params=0,
                        supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
                        weight_dtype=None):
            try:
                return _unet_choice(
                    device, model_params, supported_dtypes, weight_dtype)
            except Exception as exc:
                _report_once(
                    "unet-policy",
                    "model_management: unet policy FAILED %r" % (exc,),
                    _HINT_MM_FALLBACK)
                return original_unet_dtype(
                    device, model_params, supported_dtypes, weight_dtype)

        # Upstream resolves the manual cast of unknown / fp8 weights to bf16,
        # which is 0.55x on gfx906. Use fp16 when the model and the device
        # allow it; bf16 weights return None upstream and stay untouched, and a
        # pinned --bf16-unet is left alone.
        #
        # 未知 / fp8 权重的 manual cast 在上游会定成 bf16，这张卡上只有
        # 0.55 倍速。模型和设备都支持 fp16 时改成 fp16；bf16 权重在上游返回
        # None，这里不动；显式钉了 --bf16-unet 的也不动。
        def _manual_cast(weight_dtype, inference_device,
                         supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
            result = original_manual_cast(
                weight_dtype, inference_device, supported_dtypes)
            if (
                result == torch.bfloat16
                and not getattr(args, "bf16_unet", False)
                and torch.float16 in supported_dtypes
                and original_should_use_fp16(
                    inference_device, prioritize_performance=False)
            ):
                _report(
                    "model_management: manual cast float16 "
                    "(policy: upstream bf16, weights %s)"
                    % _weights_name(weight_dtype))
                return torch.float16
            return result

        # Install the UNet side (dtype policy + manual cast correction).
        # 装 UNet 这一侧（dtype 策略 + manual cast 修正）。
        if unet_policy:
            module.unet_dtype = _unet_dtype
            if original_manual_cast is not None:
                module.unet_manual_cast = _manual_cast

        # The VAE follows its own working_dtypes: fp16 if listed, else bf16.
        # VAE 跟随自己的 working_dtypes：列表里有 fp16 就 fp16，否则 bf16。
        def _upstream_vae(device, allowed_dtypes):
            return original_vae_dtype(device, allowed_dtypes)

        def _report_vae(choice, reason, device, allowed_dtypes):
            try:
                upstream = _upstream_vae(device, allowed_dtypes)
            except Exception:
                upstream = None
            text = "model_management: vae %s (policy: %s" % (
                _dtype_name(choice), reason)
            if upstream is not None and upstream != choice:
                text += "; comfy would pick %s" % _dtype_name(upstream)
            _report(text + ")")
            return choice

        def _delegate_vae(reason, device, allowed_dtypes):
            result = _upstream_vae(device, allowed_dtypes)
            _report("model_management: vae %s (comfy order: %s)"
                    % (_dtype_name(result), reason))
            return result

        def _vae_choice(device, allowed_dtypes):
            flag = _first_flag(args, _VAE_FLAGS)
            if flag is not None:
                return _delegate_vae(
                    "--" + flag.replace("_", "-"), device, allowed_dtypes)

            listed = _dtype_names(allowed_dtypes)
            if (torch.float16 in allowed_dtypes
                    and original_should_use_fp16(device)):
                return _report_vae(
                    torch.float16, "fp16 in %s" % listed,
                    device, allowed_dtypes)
            if (torch.bfloat16 in allowed_dtypes
                    and original_should_use_bf16(device)):
                if torch.float16 in allowed_dtypes:
                    reason = "fp16 unsupported by device, bf16 in %s" % listed
                else:
                    reason = "no fp16 in %s" % listed
                return _report_vae(
                    torch.bfloat16, reason, device, allowed_dtypes)
            return _delegate_vae("list %s" % listed, device, allowed_dtypes)

        def _vae_dtype(device=None, allowed_dtypes=[]):
            try:
                return _vae_choice(device, allowed_dtypes)
            except Exception as exc:
                _report_once(
                    "vae-policy",
                    "model_management: vae policy FAILED %r" % (exc,),
                    _HINT_MM_FALLBACK)
                return original_vae_dtype(device, allowed_dtypes)

        # Install the VAE side (list preference only).
        # 装 VAE 这一侧（只做列表偏好）。
        if vae_policy:
            module.vae_dtype = _vae_dtype

    return Outcome(
        "unet=%s | vae=%s | flags=%s"
        % ("policy" if unet_policy else "comfy-order",
           "fp16-priority" if vae_policy else "comfy-order",
           "unknown" if flags is None else (",".join(flags) or "none"))
    )


# --------------------------------------------------------------------------------------------------
# ONNX — no CUDA / TensorRT execution provider
# ONNX — 不要 CUDA / TensorRT 执行提供程序
# --------------------------------------------------------------------------------------------------
# Purpose   remove CUDA / TensorRT from ONNX Runtime's provider list.
# Cause     the installed onnxruntime-gpu is a CUDA 12 build; ZLUDA provides
#           CUDA 11 libraries, so the provider fails to load or crashes.
# Measured  n/a.
# Failure   onnxruntime: FAILED ...
# Switch    ZLUDA_KEEP_ONNX_CUDA.
#
# 作用  从 ONNX Runtime 的 provider 列表里去掉 CUDA / TensorRT。
# 原因  装的是 CUDA 12 构建的 onnxruntime-gpu，而 ZLUDA 只有 CUDA 11 的库，
#       provider 要么加载失败，要么直接崩。
# 实测  不适用。
# 失败  onnxruntime: FAILED ...
# 开关  ZLUDA_KEEP_ONNX_CUDA。
_HINT_ONNX = "ONNX Runtime 的 CUDA provider 没过滤掉：相关节点会加载失败或崩溃。"


# Filter the provider list, including the pybind copy the session path reads.
# 过滤 provider 列表，包括 session 路径实际读取的 pybind 副本。
def _apply_onnxruntime(module):
    if ZLUDA_KEEP_ONNX_CUDA:
        return Outcome("left untouched (ZLUDA_KEEP_ONNX_CUDA)")

    unavailable = ("CUDAExecutionProvider", "TensorrtExecutionProvider")
    original = module.get_available_providers

    def _filtered():
        return [p for p in original() if p not in unavailable]

    module.get_available_providers = _filtered
    try:
        # The session path reads this pybind copy, not the module-level function.
        # 建 session 时读的是这个 pybind 副本，不是模块级函数。
        module.capi._pybind_state.get_available_providers = _filtered
    except Exception:
        pass

    session = module.InferenceSession

    class _CpuInferenceSession(session):
        def __init__(self, *args, providers=None, **kwargs):
            if providers:
                providers = [p for p in providers if p not in unavailable]
                providers = providers or ["CPUExecutionProvider"]
                _report_once(
                    "onnx-cpu",
                    "onnxruntime: forcing CPUExecutionProvider",
                    "ONNX Runtime 强制使用 CPUExecutionProvider（CUDA 12 构建的 "
                    "onnxruntime-gpu 在 ZLUDA 上不可用）。",
                )
            super().__init__(*args, providers=providers, **kwargs)

    module.InferenceSession = _CpuInferenceSession
    return Outcome("CUDA/TensorRT providers filtered")


# --------------------------------------------------------------------------------------------------
# SUPPORTED_MODELS — LTXAV memory estimate
# SUPPORTED_MODELS — LTXAV 显存估算
# --------------------------------------------------------------------------------------------------
# Purpose   fix LTXAV's memory_usage_factor, upstream left it at 0.077.
# Cause     model_base.py multiplies the activation estimate by this factor, so
#           ComfyUI plans as if LTXAV needs almost nothing and then runs out.
# Measured  n/a.
# Failure   supported_models: FAILED ...
# Switch    none.
#
# 作用  修正 LTXAV 的 memory_usage_factor，上游留成了 0.077。
# 原因  model_base.py 用它乘激活显存估算，于是 ComfyUI 以为 LTXAV 几乎不吃
#       显存，然后在 16 GB 卡上 OOM。
# 实测  不适用。
# 失败  supported_models: FAILED ...
# 开关  无。
_HINT_LTXAV = "LTXAV 的显存估算没修上：16 GB 卡上可能在采样时 OOM。"


# Patch LTXAV.__init__ so the factor follows the parent formula with a margin.
# 给 LTXAV.__init__ 打补丁，让因子跟随父类公式并留出余量。
def _apply_supported_models(module):
    ltxav = getattr(module, "LTXAV", None)
    if ltxav is None:
        return Outcome("")

    original_init = ltxav.__init__
    parent = ltxav.__mro__[1] if len(ltxav.__mro__) > 1 else None

    # Read both numbers off the parent class, so an upstream change to either
    # cannot make the two formulas drift apart.
    #
    # 两个数字都从父类上取，这样上游改动也不会让两边的公式对不上。
    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        unet_config = args[0] if args else kwargs.get("unet_config", {})
        factor = getattr(parent, "memory_usage_factor", 5.5)
        dim = unet_config.get("cross_attention_dim", 2048)
        self.memory_usage_factor = (dim / 2048) * factor * 1.5

    ltxav.__init__ = _init
    return Outcome("LTXAV memory_usage_factor patched")


# --------------------------------------------------------------------------------------------------
# OPS — optional operator fallbacks, off by default
# OPS — 可选的算子兜底，默认关闭
# --------------------------------------------------------------------------------------------------
# Purpose   optional CPU fallbacks for topk / stft / jit.script.
# Cause     ZLUDA does not implement every operator; the fork replaces these
#           unconditionally, which is expensive (every topk becomes D2H + H2D).
# Measured  n/a.
# Failure   ops FAILED ...
# Switch    ZLUDA_TOPK_CPU, ZLUDA_STFT_CPU, ZLUDA_NO_JITSCRIPT.
#
# 作用  可选的 topk / stft / jit.script 兜底。
# 原因  ZLUDA 并非实现了所有算子；上游 fork 是无条件替换的，代价很大
#       （每次 topk 都变成 D2H + H2D）。
# 实测  不适用。
# 失败  ops FAILED ...
# 开关  ZLUDA_TOPK_CPU、ZLUDA_STFT_CPU、ZLUDA_NO_JITSCRIPT。
_HINT_OPS = (
    "可选算子兜底没有装上：设了 ZLUDA_TOPK_CPU / ZLUDA_STFT_CPU / "
    "ZLUDA_NO_JITSCRIPT 也不会生效。"
)


# No-op torch.jit.script; node packs that ship TorchScript need it.
# 让 torch.jit.script 变成空操作；自带 TorchScript 的节点包需要。
def _script_noop(function, *args, **kwargs):
    return function


# Install the enabled operator fallbacks; idempotent and safe to call twice.
# 安装已启用的算子兜底；幂等，重复调用也安全。
def _install_operator_fallbacks(torch_module):
    # Stay idempotent: a second call must not wrap topk twice, and must still
    # report what was applied. Read through __dict__ because torch defines a
    # module-level __getattr__.
    #
    # 保持幂等：第二次调用不能把 topk 包两层，而且要仍然报出装了什么。故意走
    # __dict__ 读，因为 torch 定义了模块级 __getattr__。
    if torch_module.__dict__.get("_zluda_ops_patched", False):
        return list(torch_module.__dict__.get("_zluda_ops_applied", ()))
    applied = []
    if ZLUDA_TOPK_CPU:
        original_topk = torch_module.topk

        def _topk(input, *args, **kwargs):
            device = input.device
            values, indices = original_topk(input.cpu(), *args, **kwargs)
            return torch_module.return_types.topk(
                (values.to(device), indices.to(device))
            )

        torch_module.topk = _topk
        applied.append("topk")
    if ZLUDA_STFT_CPU:
        original_stft = torch_module.stft
        original_istft = torch_module.istft

        # Move every tensor argument to the CPU instead of naming parameters:
        # the second positional argument of stft is n_fft, not window.
        #
        # 把所有 tensor 参数移到 CPU，而不是给参数起名：stft 的第二个位置参数
        # 是 n_fft，不是 window。
        def _to_cpu(value):
            return value.cpu() if torch_module.is_tensor(value) else value

        def _stft(input, *args, **kwargs):
            args = tuple(_to_cpu(arg) for arg in args)
            kwargs = {name: _to_cpu(value) for name, value in kwargs.items()}
            return original_stft(input.cpu(), *args, **kwargs).to(input.device)

        def _istft(input, *args, **kwargs):
            args = tuple(_to_cpu(arg) for arg in args)
            kwargs = {name: _to_cpu(value) for name, value in kwargs.items()}
            return original_istft(input.cpu(), *args, **kwargs).to(input.device)

        torch_module.stft = _stft
        torch_module.istft = _istft
        applied.append("stft")
    if ZLUDA_NO_JITSCRIPT:
        torch_module.jit.script = _script_noop
        applied.append("jit.script")
    torch_module._zluda_ops_patched = True
    torch_module._zluda_ops_applied = tuple(applied)
    return applied


# --------------------------------------------------------------------------------------------------
# MODEL_DETECTION — Anima / Cosmos-Predict2 tunes with extra blocks
# MODEL_DETECTION — 带额外层的 Anima / Cosmos-Predict2 微调
# --------------------------------------------------------------------------------------------------
# Purpose   count Anima / Cosmos-Predict2 blocks from the checkpoint.
# Cause     ComfyUI 0.30.2 hardcodes 28 / 36 from model_channels; extra blocks
#           are dropped without an error and the sampled latents become NaN.
# Measured  a 40-block BF16 Anima tune loads whole after this patch.
# Failure   model_detection: FAILED ...
# Switch    ZLUDA_ANIMA_EXTRA_BLOCKS, on by default.
#
# 作用  从检查点数 Anima / Cosmos-Predict2 的层数。
# 原因  ComfyUI 0.30.2 按 model_channels 写死 28 / 36；多出来的层被静默丢掉，
#       采样出的 latent 变成 NaN。
# 实测  一个 40 层、BF16 的 Anima 微调在补丁后能完整加载。
# 失败  model_detection: FAILED ...
# 开关  ZLUDA_ANIMA_EXTRA_BLOCKS，默认开启。
_HINT_DETECT = (
    "Anima / Cosmos-Predict2 的层数没有按检查点修正：带额外层的微调会被静默"
    "截断，出图全黑。"
)


# Wrap detect_unet_config() and replace the hardcoded block count.
# 包装 detect_unet_config()，替换写死的层数。
def _apply_model_detection(module):
    original_detect = module.detect_unet_config
    count_blocks = module.count_blocks

    def _detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        config = original_detect(state_dict, key_prefix, *args, **kwargs)
        if not ZLUDA_ANIMA_EXTRA_BLOCKS or not isinstance(config, dict):
            return config
        if config.get("image_model") not in ("anima", "cosmos_predict2"):
            return config
        counted = count_blocks(state_dict, key_prefix + "blocks.{}.")
        if counted and config.get("num_blocks") != counted:
            _report(
                "model_detection: %s num_blocks %s -> %s (counted in the checkpoint)"
                % (config.get("image_model"), config.get("num_blocks"), counted)
            )
            _hint(
                "这个检查点的层数比 ComfyUI 写死的多（或不同），补丁已按文件实际"
                "层数加载；不这么做，多出来的层会被静默丢掉，出图是全黑的。"
            )
            config["num_blocks"] = counted
        return config

    module.detect_unet_config = _detect_unet_config
    return Outcome("extra-block tunes load whole")


# ==================================================================================================
# PART 3 — ASSEMBLY
# PART 3 — 装配
# ==================================================================================================
# Everything below runs once, in this order, at import time.
# 下面所有内容在导入时按顺序执行一次。

_APPLIED = set()


# Module patch keys that already ran; the hook and the already-imported pass
# must not patch the same module twice.
#
# 已经跑过的模块补丁 key；钩子与“已导入补丁”两条路径不能把同一模块补两遍。
def _apply_module_patch(patch, module):
    if patch.key in _APPLIED:
        return
    _APPLIED.add(patch.key)
    try:
        out = patch.apply(module)
    except Exception as exc:
        _report("%s: FAILED %r" % (patch.key, exc))
        _hint(patch.failure_zh)
        return
    if out.status:
        _report("%s: %s" % (patch.key, out.status))
    if out.hint:
        _hint(out.hint)


# Build the Hook for one registered module patch.
# 为一个已注册的模块补丁构造 Hook。
def _hook_for(patch):
    return Hook(after=lambda module: _apply_module_patch(patch, module))


# Torch patches run in this order right after `import torch`.
# 这些 torch 补丁在 `import torch` 之后按这个顺序执行。
TORCH_STEPS = (
    ("cudnn", _patch_cudnn, _HINT_CUDNN),
    ("sdp", _patch_sdp, _HINT_SDP),
    ("sdpa", _patch_torch_sdpa, _HINT_SDPA),
    ("linear", _patch_torch_linear, _HINT_LINEAR),
)


# Module patches run as soon as the named module finishes executing.
# 这些模块补丁在对应模块执行完的瞬间运行。
MODULE_PATCHES = (
    ModulePatch("args", "comfy.cli_args", _apply_cli_args, _HINT_ARGS),
    ModulePatch(
        "model_management", "comfy.model_management", _apply_model_management, _HINT_MM
    ),
    ModulePatch(
        "model_detection", "comfy.model_detection", _apply_model_detection, _HINT_DETECT
    ),
    ModulePatch(
        "supported_models", "comfy.supported_models", _apply_supported_models, _HINT_LTXAV
    ),
    ModulePatch("comfy_ops", "comfy.ops", _apply_comfy_ops, _HINT_COMFY_SDPA),
    ModulePatch("onnxruntime", "onnxruntime", _apply_onnxruntime, _HINT_ONNX),
)


_HINT_TORCH_IMPORT = (
    "补丁在导入 torch 时抛了异常。后面应紧跟 Fallback hook installed；若没有，"
    "说明连回退钩子都没装上。"
)
_HINT_FALLBACK = (
    "已安装导入钩子，torch 加载完成后会补打补丁；下一次 import torch 之后应出现 "
    "Late torch= 和 ops: 两行。"
)
_HINT_GREP_FAILED = (
    "上面有 FAILED 项，说明对应的子补丁没生效。状态行看起来健康并不代表全对，"
    "请把整个 _zluda_status.txt grep 一遍 FAILED。"
)


# Report every switch once, in the canonical order of PART 0.
# 按 PART 0 的固定顺序把每个开关报一次。
def _report_switches():
    pairs = (
        ("anima_extra_blocks", ZLUDA_ANIMA_EXTRA_BLOCKS),
        ("reserve_vram", ZLUDA_RESERVE_VRAM),
        ("fp32_margin", ZLUDA_FP32_MARGIN),
        ("force_fp16", ZLUDA_FORCE_FP16),
        ("split_attention", ZLUDA_SPLIT_ATTENTION),
        ("allow_bf16", ZLUDA_ALLOW_BF16),
        ("unet_dtype", ZLUDA_UNET_DTYPE),
        ("vae_dtype", ZLUDA_VAE_DTYPE),
        ("arg_overrides", not ZLUDA_NO_ARG_OVERRIDES),
        ("onnx_cuda", ZLUDA_KEEP_ONNX_CUDA),
        ("int8_cache", ZLUDA_INT8_DEQUANT_CACHE),
        ("topk_cpu", ZLUDA_TOPK_CPU),
        ("stft_cpu", ZLUDA_STFT_CPU),
        ("no_jitscript", ZLUDA_NO_JITSCRIPT),
    )
    _report("switches: " + " | ".join("%s=%s" % pair for pair in pairs))


# Install and report the optional operator fallbacks.
# 安装并报告可选的算子兜底。
def _report_ops(torch_module):
    try:
        applied = _install_operator_fallbacks(torch_module)
    except Exception as exc:
        _report("ops FAILED %r" % (exc,))
        _hint(_HINT_OPS)
        return
    _report("ops: " + (", ".join(applied) if applied else "default"))


# Patch torch, print the aggregated line, then install the operator fallbacks.
# 给 torch 打补丁，输出汇总行，然后安装算子兜底。
def _finish_torch(torch_module, late=False):
    statuses = []
    hints = []
    for key, step, failure_zh in TORCH_STEPS:
        try:
            out = step(torch_module)
        except Exception as exc:
            out = Outcome("%s FAILED %r" % (key, exc), failure_zh)
        statuses.append(out.status)
        if out.hint:
            hints.append(out.hint)

    _report(
        "%s%s | Tensile=%s | %s"
        % (
            "Late torch=" if late else "OK torch=",
            torch_module.__version__,
            os.environ.get("ROCBLAS_TENSILE_LIBPATH"),
            " | ".join(statuses),
        )
    )
    for hint in hints:
        _hint(hint)
    if any("FAILED" in status for status in statuses):
        _hint(_HINT_GREP_FAILED)
    _report_ops(torch_module)


# The bootstrap: the only place with import-time side effects.
# 引导区：全文件唯一产生导入期副作用的地方。
def _bootstrap():
    _report_switches()
    apply_env()

    preload = preload_zluda()
    _report("Preload: " + preload.status)
    if preload.hint:
        _hint(preload.hint)

    # Install the comfy.* hooks before importing torch: if torch has to be
    # patched late, comfy.ops is still intercepted by these hooks.
    #
    # 在导入 torch 之前先装 comfy.* 钩子：如果 torch 只能晚到补丁，comfy.ops
    # 仍然会被这些钩子拦到。
    finder = _HookFinder({patch.module: _hook_for(patch) for patch in MODULE_PATCHES})
    sys.meta_path.insert(0, finder)

    # A failed direct import is not fatal: the finder gets a `torch` hook.
    # 直接导入失败不是致命的：finder 会补上一个 `torch` 钩子。
    try:
        import torch
    except Exception as exc:
        _report("Direct import failed: %r" % (exc,))
        _hint(_HINT_TORCH_IMPORT)
        finder.hooks["torch"] = Hook(
            before=apply_env,
            after=lambda module: _finish_torch(module, late=True),
        )
        _report("Fallback hook installed")
        _hint(_HINT_FALLBACK)
    else:
        _finish_torch(torch)

    for patch in MODULE_PATCHES:
        module = sys.modules.get(patch.module)
        if module is not None:
            _apply_module_patch(patch, module)

    _report("hooks: " + ", ".join(patch.module for patch in MODULE_PATCHES))


_bootstrap()

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

When it is no longer needed: delete this file and remove the bootstrap block at the top of main.py.


ZLUDA 兼容补丁（MI50 / gfx906 + ZLUDA + ComfyUI，手工添加）。

作用（全部自给自足，不依赖绘世启动器的注入机制）：
  1) 自己预加载 ZLUDA 的 DLL（nvcuda / nvml / cublas64_11 / cusparse64_11 / nvrtc64_112_0）
     —— 必须发生在 import torch 之前，否则 torch 会报"找不到 NVIDIA 驱动"
  2) 把 ROCBLAS_TENSILE_LIBPATH 指向含 gfx906 Tensile 内核的目录
  3) 关闭 ZLUDA 无法使用的 cuDNN
  4) 注意力只保留 math SDP
  5) 把"带 bias 的线性层 / addmm"拆成 matmul + 加法，绕开 cuBLASLt 的 bias 融合 GEMM

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

# Package root = the folder above the ComfyUI folder that contains this file.
# 整合包根目录 = 本文件所在的 ComfyUI 文件夹的上一级。
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# rocBLAS Tensile kernel directory; it must contain the gfx906 files.
# rocBLAS Tensile 内核目录，必须包含 gfx906 的文件。
TENSILE_DIR = r"C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library"
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


# Print to stderr and append to every status file; write failures are ignored on purpose.
# 输出到 stderr 并追加写入所有状态文件；写入失败会被有意忽略。
def _report(text):
    line = "%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), text)
    try:
        sys.stderr.write("[ZLUDA] " + text + "\n")
        sys.stderr.flush()
    except Exception:
        pass
    for path in REPORTS:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass


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
        return "No ZLUDA dir found (nvcuda.dll not preloaded / nvcuda.dll 未预加载)"

    # Newest folder first.
    # 最新的文件夹优先。
    candidates.sort(reverse=True)
    hip_root = os.environ.get("HIP_PATH") or r"C:\Program Files\AMD\ROCm\5.7"
    hip_bin = os.path.join(hip_root, "bin")

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

    if best is None:
        return "ZLUDA preload failed"
    zd, loaded, failed = best
    return "zluda_dir=%s | loaded=%d/%d%s" % (
        zd,
        len(loaded),
        len(ZLUDA_DLLS),
        (" | failed=" + ";".join(failed[:3])) if failed else "",
    )


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

    # F.linear / addmm with a bias goes through the bias-fused GEMM of cuBLASLt, which ZLUDA does
    # not implement (CUBLAS_STATUS_NOT_SUPPORTED), so split it into matmul + add.
    #
    # 带 bias 的 F.linear / addmm 会走 cuBLASLt 的 bias 融合 GEMM，而 ZLUDA 没有实现它
    # （CUBLAS_STATUS_NOT_SUPPORTED），所以把它拆成 matmul + 加法。
    try:
        import torch.nn.functional as F

        if getattr(F.linear, "__name__", "") != "_linear":
            _orig_linear = F.linear

            def _linear(input, weight, bias=None):
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

    return msgs


# Environment variables and DLL preloading must be done before torch is imported.
# 环境变量设置与 DLL 预加载必须在导入 torch 之前完成。
apply_env()
_pre = preload_zluda()
_report("Preload: " + _pre)

try:
    import torch

    _report(
        "OK torch=%s | Tensile=%s | %s"
        % (
            torch.__version__,
            os.environ.get("ROCBLAS_TENSILE_LIBPATH"),
            " | ".join(patch_torch(torch)),
        )
    )
except Exception as _e:
    _report("Direct import failed: %r" % (_e,))

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
            _report("Late torch=%s | %s" % (module.__version__, " | ".join(patch_torch(module))))

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

"""Check ZLUDA, Torch and patch status from the package root.

Usage (run from the package root):
    python zluda_env_check.py

The script finds the adjacent ComfyUI folder automatically. It performs the
normal DLL preload and Torch initialization, but does not start the ComfyUI UI.


在整合包根目录检查 ZLUDA、Torch 与补丁状态。

用法（在整合包根目录运行）：
    python zluda_env_check.py

脚本会自动找到旁边的 ComfyUI 文件夹。它会执行正常的 DLL 预加载和 Torch
初始化，但不会启动 ComfyUI 界面。
"""

import ctypes
import glob
import os
import sys
import tempfile

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isfile(os.path.join(HERE, "ComfyUI", "zluda_patch.py")):
    COMFY_DIR = os.path.join(HERE, "ComfyUI")
elif os.path.isfile(os.path.join(HERE, "zluda_patch.py")):
    COMFY_DIR = HERE
else:
    COMFY_DIR = HERE
sys.path.insert(0, COMFY_DIR)


def heading(number, english, chinese):
    print("\n" + "=" * 90)
    print("%d. %s" % (number, english))
    print("   %s" % chinese)
    print("=" * 90)


checks = []


def record(status, name):
    checks.append((status, name))


def finish():
    passed = sum(1 for status, _ in checks if status == "PASS")
    warned = sum(1 for status, _ in checks if status == "WARN")
    failed = sum(1 for status, _ in checks if status == "FAIL")
    skipped = sum(1 for status, _ in checks if status == "SKIP")
    if failed:
        status = "FAIL"
        summary = "环境异常"
    elif warned:
        status = "WARN"
        summary = "有需要处理的警告"
    else:
        status = "PASS"
        summary = "环境正常"
    print()
    for item_status, name in checks:
        marker = "FAILED" if item_status == "FAIL" else item_status
        print("[%s] %s" % (marker, name))
    print()
    print("RESULT: %s | %d passed, %d failed, %d warnings, %d skipped | %s" % (
        status, passed, failed, warned, skipped, summary,
    ))
    print("Diagnostic complete. Include this output and the latest status file "
          "when asking for help.")
    print("诊断结束。若需要求助，请提供以上输出和最新一次运行的状态文件。")
    return 1 if failed else 0


heading(1, "Python and working directory", "Python 与工作目录")
print("  python      = %s" % sys.executable)
print("  cwd         = %s" % os.getcwd())
print("  script dir  = %s" % HERE)
print("  ComfyUI dir = %s" % COMFY_DIR)
record("PASS", "Python and ComfyUI paths")

heading(2, "Environment variables", "关键环境变量")
for key in (
    "TEMP", "TMP", "LOCALAPPDATA", "HIP_PATH", "ROCBLAS_TENSILE_LIBPATH",
    "CUDA_PATH", "CUDA_HOME", "TORCH_BLAS_PREFER_CUBLASLT",
    "PYTORCH_CUDA_ALLOC_CONF",
):
    print("  %-28s = %r" % (key, os.environ.get(key)))

heading(3, "HIP / ZLUDA entries in PATH", "PATH 中的 HIP / ZLUDA 条目")
for entry in os.environ.get("PATH", "").split(os.pathsep):
    if entry and (
        "ROCm" in entry or "AMD" in entry
        or "zluda" in entry.lower() or "Temp" in entry
    ):
        print("  %s" % entry)

heading(4, "ZLUDA candidates in temporary directories",
        "临时目录中的 ZLUDA 候选")
roots = [os.environ.get(key) for key in ("TEMP", "TMP")]
roots.append(tempfile.gettempdir())
local_appdata = os.environ.get("LOCALAPPDATA")
if local_appdata:
    roots.append(os.path.join(local_appdata, "Temp"))

seen = set()
found = []
for root in roots:
    if not root:
        continue
    try:
        root = os.path.abspath(root)
    except Exception:
        continue
    if root in seen or not os.path.isdir(root):
        continue
    seen.add(root)
    for folder in glob.glob(os.path.join(root, "*")):
        if os.path.isfile(os.path.join(folder, "cublas64_11.dll")):
            found.append(folder)
            print("  找到: %s" % folder)
if not found:
    print("  未找到包含 cublas64_11.dll 的目录")

heading(5, "Patch selection", "补丁实际选择")
try:
    import zluda_patch
except Exception as exc:
    print("  导入 zluda_patch 失败: %s: %s" % (type(exc).__name__, exc))
    record("FAIL", "zluda_patch import")
    raise SystemExit(finish())
record("PASS", "zluda_patch import")

print("  ZLUDA_DIR       = %s" % zluda_patch.ZLUDA_DIR)
print("  ROOT_DIR        = %s" % zluda_patch.ROOT_DIR)
print("  HIP_ROOT        = %s" % zluda_patch.HIP_ROOT)
print("  TENSILE_DIR 存在 = %s" % os.path.isdir(zluda_patch.TENSILE_DIR))
for path in zluda_patch.REPORTS:
    print("  状态文件        = %s" % path)

zluda_dir = zluda_patch.ZLUDA_DIR
record("PASS" if zluda_dir else "FAIL", "ZLUDA directory selected")
if zluda_dir:
    missing = [
        name for name in zluda_patch.ZLUDA_DLLS
        if not os.path.isfile(os.path.join(zluda_dir, name))
    ]
    record("WARN" if missing else "PASS", "ZLUDA DLL files")
else:
    record("SKIP", "ZLUDA DLL files")
record(
    "PASS" if os.path.isdir(zluda_patch.TENSILE_DIR) else "WARN",
    "rocBLAS Tensile directory",
)

heading(6, "nvcuda initialization", "nvcuda 初始化")
nvcuda_ok = False
if zluda_dir and os.path.isdir(zluda_dir):
    try:
        nvcuda = ctypes.WinDLL(os.path.join(zluda_dir, "nvcuda.dll"))
        result = nvcuda.cuInit(0)
        print("  cuInit(0) = %s（0 表示成功）" % result)
        count = ctypes.c_int(0)
        count_result = nvcuda.cuDeviceGetCount(ctypes.byref(count))
        print("  cuDeviceGetCount = %s，设备数 = %s" % (
            count_result, count.value,
        ))
        nvcuda_ok = result == 0 and count_result == 0 and count.value > 0
    except Exception as exc:
        print("  失败: %s: %s" % (type(exc).__name__, exc))
else:
    print("  没有可用的 ZLUDA 目录，跳过")
record("PASS" if nvcuda_ok else "FAIL", "nvcuda device initialization")

heading(7, "Torch and patch status", "Torch 与补丁状态")
cuda_ok = False
device_ok = False
patch_ok = False
try:
    import torch
    import torch.nn.functional as functional

    print("  torch              = %s" % torch.__version__)
    print("  torch.version.cuda = %s" % torch.version.cuda)
    cuda_ok = bool(torch.cuda.is_available())
    print("  cuda.is_available  = %s" % cuda_ok)
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        print("  device             = %s" % device_name)
        device_ok = bool(device_name)
    linear_name = getattr(functional.linear, "__name__", "?")
    addmm_name = getattr(torch.addmm, "__name__", "?")
    cudnn_disabled = not torch.backends.cudnn.enabled
    print("  F.linear           = %s" % linear_name)
    print("  torch.addmm        = %s" % addmm_name)
    print("  cudnn.enabled      = %s" % torch.backends.cudnn.enabled)
    patch_ok = (
        linear_name == "_linear"
        and addmm_name == "_addmm"
        and cudnn_disabled
    )
except Exception as exc:
    print("  Torch 检查失败: %s: %s" % (type(exc).__name__, exc))
record("PASS" if cuda_ok else "FAIL", "Torch CUDA availability")
record("PASS" if device_ok else "FAIL", "CUDA device")
record("PASS" if patch_ok else "FAIL", "ZLUDA patch integration")

heading(8, "Check summary", "检查汇总")
raise SystemExit(finish())

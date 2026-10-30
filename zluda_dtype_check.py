"""Offline self-check for the dtype policy in ComfyUI/zluda_patch.py.

Importing zluda_patch is not an option here: it preloads the ZLUDA DLLs,
imports torch and installs import hooks. This script loads the same source with
the final `_bootstrap()` call removed, swaps the status reporter for a list and
drives `_apply_model_management()` with a fake `comfy.model_management` module,
so every branch is checked without a GPU, without the UI and without writing
`_zluda_status.txt`.

Usage:
    python zluda_dtype_check.py [path\\to\\zluda_patch.py]

Exit code 0 when every case passes, 1 otherwise.

ZLUDA 精度策略离线自检。

直接 import zluda_patch 会预加载 ZLUDA 的 DLL、导入 torch 并安装导入钩子，
所以这里改为读取同一份源码、去掉末尾的 `_bootstrap()` 调用、把状态输出换成
一个列表，再用一个假的 comfy.model_management 模块驱动
`_apply_model_management()`。不需要显卡、不需要开界面，也不会写
`_zluda_status.txt`。

用法：
    python zluda_dtype_check.py [补丁路径]

全部通过时退出码为 0，否则为 1。
"""

import os
import sys
import traceback
from types import SimpleNamespace

GIB = 1024.0 ** 3

# The four switches that change the behaviour under test; cleared before every
# load so the result never depends on the caller's environment.
# 会改变被测行为的四个开关；每次加载前先清空，结果不受调用者环境影响。
SWITCHES = (
    "ZLUDA_ALLOW_BF16",
    "ZLUDA_UNET_DTYPE",
    "ZLUDA_VAE_DTYPE",
    "ZLUDA_FP32_MARGIN",
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATCHES = (
    os.path.join(HERE, "ComfyUI", "zluda_patch.py"),
    os.path.join(HERE, "zluda_patch.py"),
)


# A stand-in for torch.dtype: only str() and identity comparisons are used.
# torch.dtype 的替身：只用到 str() 和同一性比较。
class DType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "torch." + self.name

    __repr__ = __str__


class _Torch:
    float16 = DType("float16")
    bfloat16 = DType("bfloat16")
    float32 = DType("float32")
    float64 = DType("float64")
    float8_e4m3fn = DType("float8_e4m3fn")
    # Integer stores, for the quantized-weight cases.
    # 整型存储，用于量化权重那几条用例。
    uint8 = DType("uint8")
    int8 = DType("int8")


TORCH = _Torch()
DEV = "cuda:0"
LOG = []


def load_patch(path, env=None):
    """Exec the patch without its bootstrap and return its namespace."""
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    if source.count("_bootstrap()") < 2:
        raise SystemExit(
            "%s: expected a _bootstrap() definition and a call" % path)
    head, tail = source.rsplit("_bootstrap()", 1)
    if tail.strip():
        raise SystemExit(
            "%s: unexpected text after _bootstrap(): %r"
            % (path, tail.strip()[:80]))

    saved = {name: os.environ.pop(name, None) for name in SWITCHES}
    try:
        os.environ.update(env or {})
        namespace = {"__file__": path, "__name__": "zluda_patch_offline"}
        exec(compile(head, path, "exec"), namespace)
    finally:
        for name, value in saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value

    # Never touch the real status files from here.
    # 从这里开始绝不碰真正的状态文件。
    namespace["_report"] = LOG.append
    return namespace


def _upstream(module):
    """Minimal copy of ComfyUI 0.30.2 unet_dtype / vae_dtype / unet_manual_cast."""
    torch = module.torch

    def unet_dtype(device=None, model_params=0,
                   supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
                   weight_dtype=None):
        args = module.args
        if args.fp32_unet:
            return torch.float32
        if args.bf16_unet:
            return torch.bfloat16
        if args.fp16_unet:
            return torch.float16
        if args.fp8_e4m3fn_unet:
            return torch.float8_e4m3fn
        if (weight_dtype in module.FLOAT8_TYPES
                and model_params * 2 > module.maximum_vram_for_weights(device)):
            return weight_dtype
        if weight_dtype == torch.float16 and torch.float16 in supported_dtypes:
            return torch.float16
        for dtype in supported_dtypes:
            if (dtype == torch.float16
                    and module.should_use_fp16(
                        device=device, model_params=model_params)):
                return torch.float16
            if (dtype == torch.bfloat16
                    and module.should_use_bf16(
                        device, model_params=model_params)):
                return torch.bfloat16
        return torch.float32

    def vae_dtype(device=None, allowed_dtypes=[]):
        args = module.args
        if args.fp16_vae:
            return torch.float16
        if args.bf16_vae:
            return torch.bfloat16
        if args.fp32_vae:
            return torch.float32
        for dtype in allowed_dtypes:
            if dtype == torch.float16 and module.should_use_fp16(device):
                return dtype
            if dtype == torch.bfloat16 and module.should_use_bf16(device):
                return dtype
        return torch.float32

    def unet_manual_cast(weight_dtype, inference_device,
                         supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
        if weight_dtype in (torch.float32, torch.float64):
            return None
        fp16_ok = module.should_use_fp16(
            inference_device, prioritize_performance=False)
        if fp16_ok and weight_dtype == torch.float16:
            return None
        bf16_ok = module.should_use_bf16(inference_device)
        if bf16_ok and weight_dtype == torch.bfloat16:
            return None
        for dtype in supported_dtypes:
            if dtype == torch.float16 and fp16_ok:
                return torch.float16
            if dtype == torch.bfloat16 and bf16_ok:
                return torch.bfloat16
        return torch.float32

    return unet_dtype, vae_dtype, unet_manual_cast


def make_module(ns, flags=None, fp16=True, bf16=True, budget_gib=11.766):
    """A fake comfy.model_management with the real signatures."""
    module = SimpleNamespace()
    module.torch = TORCH
    module.FLOAT8_TYPES = (TORCH.float8_e4m3fn,)
    module.args = SimpleNamespace()
    for name in ns["_PRECISION_FLAGS"]:
        setattr(module.args, name, False)
    for name, value in (flags or {}).items():
        setattr(module.args, name, value)
    module.should_use_fp16 = (
        lambda device=None, model_params=0, prioritize_performance=True,
        manual_cast=False: fp16)
    module.should_use_bf16 = (
        lambda device=None, model_params=0, prioritize_performance=True,
        manual_cast=False: bf16)
    module.maximum_vram_for_weights = lambda device=None: budget_gib * GIB
    unet_dtype, vae_dtype, manual_cast = _upstream(module)
    module.unet_dtype = unet_dtype
    module.vae_dtype = vae_dtype
    module.unet_manual_cast = manual_cast
    return module


def install(ns, **kwargs):
    """Run the patch unit against a fresh fake module; return it and the log."""
    del LOG[:]
    module = make_module(ns, **kwargs)
    outcome = ns["_apply_model_management"](module)
    return module, outcome, list(LOG)


class Checker:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            print("[PASS] %s" % name)
        else:
            self.failed += 1
            print("[FAIL] %s%s" % (name, (": " + detail) if detail else ""))

    def dtype(self, name, got, want, line=""):
        self.check(
            name, got == want,
            "got %s, want %s%s" % (got, want, ("; log=" + line) if line else ""))

    def text(self, name, lines, *needles):
        joined = " | ".join(lines)
        missing = [needle for needle in needles if needle not in joined]
        self.check(name, not missing,
                   "missing %r in %r" % (missing, joined) if missing else "")


def unet(ns, params, supported, weight=None, **kwargs):
    module, _, _ = install(ns, **kwargs)
    result = module.unet_dtype(DEV, params, supported, weight)
    return result, " | ".join(LOG)


def vae(ns, allowed, **kwargs):
    module, _, _ = install(ns, **kwargs)
    result = module.vae_dtype(DEV, allowed)
    return result, " | ".join(LOG)


def manual_cast(ns, weight, **kwargs):
    module, _, _ = install(ns, **kwargs)
    result = module.unet_manual_cast(weight, DEV, list(ANIMA))
    return result, " | ".join(LOG)


ANIMA = [TORCH.bfloat16, TORCH.float16, TORCH.float32]
BF16_FP32 = [TORCH.bfloat16, TORCH.float32]


def main(argv):
    path = argv[1] if len(argv) > 1 else next(
        (candidate for candidate in DEFAULT_PATCHES if os.path.isfile(candidate)),
        None)
    if path is None or not os.path.isfile(path):
        print("patch not found; pass its path as the first argument")
        return 2
    print("patch: %s" % path)
    ns = load_patch(path)
    check = Checker()

    # UNet, six branches.
    # UNet 的六条分支。
    got, log = unet(ns, 8_000_000_000, BF16_FP32, flags={"fp16_unet": True})
    check.dtype("unet: explicit --fp16-unet", got, TORCH.float16, log)
    check.text("unet: explicit flag is delegated", [log],
               "(comfy order: --fp16-unet)")

    got, log = unet(ns, 12_000_000_000, ANIMA, weight=TORCH.float8_e4m3fn)
    check.dtype("unet: fp8 weights keep upstream", got, TORCH.float8_e4m3fn, log)
    check.text("unet: fp8 reason", [log], "(comfy order: fp8 weights float8_e4m3fn)")

    got, log = unet(ns, 8_000_000_000, ANIMA, weight=TORCH.float16)
    check.dtype("unet: fp16 weights keep upstream", got, TORCH.float16, log)
    check.text("unet: fp16 weights reason", [log], "(comfy order: fp16 weights)")

    got, log = unet(ns, 8_000_000_000, ANIMA, weight=TORCH.bfloat16)
    check.dtype("unet: model lists fp16", got, TORCH.float16, log)
    check.text("unet: fp16 policy reason", [log],
               "(policy: model lists fp16", "comfy would pick bfloat16")

    got, log = unet(ns, 1_000_000_000, BF16_FP32, weight=TORCH.bfloat16)
    check.dtype("unet: fp32 fits the budget", got, TORCH.float32, log)
    check.text("unet: budget fits reason", [log], "(policy: fp32 3.7 GiB <= 10.6 GiB")

    got, log = unet(ns, 6_000_000_000, BF16_FP32, weight=TORCH.uint8)
    check.dtype("unet: bf16 when fp32 does not fit", got, TORCH.bfloat16, log)
    check.text("unet: budget miss reason", [log],
               "fp32 22.4 GiB > 10.6 GiB", "weights uint8,")

    got, log = unet(ns, 6_000_000_000, BF16_FP32, weight=TORCH.bfloat16, bf16=False)
    check.dtype("unet: fp32 handed back when bf16 is unusable", got, TORCH.float32, log)
    check.text("unet: bf16 unusable reason", [log], "bf16 unusable")

    # VAE, four branches.
    # VAE 的四条分支。
    got, log = vae(ns, ANIMA)
    check.dtype("vae: fp16 when listed", got, TORCH.float16, log)
    check.text("vae: fp16 policy reason", [log],
               "(policy: fp16 in [bfloat16, float16, float32]",
               "comfy would pick bfloat16")

    got, log = vae(ns, ANIMA, fp16=False)
    check.dtype("vae: bf16 when the device has no fp16", got, TORCH.bfloat16, log)
    check.text("vae: fp16 unsupported reason", [log], "fp16 unsupported by device")

    got, log = vae(ns, BF16_FP32)
    check.dtype("vae: bf16 when fp16 is not listed", got, TORCH.bfloat16, log)
    check.text("vae: no fp16 reason", [log], "no fp16 in [bfloat16, float32]")

    got, log = vae(ns, [TORCH.float32])
    check.dtype("vae: fp32 handed back", got, TORCH.float32, log)
    check.text("vae: single dtype reason", [log], "(comfy order: list [float32])")

    # Manual cast: unknown / fp8 weights must not stay on bf16.
    # manual cast：未知 / fp8 权重不再停在 bf16。
    got, log = manual_cast(ns, None)
    check.dtype("manual cast: unknown weights -> fp16", got, TORCH.float16, log)
    check.text("manual cast: reason", [log],
               "manual cast float16 (policy: upstream bf16, weights unknown)")

    got, log = manual_cast(ns, TORCH.float8_e4m3fn)
    check.dtype("manual cast: fp8 weights -> fp16", got, TORCH.float16, log)

    got, log = manual_cast(ns, TORCH.bfloat16)
    check.dtype("manual cast: bf16 weights untouched", got, None, log)

    got, log = manual_cast(ns, TORCH.bfloat16, flags={"bf16_unet": True})
    check.dtype("manual cast: --bf16-unet untouched", got, None, log)

    # Fail-open: an unexpected error inside the policy hands that call back to
    # ComfyUI instead of breaking the load.
    # fail-open：策略内部意外出错时，把这一次调用交回 ComfyUI，而不是搞崩加载。
    module, _, _ = install(ns)
    saved = ns["_weights_note"]

    def _boom(weight_dtype):
        raise RuntimeError("boom")

    ns["_weights_note"] = _boom
    try:
        got = module.unet_dtype(DEV, 8_000_000_000, ANIMA, None)
    finally:
        ns["_weights_note"] = saved
    check.dtype("fail-open: upstream answer", got, TORCH.bfloat16, " | ".join(LOG))
    check.text("fail-open: reported once", LOG, "unet policy FAILED", "RuntimeError")

    # One side off, the other still on.
    # 一侧关闭、另一侧仍然生效。
    ns["ZLUDA_UNET_DTYPE"], ns["ZLUDA_VAE_DTYPE"] = "comfy", "auto"
    module, outcome, _ = install(ns)
    check.check("switch: unet_dtype=comfy keeps upstream unet_dtype",
                module.unet_dtype.__name__ == "unet_dtype",
                "unet_dtype was replaced")
    check.check("switch: unet_dtype=comfy keeps upstream manual cast",
                module.unet_manual_cast.__name__ == "unet_manual_cast",
                "unet_manual_cast was replaced")
    check.check("switch: vae_dtype=auto still installs the VAE side",
                module.vae_dtype.__name__ == "_vae_dtype",
                "vae_dtype was not replaced")
    check.text("switch: status line, unet off", [outcome.status],
               "unet=comfy-order | vae=fp16-priority")

    ns["ZLUDA_UNET_DTYPE"], ns["ZLUDA_VAE_DTYPE"] = "auto", "comfy"
    module, outcome, _ = install(ns)
    check.check("switch: vae_dtype=comfy keeps upstream vae_dtype",
                module.vae_dtype.__name__ == "vae_dtype",
                "vae_dtype was replaced")
    check.check("switch: unet_dtype=auto still installs the UNet side",
                module.unet_dtype.__name__ == "_unet_dtype",
                "unet_dtype was not replaced")
    check.text("switch: status line, vae off", [outcome.status],
               "unet=policy | vae=comfy-order")

    # The alias sets both sides, but an explicit side wins.
    # 别名把两侧都设成 comfy，但显式写的那一侧优先。
    ns = load_patch(path, env={"ZLUDA_ALLOW_BF16": "1"})
    check.check("alias: allow_bf16 sets both sides",
                (ns["ZLUDA_UNET_DTYPE"], ns["ZLUDA_VAE_DTYPE"]) == ("comfy", "comfy"),
                "%r" % ((ns["ZLUDA_UNET_DTYPE"], ns["ZLUDA_VAE_DTYPE"]),))
    ns = load_patch(path, env={"ZLUDA_ALLOW_BF16": "1", "ZLUDA_UNET_DTYPE": "auto"})
    check.check("alias: explicit side wins",
                (ns["ZLUDA_UNET_DTYPE"], ns["ZLUDA_VAE_DTYPE"]) == ("auto", "comfy"),
                "%r" % ((ns["ZLUDA_UNET_DTYPE"], ns["ZLUDA_VAE_DTYPE"]),))

    # Missing args: no wrapper at all.
    # 拿不到 args：两侧都不装。
    ns = load_patch(path)
    module = make_module(ns)
    del module.args
    outcome = ns["_apply_model_management"](module)
    check.text("no args: status", [outcome.status], "unet=off | vae=off")

    print()
    print("%d passed, %d failed" % (check.passed, check.failed))
    return 0 if check.failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception:
        traceback.print_exc()
        sys.exit(2)

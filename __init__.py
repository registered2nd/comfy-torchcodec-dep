"""Make torchcodec loadable, and say out loud what is on this pod.

torchaudio 2.11+ routes every audio write through torchcodec, and both LatentSync wrappers
write a temp wav before inference, so lip sync on this deployment stands or falls on
torchcodec importing. It ships with no FFmpeg of its own: it dlopens libtorchcodec_coreN.so,
whose libav* dependencies the system loader has to satisfy. The symbol-versioned FFmpeg the
official wheels accept lives at /opt/ffmpeg7, which is not on the loader path, and setting
LD_LIBRARY_PATH or patching RPATH both need a shell on the pod.

Loading the libav* objects here with RTLD_GLOBAL does the same job from inside the process:
once they are resolved globally, the loader satisfies libtorchcodec_core's dependencies from
what is already mapped. This module is imported by ComfyUI at startup, which is before any
prompt runs and therefore before torchaudio first reaches for torchcodec.

Registers no nodes. Everything it prints goes to /internal/logs/raw.
"""
import ctypes
import glob
import os
import sys

TAG = "[torchcodec-dep]"


def _say(msg):
    print(f"{TAG} {msg}", flush=True)


def _report():
    _say(f"python={sys.version.split()[0]} exe={sys.executable}")
    try:
        import torch
        _say(f"torch={torch.__version__}")
    except Exception as e:
        _say(f"torch import failed: {e}")
    for p in ("/opt", "/root/ComfyUI/user"):
        try:
            _say(f"ls {p} -> {sorted(os.listdir(p))[:40]}")
        except Exception as e:
            _say(f"ls {p} failed: {e}")
    for pat in ("/opt/*/lib/libav*.so*", "/usr/lib64/libav*.so*", "/usr/lib/libav*.so*",
                "/usr/local/lib/libav*.so*", "/usr/lib64/libsw*.so*"):
        hits = sorted(glob.glob(pat))
        if hits:
            _say(f"{pat} -> {[os.path.basename(h) for h in hits][:20]}")
    for mod in ("torchcodec",):
        try:
            m = __import__(mod)
            _say(f"{mod} at {getattr(m, '__file__', '?')}")
        except Exception as e:
            _say(f"{mod} not importable yet: {type(e).__name__}: {str(e)[:200]}")


# libavutil underpins everything else, and libavformat depends on codec/swresample, so the
# load order is not arbitrary - a dependency loaded second is a dependency the loader has
# already failed to find once.
_ORDER = ("libavutil", "libswresample", "libswscale", "libavcodec", "libavformat",
          "libavfilter", "libavdevice")


def _preload():
    roots = sorted(glob.glob("/opt/ffmpeg*/lib")) + ["/usr/local/lib", "/usr/lib64", "/usr/lib"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        loaded, failed = [], []
        for stem in _ORDER:
            for path in sorted(glob.glob(os.path.join(root, f"{stem}.so.*"))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(path))
                    break
                except OSError as e:
                    failed.append(f"{os.path.basename(path)}: {str(e)[:80]}")
        if loaded:
            _say(f"preloaded from {root}: {loaded}")
            if failed:
                _say(f"  skipped: {failed[:6]}")
            return root
    _say("no libav* preloaded from any candidate root")
    return None


_report()
_preload()
try:
    from torchcodec.encoders import AudioEncoder  # noqa: F401
    _say("torchcodec.encoders.AudioEncoder OK - torchaudio.save will work")
except Exception as e:
    _say(f"torchcodec still failing: {type(e).__name__}: {str(e)[:600]}")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

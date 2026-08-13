"""Give torchaudio a WAV path that does not go through torchcodec.

From torchaudio 2.11 every `save` and `load` is routed to torchcodec, which has no codecs of
its own and dlopens FFmpeg's C libraries. This pod carries FFmpeg 8 only (libavutil.so.60,
libavcodec.so.62) and /opt is empty, so there is no FFmpeg 4-7 for a torchcodec wheel to bind
to and nothing importable to bind with. Both LatentSync wrappers call `torchaudio.save` to
write a temporary wav before inference, so lip sync fails at the write step with the model
loaded and the weights in place.

A wav needs no codec. This replaces `save` and `load` with the standard library's `wave`
module for wav paths only, and defers to the original implementation for every other format,
so the patch can only turn an exception into a working call and never changes a path that
already worked. ComfyUI imports this at startup, before any prompt runs.

Registers no nodes. Everything it prints goes to /internal/logs/raw.
"""
import os
import sys
import wave

TAG = "[torchcodec-dep]"


def _say(msg):
    print(f"{TAG} {msg}", flush=True)


def _is_wav(uri):
    return isinstance(uri, (str, os.PathLike)) and str(uri).lower().endswith(".wav")


def _patch():
    import numpy as np
    import torch
    import torchaudio

    orig_save, orig_load = torchaudio.save, torchaudio.load

    def save(uri, src, sample_rate, channels_first=True, format=None, **kw):
        if not _is_wav(uri) or format not in (None, "wav"):
            return orig_save(uri, src, sample_rate, channels_first=channels_first,
                             format=format, **kw)
        x = src.detach().cpu()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if not channels_first:
            x = x.transpose(0, 1)
        # float tensors are [-1, 1]; clamp before scaling or a hot sample wraps to the
        # opposite polarity and the mouth gets driven by a click.
        if x.is_floating_point():
            x = (x.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)
        else:
            x = x.to(torch.int16)
        data = x.transpose(0, 1).contiguous().numpy().astype("<i2")
        with wave.open(str(uri), "wb") as w:
            w.setnchannels(int(x.shape[0]))
            w.setsampwidth(2)
            w.setframerate(int(sample_rate))
            w.writeframes(data.tobytes())
        return None

    def load(uri, *a, channels_first=True, **kw):
        if not _is_wav(uri):
            return orig_load(uri, *a, channels_first=channels_first, **kw)
        with wave.open(str(uri), "rb") as r:
            ch, width, rate, n = r.getnchannels(), r.getsampwidth(), r.getframerate(), r.getnframes()
            raw = r.readframes(n)
        if width != 2:
            return orig_load(uri, *a, channels_first=channels_first, **kw)
        arr = np.frombuffer(raw, dtype="<i2").reshape(-1, ch).astype("float32") / 32768.0
        t = torch.from_numpy(arr.copy())
        return (t.transpose(0, 1).contiguous() if channels_first else t), rate

    torchaudio.save, torchaudio.load = save, load
    _say(f"torchaudio {torchaudio.__version__} save/load patched for wav via stdlib wave")


try:
    _patch()
except Exception as e:
    _say(f"patch failed: {type(e).__name__}: {e}")

# Prove it round-trips here rather than discovering it inside a 12-minute render.
try:
    import tempfile

    import torch
    import torchaudio
    p = os.path.join(tempfile.gettempdir(), "torchcodec_dep_selftest.wav")
    sig = torch.sin(torch.arange(16000, dtype=torch.float32) * 0.05).unsqueeze(0) * 0.5
    torchaudio.save(p, sig, 16000)
    back, sr = torchaudio.load(p)
    err = (back - sig).abs().max().item()
    _say(f"self-test: wrote+read {back.shape} at {sr} Hz, max error {err:.5f}")
    os.remove(p)
except Exception as e:
    _say(f"self-test FAILED: {type(e).__name__}: {e}")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

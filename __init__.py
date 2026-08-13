"""No nodes. This package exists so ComfyUI Manager's git-URL installer runs its
requirements.txt, which is the only route on this deployment that pip-installs into the
environment ComfyUI actually runs. torchaudio 2.11+ routes every audio write through
torchcodec and both LatentSync wrappers write a temp wav before inference, so without it
lip sync cannot run."""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

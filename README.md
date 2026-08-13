# comfy-torchcodec-dep

An empty ComfyUI custom-node package whose only purpose is its `requirements.txt`.

Installing it through ComfyUI Manager's *Install via Git URL* makes Manager run
`pip install -r requirements.txt` in the interpreter ComfyUI runs, which is how the
`torchcodec` wheel gets into an environment where Manager's own *Install PIP packages*
dialog silently does nothing.

Registers no nodes and imports nothing.

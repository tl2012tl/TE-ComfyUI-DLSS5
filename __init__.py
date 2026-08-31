"""TE-ComfyUI-DLSS5

ComfyUI nodes for streaming video through a separately built NVIDIA NGX
backend.  The Python package intentionally contains no NVIDIA SDK code.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

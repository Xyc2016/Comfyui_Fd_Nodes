"""Top-level package for Comfyui_Fd_Nodes."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",

]

__author__ = """xuyucai"""
__email__ = "986327386@qq.com"
__version__ = "53.3.2"

try:
    from .src.Comfyui_Fd_Nodes.nodes import NODE_CLASS_MAPPINGS
    from .src.Comfyui_Fd_Nodes.nodes import NODE_DISPLAY_NAME_MAPPINGS
except ModuleNotFoundError as exc:
    if not exc.name.startswith("comfy"):
        raise
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

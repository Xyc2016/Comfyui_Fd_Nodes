import os
import sys
import types

os.environ.setdefault("FD_LITELLM_BASE_URL", "https://example.com")
os.environ.setdefault("FD_LITELLM_API_KEY", "test-key")

# Add the project root directory to Python path
# This allows the tests to import the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _install_comfy_test_stubs():
    if "comfy.comfy_types.node_typing" in sys.modules:
        return

    comfy_module = sys.modules.get("comfy") or types.ModuleType("comfy")
    comfy_types_module = types.ModuleType("comfy.comfy_types")
    node_typing_module = types.ModuleType("comfy.comfy_types.node_typing")
    utils_module = types.ModuleType("comfy.utils")

    class IO:
        STRING = "STRING"
        COMBO = "COMBO"
        INT = "INT"
        FLOAT = "FLOAT"
        BOOLEAN = "BOOLEAN"
        IMAGE = "IMAGE"

    class ComfyNodeABC:
        pass

    def common_upscale(samples, width, height, _upscale_method, _crop):
        import torch.nn.functional as F

        return F.interpolate(samples, size=(height, width), mode="bilinear", align_corners=False)

    node_typing_module.IO = IO
    node_typing_module.ComfyNodeABC = ComfyNodeABC
    node_typing_module.InputTypeDict = dict
    utils_module.common_upscale = common_upscale
    comfy_module.comfy_types = comfy_types_module
    comfy_module.utils = utils_module
    comfy_types_module.node_typing = node_typing_module

    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.comfy_types"] = comfy_types_module
    sys.modules["comfy.comfy_types.node_typing"] = node_typing_module
    sys.modules["comfy.utils"] = utils_module


_install_comfy_test_stubs()

"""Deployment routing, legacy graph isolation and refined pixel preservation."""

import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from src.Comfyui_Fd_Nodes import zhiyi_rmbg_segment_node as module
from src.Comfyui_Fd_Nodes.v2.rmbg_segment_v2_node import ZhiYiRMBGNodeV2
from tests.test_rmbg_mask_only import call_node, mask_response


@pytest.mark.parametrize("runtime,quality,host", [
    ("dev", "png", "10.1.0.230:8003"),
    ("domestic", "png", "model-api-dwpose-svc.online-server-gray:8001"),
    ("overseas", "jpeg_q95", "overseas.example/model-api-dwpose"),
])
@pytest.mark.parametrize("override", [False, True])
def test_runtime_config(runtime: str, quality: str, host: str, override: bool) -> None:
    env = {**os.environ, "FD_COMFY_RUNTIME_ENV": runtime, "FD_OVERSEAS_GPU_GATEWAY": "http://overseas.example"}
    env.pop("FD_RMBG_URL", None)
    if override:
        env["FD_RMBG_URL"] = "http://configured.example/v1/rmbg"
    config = Path(module.__file__).with_name("config.py")
    code = "import runpy,json,sys; c=runpy.run_path(sys.argv[1]); print(json.dumps([c['RMBG_RUNTIME_URL'],c['RMBG_DEFAULT_IMAGE_ENCODING']]))"
    result = subprocess.run([sys.executable, "-c", code, str(config)], env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [f"http://{host}/v1/rmbg" if not override else env["FD_RMBG_URL"], quality]


def test_invalid_runtime_fails_on_load() -> None:
    env = {**os.environ, "FD_COMFY_RUNTIME_ENV": "typo"}
    result = subprocess.run([sys.executable, str(Path(module.__file__).with_name("config.py"))],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "FD_COMFY_RUNTIME_ENV must be" in result.stderr


@pytest.mark.parametrize("node_cls", [module.ZhiYiRMBGNode, ZhiYiRMBGNodeV2])
@pytest.mark.parametrize("background", ["Alpha", "Color"])
def test_refine_preserves_server_pixels(monkeypatch, node_cls: type, background: str) -> None:
    node = node_cls()
    mode = "RGBA" if background == "Alpha" else "RGB"
    color = (37, 111, 207, 128) if mode == "RGBA" else (37, 111, 207)
    refined = Image.new(mode, (5, 3), color)
    buffer = io.BytesIO()
    refined.save(buffer, format="PNG")
    response = mask_response()
    response["processed_image"] = base64.b64encode(buffer.getvalue()).decode()

    def post(url: str, payload: dict, timeout: int, label: str) -> dict:
        assert (payload["return_image"], payload["return_mask"], payload["return_mask_image"]) == (True, True, False)
        assert payload["refine_foreground"] is True
        return response

    monkeypatch.setattr(node, "_post_process", post)
    image = torch.zeros(1, 3, 5, 3)
    output, mask, preview, text = call_node(node, image, refine_foreground=True, background=background,
                                          return_image=False, return_mask=False)
    assert torch.equal(output, torch.from_numpy(np.asarray(refined).copy().astype(np.float32) / 255).unsqueeze(0))
    assert torch.equal(preview, mask.unsqueeze(-1).expand(-1, -1, -1, 3))
    assert json.loads(text)["mask_only_response"] is False


def test_refine_missing_image_fails(monkeypatch) -> None:
    node = ZhiYiRMBGNodeV2()
    monkeypatch.setattr(node, "_post_process", lambda *args: mask_response())
    with pytest.raises(RuntimeError, match="processed_image"):
        call_node(node, torch.zeros(1, 3, 5, 3), refine_foreground=True)


@pytest.mark.parametrize("node_cls", [module.ZhiYiRMBGNode, ZhiYiRMBGNodeV2])
def test_retired_controls_are_not_node_inputs(node_cls: type) -> None:
    fields = node_cls.INPUT_TYPES()
    assert not {"env", "service_url", "service_url_preset", "return_image", "return_mask", "return_mask_image"} & (
        fields["required"].keys() | fields["optional"].keys())

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


@pytest.mark.parametrize("runtime,host", [
    ("dev", "10.1.0.230:8003"),
    ("domestic", "model-api-dwpose-svc.online-server-gray:8001"),
    ("overseas", "overseas.example/model-api-dwpose"),
])
@pytest.mark.parametrize("override", [False, True])
def test_runtime_config(runtime: str, host: str, override: bool) -> None:
    env = {**os.environ, "FD_COMFY_RUNTIME_ENV": runtime, "FD_OVERSEAS_GPU_GATEWAY": "http://overseas.example"}
    env.pop("FD_RMBG_URL", None)
    if override:
        env["FD_RMBG_URL"] = "http://configured.example/v1/rmbg"
    config = Path(module.__file__).with_name("config.py")
    code = "import runpy,json,sys; c=runpy.run_path(sys.argv[1]); print(json.dumps([c['RMBG_RUNTIME_URL']]))"
    result = subprocess.run([sys.executable, "-c", code, str(config)], env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [f"http://{host}/v1/rmbg" if not override else env["FD_RMBG_URL"]]


def _mask_response(size: tuple[int, int] = (5, 3), source_size: tuple[int, int] = (5, 3)) -> dict:
    values = np.linspace(0, 255, size[0] * size[1], dtype=np.uint8).reshape(size[1], size[0])
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "mask": f"data:image/png;base64,{encoded}",
        "processed_image": None,
        "mask_image": None,
        "image_size": {"width": source_size[0], "height": source_size[1]},
        "inference_size": {"width": size[0], "height": size[1]},
        "output_size": {"width": size[0], "height": size[1]},
    }


def _call_node(node, image, **overrides):
    kwargs = {
        name: spec[1]["default"]
        for name, spec in node.INPUT_TYPES()["required"].items()
        if name != "image"
    }
    kwargs.update(image=image, service_url="http://overseas.example", env="overseas")
    kwargs.update(overrides)
    return getattr(node, node.FUNCTION)(**kwargs)


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
    response = _mask_response()
    response["processed_image"] = base64.b64encode(buffer.getvalue()).decode()

    def post(url: str, payload: dict, timeout: int, label: str) -> dict:
        assert (payload["return_image"], payload["return_mask"], payload["return_mask_image"]) == (True, True, False)
        assert payload["refine_foreground"] is True
        return response

    monkeypatch.setattr(node, "_post_process", post)
    image = torch.zeros(1, 3, 5, 3)
    output, mask, preview, text = _call_node(node, image, refine_foreground=True, background=background,
                                          return_image=False, return_mask=False)
    assert torch.equal(output, torch.from_numpy(np.asarray(refined).copy().astype(np.float32) / 255).unsqueeze(0))
    assert torch.equal(preview, mask.unsqueeze(-1).expand(-1, -1, -1, 3))


def test_refine_missing_image_fails(monkeypatch) -> None:
    node = ZhiYiRMBGNodeV2()
    monkeypatch.setattr(node, "_post_process", lambda *args: _mask_response())
    with pytest.raises(RuntimeError, match="processed_image"):
        _call_node(node, torch.zeros(1, 3, 5, 3), refine_foreground=True)


@pytest.mark.parametrize("node_cls", [module.ZhiYiRMBGNode, ZhiYiRMBGNodeV2])
def test_retired_controls_are_not_node_inputs(node_cls: type) -> None:
    fields = node_cls.INPUT_TYPES()
    assert not {"env", "service_url", "service_url_preset", "return_image", "return_mask", "return_mask_image"} & (
        fields["required"].keys() | fields["optional"].keys())

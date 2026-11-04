import base64
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes import zhiyi_rmbg_segment_node as module
from src.Comfyui_Fd_Nodes.v2.rmbg_segment_v2_node import ZhiYiRMBGNodeV2
from src.Comfyui_Fd_Nodes.config import FD_OVERSEAS_DWPOSE_URL


def mask_response(size: tuple[int, int] = (5, 3), source_size: tuple[int, int] = (5, 3)) -> dict:
    values = np.linspace(0, 255, size[0] * size[1], dtype=np.uint8).reshape(size[1], size[0])
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    return {
        "mask": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        "processed_image": None,
        "mask_image": None,
        "image_size": {"width": source_size[0], "height": source_size[1]},
        "inference_size": {"width": size[0], "height": size[1]},
        "output_size": {"width": size[0], "height": size[1]},
    }


def call_node(node: module._RmbgSegmentApiBase, image: torch.Tensor, **overrides: object) -> tuple:
    kwargs = {
        name: spec[1]["default"]
        for name, spec in node.INPUT_TYPES()["required"].items()
        if name != "image"
    }
    kwargs.update(image=image, service_url=FD_OVERSEAS_DWPOSE_URL)
    if "env" in kwargs:
        kwargs["env"] = "overseas"
    kwargs.update(overrides)
    return getattr(node, node.FUNCTION)(**kwargs)


@pytest.mark.parametrize("node_cls", [module.ZhiYiRMBGNode, ZhiYiRMBGNodeV2])
@pytest.mark.parametrize("background", ["Alpha", "Color"])
def test_overseas_returns_only_mask_and_composes_original(monkeypatch, node_cls: type, background: str) -> None:
    node = node_cls()
    image = torch.arange(45, dtype=torch.float32).reshape(1, 3, 5, 3) / 255
    original = image.clone()
    result = mask_response()
    captured = []

    def post(url: str, payload: dict, timeout: int, label: str) -> dict:
        captured.append(payload)
        assert (payload["return_image"], payload["return_mask"], payload["return_mask_image"]) == (False, True, False)
        assert payload["image_format"] == "png_base64"
        assert payload["process_res"] == 1024
        return result

    monkeypatch.setattr(node, "_post_process", post)
    monkeypatch.setattr(module, "tensor_to_jpeg_q95_payload", lambda *args: "https://example.com/inference.jpg")
    output, mask, preview, info_text = call_node(
        node, image, background=background, background_color="#123456",
        return_image=True, return_mask=False, return_mask_image=True,
    )
    alpha = torch.from_numpy(np.linspace(0, 255, 15, dtype=np.uint8).reshape(1, 3, 5).astype(np.float32) / 255)
    assert len(captured) == 1
    assert torch.equal(image, original)
    assert torch.equal(mask, alpha)
    assert torch.equal(preview, alpha.unsqueeze(-1).expand(-1, -1, -1, 3))
    if background == "Alpha":
        assert output.shape == (1, 3, 5, 4)
        assert torch.equal(output[..., 3], alpha)
        assert torch.equal(output[..., :3], image * (alpha.unsqueeze(-1) > 0))
        # Soft edge RGB stays straight alpha; it must not be multiplied twice.
        assert torch.equal(output[0, 0, 1, :3], image[0, 0, 1])
    else:
        color = torch.tensor([18, 52, 86], dtype=torch.float32) / 255
        assert torch.equal(output, image * alpha.unsqueeze(-1) + color * (1 - alpha.unsqueeze(-1)))
    info = json.loads(info_text)
    assert info["mask_only_response"] is True
    assert (info["return_image"], info["return_mask"], info["return_mask_image"]) == (False, True, False)


def test_other_segment_nodes_keep_original_return_flags(monkeypatch) -> None:
    node = module.ZhiYiBodySegmentNode()
    monkeypatch.setattr(module, "tensor_to_image_payload", lambda *args: "data:image/png;base64,input")

    def post(url: str, payload: dict, timeout: int, label: str) -> dict:
        assert (payload["return_image"], payload["return_mask"], payload["return_mask_image"]) == (True, False, True)
        return mask_response()

    monkeypatch.setattr(node, "_post_process", post)
    *_, info = call_node(node, torch.zeros(1, 3, 5, 3), return_image=True, return_mask=False, return_mask_image=True)
    assert json.loads(info)["mask_only_response"] is False


@pytest.mark.parametrize("node_cls", [module.ZhiYiRMBGNode, ZhiYiRMBGNodeV2])
@pytest.mark.parametrize("image_encoding", ["png", "jpeg_q95"])
@pytest.mark.parametrize("old_flag", [None, "0", "1"])
def test_node_quality_ignores_old_environment_switch(monkeypatch, node_cls: type, image_encoding: str,
                                                     old_flag: str | None, caplog) -> None:
    if old_flag is None:
        monkeypatch.delenv("FD_RMBG_REMOTE_OPTIMIZATION", raising=False)
    else:
        monkeypatch.setenv("FD_RMBG_REMOTE_OPTIMIZATION", old_flag)
    monkeypatch.setattr(module, "tensor_to_image_payload", lambda *args: "data:image/png;base64,input")
    monkeypatch.setattr(module, "tensor_to_jpeg_q95_payload", lambda *args: "data:image/jpeg;base64,input")
    node = node_cls()
    monkeypatch.setattr(module, "RMBG_RUNTIME_URL", "http://domestic.example/v1/rmbg")

    def post(url: str, payload: dict, timeout: int, label: str) -> dict:
        assert url == "http://domestic.example/v1/rmbg"
        assert payload["image"].startswith("data:image/jpeg;" if image_encoding == "jpeg_q95" else "data:image/png;")
        assert (payload["return_image"], payload["return_mask"], payload["return_mask_image"]) == (False, True, False)
        return mask_response()

    monkeypatch.setattr(node, "_post_process", post)
    overrides = {"env": "自定义（使用 service_url）"} if node_cls is ZhiYiRMBGNodeV2 else {}
    with caplog.at_level("INFO", logger=module.__name__):
        *_, info_text = call_node(node, torch.zeros(1, 3, 5, 3), service_url="http://domestic.example/v1/rmbg",
                                 image_encoding=image_encoding, return_image=True, return_mask=False,
                                 return_mask_image=True, **overrides)
    info = json.loads(info_text)
    assert info["mask_only_response"] is True
    assert info["image_encoding"] == image_encoding
    assert f"image_encoding={image_encoding} mask_only_response=True" in caplog.text


def test_declared_inference_mask_is_restored_with_bilinear() -> None:
    node = module.ZhiYiRMBGNode()
    response = mask_response(size=(2, 2), source_size=(5, 3))
    result = node._rmbg_mask_only_to_tensor(response, (5, 3))
    expected = np.array(Image.fromarray(np.array([[0, 85], [170, 255]], dtype=np.uint8)).resize((5, 3), Image.Resampling.BILINEAR))
    assert torch.equal(result, torch.from_numpy(expected.astype(np.float32) / 255).unsqueeze(0))


@pytest.mark.parametrize("field,value", [
    ("mask", None), ("mask", "broken-base64"),
    ("image_size", None), ("image_size", {"width": 3, "height": 5}),
    ("image_size", {"width": True, "height": 3}),
    ("inference_size", {"width": 0, "height": 3}),
    ("output_size", None), ("output_size", {"width": 99, "height": 3}),
    ("processed_image", "unexpected-image"), ("mask_image", "unexpected-preview"),
])
def test_invalid_mask_only_response_fails(field: str, value: object) -> None:
    response = mask_response()
    response[field] = value
    with pytest.raises(RuntimeError, match="mask|image_size|inference_size|output_size"):
        module.ZhiYiRMBGNode()._rmbg_mask_only_to_tensor(response, (5, 3))


@pytest.mark.parametrize("mode,format", [("RGB", "PNG"), ("L", "JPEG"), ("I;16", "PNG")])
def test_mask_encoding_must_be_lossless_grayscale(mode: str, format: str) -> None:
    response = mask_response()
    buffer = io.BytesIO()
    Image.new(mode, (5, 3)).save(buffer, format=format)
    response["mask"] = base64.b64encode(buffer.getvalue()).decode("ascii")
    with pytest.raises(RuntimeError, match="8 位灰度 PNG"):
        module.ZhiYiRMBGNode()._rmbg_mask_only_to_tensor(response, (5, 3))


def test_undeclared_output_size_fails() -> None:
    response = mask_response(size=(2, 2), source_size=(5, 3))
    response["inference_size"] = {"width": 4, "height": 2}
    with pytest.raises(RuntimeError, match="mask 尺寸异常"):
        module.ZhiYiRMBGNode()._rmbg_mask_only_to_tensor(response, (5, 3))

"""Q95 transport must preserve dimensions and compose from the original RGB."""

import base64
import hashlib
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image, JpegImagePlugin

from src.Comfyui_Fd_Nodes import zhiyi_rmbg_segment_node as module
from src.Comfyui_Fd_Nodes.utils import rmbg_image_payload as encoding
from src.Comfyui_Fd_Nodes.v2.rmbg_segment_v2_node import ZhiYiRMBGNodeV2


def source_tensor() -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(7).random((1, 19, 31, 4), dtype=np.float32))


def call_node(node: ZhiYiRMBGNodeV2, image: torch.Tensor, **overrides: object) -> tuple:
    kwargs = {name: spec[1]["default"] for name, spec in node.INPUT_TYPES()["required"].items() if name != "image"}
    kwargs.update(image=image, env="海外环境", image_encoding="jpeg_q95", health_check=False)
    kwargs.update(overrides)
    return node.remove_background(**kwargs)


def test_q95_is_rgb_same_size_and_does_not_mutate_source() -> None:
    image = source_tensor()
    original = image.clone()
    data = encoding.tensor_to_jpeg_q95_bytes(image)
    with Image.open(io.BytesIO(data)) as jpeg:
        jpeg.load()
        assert jpeg.format == "JPEG" and jpeg.mode == "RGB" and jpeg.size == (31, 19)
        assert JpegImagePlugin.get_sampling(jpeg) == 2
        assert jpeg.quantization[0][0] == 2  # Q95 luminance DC; Q100 would be 1.
        assert not jpeg.info.get("progressive")
        assert not np.array_equal(np.asarray(jpeg), (image[0, ..., :3].numpy() * 255).astype(np.uint8))
    assert torch.equal(image, original)


def test_base64_identifies_jpeg(monkeypatch) -> None:
    monkeypatch.delenv("FD_OVERSEAS_GPU_GATEWAY", raising=False)
    monkeypatch.delenv("FD_IMAGE_DELIVERY", raising=False)
    image = source_tensor()
    payload = encoding.tensor_to_jpeg_q95_payload(image, "base64")
    prefix, data = payload.split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    assert base64.b64decode(data, validate=True) == encoding.tensor_to_jpeg_q95_bytes(image)


def test_url_upload_has_jpeg_suffix_and_content_hash(monkeypatch) -> None:
    captured = []

    def upload(path: str, data: bytes) -> str:
        captured.append((path, data))
        return "https://oss.example.com/" + path

    monkeypatch.setattr(encoding, "upload_bytes_to_oss", upload)
    image = source_tensor()
    payload = encoding.tensor_to_jpeg_q95_payload(image, "url")
    path, data = captured[0]
    assert len(captured) == 1
    assert path.endswith("/" + hashlib.md5(data).hexdigest() + ".jpg")
    assert data == encoding.tensor_to_jpeg_q95_bytes(image)
    assert payload == "https://oss.example.com/" + path


@pytest.mark.parametrize("uploaded", [None, "", "file:///tmp/image.jpg", "https://bad host/image.jpg"])
def test_url_upload_invalid_result_fails(monkeypatch, uploaded: object) -> None:
    monkeypatch.setattr(encoding, "upload_bytes_to_oss", lambda *_: uploaded)
    with pytest.raises(RuntimeError, match="HTTP"):
        encoding.tensor_to_jpeg_q95_payload(source_tensor(), "url")


def test_url_upload_error_does_not_fall_back_to_base64(monkeypatch) -> None:
    def upload(*args: object) -> str:
        raise OSError("upload unavailable")

    monkeypatch.setattr(encoding, "upload_bytes_to_oss", upload)
    with pytest.raises(RuntimeError, match="JPEG Q95 上传 OSS 失败"):
        encoding.tensor_to_jpeg_q95_payload(source_tensor(), "url")


@pytest.mark.parametrize("background", ["Alpha", "Color"])
def test_q95_node_posts_inference_copy_but_composes_original(monkeypatch, background: str) -> None:
    image = source_tensor()
    original = image.clone()
    node = ZhiYiRMBGNodeV2()
    buffer = io.BytesIO()
    Image.new("L", (31, 19), 128).save(buffer, format="PNG")
    captured = []

    def upload(path: str, data: bytes) -> str:
        captured.append(data)
        return "https://oss.example.com/" + path

    def post(url: str, payload: dict, timeout: int, label: str) -> dict:
        assert url == module.FD_OVERSEAS_DWPOSE_URL + "/v1/rmbg"
        assert payload["image"].endswith(".jpg")
        assert payload["image_format"] == "png_base64"
        assert (payload["return_image"], payload["return_mask"], payload["return_mask_image"]) == (False, True, False)
        return {"mask": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                "image_size": {"width": 31, "height": 19},
                "inference_size": {"width": 31, "height": 19},
                "output_size": {"width": 31, "height": 19}}

    monkeypatch.setattr(encoding, "upload_bytes_to_oss", upload)
    monkeypatch.setattr(node, "_post_process", post)
    output, mask, _, info = call_node(node, image, image_delivery="url", background=background, background_color="#ffffff")
    assert captured[0] == encoding.tensor_to_jpeg_q95_bytes(image)
    assert torch.equal(image, original)
    if background == "Alpha":
        assert torch.equal(output[..., :3], image[..., :3])
        assert torch.equal(output[..., 3], mask)
    else:
        rgb = torch.from_numpy((image[..., :3].numpy() * 255).astype(np.uint8).astype(np.float32) / 255)
        assert torch.allclose(output, rgb * mask.unsqueeze(-1) + (1 - mask.unsqueeze(-1)), atol=1e-7, rtol=0)
    assert json.loads(info)["image_encoding"] == "jpeg_q95"


def test_unknown_encoding_fails_before_http() -> None:
    with pytest.raises(RuntimeError, match="image_encoding"):
        call_node(ZhiYiRMBGNodeV2(), source_tensor(), image_encoding="jpeg_q100")

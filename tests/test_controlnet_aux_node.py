import base64
import io
import json

import pytest
import requests
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes import zhiyi_controlnet_aux_node as aux_module
from src.Comfyui_Fd_Nodes.config import CUSTOM_SERVICE_URL_PRESET, _derive_dwpose_preprocess_url
from src.Comfyui_Fd_Nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from src.Comfyui_Fd_Nodes.zhiyi_controlnet_aux_node import (
    ZhiYiDepthAnythingV2PreprocessorNode,
    ZhiYiLineArtPreprocessorNode,
    _health_url,
    _normalize_preprocess_url,
)


def _image_data_url(size=(2, 2), color=(255, 0, 0)):
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class DummyResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data
        self.text = text

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def _preprocess_response(request_id="pre-1", size=(2, 2), preprocessor="LineArtPreprocessor"):
    model_name = "Realistic Lineart" if preprocessor == "LineArtPreprocessor" else "Depth Anything V2"
    model_type = "lineart_preprocessor" if preprocessor == "LineArtPreprocessor" else "depth_preprocessor"
    model_filename = "sk_model.pth" if preprocessor == "LineArtPreprocessor" else "depth_anything_v2_vitb.pth"
    return {
        "image_size": {"width": size[0], "height": size[1]},
        "inference_size": {"width": size[0], "height": size[1]},
        "output_size": {"width": size[0], "height": size[1]},
        "processed_image": _image_data_url(size=size, color=(0, 255, 0)),
        "request_id": request_id,
        "preprocessor": preprocessor,
        "model_name": model_name,
        "model_type": model_type,
        "model_filename": model_filename,
        "model_path": f"/models/{model_filename}",
        "device": "cpu",
        "inference_ms": 12.3,
        "total_ms": 45.6,
    }


def test_lineart_node_metadata():
    input_types = ZhiYiLineArtPreprocessorNode.INPUT_TYPES()

    assert input_types["required"]["image"][0] == "IMAGE"
    assert input_types["required"]["service_url"][1]["default"].endswith("/v1/lineart")
    assert input_types["required"]["coarse"][1]["default"] is False
    assert "health_check" in input_types["optional"]
    assert list(input_types["optional"])[-1] == "service_url_preset"
    preset_options, preset_config = input_types["optional"]["service_url_preset"]
    assert preset_options == [CUSTOM_SERVICE_URL_PRESET, "K8s 灰度", "10.1.0.230"]
    assert preset_config["default"] == CUSTOM_SERVICE_URL_PRESET
    assert ZhiYiLineArtPreprocessorNode.RETURN_TYPES == ("IMAGE", "JSON")
    assert ZhiYiLineArtPreprocessorNode.RETURN_NAMES == ("LINEART_IMAGE", "INFO")
    assert ZhiYiLineArtPreprocessorNode.CATEGORY == "知衣/ControlNet预处理"
    assert NODE_CLASS_MAPPINGS["ZhiYiLineArtPreprocessorNode"] is ZhiYiLineArtPreprocessorNode
    assert NODE_DISPLAY_NAME_MAPPINGS["ZhiYiLineArtPreprocessorNode"] == "知衣-LineArt线稿预处理"


def test_depth_node_metadata():
    input_types = ZhiYiDepthAnythingV2PreprocessorNode.INPUT_TYPES()

    assert input_types["required"]["image"][0] == "IMAGE"
    assert input_types["required"]["service_url"][1]["default"].endswith("/v1/depth-anything-v2")
    assert input_types["required"]["max_depth"][1]["default"] == 1.0
    assert "health_check" in input_types["optional"]
    assert list(input_types["optional"])[-1] == "service_url_preset"
    preset_options, preset_config = input_types["optional"]["service_url_preset"]
    assert preset_options == [CUSTOM_SERVICE_URL_PRESET, "K8s 灰度", "10.1.0.230"]
    assert preset_config["default"] == CUSTOM_SERVICE_URL_PRESET
    assert ZhiYiDepthAnythingV2PreprocessorNode.RETURN_TYPES == ("IMAGE", "JSON")
    assert ZhiYiDepthAnythingV2PreprocessorNode.RETURN_NAMES == ("DEPTH_IMAGE", "INFO")
    assert ZhiYiDepthAnythingV2PreprocessorNode.CATEGORY == "知衣/ControlNet预处理"
    assert NODE_CLASS_MAPPINGS["ZhiYiDepthAnythingV2PreprocessorNode"] is ZhiYiDepthAnythingV2PreprocessorNode
    assert NODE_DISPLAY_NAME_MAPPINGS["ZhiYiDepthAnythingV2PreprocessorNode"] == "知衣-DepthAnythingV2深度图预处理"


def test_normalize_preprocess_url_variants():
    assert (
        _normalize_preprocess_url("model-api-dwpose-svc.online-server-gray:8001", "http://unused/v1/lineart", "lineart")
        == "http://model-api-dwpose-svc.online-server-gray:8001/v1/lineart"
    )
    assert _normalize_preprocess_url("http://example.com:8001", "http://unused/v1/lineart", "lineart") == "http://example.com:8001/v1/lineart"
    assert _normalize_preprocess_url("http://example.com:8001/v1", "http://unused/v1/lineart", "lineart") == "http://example.com:8001/v1/lineart"
    assert _normalize_preprocess_url("https://example.com/api/v1", "http://unused/v1/lineart", "lineart") == "https://example.com/api/v1/lineart"
    assert _normalize_preprocess_url("https://example.com/v1/lineart/", "http://unused/v1/lineart", "lineart") == "https://example.com/v1/lineart"
    assert _normalize_preprocess_url("https://example.com/api", "http://unused/v1/depth-anything-v2", "depth-anything-v2") == (
        "https://example.com/api/v1/depth-anything-v2"
    )
    assert _normalize_preprocess_url("https://example.com/api/v1", "http://unused/v1/depth-anything-v2", "depth-anything-v2") == (
        "https://example.com/api/v1/depth-anything-v2"
    )
    assert _health_url("https://example.com/v1/lineart") == "https://example.com/health"
    assert _health_url("https://example.com/api/v1/depth-anything-v2") == "https://example.com/api/health"


def test_controlnet_aux_defaults_are_derived_from_dwpose_service_root(monkeypatch):
    monkeypatch.setattr("src.Comfyui_Fd_Nodes.config.FD_DWPOSE_POSE_URL", "dwpose.example.com:8001/v1/pose")
    assert _derive_dwpose_preprocess_url("lineart") == "http://dwpose.example.com:8001/v1/lineart"
    assert _derive_dwpose_preprocess_url("depth-anything-v2") == "http://dwpose.example.com:8001/v1/depth-anything-v2"

    monkeypatch.setattr("src.Comfyui_Fd_Nodes.config.FD_DWPOSE_POSE_URL", "https://dwpose.example.com/api/v1/pose")
    assert _derive_dwpose_preprocess_url("lineart") == "https://dwpose.example.com/api/v1/lineart"


def test_lineart_posts_payload_and_decodes_outputs(monkeypatch):
    node = ZhiYiLineArtPreprocessorNode()
    images = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url, timeout, kwargs))
        assert method == "POST"
        assert url == "http://example.com:8001/v1/lineart"
        assert kwargs["headers"] == {"Content-Type": "application/json"}
        payload = kwargs["json"]
        assert payload["image"].startswith("data:image/png;base64,")
        assert payload["resolution"] == 768
        assert payload["coarse"] is True
        assert payload["upscale_method"] == "INTER_AREA"
        assert payload["return_image"] is True
        assert payload["image_format"] == "png_base64"
        return DummyResponse(data=_preprocess_response(request_id="line-1", size=(2, 2)))

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    lineart_image, info_text = node.preprocess(
        image=images,
        service_url="example.com:8001",
        resolution=768,
        coarse=True,
        upscale_method="INTER_AREA",
        max_concurrency=1,
        timeout=30,
        health_check=False,
        node_switch=0,
    )

    info = json.loads(info_text)
    assert len(calls) == 1
    assert lineart_image.shape == (1, 2, 2, 3)
    assert torch.equal(lineart_image[0, 0, 0], torch.tensor([0.0, 1.0, 0.0]))
    assert info["service_url"] == "http://example.com:8001/v1/lineart"
    assert info["coarse"] is True
    assert info["items"][0]["request_id"] == "line-1"
    assert info["items"][0]["preprocessor"] == "LineArtPreprocessor"


def test_lineart_empty_service_url_uses_configured_default(monkeypatch):
    node = ZhiYiLineArtPreprocessorNode()
    monkeypatch.setattr(aux_module, "FD_CONTROLNET_AUX_LINEART_URL", "https://dwpose.example.com/v1/lineart")

    def fake_request(method, url, timeout, **kwargs):
        assert url == "https://dwpose.example.com/v1/lineart"
        return DummyResponse(data=_preprocess_response(request_id="line-default", size=(2, 2)))

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    _, info_text = node.preprocess(
        image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        service_url="",
        resolution=512,
        coarse=False,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
        health_check=False,
        node_switch=0,
    )

    assert json.loads(info_text)["service_url"] == "https://dwpose.example.com/v1/lineart"


def test_depth_posts_payload_and_decodes_outputs(monkeypatch):
    node = ZhiYiDepthAnythingV2PreprocessorNode()
    images = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url, timeout, kwargs))
        payload = kwargs["json"]
        assert method == "POST"
        assert url == "https://example.com/v1/depth-anything-v2"
        assert payload["max_depth"] == 2.5
        assert payload["resolution"] == 512
        assert payload["upscale_method"] == "INTER_CUBIC"
        return DummyResponse(data=_preprocess_response(request_id="depth-1", size=(2, 2), preprocessor="DepthAnythingV2Preprocessor"))

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    depth_image, info_text = node.preprocess(
        image=images,
        service_url="https://example.com/v1/depth-anything-v2",
        resolution=512,
        max_depth=2.5,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
        health_check=False,
        node_switch=0,
    )

    info = json.loads(info_text)
    assert len(calls) == 1
    assert depth_image.shape == (1, 2, 2, 3)
    assert info["max_depth"] == 2.5
    assert info["items"][0]["request_id"] == "depth-1"
    assert info["items"][0]["preprocessor"] == "DepthAnythingV2Preprocessor"


def test_controlnet_presets_generate_node_specific_endpoints(monkeypatch):
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append(url)
        preprocessor = "LineArtPreprocessor" if url.endswith("/lineart") else "DepthAnythingV2Preprocessor"
        return DummyResponse(data=_preprocess_response(preprocessor=preprocessor))

    monkeypatch.setattr(aux_module.requests, "request", fake_request)
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

    _, lineart_info = ZhiYiLineArtPreprocessorNode().preprocess(
        image=image,
        service_url="https://custom.example.com/v1/lineart",
        resolution=512,
        coarse=False,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
        service_url_preset="10.1.0.230",
    )
    _, depth_info = ZhiYiDepthAnythingV2PreprocessorNode().preprocess(
        image=image,
        service_url="https://custom.example.com/v1/depth-anything-v2",
        resolution=512,
        max_depth=1.0,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
        service_url_preset="K8s 灰度",
    )

    assert calls == [
        "http://10.1.0.230:8003/v1/lineart",
        "http://model-api-dwpose-svc.online-server-gray:8001/v1/depth-anything-v2",
    ]
    assert json.loads(lineart_info)["service_url"] == calls[0]
    assert json.loads(depth_info)["service_url"] == calls[1]


def test_batch_keeps_order_and_resizes_processed_image(monkeypatch):
    node = ZhiYiLineArtPreprocessorNode()
    images = torch.zeros((2, 2, 3, 3), dtype=torch.float32)
    call_count = 0

    def fake_request(method, url, timeout, **kwargs):
        nonlocal call_count
        call_count += 1
        return DummyResponse(data=_preprocess_response(request_id=f"line-{call_count}", size=(1, 1)))

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    lineart_image, info_text = node.preprocess(
        image=images,
        service_url="https://example.com/v1/lineart",
        resolution=512,
        coarse=False,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
        health_check=False,
        node_switch=0,
    )

    info = json.loads(info_text)
    assert call_count == 2
    assert lineart_image.shape == (2, 2, 3, 3)
    assert info["items"][0]["request_id"] == "line-1"
    assert info["items"][1]["request_id"] == "line-2"


def test_node_switch_returns_original_image_without_request(monkeypatch):
    node = ZhiYiDepthAnythingV2PreprocessorNode()
    image = torch.ones((1, 2, 2, 3), dtype=torch.float32)

    def fake_request(method, url, timeout, **kwargs):
        raise AssertionError("request should not be called")

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    depth_image, info_text = node.preprocess(
        image=image,
        service_url="https://example.com/v1/depth-anything-v2",
        resolution=512,
        max_depth=1.0,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
        health_check=True,
        node_switch=1,
    )

    assert torch.equal(depth_image, image)
    assert json.loads(info_text)["skipped"] is True


def test_health_check_uses_service_root(monkeypatch):
    node = ZhiYiDepthAnythingV2PreprocessorNode()
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url))
        if method == "GET":
            return DummyResponse(data={"status": "ok", "depth_anything_v2": {"loaded": True}})
        return DummyResponse(data=_preprocess_response(request_id="depth-health", preprocessor="DepthAnythingV2Preprocessor"))

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    node.preprocess(
        image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        service_url="https://example.com/v1/depth-anything-v2",
        resolution=512,
        max_depth=1.0,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
        health_check=True,
        node_switch=0,
    )

    assert calls[0] == ("GET", "https://example.com/health")
    assert calls[1] == ("POST", "https://example.com/v1/depth-anything-v2")


def test_health_check_rejects_unloaded(monkeypatch):
    node = ZhiYiLineArtPreprocessorNode()

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(data={"status": "ok", "lineart": {"loaded": False, "load_error": "not loaded"}})

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="服务未就绪"):
        node.preprocess(
            image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            service_url="https://example.com/v1/lineart",
            resolution=512,
            coarse=False,
            upscale_method="INTER_CUBIC",
            max_concurrency=1,
            timeout=30,
            health_check=True,
            node_switch=0,
        )


def test_health_check_requires_preprocessor_section(monkeypatch):
    node = ZhiYiLineArtPreprocessorNode()

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(data={"status": "ok"})

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="缺少 lineart 信息"):
        node.preprocess(
            image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            service_url="https://example.com/v1/lineart",
            resolution=512,
            coarse=False,
            upscale_method="INTER_CUBIC",
            max_concurrency=1,
            timeout=30,
            health_check=True,
            node_switch=0,
        )


def test_post_preprocess_uses_service_error_message():
    node = ZhiYiLineArtPreprocessorNode()

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(
            status_code=500,
            data={"error": {"code": "MODEL_UNAVAILABLE", "message": "not loaded", "request_id": "req-err"}},
        )

    original_request = aux_module.requests.request
    aux_module.requests.request = fake_request
    try:
        with pytest.raises(RuntimeError, match="MODEL_UNAVAILABLE: not loaded: request_id=req-err"):
            node._post_preprocess("https://example.com/v1/lineart", {"image": "x"}, timeout=30)
    finally:
        aux_module.requests.request = original_request


def test_post_preprocess_rejects_non_json_response(monkeypatch):
    node = ZhiYiLineArtPreprocessorNode()

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(data=ValueError("not json"), text="<html>bad</html>")

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="响应不是 JSON"):
        node._post_preprocess("https://example.com/v1/lineart", {"image": "x"}, timeout=30)


def test_post_preprocess_requires_processed_image(monkeypatch):
    node = ZhiYiLineArtPreprocessorNode()

    def fake_request(method, url, timeout, **kwargs):
        data = _preprocess_response()
        data["processed_image"] = None
        return DummyResponse(data=data)

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="缺少 processed_image"):
        node._post_preprocess("https://example.com/v1/lineart", {"image": "x"}, timeout=30)


def test_timeout_is_normalized(monkeypatch):
    node = ZhiYiDepthAnythingV2PreprocessorNode()

    def fake_request(method, url, timeout, **kwargs):
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr(aux_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="TIMEOUT"):
        node.preprocess(
            image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            service_url="https://example.com/v1/depth-anything-v2",
            resolution=512,
            max_depth=1.0,
            upscale_method="INTER_CUBIC",
            max_concurrency=1,
            timeout=30,
            health_check=False,
            node_switch=0,
        )

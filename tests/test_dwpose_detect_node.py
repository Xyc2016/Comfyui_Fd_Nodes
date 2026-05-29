import base64
import io
import json

import pytest
import requests
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes import zhiyi_dwpose_detect_node as dwpose_module
from src.Comfyui_Fd_Nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from src.Comfyui_Fd_Nodes.zhiyi_dwpose_detect_node import (
    ZhiYiDWPoseDetectNode,
    _health_url,
    _normalize_pose_url,
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


def _pose_response(request_id="pose-1", size=(2, 2), people_count=1):
    return {
        "image_size": {"width": size[0], "height": size[1]},
        "inference_size": {"width": size[0], "height": size[1]},
        "pose_image": _image_data_url(size=size, color=(0, 255, 0)),
        "openpose_json": {
            "people": [{"pose_keypoints_2d": [1, 2, 0.9]}],
            "canvas_width": size[0],
            "canvas_height": size[1],
        },
        "people_count": people_count,
        "request_id": request_id,
        "model_name": "DWPose",
        "detector_filename": "yolox_l.onnx",
        "pose_filename": "dw-ll_ucoco_384.onnx",
        "device": "cpu",
        "inference_ms": 12.3,
        "total_ms": 45.6,
    }


def test_dwpose_node_metadata():
    input_types = ZhiYiDWPoseDetectNode.INPUT_TYPES()

    assert input_types["required"]["image"][0] == "IMAGE"
    assert input_types["required"]["service_url"][1]["default"].endswith("/v1/pose")
    assert input_types["required"]["detect_face"][1]["default"] is False
    assert "health_check" in input_types["optional"]
    assert "node_switch" in input_types["optional"]
    assert ZhiYiDWPoseDetectNode.RETURN_TYPES == ("IMAGE", "POSE_KEYPOINT", "JSON")
    assert ZhiYiDWPoseDetectNode.RETURN_NAMES == ("POSE_IMAGE", "POSE_KEYPOINT", "INFO")
    assert ZhiYiDWPoseDetectNode.CATEGORY == "知衣/姿态检测"
    assert NODE_CLASS_MAPPINGS["ZhiYiDWPoseDetectNode"] is ZhiYiDWPoseDetectNode
    assert NODE_DISPLAY_NAME_MAPPINGS["ZhiYiDWPoseDetectNode"] == "知衣-DWPose姿态检测"


def test_normalize_pose_url_variants():
    assert (
        _normalize_pose_url("model-api-dwpose-svc.online-server-gray:8001")
        == "http://model-api-dwpose-svc.online-server-gray:8001/v1/pose"
    )
    assert _normalize_pose_url("http://example.com:8001") == "http://example.com:8001/v1/pose"
    assert _normalize_pose_url("https://example.com/v1/pose/") == "https://example.com/v1/pose"
    assert _normalize_pose_url("https://example.com/api") == "https://example.com/api/v1/pose"
    assert _health_url("https://example.com/v1/pose") == "https://example.com/health"
    assert _health_url("https://example.com/api/v1/pose") == "https://example.com/api/health"


def test_tensor_data_url_roundtrip():
    node = ZhiYiDWPoseDetectNode()
    image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    image[:, :, :, 0] = 1.0

    data_url = node._tensor_to_png_data_url(image)
    decoded = node._decode_png_data_url(data_url, "pose_image")
    tensor = node._pil_to_image_tensor(decoded)

    assert data_url.startswith("data:image/png;base64,")
    assert decoded.size == (2, 2)
    assert tensor.shape == (1, 2, 2, 3)
    assert torch.equal(tensor[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_detect_pose_posts_payload_and_decodes_outputs(monkeypatch):
    node = ZhiYiDWPoseDetectNode()
    images = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url, timeout, kwargs))
        assert method == "POST"
        assert url == "http://example.com:8001/v1/pose"
        assert kwargs["headers"] == {"Content-Type": "application/json"}
        payload = kwargs["json"]
        assert payload["image"].startswith("data:image/png;base64,")
        assert payload["resolution"] == 768
        assert payload["detect_body"] is True
        assert payload["detect_hand"] is False
        assert payload["detect_face"] is False
        assert payload["xinsr_stick_scaling"] is True
        assert payload["upscale_method"] == "INTER_AREA"
        assert payload["return_pose_image"] is True
        assert payload["return_openpose_json"] is True
        assert payload["image_format"] == "png_base64"
        return DummyResponse(data=_pose_response(request_id="pose-1", size=(2, 2)))

    monkeypatch.setattr(dwpose_module.requests, "request", fake_request)

    pose_image, pose_keypoint, info_text = node.detect_pose(
        image=images,
        service_url="example.com:8001",
        resolution=768,
        detect_body=True,
        detect_hand=False,
        detect_face=False,
        xinsr_stick_scaling=True,
        upscale_method="INTER_AREA",
        max_concurrency=1,
        timeout=30,
        health_check=False,
        node_switch=0,
    )

    info = json.loads(info_text)
    assert len(calls) == 1
    assert pose_image.shape == (1, 2, 2, 3)
    assert torch.equal(pose_image[0, 0, 0], torch.tensor([0.0, 1.0, 0.0]))
    assert pose_keypoint[0]["people"][0]["pose_keypoints_2d"] == [1, 2, 0.9]
    assert info["service_url"] == "http://example.com:8001/v1/pose"
    assert info["resolution"] == 768
    assert info["detect_hand"] is False
    assert info["items"][0]["request_id"] == "pose-1"
    assert info["items"][0]["people_count"] == 1


def test_detect_pose_batch_keeps_order_and_resizes_pose_image(monkeypatch):
    node = ZhiYiDWPoseDetectNode()
    images = torch.zeros((2, 2, 3, 3), dtype=torch.float32)
    call_count = 0

    def fake_request(method, url, timeout, **kwargs):
        nonlocal call_count
        call_count += 1
        return DummyResponse(data=_pose_response(request_id=f"pose-{call_count}", size=(1, 1), people_count=call_count))

    monkeypatch.setattr(dwpose_module.requests, "request", fake_request)

    pose_image, pose_keypoint, info_text = node.detect_pose(
        image=images,
        service_url="https://example.com/v1/pose",
        resolution=512,
        detect_body=True,
        detect_hand=True,
        detect_face=False,
        xinsr_stick_scaling=False,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
        health_check=False,
        node_switch=0,
    )

    info = json.loads(info_text)
    assert call_count == 2
    assert pose_image.shape == (2, 2, 3, 3)
    assert len(pose_keypoint) == 2
    assert info["items"][0]["request_id"] == "pose-1"
    assert info["items"][1]["request_id"] == "pose-2"


def test_node_switch_returns_original_image_without_request(monkeypatch):
    node = ZhiYiDWPoseDetectNode()
    image = torch.ones((1, 2, 2, 3), dtype=torch.float32)

    def fake_request(method, url, timeout, **kwargs):
        raise AssertionError("request should not be called")

    monkeypatch.setattr(dwpose_module.requests, "request", fake_request)

    pose_image, pose_keypoint, info_text = node.detect_pose(
        image=image,
        service_url="https://example.com/v1/pose",
        resolution=512,
        detect_body=True,
        detect_hand=True,
        detect_face=False,
        xinsr_stick_scaling=False,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
        health_check=True,
        node_switch=1,
    )

    assert torch.equal(pose_image, image)
    assert pose_keypoint == []
    assert json.loads(info_text)["skipped"] is True


def test_health_check_uses_service_root(monkeypatch):
    node = ZhiYiDWPoseDetectNode()
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url))
        if method == "GET":
            return DummyResponse(data={"status": "ok", "loaded": True})
        return DummyResponse(data=_pose_response(request_id="pose-health"))

    monkeypatch.setattr(dwpose_module.requests, "request", fake_request)

    node.detect_pose(
        image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        service_url="https://example.com/v1/pose",
        resolution=512,
        detect_body=True,
        detect_hand=True,
        detect_face=False,
        xinsr_stick_scaling=False,
        upscale_method="INTER_CUBIC",
        max_concurrency=1,
        timeout=30,
        health_check=True,
        node_switch=0,
    )

    assert calls[0] == ("GET", "https://example.com/health")
    assert calls[1] == ("POST", "https://example.com/v1/pose")


def test_health_check_rejects_unloaded(monkeypatch):
    node = ZhiYiDWPoseDetectNode()

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(data={"status": "loading_failed", "loaded": False})

    monkeypatch.setattr(dwpose_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="服务未就绪"):
        node.detect_pose(
            image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            service_url="https://example.com/v1/pose",
            resolution=512,
            detect_body=True,
            detect_hand=True,
            detect_face=False,
            xinsr_stick_scaling=False,
            upscale_method="INTER_CUBIC",
            max_concurrency=1,
            timeout=30,
            health_check=True,
            node_switch=0,
        )


def test_post_pose_uses_service_error_message():
    node = ZhiYiDWPoseDetectNode()

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(
            status_code=500,
            data={"error": {"code": "MODEL_UNAVAILABLE", "message": "not loaded", "request_id": "req-err"}},
        )

    original_request = dwpose_module.requests.request
    dwpose_module.requests.request = fake_request
    try:
        with pytest.raises(RuntimeError, match="MODEL_UNAVAILABLE: not loaded: request_id=req-err"):
            node._post_pose("https://example.com/v1/pose", {"image": "x"}, timeout=30)
    finally:
        dwpose_module.requests.request = original_request


def test_post_pose_rejects_non_json_response(monkeypatch):
    node = ZhiYiDWPoseDetectNode()

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(data=ValueError("not json"), text="<html>bad</html>")

    monkeypatch.setattr(dwpose_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="响应不是 JSON"):
        node._post_pose("https://example.com/v1/pose", {"image": "x"}, timeout=30)


def test_post_pose_requires_pose_image(monkeypatch):
    node = ZhiYiDWPoseDetectNode()

    def fake_request(method, url, timeout, **kwargs):
        data = _pose_response()
        data["pose_image"] = None
        return DummyResponse(data=data)

    monkeypatch.setattr(dwpose_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="缺少 pose_image"):
        node._post_pose("https://example.com/v1/pose", {"image": "x"}, timeout=30)


def test_timeout_is_normalized(monkeypatch):
    node = ZhiYiDWPoseDetectNode()

    def fake_request(method, url, timeout, **kwargs):
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr(dwpose_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="TIMEOUT"):
        node.detect_pose(
            image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            service_url="https://example.com/v1/pose",
            resolution=512,
            detect_body=True,
            detect_hand=True,
            detect_face=False,
            xinsr_stick_scaling=False,
            upscale_method="INTER_CUBIC",
            max_concurrency=1,
            timeout=30,
            health_check=False,
            node_switch=0,
        )

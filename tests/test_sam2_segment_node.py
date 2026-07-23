import base64
import io
import json

import pytest
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes import sam2_segment_node as sam2_module
from src.Comfyui_Fd_Nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from src.Comfyui_Fd_Nodes.sam2_segment_node import ZhiYiSAM2SegmentNode


def _mask_data_url(size=(2, 2), white_pixels=None):
    image = Image.new("L", size, 0)
    for xy in white_pixels or []:
        image.putpixel(xy, 255)
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


def test_sam2_node_metadata():
    input_types = ZhiYiSAM2SegmentNode.INPUT_TYPES()

    assert input_types["required"]["images"][0] == "IMAGE"
    assert input_types["required"]["bboxes"][0] == "BBOXES"
    assert input_types["required"]["service_url"][1]["default"].endswith("/v1/segment")
    assert ZhiYiSAM2SegmentNode.RETURN_TYPES == ("IMAGE", "MASK", "IMAGE", "JSON")
    assert ZhiYiSAM2SegmentNode.RETURN_NAMES == ("IMAGE", "MASK", "MASK_IMAGE", "INFO")
    assert ZhiYiSAM2SegmentNode.CATEGORY == "知衣/抠图"
    assert NODE_CLASS_MAPPINGS["ZhiYiSAM2SegmentNode"] is ZhiYiSAM2SegmentNode
    assert NODE_DISPLAY_NAME_MAPPINGS["ZhiYiSAM2SegmentNode"] == "知衣-SAM2抠图"


def test_normalize_bboxes_batch_single_group_reused_for_batch():
    node = ZhiYiSAM2SegmentNode()

    result = node._normalize_bboxes_batch([[[0, 0, 2, 2]]], image_count=2)

    assert result == [[[0.0, 0.0, 2.0, 2.0]], [[0.0, 0.0, 2.0, 2.0]]]
    assert result[0] is not result[1]
    assert result[0][0] is not result[1][0]


def test_normalize_bboxes_batch_per_image_groups():
    node = ZhiYiSAM2SegmentNode()

    result = node._normalize_bboxes_batch([[[0, 0, 2, 2]], [[1, 1, 3, 3]]], image_count=2)

    assert result == [[[0.0, 0.0, 2.0, 2.0]], [[1.0, 1.0, 3.0, 3.0]]]


def test_normalize_bboxes_batch_rejects_mismatch_and_reuses_empty_group():
    node = ZhiYiSAM2SegmentNode()

    with pytest.raises(RuntimeError, match="必须为 1 或等于图片数量"):
        node._normalize_bboxes_batch([[[0, 0, 2, 2]], [[1, 1, 3, 3]], [[2, 2, 4, 4]]], image_count=2)

    assert node._normalize_bboxes_batch([], image_count=1) == [[]]
    assert node._normalize_bboxes_batch([[]], image_count=2) == [[], []]


def test_normalize_bboxes_batch_preserves_mixed_groups():
    node = ZhiYiSAM2SegmentNode()

    result = node._normalize_bboxes_batch([[], [[1, 1, 3, 3]]], image_count=2)

    assert result == [[], [[1.0, 1.0, 3.0, 3.0]]]


def test_normalize_bboxes_batch_rejects_invalid_box():
    node = ZhiYiSAM2SegmentNode()

    with pytest.raises(RuntimeError, match="x2>x1"):
        node._normalize_bboxes_batch([[[2, 0, 1, 2]]], image_count=1)


def test_segment_posts_payload_and_decodes_outputs(monkeypatch):
    node = ZhiYiSAM2SegmentNode()
    images = torch.ones((1, 2, 2, 3), dtype=torch.float32)
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url, timeout, kwargs))
        assert method == "POST"
        payload = kwargs["json"]
        assert payload["image"].startswith("data:image/png;base64,")
        assert payload["bboxes"] == [[0.0, 0.0, 2.0, 2.0]]
        assert payload["return_merged_mask"] is True
        assert payload["return_masked_image"] is False
        assert payload["dilate"] == 2
        assert payload["blur"] == 1
        return DummyResponse(data={
            "image_size": {"width": 2, "height": 2},
            "mask": _mask_data_url(white_pixels=[(0, 0), (1, 1)]),
            "masks": [{"bbox": [0, 0, 2, 2], "score": 0.98, "mask": _mask_data_url()}],
            "masked_image": None,
            "count": 1,
            "request_id": "req-1",
        })

    monkeypatch.setattr(sam2_module.requests, "request", fake_request)

    result_image, mask, mask_image, info_text = node.segment(
        images=images,
        bboxes=[[[0, 0, 2, 2]]],
        service_url="https://example.com/v1/segment",
        dilate=2,
        blur=1,
        background="Color",
        background_color="#000000",
        max_concurrency=1,
        timeout=30,
        invert_output=False,
        health_check=False,
    )

    info = json.loads(info_text)
    assert len(calls) == 1
    assert result_image.shape == (1, 2, 2, 3)
    assert mask.shape == (1, 2, 2)
    assert mask_image.shape == (1, 2, 2, 3)
    assert torch.equal(mask, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    assert torch.equal(result_image[0, 0, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(result_image[0, 0, 1], torch.tensor([0.0, 0.0, 0.0]))
    assert info["items"][0]["request_id"] == "req-1"
    assert info["items"][0]["count"] == 1
    assert info["items"][0]["scores"] == [0.98]


def test_segment_empty_bbox_skips_api_and_returns_fallback(monkeypatch, caplog):
    node = ZhiYiSAM2SegmentNode()
    images = torch.tensor(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 0.0, 0.5]]]],
        dtype=torch.float32,
    )
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url))
        raise AssertionError("空 bbox 不应访问 SAM2 服务")

    monkeypatch.setattr(sam2_module.requests, "request", fake_request)

    with caplog.at_level("WARNING", logger=sam2_module.__name__):
        result_image, mask, mask_image, info_text = node.segment(
            images=images,
            bboxes=[[]],
            service_url="https://example.com/v1/segment",
            dilate=0,
            blur=0,
            background="Color",
            background_color="#ff0000",
            max_concurrency=1,
            timeout=30,
            invert_output=True,
            health_check=False,
        )

    info = json.loads(info_text)
    assert calls == []
    assert torch.equal(result_image, images)
    assert torch.equal(mask, torch.ones((1, 2, 2)))
    assert torch.equal(mask_image, torch.ones((1, 2, 2, 3)))
    assert "第 1 张图片无 bbox，跳过 SAM2 API，原图直出且返回全白 mask" in caplog.text
    assert info["items"] == [{"index": 0, "bboxes": [], "skipped": True, "reason": "no_bbox"}]


def test_segment_mixed_batch_only_posts_non_empty_bbox(monkeypatch):
    node = ZhiYiSAM2SegmentNode()
    images = torch.stack((
        torch.full((2, 2, 3), 0.25),
        torch.full((2, 2, 3), 0.75),
    ))
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, kwargs["json"]["bboxes"]))
        return DummyResponse(data={
            "image_size": {"width": 2, "height": 2},
            "mask": _mask_data_url(white_pixels=[(0, 0)]),
            "masks": [{"score": 0.9}],
            "masked_image": None,
            "count": 1,
            "request_id": "req-mixed",
        })

    monkeypatch.setattr(sam2_module.requests, "request", fake_request)

    result_image, mask, mask_image, info_text = node.segment(
        images=images,
        bboxes=[[], [[0, 0, 2, 2]]],
        service_url="https://example.com/v1/segment",
        dilate=0,
        blur=0,
        background="Original",
        background_color="#000000",
        max_concurrency=2,
        timeout=30,
        invert_output=False,
        health_check=False,
    )

    info = json.loads(info_text)
    assert calls == [("POST", [[0.0, 0.0, 2.0, 2.0]])]
    assert torch.equal(result_image[0:1], images[0:1])
    assert torch.allclose(result_image[1:2], torch.full((1, 2, 2, 3), 0.7490196))
    assert torch.equal(mask[0], torch.ones((2, 2)))
    assert torch.equal(mask[1], torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    assert torch.equal(mask_image[0], torch.ones((2, 2, 3)))
    assert info["items"][0]["index"] == 0
    assert info["items"][0]["skipped"] is True
    assert info["items"][0]["reason"] == "no_bbox"
    assert info["items"][1]["index"] == 1
    assert info["items"][1]["request_id"] == "req-mixed"


def test_segment_all_empty_batch_skips_health_and_api(monkeypatch):
    node = ZhiYiSAM2SegmentNode()
    images = torch.stack((
        torch.zeros((2, 3, 3)),
        torch.full((2, 3, 3), 0.5),
    ))
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url))
        raise AssertionError("全空 bbox 批次不应访问 SAM2 服务")

    monkeypatch.setattr(sam2_module.requests, "request", fake_request)

    result_image, mask, mask_image, info_text = node.segment(
        images=images,
        bboxes=[[]],
        service_url="https://example.com/v1/segment",
        dilate=0,
        blur=0,
        background="Alpha",
        background_color="#000000",
        max_concurrency=2,
        timeout=30,
        invert_output=True,
        health_check=True,
    )

    info = json.loads(info_text)
    assert calls == []
    assert torch.equal(result_image, images)
    assert torch.equal(mask, torch.ones((2, 2, 3)))
    assert torch.equal(mask_image, torch.ones((2, 2, 3, 3)))
    assert [item["index"] for item in info["items"]] == [0, 1]
    assert all(item["skipped"] is True and item["reason"] == "no_bbox" for item in info["items"])


def test_segment_inverts_mask(monkeypatch):
    node = ZhiYiSAM2SegmentNode()
    images = torch.ones((1, 2, 2, 3), dtype=torch.float32)

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(data={
            "image_size": {"width": 2, "height": 2},
            "mask": _mask_data_url(white_pixels=[(0, 0)]),
            "masks": [],
            "masked_image": None,
            "count": 1,
            "request_id": "req-2",
        })

    monkeypatch.setattr(sam2_module.requests, "request", fake_request)

    _, mask, _, _ = node.segment(
        images=images,
        bboxes=[[[0, 0, 2, 2]]],
        service_url="https://example.com/v1/segment",
        dilate=0,
        blur=0,
        background="Original",
        background_color="#000000",
        max_concurrency=1,
        timeout=30,
        invert_output=True,
        health_check=False,
    )

    assert torch.equal(mask, torch.tensor([[[0.0, 1.0], [1.0, 1.0]]]))


def test_health_check_uses_service_root(monkeypatch):
    node = ZhiYiSAM2SegmentNode()
    images = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url))
        if method == "GET":
            return DummyResponse(data={"status": "ok", "loaded": True})
        return DummyResponse(data={
            "image_size": {"width": 2, "height": 2},
            "mask": _mask_data_url(white_pixels=[(0, 0)]),
            "masks": [],
            "masked_image": None,
            "count": 1,
            "request_id": "req-3",
        })

    monkeypatch.setattr(sam2_module.requests, "request", fake_request)

    node.segment(
        images=images,
        bboxes=[[[0, 0, 2, 2]]],
        service_url="https://example.com/v1/segment",
        dilate=0,
        blur=0,
        background="Original",
        background_color="#000000",
        max_concurrency=1,
        timeout=30,
        invert_output=False,
        health_check=True,
    )

    assert calls[0] == ("GET", "https://example.com/health")
    assert calls[1] == ("POST", "https://example.com/v1/segment")


def test_health_check_rejects_unloaded(monkeypatch):
    node = ZhiYiSAM2SegmentNode()

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(data={"status": "loading_failed", "loaded": False})

    monkeypatch.setattr(sam2_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="服务未就绪"):
        node.segment(
            images=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            bboxes=[[[0, 0, 2, 2]]],
            service_url="https://example.com/v1/segment",
            dilate=0,
            blur=0,
            background="Original",
            background_color="#000000",
            max_concurrency=1,
            timeout=30,
            invert_output=False,
            health_check=True,
        )


def test_post_segment_uses_service_error_message():
    node = ZhiYiSAM2SegmentNode()

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(
            status_code=400,
            data={"error": {"code": "INVALID_BBOX", "message": "bad box", "request_id": "req-err"}},
        )

    original_request = sam2_module.requests.request
    sam2_module.requests.request = fake_request
    try:
        with pytest.raises(RuntimeError, match="INVALID_BBOX: bad box: request_id=req-err"):
            node._post_segment("https://example.com/v1/segment", {"image": "x"}, timeout=30)
    finally:
        sam2_module.requests.request = original_request


def test_post_segment_rejects_non_json_response(monkeypatch):
    node = ZhiYiSAM2SegmentNode()

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(data=ValueError("not json"), text="<html>bad</html>")

    monkeypatch.setattr(sam2_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="响应不是 JSON"):
        node._post_segment("https://example.com/v1/segment", {"image": "x"}, timeout=30)

import base64
import io
import json

import pytest
import requests
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes import zhiyi_rmbg_segment_node as rmbg_module
from src.Comfyui_Fd_Nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from src.Comfyui_Fd_Nodes.zhiyi_rmbg_segment_node import (
    BODY_CLASSES,
    CLOTHES_CLASSES,
    FASHION_CLASSES,
    ZhiYiBodySegmentNode,
    ZhiYiClothesSegmentNode,
    ZhiYiFashionSegmentNode,
    ZhiYiRMBGNode,
    _health_url,
    _normalize_rmbg_segment_url,
)


def _image_data_url(size=(2, 2), color=(0, 255, 0), mode="RGB"):
    image = Image.new(mode, size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


_MISSING = object()


def _segment_response(
    *,
    request_id="seg-1",
    size=(2, 2),
    processed_image=_MISSING,
    mask=None,
    mask_image=None,
    classes=None,
    preprocessor="ClothesSegmentPreprocessor",
):
    return {
        "image_size": {"width": size[0], "height": size[1]},
        "inference_size": {"width": size[0], "height": size[1]},
        "output_size": {"width": size[0], "height": size[1]},
        "processed_image": _image_data_url(size=size, color=(0, 255, 0)) if processed_image is _MISSING else processed_image,
        "mask": mask if mask is not None else _mask_data_url(size=size, white_pixels=[(0, 0), (1, 1)]),
        "mask_image": mask_image,
        "classes": classes or [],
        "request_id": request_id,
        "preprocessor": preprocessor,
        "model_name": "SegFormer Clothes",
        "model_type": "semantic_segmentation",
        "model_repo": "1038lab/segformer_clothes",
        "model_filename": "model.safetensors",
        "model_path": "/models/dwpose/1038lab/segformer_clothes/model.safetensors",
        "device": "cpu",
        "inference_ms": 12.3,
        "total_ms": 45.6,
    }


def test_rmbg_and_segment_node_metadata():
    rmbg_inputs = ZhiYiRMBGNode.INPUT_TYPES()
    clothes_inputs = ZhiYiClothesSegmentNode.INPUT_TYPES()
    fashion_inputs = ZhiYiFashionSegmentNode.INPUT_TYPES()
    body_inputs = ZhiYiBodySegmentNode.INPUT_TYPES()

    assert rmbg_inputs["required"]["service_url"][1]["default"].endswith("/v1/rmbg")
    assert rmbg_inputs["required"]["process_res"][1]["default"] == 1024
    assert rmbg_inputs["required"]["sensitivity"][1]["default"] == 1.0
    assert rmbg_inputs["required"]["refine_foreground"][1]["default"] is False
    assert rmbg_inputs["required"]["return_mask_image"][1]["default"] is False
    assert clothes_inputs["required"]["service_url"][1]["default"].endswith("/v1/segment/clothes")
    assert fashion_inputs["required"]["service_url"][1]["default"].endswith("/v1/segment/fashion")
    assert body_inputs["required"]["service_url"][1]["default"].endswith("/v1/segment/body")
    assert "忽略 process_res" in body_inputs["required"]["process_res"][1]["tooltip"]

    assert len(CLOTHES_CLASSES) == 18
    assert len(FASHION_CLASSES) == 47
    assert len(BODY_CLASSES) == 12
    assert "shirt, blouse" in FASHION_CLASSES
    assert "top, t-shirt, sweatshirt" in FASHION_CLASSES
    assert "Torso-skin" in BODY_CLASSES

    for node_cls in (ZhiYiRMBGNode, ZhiYiClothesSegmentNode, ZhiYiFashionSegmentNode, ZhiYiBodySegmentNode):
        assert node_cls.RETURN_TYPES == ("IMAGE", "MASK", "IMAGE", "JSON")
        assert node_cls.RETURN_NAMES == ("IMAGE", "MASK", "MASK_IMAGE", "INFO")

    assert ZhiYiRMBGNode.CATEGORY == "知衣/抠图"
    assert ZhiYiClothesSegmentNode.CATEGORY == "知衣/语义分割"
    assert NODE_CLASS_MAPPINGS["ZhiYiRMBGNode"] is ZhiYiRMBGNode
    assert NODE_CLASS_MAPPINGS["ZhiYiClothesSegmentNode"] is ZhiYiClothesSegmentNode
    assert NODE_CLASS_MAPPINGS["ZhiYiFashionSegmentNode"] is ZhiYiFashionSegmentNode
    assert NODE_CLASS_MAPPINGS["ZhiYiBodySegmentNode"] is ZhiYiBodySegmentNode
    assert NODE_DISPLAY_NAME_MAPPINGS["ZhiYiRMBGNode"] == "知衣-RMBG2.0背景去除"
    assert NODE_DISPLAY_NAME_MAPPINGS["ZhiYiClothesSegmentNode"] == "知衣-衣物语义分割"
    assert NODE_DISPLAY_NAME_MAPPINGS["ZhiYiFashionSegmentNode"] == "知衣-时尚单品分割"
    assert NODE_DISPLAY_NAME_MAPPINGS["ZhiYiBodySegmentNode"] == "知衣-身体部位分割"


def test_normalize_rmbg_segment_url_variants():
    assert _normalize_rmbg_segment_url("10.1.0.230:8003", "http://unused/v1/rmbg", "rmbg") == "http://10.1.0.230:8003/v1/rmbg"
    assert _normalize_rmbg_segment_url("https://example.com/api", "http://unused/v1/rmbg", "rmbg") == "https://example.com/api/v1/rmbg"
    assert (
        _normalize_rmbg_segment_url("https://example.com/api/v1", "http://unused/v1/segment/clothes", "segment/clothes")
        == "https://example.com/api/v1/segment/clothes"
    )
    assert (
        _normalize_rmbg_segment_url(
            "https://example.com/api/v1/segment/fashion/",
            "http://unused/v1/segment/fashion",
            "segment/fashion",
        )
        == "https://example.com/api/v1/segment/fashion"
    )
    assert _health_url("https://example.com/v1/rmbg") == "https://example.com/health"
    assert _health_url("https://example.com/api/v1/segment/body") == "https://example.com/api/health"


def test_parse_classes_preserves_fashion_comma_names():
    clothes = ZhiYiClothesSegmentNode()
    fashion = ZhiYiFashionSegmentNode()
    body = ZhiYiBodySegmentNode()

    assert clothes._parse_classes_text("Upper-clothes, Pants", CLOTHES_CLASSES, allow_comma_split=True) == ["Upper-clothes", "Pants"]
    assert fashion._parse_classes_text("shirt, blouse\npants", FASHION_CLASSES, allow_comma_split=False) == ["shirt, blouse", "pants"]
    assert fashion._parse_classes_text('["top, t-shirt, sweatshirt", "bag, wallet"]', FASHION_CLASSES) == [
        "top, t-shirt, sweatshirt",
        "bag, wallet",
    ]
    assert body._parse_classes_text("", BODY_CLASSES, allow_comma_split=True) == []

    with pytest.raises(RuntimeError, match="不支持的分割类别"):
        fashion._parse_classes_text("shirt, blouse, pants", FASHION_CLASSES, allow_comma_split=False)


def test_rmbg_posts_full_payload_and_decodes_outputs(monkeypatch):
    node = ZhiYiRMBGNode()
    images = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url, timeout, kwargs))
        assert method == "POST"
        assert url == "http://example.com:8003/v1/rmbg"
        assert kwargs["headers"] == {"Content-Type": "application/json"}
        payload = kwargs["json"]
        assert payload["image"].startswith("data:image/png;base64,")
        assert payload["process_res"] == 1536
        assert payload["sensitivity"] == 0.85
        assert payload["mask_blur"] == 4
        assert payload["mask_offset"] == -2
        assert payload["invert_output"] is True
        assert payload["refine_foreground"] is True
        assert payload["background"] == "Color"
        assert payload["background_color"] == "#ffffff"
        assert payload["return_image"] is True
        assert payload["return_mask"] is True
        assert payload["return_mask_image"] is True
        assert payload["image_format"] == "png_base64"
        data = _segment_response(
            request_id="rmbg-1",
            classes=[],
            preprocessor="RMBGPreprocessor",
            processed_image=_image_data_url(color=(0, 255, 0)),
            mask_image=_image_data_url(color=(255, 0, 0)),
        )
        data["model_name"] = "RMBG-2.0"
        data["model_type"] = "background_removal"
        data["model_repo"] = "1038lab/RMBG-2.0"
        return DummyResponse(data=data)

    monkeypatch.setattr(rmbg_module.requests, "request", fake_request)

    result_image, mask, mask_image, info_text = node.remove_background(
        image=images,
        service_url="example.com:8003",
        process_res=1536,
        sensitivity=0.85,
        mask_blur=4,
        mask_offset=-2,
        invert_output=True,
        refine_foreground=True,
        background="Color",
        background_color="#ffffff",
        return_image=True,
        return_mask=True,
        return_mask_image=True,
        max_concurrency=1,
        timeout=30,
        health_check=False,
        node_switch=0,
    )

    info = json.loads(info_text)
    assert len(calls) == 1
    assert result_image.shape == (1, 2, 2, 3)
    assert mask.shape == (1, 2, 2)
    assert mask_image.shape == (1, 2, 2, 3)
    assert torch.equal(result_image[0, 0, 0], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(mask, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    assert torch.equal(mask_image[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert info["service_url"] == "http://example.com:8003/v1/rmbg"
    assert info["sensitivity"] == 0.85
    assert info["refine_foreground"] is True
    assert info["items"][0]["request_id"] == "rmbg-1"
    assert info["items"][0]["preprocessor"] == "RMBGPreprocessor"


def test_clothes_segment_posts_classes_and_composes_from_mask_when_image_missing(monkeypatch):
    node = ZhiYiClothesSegmentNode()
    images = torch.ones((1, 2, 2, 3), dtype=torch.float32)

    def fake_request(method, url, timeout, **kwargs):
        assert method == "POST"
        assert url == "https://example.com/v1/segment/clothes"
        payload = kwargs["json"]
        assert payload["classes"] == ["Upper-clothes", "Pants"]
        assert payload["process_res"] == 512
        assert payload["return_image"] is False
        assert payload["return_mask"] is True
        return DummyResponse(data=_segment_response(
            request_id="cloth-1",
            processed_image=None,
            mask=_mask_data_url(white_pixels=[(0, 0), (1, 1)]),
            classes=["Upper-clothes", "Pants"],
        ))

    monkeypatch.setattr(rmbg_module.requests, "request", fake_request)

    result_image, mask, _, info_text = node.segment(
        image=images,
        service_url="https://example.com/v1/segment/clothes",
        classes="Upper-clothes\nPants",
        process_res=512,
        mask_blur=0,
        mask_offset=0,
        invert_output=False,
        background="Color",
        background_color="#000000",
        return_image=False,
        return_mask=True,
        return_mask_image=False,
        max_concurrency=1,
        timeout=30,
        health_check=False,
        node_switch=0,
    )

    info = json.loads(info_text)
    assert torch.equal(mask, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    assert torch.equal(result_image[0, 0, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(result_image[0, 0, 1], torch.tensor([0.0, 0.0, 0.0]))
    assert info["classes"] == ["Upper-clothes", "Pants"]
    assert info["items"][0]["classes"] == ["Upper-clothes", "Pants"]


def test_body_health_check_and_process_res_ignored_notice(monkeypatch):
    node = ZhiYiBodySegmentNode()
    calls = []

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url))
        if method == "GET":
            return DummyResponse(data={"status": "ok", "body_segment": {"loaded": True}})
        payload = kwargs["json"]
        assert payload["classes"] == ["Face", "Torso-skin"]
        assert payload["process_res"] == 2048
        return DummyResponse(data=_segment_response(
            request_id="body-1",
            classes=["Face", "Torso-skin"],
            preprocessor="BodySegmentPreprocessor",
        ))

    monkeypatch.setattr(rmbg_module.requests, "request", fake_request)

    _, _, _, info_text = node.segment(
        image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        service_url="https://example.com/api/v1/segment/body",
        classes='["Face", "Torso-skin"]',
        process_res=2048,
        mask_blur=0,
        mask_offset=0,
        invert_output=False,
        background="Alpha",
        background_color="#222222",
        return_image=True,
        return_mask=True,
        return_mask_image=False,
        max_concurrency=1,
        timeout=30,
        health_check=True,
        node_switch=0,
    )

    info = json.loads(info_text)
    assert calls[0] == ("GET", "https://example.com/api/health")
    assert calls[1] == ("POST", "https://example.com/api/v1/segment/body")
    assert info["process_res_ignored"] is True
    assert info["items"][0]["preprocessor"] == "BodySegmentPreprocessor"


def test_node_switch_returns_original_and_empty_mask_without_request(monkeypatch):
    node = ZhiYiFashionSegmentNode()
    image = torch.ones((1, 2, 2, 3), dtype=torch.float32)

    def fake_request(method, url, timeout, **kwargs):
        raise AssertionError("request should not be called")

    monkeypatch.setattr(rmbg_module.requests, "request", fake_request)

    result_image, mask, mask_image, info_text = node.segment(
        image=image,
        service_url="https://example.com/v1/segment/fashion",
        classes="shirt, blouse",
        process_res=512,
        mask_blur=0,
        mask_offset=0,
        invert_output=False,
        background="Alpha",
        background_color="#222222",
        return_image=True,
        return_mask=True,
        return_mask_image=False,
        max_concurrency=1,
        timeout=30,
        health_check=True,
        node_switch=1,
    )

    assert torch.equal(result_image, image)
    assert torch.equal(mask, torch.zeros((1, 2, 2), dtype=torch.float32))
    assert torch.equal(mask_image, torch.zeros((1, 2, 2, 3), dtype=torch.float32))
    assert json.loads(info_text)["skipped"] is True


def test_post_process_uses_service_error_message():
    node = ZhiYiClothesSegmentNode()

    def fake_request(method, url, timeout, **kwargs):
        return DummyResponse(
            status_code=422,
            data={"error": {"code": "INVALID_CLASSES", "message": "bad class", "request_id": "req-err"}},
        )

    original_request = rmbg_module.requests.request
    rmbg_module.requests.request = fake_request
    try:
        with pytest.raises(RuntimeError, match="INVALID_CLASSES: bad class: request_id=req-err"):
            node._post_process("https://example.com/v1/segment/clothes", {"image": "x"}, timeout=30, label="知衣衣物语义分割")
    finally:
        rmbg_module.requests.request = original_request


def test_timeout_is_normalized(monkeypatch):
    node = ZhiYiRMBGNode()

    def fake_request(method, url, timeout, **kwargs):
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr(rmbg_module.requests, "request", fake_request)

    with pytest.raises(RuntimeError, match="TIMEOUT"):
        node.remove_background(
            image=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
            service_url="https://example.com/v1/rmbg",
            process_res=1024,
            sensitivity=1.0,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            refine_foreground=False,
            background="Alpha",
            background_color="#222222",
            return_image=True,
            return_mask=True,
            return_mask_image=False,
            max_concurrency=1,
            timeout=30,
            health_check=False,
            node_switch=0,
        )

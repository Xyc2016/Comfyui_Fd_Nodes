"""Tests for the four-grid ComfyUI node."""

import json

import pytest
import requests

import numpy as np
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes import four_grid_crop_detector
from src.Comfyui_Fd_Nodes import zhiyi_four_grid_crop_node as node_module
from src.Comfyui_Fd_Nodes.four_grid_crop_core import PersonBox
from src.Comfyui_Fd_Nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from src.Comfyui_Fd_Nodes.zhiyi_four_grid_crop_node import ZhiYiFourGridCropNode


def _image_tensor(color, size=(100, 100)):
    image = Image.new("RGB", size, color)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _recognition_payload(width=100, height=100):
    boxes = [
        (0.05, 0.05, 0.45, 0.45, 0.9),
        (0.55, 0.05, 0.95, 0.45, 0.9),
        (0.05, 0.55, 0.45, 0.95, 0.9),
        (0.55, 0.55, 0.95, 0.95, 0.9),
    ]
    return {
        "status": True,
        "result": [
            {
                "status": True,
                "img_w": width,
                "img_h": height,
                "body_positions": [
                    {
                        "person": {
                            "label": "person",
                            "xmin": x1,
                            "ymin": y1,
                            "xmax": x2,
                            "ymax": y2,
                            "score": score,
                        }
                    }
                    for x1, y1, x2, y2, score in boxes
                ],
            }
        ],
    }


def test_node_metadata_and_registration():
    assert NODE_CLASS_MAPPINGS["ZhiYiFourGridCropNode"] is ZhiYiFourGridCropNode
    assert NODE_DISPLAY_NAME_MAPPINGS["ZhiYiFourGridCropNode"] == "知衣-四宫格识别拆分"
    assert ZhiYiFourGridCropNode.RETURN_TYPES == ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    assert ZhiYiFourGridCropNode.RETURN_NAMES == ("top_left", "top_right", "bottom_left", "bottom_right", "details")


def test_detector_builds_expected_request(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return _recognition_payload()

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(four_grid_crop_detector.requests, "post", fake_post)
    boxes = four_grid_crop_detector.detect_person_boxes(
        "https://example.com/source.png",
        (100, 100),
        detector_url="https://detector.example/v1/recognition",
        timeout=12,
    )
    assert captured["url"] == "https://detector.example/v1/recognition"
    assert captured["timeout"] == 12
    assert captured["json"]["images"][0]["url"] == "https://example.com/source.png"
    assert len(boxes) == 4


def test_node_processes_batch_and_uploads_each_frame(monkeypatch):
    uploaded = []

    def fake_upload(path, data):
        uploaded.append((path, data))
        return f"https://oss.example/{len(uploaded)}.png"

    monkeypatch.setattr(node_module, "upload_bytes_to_oss", fake_upload)
    monkeypatch.setattr(
        node_module,
        "detect_person_boxes",
        lambda url, image_size, **kwargs: [
            PersonBox((5, 5, 45, 45), 0.9),
            PersonBox((55, 5, 95, 45), 0.9),
            PersonBox((5, 55, 45, 95), 0.9),
            PersonBox((55, 55, 95, 95), 0.9),
        ],
    )
    image = torch.cat([_image_tensor("white"), _image_tensor("white")], dim=0)
    outputs = ZhiYiFourGridCropNode().crop(
        image,
        padding=0,
        detector_url="https://detector.example/v1/recognition",
    )
    assert all(output.shape == (2, 40, 40, 3) for output in outputs[:4])
    assert len(uploaded) == 2
    details = json.loads(outputs[4])
    assert details["batch_size"] == 2
    assert len(details["items"]) == 2
    assert all(path.endswith(".png") for path, _data in uploaded)


@pytest.mark.parametrize("gradient", [False, True])
def test_batch_rebuilds_background_from_source_without_extra_requests(monkeypatch, gradient):
    height, width = 90, 120
    array = np.zeros((2, height, width, 3), dtype=np.uint8)
    array[0] = (80, 110, 140)
    array[1] = (100, 130, 160)
    if gradient:
        for y in range(height):
            for x in range(width):
                array[0, y, x] = (40 + x, 60 + y, 80)
    array[:, 5:25, 10:30] = (255, 0, 0)
    frames = torch.from_numpy(array.astype(np.float32) / 255.0)
    calls = []
    uploads = []
    monkeypatch.setattr(node_module, "upload_bytes_to_oss", lambda path, data: uploads.append(path) or "https://example.com/in.png")

    def detect(url, image_size, **kwargs):
        side = 20 if not calls else 30
        calls.append(url)
        return [PersonBox((10 + x, 5 + y, 10 + x + side, 5 + y + side)) for x, y in ((0, 0), (60, 0), (0, 45), (60, 45))]

    monkeypatch.setattr(node_module, "detect_person_boxes", detect)
    results = ZhiYiFourGridCropNode().crop(frames, padding=0)
    assert len(calls) == len(uploads) == 2
    assert all(result.shape == (2, 30, 40, 3) for result in results[:4])
    details = json.loads(results[4])
    assert details["items"][0]["canvas_size"] == [28, 21]
    assert details["items"][0]["output_canvas_size"] == [40, 30]
    # The centered subject stays exactly 20x20 pixels after the larger batch canvas is built.
    assert torch.equal(results[0][0, 5:25, 10:30], frames[0, 5:25, 10:30])
    pixel = tuple((results[0][0, 0, 0] * 255).round().int().tolist())
    if gradient:
        assert pixel == (40, 60, 80)
        assert results[0][0, 0, -1, 0] > results[0][0, 0, 0, 0]
    else:
        assert pixel == (80, 110, 140)


def test_detector_generic_protocol(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"persons": [[1, 2, 10, 20, 0.8]]}}

    def post(url, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(four_grid_crop_detector.requests, "post", post)
    boxes = four_grid_crop_detector.detect_person_boxes(
        "https://example.com/input.png", (100, 100), detector_protocol="combine_detect_batch", boxes_path="data.persons", timeout=(5, 30)
    )
    assert boxes == [PersonBox((1, 2, 10, 20), 0.8)]
    assert captured["json"]["internal"] is True
    assert captured["json"]["image_list"][0]["url"] == "https://example.com/input.png"
    assert captured["timeout"] == (5, 30)


@pytest.mark.parametrize(
    "error, code",
    [
        (requests.Timeout("timeout"), "detector_timeout"),
        (requests.HTTPError("bad gateway"), "detector_unavailable"),
        (requests.exceptions.JSONDecodeError("invalid", "x", 0), "detector_schema_error"),
    ],
)
def test_detector_errors_are_explicit(monkeypatch, error, code):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(four_grid_crop_detector.requests, "post", fail)
    with pytest.raises(four_grid_crop_detector.CropError) as caught:
        four_grid_crop_detector.detect_person_boxes("https://example.com/input.png", (100, 100))
    assert caught.value.code == code


def test_saved_node_input_order_is_unchanged():
    inputs = ZhiYiFourGridCropNode.INPUT_TYPES()
    assert list(inputs["required"]) == ["image", "padding", "min_confidence"]
    assert list(inputs["optional"]) == ["detector_url", "detector_timeout"]

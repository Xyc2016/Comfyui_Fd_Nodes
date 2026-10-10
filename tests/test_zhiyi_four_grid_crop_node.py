"""Tests for the four-grid ComfyUI node."""

import json

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

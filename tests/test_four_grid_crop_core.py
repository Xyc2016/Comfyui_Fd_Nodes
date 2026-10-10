"""Tests for four-grid crop core algorithms."""

import pytest
from PIL import Image, ImageDraw

from src.Comfyui_Fd_Nodes.four_grid_crop_core import (
    CropError,
    PersonBox,
    crop_to_white_canvases,
    parse_person_boxes,
)


def _grid_image(size=(100, 100)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, 44, 44), fill="red")
    draw.rectangle((55, 5, 94, 44), fill="green")
    draw.rectangle((5, 55, 44, 94), fill="blue")
    draw.rectangle((55, 55, 94, 94), fill="black")
    return image


def test_crop_uses_four_quadrants_and_fallbacks():
    image = _grid_image()
    boxes = [
        PersonBox((5, 5, 45, 45), 0.9),
        PersonBox((55, 5, 95, 45), 0.8),
        PersonBox((5, 55, 45, 95), 0.7),
    ]

    outputs, details = crop_to_white_canvases(image, boxes, padding=0)

    assert len(outputs) == 4
    assert details["grid_split"] == [50, 50]
    assert [item["position"] for item in details["items"]] == ["top_left", "top_right", "bottom_left", "bottom_right"]
    assert [item["fallback"] for item in details["items"]] == [False, False, False, True]
    assert all(output.size == (50, 50) for output in outputs)
    assert outputs[0].getpixel((10, 10)) == (255, 0, 0)
    assert outputs[1].getpixel((10, 10)) == (0, 128, 0)
    assert outputs[2].getpixel((10, 10)) == (0, 0, 255)
    assert outputs[3].getpixel((10, 10)) == (0, 0, 0)
    image.close()
    for output in outputs:
        output.close()


def test_low_confidence_box_is_ignored():
    image = _grid_image()
    outputs, details = crop_to_white_canvases(
        image,
        [PersonBox((5, 5, 45, 45), 0.1)],
        min_confidence=0.25,
    )
    assert details["items"][0]["fallback"] is True
    assert outputs[0].size == (50, 50)
    image.close()
    for output in outputs:
        output.close()


def test_parse_recognition_payload_converts_normalized_coordinates():
    payload = {
        "status": True,
        "result": [
            {
                "status": True,
                "img_w": 100,
                "img_h": 80,
                "body_positions": [
                    {
                        "person": {
                            "label": "person",
                            "xmin": 0.1,
                            "ymin": 0.2,
                            "xmax": 0.5,
                            "ymax": 0.8,
                            "score": 0.9,
                        }
                    }
                ],
            }
        ],
    }
    boxes = parse_person_boxes(payload, (100, 80))
    assert boxes[0].xyxy == (10.0, 16.0, 50.0, 64.0)
    assert boxes[0].confidence == 0.9


def test_parse_recognition_rejects_size_mismatch():
    payload = {
        "status": True,
        "result": [{"status": True, "img_w": 99, "img_h": 80, "body_positions": []}],
    }
    with pytest.raises(CropError, match="尺寸"):
        parse_person_boxes(payload, (100, 80))

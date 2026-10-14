"""Regression tests for adjacent-background patch rendering."""

import random

import pytest
from PIL import Image

from src.Comfyui_Fd_Nodes.four_grid_crop_core import PersonBox, _source_background_canvas


def test_texture_does_not_turn_into_flat_horizontal_stripes():
    rng = random.Random(21)
    image = Image.new("RGB", (160, 200))
    image.putdata(
        [
            (value, value, value)
            for y in range(200)
            for x in range(160)
            if (value := min(255, 250 + y // 80 + rng.randrange(-2, 3))) is not None
        ]
    )
    bounds = (65, 10, 95, 190)
    size = (150, 190)
    canvas = _source_background_canvas(image, (0, 0, 160, 200), bounds, size)
    try:
        offset_x, offset_y = (size[0] - 30) // 2, (size[1] - 180) // 2
        patch = canvas.crop((0, offset_y, offset_x, offset_y + 180)).convert("L")
        raw = patch.tobytes()
        width = patch.width
        flat = sum(max(raw[y * width : (y + 1) * width]) - min(raw[y * width : (y + 1) * width]) <= 1 for y in range(patch.height))
        assert flat < patch.height // 10
        with image.crop((bounds[0] - offset_x, bounds[1], bounds[0], bounds[3])) as expected:
            assert patch.tobytes() == expected.convert("L").tobytes()
        assert canvas.getpixel((offset_x - 1, offset_y + 40)) == image.getpixel((bounds[0] - 1, bounds[1] + 40))
        patch.close()
    finally:
        canvas.close()
        image.close()


def test_background_stretch_keeps_texture_and_never_copies_neighbor_grid():
    image = Image.new("RGB", (100, 80), (251, 251, 251))
    for y in range(80):
        for x in range(50):
            image.putpixel((x, y), (249 + (x + y) % 5,) * 3)
    image.paste("red", (50, 0, 100, 80))
    canvas = _source_background_canvas(image, (0, 0, 50, 80), (5, 5, 45, 75), (120, 90))
    try:
        assert all(red == green == blue and 249 <= red <= 253 for red, green, blue in canvas.getdata())
        rows = [canvas.crop((0, y, 40, y + 1)).convert("L").getextrema() for y in range(10, 70)]
        assert all(high - low >= 2 for low, high in rows)
    finally:
        canvas.close()
        image.close()


@pytest.mark.parametrize("bounds", [(0, 0, 30, 40), (20, 0, 50, 40), (0, 40, 30, 80), (20, 40, 50, 80)])
def test_empty_background_strip_does_not_reflect_subject_into_padding(bounds):
    image = Image.new("RGB", (50, 80), (251, 251, 251))
    image.paste("black", bounds)
    canvas = _source_background_canvas(image, (0, 0, 50, 80), bounds, (70, 100))
    try:
        assert min(canvas.convert("L").getextrema()) >= 245
    finally:
        canvas.close()
        image.close()


def test_no_resizing_of_person_even_when_background_requires_stretch():
    image = Image.new("RGB", (100, 100), (251, 251, 251))
    image.paste("black", (1, 2, 12, 45))
    canvas, details = __import__(
        "src.Comfyui_Fd_Nodes.four_grid_crop_core",
        fromlist=["crop_to_white_canvases"],
    ).crop_to_white_canvases(image, [PersonBox((1, 2, 12, 45))])
    try:
        output = canvas[0]
        crop = image.crop(details["items"][0]["box"])
        x, y = (output.width - crop.width) // 2, (output.height - crop.height) // 2
        assert output.crop((x, y, x + crop.width, y + crop.height)).tobytes() == crop.tobytes()
        crop.close()
    finally:
        for output in canvas:
            output.close()
        image.close()

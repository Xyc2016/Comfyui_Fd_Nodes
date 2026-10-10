"""四宫格人物框裁剪算法。"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from typing import Any

from PIL import Image, ImageChops


ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")


class CropError(Exception):
    """可向 ComfyUI 节点调用方展示的裁剪错误。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class PersonBox:
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if len(self.xyxy) != 4 or not all(math.isfinite(value) for value in self.xyxy):
            raise ValueError("box must have four finite coordinates")
        x1, y1, x2, y2 = self.xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box must have positive width and height")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")


def parse_person_boxes(payload: Any, image_size: tuple[int, int]) -> list[PersonBox]:
    """解析 recognition 接口返回的归一化人物框。"""
    if not isinstance(payload, dict) or payload.get("status") is not True:
        raise CropError("detector_failed", "识别服务返回失败")

    results = payload.get("result")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise CropError("detector_schema_error", "单图识别响应必须包含一个 result")

    result = results[0]
    if result.get("status") is not True:
        raise CropError("detector_failed", "识别服务返回图片处理失败")

    try:
        reported_size = (float(result["img_w"]), float(result["img_h"]))
        if reported_size != image_size:
            raise CropError("detector_image_size_mismatch", "检测图片尺寸与下载原图不一致")
    except CropError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CropError("detector_schema_error", "识别响应缺少有效图片尺寸") from exc

    positions = result.get("body_positions")
    if not isinstance(positions, list):
        raise CropError("detector_schema_error", "识别响应缺少 body_positions 列表")

    width, height = image_size
    boxes: list[PersonBox] = []
    for position in positions:
        if not isinstance(position, dict) or not isinstance(position.get("person"), dict):
            raise CropError("detector_schema_error", "body_positions 中缺少 person 对象")
        person = position["person"]
        if person.get("label") != "person":
            raise CropError("detector_schema_error", "人物框类别必须为 person")
        try:
            values = tuple(float(person[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                raise ValueError("coordinates must be finite normalized values")
            x1, y1, x2, y2 = values
            confidence = float(person["score"])
            boxes.append(PersonBox((x1 * width, y1 * height, x2 * width, y2 * height), confidence))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise CropError("detector_schema_error", "人物框归一化坐标或置信度无效") from exc
    return boxes


def _white_gutter(image: Image.Image, axis: int) -> int:
    length = image.size[axis]
    middle = length // 2
    radius = max(1, length // 10)
    start, stop = max(1, middle - radius), min(length - 1, middle + radius + 1)
    channels = image.split()
    darkest = ImageChops.darker(ImageChops.darker(channels[0], channels[1]), channels[2])
    mask = darkest.point([255 if value < 245 else 0 for value in range(256)])
    white: list[bool] = []
    for coordinate in range(start, stop):
        bounds = (coordinate, 0, coordinate + 1, image.height) if axis == 0 else (0, coordinate, image.width, coordinate + 1)
        white.append(mask.crop(bounds).getbbox() is None)

    runs: list[int] = []
    run_start: int | None = None
    for offset, is_white in enumerate(white + [False]):
        coordinate = start + offset
        if is_white and run_start is None:
            run_start = coordinate
        elif not is_white and run_start is not None:
            if coordinate - run_start >= 3:
                if run_start <= middle < coordinate:
                    return middle
                runs.append((run_start + coordinate) // 2)
            run_start = None
    return min(runs, key=lambda value: abs(value - middle)) if runs else middle


def _retain_white_background_content(
    image: Image.Image,
    region: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = region
    tile = image.crop(region)
    red, green, blue = tile.split()
    darkest = ImageChops.darker(ImageChops.darker(red, green), blue)
    histogram = darkest.histogram()
    if sum(histogram[245:]) < tile.width * tile.height / 2:
        return bounds
    foreground = darkest.point([255 if value < 245 else 0 for value in range(256)]).getbbox()
    if foreground is None:
        return bounds
    x1, y1, x2, y2 = foreground
    return (
        max(left, min(bounds[0], left + x1 - padding)),
        max(top, min(bounds[1], top + y1 - padding)),
        min(right, max(bounds[2], left + x2 + padding)),
        min(bottom, max(bounds[3], top + y2 + padding)),
    )


def crop_to_white_canvases(
    image: Image.Image,
    boxes: list[PersonBox],
    *,
    padding: int = 5,
    min_confidence: float = 0.25,
) -> tuple[list[Image.Image], dict[str, Any]]:
    """按人物框裁剪四格，并将本帧四个结果统一到白底画布。"""
    if padding < 0 or not 0 <= min_confidence <= 1:
        raise ValueError("invalid crop configuration")
    width, height = image.size
    if width < 2 or height < 2:
        raise CropError("invalid_image", "图片宽高必须至少为 2 像素")

    rgba = image.convert("RGBA")
    composite = Image.new("RGBA", image.size, (255, 255, 255, 255))
    composite.alpha_composite(rgba)
    rgb = composite.convert("RGB")
    split_x, split_y = _white_gutter(rgb, 0), _white_gutter(rgb, 1)
    regions = [
        (0, 0, split_x, split_y),
        (split_x, 0, width, split_y),
        (0, split_y, split_x, height),
        (split_x, split_y, width, height),
    ]

    selected: list[PersonBox | None] = [None] * 4
    for box in boxes:
        if box.confidence < min_confidence:
            continue
        x1, y1, x2, y2 = box.xyxy
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        index = (2 if center_y >= split_y else 0) + (1 if center_x >= split_x else 0)
        candidate = PersonBox((x1, y1, x2, y2), box.confidence)
        previous = selected[index]

        def rank(item: PersonBox) -> tuple[float, float]:
            left, top, right, bottom = item.xyxy
            return item.confidence, (right - left) * (bottom - top)

        if previous is None or rank(candidate) > rank(previous):
            selected[index] = candidate

    crops: list[Image.Image] = []
    details: list[dict[str, Any]] = []
    for index, (region, box) in enumerate(zip(regions, selected)):
        left, top, right, bottom = region
        if box is None:
            bounds = region
            detector_bounds = None
        else:
            x1, y1, x2, y2 = box.xyxy
            bounds = (
                max(left, math.floor(x1) - padding),
                max(top, math.floor(y1) - padding),
                min(right, math.ceil(x2) + padding),
                min(bottom, math.ceil(y2) + padding),
            )
            detector_bounds = bounds
            bounds = _retain_white_background_content(rgb, region, bounds, padding)
        crops.append(rgb.crop(bounds))
        details.append(
            {
                "position": ORDER[index],
                "box": list(bounds),
                "fallback": box is None,
                "confidence": box.confidence if box else None,
                "detector_crop_box": list(detector_bounds) if detector_bounds else None,
                "content_expanded": detector_bounds is not None and bounds != detector_bounds,
            }
        )

    canvas_size = (max(crop.width for crop in crops), max(crop.height for crop in crops))
    outputs: list[Image.Image] = []
    for crop in crops:
        canvas = Image.new("RGB", canvas_size, "white")
        canvas.paste(crop, ((canvas_size[0] - crop.width) // 2, (canvas_size[1] - crop.height) // 2))
        outputs.append(canvas)

    rgb.close()
    composite.close()
    rgba.close()
    return outputs, {
        "source_size": [width, height],
        "canvas_size": list(canvas_size),
        "grid_split": [split_x, split_y],
        "items": details,
    }


def encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()

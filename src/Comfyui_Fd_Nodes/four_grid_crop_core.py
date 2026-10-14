"""四宫格裁剪：坐标均为原图像素 xyxy，所有输出保持原始像素比例。"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from typing import Any
from PIL import Image, ImageChops, ImageFilter

ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")


class CropError(Exception):
    def __init__(self, code: str, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


@dataclass(frozen=True)
class PersonBox:
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if len(self.xyxy) != 4 or not all(math.isfinite(v) for v in self.xyxy):
            raise ValueError("box must have four finite coordinates")
        x1, y1, x2, y2 = self.xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box must have positive width and height")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")


def parse_person_boxes(
    payload: Any, *, boxes_path: str = "", box_format: str = "xyxy", normalized: bool = False, image_size: tuple[int, int] | None = None
) -> list[PersonBox]:
    """Strict adapter: unknown schema is an error, not four silent fallbacks.

    Supports person/persons/person_boxes containers and labelled detections.
    boxes_path selects a dot-separated path (list indexes allowed) when required.
    A selected path is explicitly understood as a person-only container.
    """
    if isinstance(payload, dict):
        if payload.get("success") is False:
            raise CropError("detector_failed", "检测服务返回失败")
        if "code" in payload and str(payload["code"]) not in {"0", "200"}:
            raise CropError("detector_failed", "检测服务返回非成功 code")
    if box_format not in {"xyxy", "xywh"}:
        raise ValueError("DETECTOR_BOX_FORMAT must be xyxy or xywh")
    if normalized and image_size is None:
        raise ValueError("image_size is required for normalized coordinates")
    forced = bool(boxes_path)
    if forced:
        try:
            for component in boxes_path.split("."):
                payload = payload[int(component)] if isinstance(payload, list) else payload[component]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise CropError("detector_schema_error", "配置的检测框路径不存在") from exc
    boxes: list[PersonBox] = []
    recognized = forced
    person_keys = {"person", "persons", "person_boxes"}
    detection_keys = {"detections", "objects", "predictions"}

    def add(raw: Any, confidence: Any = 1.0) -> None:
        try:
            if isinstance(raw, dict):
                coords = [raw[k] for k in ("x1", "y1", "x2", "y2")]
            else:
                if not isinstance(raw, (list, tuple)) or len(raw) not in {4, 5}:
                    raise ValueError("expected four coordinates and optional score")
                coords = raw[:4]
                if len(raw) == 5:
                    confidence = raw[4]
            x1, y1, a, b = map(float, coords)
            x2, y2 = (x1 + a, y1 + b) if box_format == "xywh" else (a, b)
            if normalized:
                assert image_size is not None
                w, h = image_size
                x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
            boxes.append(PersonBox((x1, y1, x2, y2), float(confidence)))
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise CropError("detector_schema_error", "人物框坐标或置信度无效") from exc

    def visit(node: Any, person_context: bool = False) -> None:
        nonlocal recognized
        if isinstance(node, dict):
            label = next((node[k] for k in ("label", "class_name", "category", "name", "class_id") if k in node), None)
            labelled = label is not None
            is_person = str(label).lower() in {"person", "0"} if labelled else person_context
            raw = next((node[k] for k in ("bbox", "box", "xyxy") if k in node), None)
            if raw is None and all(k in node for k in ("x1", "y1", "x2", "y2")):
                raw = node
            if raw is not None:
                if labelled:
                    recognized = True
                if is_person:
                    recognized = True
                    add(raw, node.get("confidence", node.get("score", node.get("conf", 1.0))))
                elif person_context:
                    raise CropError("detector_schema_error", "人物容器中出现非人物类别")
                return
            if person_context and not any(isinstance(v, (dict, list)) for v in node.values()):
                raise CropError("detector_schema_error", "人物容器格式不正确，请用空列表表示漏检")
            for key, value in node.items():
                if key in person_keys:
                    recognized = True
                    visit(value, True)
                elif key in detection_keys:
                    recognized = True
                    if not isinstance(value, list):
                        raise CropError("detector_schema_error", "通用检测容器必须为带类别标签的对象列表")
                    for detection in value:
                        if (
                            not isinstance(detection, dict)
                            or not any(k in detection for k in ("label", "class_name", "category", "name", "class_id"))
                            or not (
                                any(k in detection for k in ("bbox", "box", "xyxy"))
                                or all(k in detection for k in ("x1", "y1", "x2", "y2"))
                            )
                        ):
                            raise CropError("detector_schema_error", "通用检测结果缺少类别标签或坐标")
                        visit(detection, False)
                elif isinstance(value, (dict, list)):
                    visit(value, person_context)
        elif isinstance(node, list):
            if person_context and node and all(isinstance(v, (int, float)) for v in node):
                add(node)
            else:
                for child in node:
                    visit(child, person_context)
        elif person_context:
            raise CropError("detector_schema_error", "人物容器必须为列表或对象")

    visit(payload, forced)
    if not recognized:
        raise CropError("detector_schema_error", "检测响应未找到可识别的人物框，请配置 DETECTOR_BOXES_PATH")
    return boxes


def _white_gutter(image: Image.Image, axis: int) -> int:
    """Locate a clear white separator near the midpoint, never inside visible ink.

    Require a full-width/height white stripe (all channels >=245), at least
    three pixels wide. If the midpoint is already white, keep it unchanged.
    Non-white backgrounds or absent gutters retain the original half split.
    """
    length = image.size[axis]
    middle = length // 2
    radius = max(1, length // 10)
    start, stop = max(1, middle - radius), min(length - 1, middle + radius + 1)
    channels = image.split()
    darkest = ImageChops.darker(ImageChops.darker(channels[0], channels[1]), channels[2])
    mask = darkest.point([255 if value < 245 else 0 for value in range(256)])
    white = []
    for coordinate in range(start, stop):
        bounds = (coordinate, 0, coordinate + 1, image.height) if axis == 0 else (0, coordinate, image.width, coordinate + 1)
        white.append(mask.crop(bounds).getbbox() is None)
    runs = []
    run_start = None
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
    image: Image.Image, region: tuple[int, int, int, int], bounds: tuple[int, int, int, int], padding: int
) -> tuple[int, int, int, int]:
    """Expand a detector crop to retain visible content on predominantly white grids.

    This is a conservative pixel check, not person segmentation: shadows and
    other visible content may also be retained. Colored backgrounds keep the
    detector crop. Never shrink the original crop or cross a grid separator.
    """
    left, top, right, bottom = region
    tile = image.crop(region)
    r, g, b = tile.split()
    darkest = ImageChops.darker(ImageChops.darker(r, g), b)
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


def _source_background_canvas(
    image: Image.Image,
    region: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
    size: tuple[int, int],
) -> Image.Image:
    """用裁剪框周围的相邻背景生成画布，避免采到人物颜色。"""
    cw, ch = size
    x1, y1, x2, y2 = bounds
    offset_x, offset_y = (cw - (x2 - x1)) // 2, (ch - (y2 - y1)) // 2
    source_x = (region[0], x1, x2, region[2])
    source_y = (region[1], y1, y2, region[3])
    target_x = (0, offset_x, offset_x + x2 - x1, cw)
    target_y = (0, offset_y, offset_y + y2 - y1, ch)
    canvas = _edge_background_fallback(image, region, bounds, size)

    def adjacent_segment(start: int, stop: int, required: int, segment: int) -> tuple[int, int]:
        if stop - start > required:
            if segment == 0:
                start = stop - required
            elif segment == 2:
                stop = start + required
        return start, stop

    for row in range(3):
        for column in range(3):
            if row == column == 1:
                continue
            patch_width = target_x[column + 1] - target_x[column]
            patch_height = target_y[row + 1] - target_y[row]
            if patch_width <= 0 or patch_height <= 0:
                continue
            sx1, sx2 = adjacent_segment(source_x[column], source_x[column + 1], patch_width, column)
            sy1, sy2 = adjacent_segment(source_y[row], source_y[row + 1], patch_height, row)
            if sx2 <= sx1 or sy2 <= sy1:
                continue
            with image.crop((sx1, sy1, sx2, sy2)) as patch:
                if patch.size != (patch_width, patch_height):
                    resized = patch.resize((patch_width, patch_height), Image.Resampling.BILINEAR)
                    canvas.paste(resized, (target_x[column], target_y[row]))
                    resized.close()
                else:
                    canvas.paste(patch, (target_x[column], target_y[row]))
    return canvas


def _edge_background_fallback(
    image: Image.Image,
    region: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
    size: tuple[int, int],
) -> Image.Image:
    """边缘无可用相邻背景时，避开裁剪框并平滑估计背景。"""
    left, top, right, bottom = region
    cw, ch = size
    crop_width, crop_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    offset_x, offset_y = (cw - crop_width) // 2, (ch - crop_height) // 2
    strip = max(1, min(32, (right - left) // 20))
    left_x, right_x = left + (strip - 1) / 2, right - 1 - (strip - 1) / 2

    def fraction(x: float) -> float:
        return min(1.0, max(0.0, (x - left_x) / max(1.0, right_x - left_x)))

    t0 = fraction(bounds[0] - offset_x)
    t1 = fraction(bounds[0] - offset_x + cw - 1)
    samples: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    rows: list[tuple[int, int, int]] = []
    for y in range(ch):
        sample_y = min(bottom - 1, max(top, bounds[1] + y - offset_y))
        if sample_y not in samples:
            intervals = [(left, left + strip), (right - strip, right)]
            if bounds[1] <= sample_y < bounds[3]:
                intervals = [
                    (left, min(left + strip, bounds[0])),
                    (max(right - strip, bounds[2]), right),
                ]
                intervals = [(start, stop) for start, stop in intervals if stop > start]
                if not intervals:
                    outside_rows = [row for row in (bounds[1] - 1, bounds[3]) if top <= row < bottom]
                    if outside_rows:
                        sample_y = min(outside_rows, key=lambda row: abs(row - sample_y))
                    intervals = [(left, left + strip), (right - strip, right)]
            colors: list[tuple[int, ...]] = []
            for start, stop in intervals:
                histogram = image.crop((start, sample_y, stop, sample_y + 1)).histogram()
                channels: list[int] = []
                for channel in range(3):
                    count = 0
                    for value, amount in enumerate(histogram[channel * 256 : (channel + 1) * 256]):
                        count += amount
                        if count > (stop - start) // 2:
                            channels.append(value)
                            break
                colors.append(tuple(channels))
            samples[sample_y] = (colors[0], colors[-1])
        first, second = samples[sample_y]
        rows.extend(tuple(round(first[channel] + (second[channel] - first[channel]) * t) for channel in range(3)) for t in (t0, t1))
    background = Image.new("RGB", (2, ch))
    background.putdata(rows)
    smoothed = background.filter(ImageFilter.GaussianBlur(radius=8))
    background.close()
    canvas = smoothed.resize((cw, ch), Image.Resampling.BILINEAR, box=(0.5, 0, 1.5, ch))
    smoothed.close()
    return canvas


def crop_to_white_canvases(
    image: Image.Image, boxes: list[PersonBox], *, padding: int = 5, min_confidence: float = 0.25
) -> tuple[list[Image.Image], dict[str, Any]]:
    if padding < 0 or not 0 <= min_confidence <= 1:
        raise ValueError("invalid crop configuration")
    w, h = image.size
    if w < 2 or h < 2:
        raise CropError("invalid_image", "图片宽高必须至少为 2 像素", 422)
    # Composite transparent source pixels onto white before cropping.
    rgba = image.convert("RGBA")
    composite = Image.new("RGBA", image.size, (255, 255, 255, 255))
    composite.alpha_composite(rgba)
    rgb = composite.convert("RGB")
    composite.close()
    rgba.close()
    mx, my = _white_gutter(rgb, 0), _white_gutter(rgb, 1)
    regions = [(0, 0, mx, my), (mx, 0, w, my), (0, my, mx, h), (mx, my, w, h)]
    selected: list[PersonBox | None] = [None] * 4
    for box in boxes:
        if box.confidence < min_confidence:
            continue
        x1, y1, x2, y2 = box.xyxy
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        index = (2 if cy >= my else 0) + (1 if cx >= mx else 0)
        candidate = PersonBox((x1, y1, x2, y2), box.confidence)
        previous = selected[index]

        def rank(b: PersonBox) -> tuple[float, float]:
            a, c, d, e = b.xyxy
            return b.confidence, (d - a) * (e - c)

        if previous is None or rank(candidate) > rank(previous):
            selected[index] = candidate
    crops, details = [], []
    for i, (region, box) in enumerate(zip(regions, selected)):
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
        crop = rgb.crop(bounds)
        crops.append(crop)
        details.append(
            {
                "position": ORDER[i],
                "box": list(bounds),
                "fallback": box is None,
                "confidence": box.confidence if box else None,
                "detector_crop_box": list(detector_bounds) if detector_bounds else None,
                "content_expanded": detector_bounds is not None and bounds != detector_bounds,
            }
        )
    # Smallest integer canvas with the exact source aspect ratio that fits
    # every crop. Integer multiples avoid floating-point rounding errors.
    divisor = math.gcd(w, h)
    ratio_w, ratio_h = w // divisor, h // divisor
    max_w, max_h = max(c.width for c in crops), max(c.height for c in crops)
    multiple = max((max_w + ratio_w - 1) // ratio_w, (max_h + ratio_h - 1) // ratio_h)
    size = (ratio_w * multiple, ratio_h * multiple)
    result_details = {
        "source_size": [w, h],
        "canvas_size": list(size),
        "grid_split": [mx, my],
        "aspect_ratio": [ratio_w, ratio_h],
        "items": details,
    }
    try:
        outputs = render_crop_canvases(rgb, result_details, size)
    finally:
        rgb.close()
        for crop in crops:
            crop.close()
    return outputs, result_details


def render_crop_canvases(
    image: Image.Image,
    details: dict[str, Any],
    size: tuple[int, int],
) -> list[Image.Image]:
    """从原图重建指定尺寸的背景，只复制原始裁剪像素，不缩放人物。"""
    with image.convert("RGBA") as rgba, Image.new("RGBA", image.size, "white") as composite:
        composite.alpha_composite(rgba)
        rgb = composite.convert("RGB")
    w, h = rgb.size
    mx, my = details["grid_split"]
    regions = [(0, 0, mx, my), (mx, 0, w, my), (0, my, mx, h), (mx, my, w, h)]
    outputs: list[Image.Image] = []
    try:
        for region, item in zip(regions, details["items"]):
            bounds = tuple(item["box"])
            with rgb.crop(bounds) as crop:
                if crop.width > size[0] or crop.height > size[1]:
                    raise ValueError("画布尺寸不能小于裁剪内容")
                canvas = _source_background_canvas(rgb, region, bounds, size)
                canvas.paste(crop, ((size[0] - crop.width) // 2, (size[1] - crop.height) // 2))
                outputs.append(canvas)
            item["background_method"] = "source_adjacent_background_patches"
    except Exception:
        for canvas in outputs:
            canvas.close()
        raise
    finally:
        rgb.close()
    return outputs


def encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()

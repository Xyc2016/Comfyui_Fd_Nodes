"""四宫格裁剪使用的知衣检测 HTTP 适配器。"""

from __future__ import annotations

import math
import os
from typing import Any

import requests

from .four_grid_crop_core import CropError, PersonBox, parse_person_boxes


DEFAULT_DETECTOR_URL = os.getenv(
    "FD_FOUR_GRID_DETECTOR_URL",
    "http://image-server-internal.zhiyi.com.cn/trition-servings-gray/image/recognition",
)


def build_detector_payload(url: str, detector_protocol: str = "recognition") -> dict[str, Any]:
    if detector_protocol == "recognition":
        return {
            "add_rule_flag": False,
            "label_convert": True,
            "do_flag": [
                "combine_detect",
                "detection_box_quality",
                "crop_recognition",
                "fashion_detection_merge",
            ],
            "images": [{"url": url, "platform_id": 0, "union_id": "0"}],
            "filter_no_category_box_on_client": False,
        }
    if detector_protocol == "combine_detect_batch":
        return {
            "image_list": [
                {
                    "feature_save": False,
                    "platform_id": 1,
                    "union_id": "123",
                    "url": url,
                }
            ],
            "internal": True,
        }
    raise ValueError("detector_protocol 必须为 recognition 或 combine_detect_batch")


def parse_recognition_response(payload: dict[str, Any], image_size: tuple[int, int]) -> list[PersonBox]:
    if not isinstance(payload, dict) or "status" not in payload:
        raise CropError("detector_schema_error", "识别响应缺少 status")
    if payload["status"] is not True:
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
            boxes.append(PersonBox((x1 * width, y1 * height, x2 * width, y2 * height), float(person["score"])))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise CropError("detector_schema_error", "人物框归一化坐标或置信度无效") from exc
    return boxes


def detect_person_boxes(
    url: str,
    image_size: tuple[int, int],
    *,
    detector_url: str = DEFAULT_DETECTOR_URL,
    timeout: float | tuple[float, float] = 30.0,
    detector_protocol: str = "recognition",
    boxes_path: str = "",
    box_format: str = "xyxy",
    normalized: bool = False,
) -> list[PersonBox]:
    """调用检测服务并返回像素坐标的人物框。"""
    if not detector_url.strip():
        raise CropError("detector_config_error", "检测服务地址不能为空")
    try:
        response = requests.post(
            detector_url,
            json=build_detector_payload(url, detector_protocol),
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise CropError("detector_timeout", "调用检测模型超时", 504) from exc
    except requests.exceptions.JSONDecodeError as exc:
        raise CropError("detector_schema_error", "检测服务返回非 JSON 内容") from exc
    except requests.RequestException as exc:
        raise CropError("detector_unavailable", "无法调用检测模型") from exc

    if detector_protocol == "recognition":
        return parse_recognition_response(payload, image_size)
    return parse_person_boxes(
        payload,
        boxes_path=boxes_path,
        box_format=box_format,
        normalized=normalized,
        image_size=image_size,
    )

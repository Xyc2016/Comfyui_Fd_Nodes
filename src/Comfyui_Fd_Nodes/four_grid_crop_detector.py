"""四宫格裁剪使用的知衣 recognition HTTP 适配器。"""

from __future__ import annotations

import os
from typing import Any

import requests

from .four_grid_crop_core import CropError, PersonBox, parse_person_boxes


DEFAULT_DETECTOR_URL = os.getenv(
    "FD_FOUR_GRID_DETECTOR_URL",
    "http://image-server-internal.zhiyi.com.cn/trition-servings-gray/image/recognition",
)


def build_detector_payload(url: str) -> dict[str, Any]:
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


def detect_person_boxes(
    url: str,
    image_size: tuple[int, int],
    *,
    detector_url: str = DEFAULT_DETECTOR_URL,
    timeout: float = 30.0,
) -> list[PersonBox]:
    """调用 recognition 接口并返回像素坐标的人物框。"""
    if not detector_url.strip():
        raise CropError("detector_config_error", "检测服务地址不能为空")
    try:
        response = requests.post(
            detector_url,
            json=build_detector_payload(url),
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise CropError("detector_timeout", "调用检测模型超时") from exc
    except requests.RequestException as exc:
        raise CropError("detector_unavailable", "无法调用检测模型") from exc

    try:
        payload = response.json()
    except (ValueError, requests.RequestException) as exc:
        raise CropError("detector_schema_error", "检测服务返回非 JSON 内容") from exc
    return parse_person_boxes(payload, image_size)

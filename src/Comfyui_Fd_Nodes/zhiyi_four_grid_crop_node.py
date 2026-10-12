"""知衣四宫格图片识别拆分 ComfyUI 节点。"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import FD_OSS_URL_PATH_PREFIX_BEFORE_GEN
from .four_grid_crop_core import crop_to_white_canvases, encode_png, render_crop_canvases
from .four_grid_crop_detector import DEFAULT_DETECTOR_URL, detect_person_boxes
from .utils.common_util import bytes_calculate_hex_md5
from .utils.oss_client import upload_bytes_to_oss


logger = logging.getLogger(__name__)


class ZhiYiFourGridCropNode:
    """识别四宫格人物并输出左上、右上、左下、右下四张图。"""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "需要识别拆分的四宫格图片"}),
                "padding": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "人物检测框四边额外保留的像素",
                    },
                ),
                "min_confidence": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "低于该置信度的人物框会被忽略",
                    },
                ),
            },
            "optional": {
                "detector_url": (
                    "STRING",
                    {
                        "default": DEFAULT_DETECTOR_URL,
                        "tooltip": "知衣 recognition 检测服务地址",
                    },
                ),
                "detector_timeout": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 1.0,
                        "max": 600.0,
                        "step": 1.0,
                        "tooltip": "检测服务请求超时时间（秒）",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right", "details")
    FUNCTION = "crop"
    CATEGORY = "知衣/图像处理"
    OUTPUT_NODE = False
    DESCRIPTION = "四宫格人物裁剪；输出保持原图宽高比，使用对应宫格边缘背景补边，人物不缩放。"

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
        if image.ndim == 4:
            image = image[0]
        array = (image.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError("图片 tensor 必须为 [H,W,3] 或 [H,W,4]")
        return Image.fromarray(array, mode="RGBA" if array.shape[-1] == 4 else "RGB")

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    @staticmethod
    def _normalize_batch(image: torch.Tensor) -> list[torch.Tensor]:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image 必须是 torch.Tensor")
        if image.ndim == 3:
            return [image.unsqueeze(0)]
        if image.ndim == 4:
            if image.shape[0] == 0:
                raise ValueError("图片 batch 不能为空")
            return [image[index : index + 1] for index in range(image.shape[0])]
        raise ValueError(f"图片 tensor 维度错误: {image.ndim}")

    @staticmethod
    def _upload_source(image: Image.Image) -> str:
        image_bytes = encode_png(image)
        object_path = f"{FD_OSS_URL_PATH_PREFIX_BEFORE_GEN}/{bytes_calculate_hex_md5(image_bytes)}.png"
        return upload_bytes_to_oss(object_path, image_bytes)

    def _crop_one(
        self,
        image_tensor: torch.Tensor,
        *,
        detector_url: str,
        detector_timeout: float,
        padding: int,
        min_confidence: float,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        source = self._tensor_to_pil(image_tensor)
        try:
            source_url = self._upload_source(source)
            boxes = detect_person_boxes(
                source_url,
                source.size,
                detector_url=detector_url,
                timeout=detector_timeout,
            )
            crops, details = crop_to_white_canvases(
                source,
                boxes,
                padding=padding,
                min_confidence=min_confidence,
            )
            details["detected_box_count"] = len(boxes)
            try:
                output_tensors = [self._pil_to_tensor(crop) for crop in crops]
            finally:
                for crop in crops:
                    crop.close()
            return output_tensors, details
        finally:
            source.close()

    def crop(
        self,
        image: torch.Tensor,
        padding: int = 5,
        min_confidence: float = 0.25,
        detector_url: str = DEFAULT_DETECTOR_URL,
        detector_timeout: float = 30.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        if padding < 0:
            raise ValueError("padding 不能小于 0")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence 必须在 0 到 1 之间")
        if detector_timeout <= 0:
            raise ValueError("detector_timeout 必须大于 0")

        frames = self._normalize_batch(image)
        outputs: list[list[torch.Tensor]] = [[], [], [], []]
        details: list[dict[str, Any]] = []
        max_width, max_height = 0, 0
        for index, frame in enumerate(frames):
            frame_outputs, frame_details = self._crop_one(
                frame,
                detector_url=detector_url,
                detector_timeout=detector_timeout,
                padding=padding,
                min_confidence=min_confidence,
            )
            details.append({"batch_index": index, **frame_details})
            max_width = max(max_width, *(output.shape[2] for output in frame_outputs))
            max_height = max(max_height, *(output.shape[1] for output in frame_outputs))
            for output_index, output in enumerate(frame_outputs):
                outputs[output_index].append(output)

        target_size = (max_width, max_height)
        for index, (frame, frame_details) in enumerate(zip(frames, details)):
            if tuple(frame_details["canvas_size"]) != target_size:
                with self._tensor_to_pil(frame) as source:
                    canvases = render_crop_canvases(source, frame_details, target_size)
                try:
                    for output_index, canvas in enumerate(canvases):
                        outputs[output_index][index] = self._pil_to_tensor(canvas)
                finally:
                    for canvas in canvases:
                        canvas.close()
            frame_details["output_canvas_size"] = list(target_size)
        result = tuple(torch.cat(items, dim=0) for items in outputs)
        return (
            *result,
            json.dumps({"batch_size": len(frames), "canvas_size": [max_width, max_height], "items": details}, ensure_ascii=False),
        )

import base64
import io
import json
import logging
import math
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse

import numpy as np
import requests
import torch
from PIL import Image, ImageColor

from .config import FD_SAM2_SEGMENT_URL
from .utils.error_utils import normalize_error_message
from .utils.logging_utils import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)


class ZhiYiSAM2SegmentNode:
    """知衣 SAM2 bbox 抠图节点 - 纯 API 调用，不加载本地模型。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "bboxes": ("BBOXES",),
                "service_url": ("STRING", {
                    "default": FD_SAM2_SEGMENT_URL or "",
                    "multiline": False,
                    "tooltip": "SAM2 分割接口地址，例如 http://host:8000/v1/segment",
                }),
                "dilate": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                }),
                "blur": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                }),
                "background": (["Alpha", "Color", "Original"], {"default": "Alpha"}),
                "background_color": ("STRING", {"default": "#222222"}),
                "max_concurrency": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                }),
                "timeout": ("INT", {
                    "default": 120,
                    "min": 10,
                    "max": 600,
                    "step": 10,
                }),
            },
            "optional": {
                "invert_output": ("BOOLEAN", {"default": False}),
                "health_check": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "JSON")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "INFO")
    FUNCTION = "segment"
    CATEGORY = "知衣/抠图"
    OUTPUT_NODE = False

    def _expand_images(self, images):
        if images is None:
            return []
        if not isinstance(images, torch.Tensor):
            raise RuntimeError(f"图片必须是 torch.Tensor，实际为 {type(images).__name__}")
        if images.ndim == 4:
            return [images[i:i + 1] for i in range(images.shape[0])]
        if images.ndim == 3:
            return [images.unsqueeze(0)]
        raise RuntimeError(f"图片 tensor 维度错误: {images.ndim}")

    def _image_tensor_to_rgb_pil(self, image_tensor):
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]
        arr = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise RuntimeError(f"图片 tensor shape 错误: {tuple(image_tensor.shape)}")
        return Image.fromarray(arr[..., :3]).convert("RGB")

    def _tensor_to_png_data_url(self, image_tensor):
        image = self._image_tensor_to_rgb_pil(image_tensor)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _decode_png_data_url(self, value, field_name):
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"SAM2 响应缺少 {field_name}")
        if "," in value and value.startswith("data:"):
            encoded = value.split(",", 1)[1]
        else:
            encoded = value
        try:
            payload = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(payload))
            image.load()
            return image
        except Exception as exc:
            raise RuntimeError(f"SAM2 响应 {field_name} 不是有效 PNG base64") from exc

    def _mask_data_url_to_tensor(self, value, size, invert_output):
        mask_image = self._decode_png_data_url(value, "mask").convert("L")
        if mask_image.size != size:
            mask_image = mask_image.resize(size, Image.Resampling.BILINEAR)
        mask_array = np.array(mask_image).astype(np.float32) / 255.0
        if invert_output:
            mask_array = 1.0 - mask_array
        mask_array = np.clip(mask_array, 0.0, 1.0)
        return torch.from_numpy(mask_array).unsqueeze(0)

    def _mask_to_image_tensor(self, mask_tensor):
        mask = mask_tensor
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return mask.unsqueeze(-1).expand(-1, -1, -1, 3).to(dtype=torch.float32)

    def _parse_hex_color(self, background_color):
        try:
            return ImageColor.getrgb((background_color or "#222222").strip())
        except ValueError as exc:
            raise RuntimeError(f"background_color 不是有效 hex 颜色: {background_color}") from exc

    def _compose_result_image(self, original_image_tensor, mask_tensor, background, background_color):
        original = self._image_tensor_to_rgb_pil(original_image_tensor)
        original_np = np.array(original).astype(np.float32) / 255.0
        mask_np = mask_tensor[0].detach().cpu().numpy().astype(np.float32)

        if background == "Original":
            result = original_np
        elif background == "Color":
            color = np.array(self._parse_hex_color(background_color), dtype=np.float32) / 255.0
            bg = np.ones_like(original_np, dtype=np.float32) * color
            result = original_np * mask_np[..., None] + bg * (1.0 - mask_np[..., None])
        else:
            result = original_np * mask_np[..., None]

        return torch.from_numpy(np.clip(result, 0.0, 1.0).astype(np.float32)).unsqueeze(0)

    def _is_box(self, value):
        if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)) or len(value) != 4:
            return False
        return not any(isinstance(item, (list, tuple, dict)) for item in value)

    def _coerce_box(self, value):
        if not self._is_box(value):
            raise RuntimeError("bbox 必须是 [x1, y1, x2, y2]")
        try:
            box = [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("bbox 坐标必须是数字") from exc
        if not all(math.isfinite(coord) for coord in box):
            raise RuntimeError("bbox 坐标必须是有限数字")
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"bbox 无效，要求 x2>x1 且 y2>y1: {value}")
        return box

    def _coerce_group(self, value):
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise RuntimeError("BBOXES 每组 bbox 必须是 list")
        return [self._coerce_box(box) for box in value]

    def _normalize_bboxes_batch(self, bboxes, image_count):
        if not isinstance(bboxes, list):
            raise RuntimeError("bboxes 必须是 BBOXES list")

        if not bboxes:
            groups = [[]]
        elif self._is_box(bboxes):
            groups = [[self._coerce_box(bboxes)]]
        elif self._is_box(bboxes[0]):
            groups = [self._coerce_group(bboxes)]
        else:
            groups = [self._coerce_group(group) for group in bboxes]

        if image_count > 1 and len(groups) == 1:
            groups = [[box[:] for box in groups[0]] for _ in range(image_count)]
        elif len(groups) != image_count:
            raise RuntimeError(f"BBOXES 数量({len(groups)})必须为 1 或等于图片数量({image_count})")

        return groups

    def _build_payload(self, image_data_url, bboxes, dilate, blur):
        return {
            "image": image_data_url,
            "bboxes": bboxes,
            "bbox_format": "xyxy",
            "return_individual_masks": True,
            "return_merged_mask": True,
            "return_masked_image": False,
            "mask_format": "png_base64",
            "dilate": int(dilate),
            "blur": int(blur),
        }

    def _extract_response_error(self, response):
        try:
            data = response.json()
        except ValueError:
            return response.text[:1000]
        if not isinstance(data, dict):
            return str(data)[:1000]
        error = data.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            request_id = error.get("request_id")
            parts = []
            if code:
                parts.append(str(code))
            if message:
                parts.append(str(message))
            if request_id:
                parts.append(f"request_id={request_id}")
            if parts:
                return ": ".join(parts)
        message = data.get("message")
        if message:
            return str(message)
        return json.dumps(data, ensure_ascii=False)[:1000]

    def _request_with_retry(self, method, url, *, timeout, retry_count=1, **kwargs):
        last_exc = None
        for attempt in range(retry_count + 1):
            try:
                response = requests.request(method, url, timeout=timeout, **kwargs)
                if response.status_code >= 500 and attempt < retry_count:
                    time.sleep(0.5)
                    continue
                return response
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                if attempt >= retry_count:
                    raise
                time.sleep(0.5)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("请求失败")

    def _post_segment(self, service_url, payload, timeout):
        response = self._request_with_retry(
            "POST",
            service_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
            retry_count=1,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"SAM2 API 请求失败: {response.status_code}\n{self._extract_response_error(response)}")
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"SAM2 API 响应不是 JSON: {response.text[:500]}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"SAM2 API 响应格式错误: 期望 dict，实际 {type(data).__name__}")
        if not data.get("mask"):
            raise RuntimeError("SAM2 API 响应缺少 mask")
        return data

    def _health_url(self, service_url):
        url = service_url.rstrip("/")
        for suffix in ("/v1/segment", "/segment"):
            if url.endswith(suffix):
                return f"{url[:-len(suffix)]}/health"
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            return urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))
        return f"{url}/health"

    def _check_health(self, service_url, timeout):
        health_url = self._health_url(service_url)
        response = self._request_with_retry("GET", health_url, timeout=timeout, retry_count=0)
        if response.status_code >= 400:
            raise RuntimeError(f"SAM2 health check 失败: {response.status_code}\n{response.text[:500]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"SAM2 health check 响应不是 JSON: {response.text[:500]}") from exc
        if data.get("status") != "ok" or data.get("loaded") is not True:
            raise RuntimeError(f"SAM2 服务未就绪: {json.dumps(data, ensure_ascii=False)[:1000]}")

    def _empty_bbox_result(self, idx, image_tensor):
        height, width = image_tensor.shape[-3:-1]
        mask = torch.ones(
            (1, height, width),
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        )
        logger.warning(
            "第 %s 张图片无 bbox，跳过 SAM2 API，原图直出且返回全白 mask",
            idx + 1,
        )
        return {
            "result_image": image_tensor,
            "mask": mask,
            "mask_image": self._mask_to_image_tensor(mask),
            "info": {
                "index": idx,
                "bboxes": [],
                "skipped": True,
                "reason": "no_bbox",
            },
        }

    def _single_request(self, idx, image_tensor, bboxes, service_url, dilate, blur, background, background_color, invert_output, timeout):
        image_data_url = self._tensor_to_png_data_url(image_tensor)
        payload = self._build_payload(image_data_url, bboxes, dilate, blur)
        logger.info("Calling SAM2 API idx=%s bbox_count=%s service_url=%s", idx, len(bboxes), service_url)
        result = self._post_segment(service_url, payload, timeout)

        image = self._image_tensor_to_rgb_pil(image_tensor)
        mask = self._mask_data_url_to_tensor(result["mask"], image.size, invert_output)
        mask_image = self._mask_to_image_tensor(mask)
        result_image = self._compose_result_image(image_tensor, mask, background, background_color)
        scores = []
        masks = result.get("masks") if isinstance(result.get("masks"), list) else []
        for item in masks:
            if isinstance(item, dict):
                scores.append(item.get("score"))

        return {
            "result_image": result_image,
            "mask": mask,
            "mask_image": mask_image,
            "info": {
                "index": idx,
                "request_id": result.get("request_id"),
                "count": result.get("count"),
                "image_size": result.get("image_size"),
                "bboxes": bboxes,
                "scores": scores,
            },
        }

    def _run_concurrent(self, tasks, max_workers):
        results = [None] * len(tasks)
        errors = []
        workers = min(len(tasks), max(1, int(max_workers)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fn, *args): idx
                for idx, fn, args in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    message = normalize_error_message(exc)
                    errors.append(f"[请求 {idx + 1}] {type(exc).__name__}: {message}")
                    print(f"[知衣SAM2抠图] 请求 {idx + 1} 失败: {message}")
                    traceback.print_exc()
        if errors:
            raise RuntimeError("SAM2 抠图失败:\n" + "\n".join(errors))
        return results

    def segment(
        self,
        images,
        bboxes,
        service_url,
        dilate,
        blur,
        background,
        background_color,
        max_concurrency,
        timeout,
        invert_output=False,
        health_check=False,
    ):
        service_url = (service_url or FD_SAM2_SEGMENT_URL or "").strip()
        if not service_url:
            raise RuntimeError("未配置 SAM2 服务地址，请设置 FD_SAM2_SEGMENT_URL 或在节点 service_url 中填写完整接口地址")

        image_tensors = self._expand_images(images)
        if not image_tensors:
            raise RuntimeError("未提供图片")

        bbox_groups = self._normalize_bboxes_batch(bboxes, len(image_tensors))
        if health_check and any(bbox_groups):
            self._check_health(service_url, timeout)

        tasks = []
        for idx, image_tensor in enumerate(image_tensors):
            if bbox_groups[idx]:
                task = (
                    idx,
                    self._single_request,
                    (
                        idx,
                        image_tensor,
                        bbox_groups[idx],
                        service_url,
                        dilate,
                        blur,
                        background,
                        background_color,
                        invert_output,
                        timeout,
                    ),
                )
            else:
                task = (idx, self._empty_bbox_result, (idx, image_tensor))
            tasks.append(task)

        print(f"[知衣SAM2抠图] 处理 {len(tasks)} 张图片，并发上限 {max_concurrency}")
        results = self._run_concurrent(tasks, max_concurrency)
        info = {
            "service_url": service_url,
            "background": background,
            "dilate": int(dilate),
            "blur": int(blur),
            "items": [result["info"] for result in results],
        }

        return (
            torch.cat([result["result_image"] for result in results], dim=0),
            torch.cat([result["mask"] for result in results], dim=0),
            torch.cat([result["mask_image"] for result in results], dim=0),
            json.dumps(info, ensure_ascii=False),
        )

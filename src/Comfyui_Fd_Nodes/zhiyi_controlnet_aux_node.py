import base64
import io
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse

import numpy as np
import requests
import torch
from PIL import Image

from .config import (
    CUSTOM_SERVICE_URL_PRESET,
    DWPOSE_SERVICE_URL_PRESETS,
    FD_CONTROLNET_AUX_DEPTH_ANYTHING_V2_URL,
    FD_CONTROLNET_AUX_LINEART_URL,
)
from .utils.error_utils import ERROR_TIMEOUT, normalize_error_message
from .utils.logging_utils import configure_default_logging
from .utils.service_url import resolve_service_url_preset, service_url_preset_options

configure_default_logging()
logger = logging.getLogger(__name__)


DEFAULT_LINEART_URL = "http://model-api-dwpose-svc.online-server-gray:8001/v1/lineart"
DEFAULT_DEPTH_ANYTHING_V2_URL = "http://model-api-dwpose-svc.online-server-gray:8001/v1/depth-anything-v2"
UPSCALE_METHODS = [
    "INTER_NEAREST",
    "INTER_LINEAR",
    "INTER_AREA",
    "INTER_CUBIC",
    "INTER_LANCZOS4",
]


def _normalize_preprocess_url(service_url: str, default_url: str, endpoint: str) -> str:
    url = (service_url or default_url).strip()
    if not url:
        raise RuntimeError(f"未配置 ControlNet 预处理服务地址，请在节点 service_url 中填写 {endpoint} 接口地址")
    if "://" not in url:
        url = f"http://{url}"

    parsed = urlparse(url.rstrip("/"))
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"ControlNet 预处理服务地址无效: {service_url}")

    endpoint_path = f"/v1/{endpoint}"
    path = parsed.path.rstrip("/")
    if not path:
        path = endpoint_path
    elif path.endswith("/v1"):
        path = f"{path}/{endpoint}"
    elif not (path.endswith(endpoint_path) or path.endswith(f"/{endpoint}")):
        path = f"{path}{endpoint_path}"

    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _health_url(service_url: str) -> str:
    parsed = urlparse(service_url)
    path = parsed.path.rstrip("/")
    for suffix in ("/v1/lineart", "/lineart", "/v1/depth-anything-v2", "/depth-anything-v2"):
        if path.endswith(suffix):
            base_path = path[:-len(suffix)].rstrip("/")
            health_path = f"{base_path}/health" if base_path else "/health"
            return urlunparse((parsed.scheme, parsed.netloc, health_path, "", "", ""))
    return urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))


class _ControlNetAuxApiBase:
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

    def _normalize_image_batch(self, image):
        if not isinstance(image, torch.Tensor):
            raise RuntimeError(f"图片必须是 torch.Tensor，实际为 {type(image).__name__}")
        if image.ndim == 4:
            return image
        if image.ndim == 3:
            return image.unsqueeze(0)
        raise RuntimeError(f"图片 tensor 维度错误: {image.ndim}")

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
            raise RuntimeError(f"ControlNet 预处理响应缺少 {field_name}")
        if "," in value and value.startswith("data:"):
            encoded = value.split(",", 1)[1]
        else:
            encoded = value
        try:
            payload = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(payload))
            image.load()
            return image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"ControlNet 预处理响应 {field_name} 不是有效 PNG base64") from exc

    def _pil_to_image_tensor(self, image):
        arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32)).unsqueeze(0)

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

    def _post_preprocess(self, service_url, payload, timeout):
        response = self._request_with_retry(
            "POST",
            service_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
            retry_count=1,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"ControlNet 预处理 API 请求失败: {response.status_code}\n{self._extract_response_error(response)}")
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"ControlNet 预处理 API 响应不是 JSON: {response.text[:500]}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"ControlNet 预处理 API 响应格式错误: 期望 dict，实际 {type(data).__name__}")
        if not data.get("processed_image"):
            raise RuntimeError("ControlNet 预处理 API 响应缺少 processed_image")
        return data

    def _check_health(self, service_url, timeout, health_key):
        health_url = _health_url(service_url)
        response = self._request_with_retry("GET", health_url, timeout=timeout, retry_count=0)
        if response.status_code >= 400:
            raise RuntimeError(f"ControlNet 预处理 health check 失败: {response.status_code}\n{response.text[:500]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"ControlNet 预处理 health check 响应不是 JSON: {response.text[:500]}") from exc
        section = data.get(health_key)
        if not isinstance(section, dict):
            raise RuntimeError(f"ControlNet 预处理 health check 缺少 {health_key} 信息: {json.dumps(data, ensure_ascii=False)[:1000]}")
        if section.get("loaded") is not True:
            raise RuntimeError(f"ControlNet 预处理服务未就绪: {json.dumps(section, ensure_ascii=False)[:1000]}")

    def _build_common_payload(self, image_data_url, resolution, upscale_method):
        if upscale_method not in UPSCALE_METHODS:
            raise RuntimeError(f"upscale_method 无效: {upscale_method}")
        return {
            "image": image_data_url,
            "resolution": int(resolution),
            "upscale_method": upscale_method,
            "return_image": True,
            "image_format": "png_base64",
        }

    def _run_concurrent(self, tasks, max_workers, label):
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
                except requests.exceptions.Timeout as exc:
                    message = normalize_error_message(
                        exc,
                        category=ERROR_TIMEOUT,
                        fallback_detail="ControlNet preprocessor request timed out",
                    )
                    errors.append(f"[请求 {idx + 1}] {type(exc).__name__}: {message}")
                    print(f"[{label}] 请求 {idx + 1} 失败: {message}")
                    traceback.print_exc()
                except Exception as exc:
                    message = normalize_error_message(exc)
                    errors.append(f"[请求 {idx + 1}] {type(exc).__name__}: {message}")
                    print(f"[{label}] 请求 {idx + 1} 失败: {message}")
                    traceback.print_exc()
        if errors:
            raise RuntimeError(f"{label} 失败:\n" + "\n".join(errors))
        return results

    def _single_request(self, idx, image_tensor, service_url, payload_extra, timeout):
        original_image = self._image_tensor_to_rgb_pil(image_tensor)
        payload = self._build_common_payload(
            self._tensor_to_png_data_url(image_tensor),
            payload_extra["resolution"],
            payload_extra["upscale_method"],
        )
        payload.update(payload_extra["extra"])
        logger.info("Calling %s API idx=%s service_url=%s resolution=%s", payload_extra["label"], idx, service_url, payload_extra["resolution"])
        result = self._post_preprocess(service_url, payload, timeout)

        processed_image = self._decode_png_data_url(result["processed_image"], "processed_image")
        if processed_image.size != original_image.size:
            processed_image = processed_image.resize(original_image.size, Image.Resampling.BILINEAR)

        return {
            "image": self._pil_to_image_tensor(processed_image),
            "info": {
                "index": idx,
                "request_id": result.get("request_id"),
                "image_size": result.get("image_size"),
                "inference_size": result.get("inference_size"),
                "output_size": result.get("output_size"),
                "preprocessor": result.get("preprocessor"),
                "model_name": result.get("model_name"),
                "model_type": result.get("model_type"),
                "model_filename": result.get("model_filename"),
                "device": result.get("device"),
                "inference_ms": result.get("inference_ms"),
                "total_ms": result.get("total_ms"),
            },
        }

    def _execute(
        self,
        *,
        image,
        service_url,
        endpoint,
        default_url,
        health_key,
        label,
        resolution,
        upscale_method,
        max_concurrency,
        timeout,
        health_check,
        node_switch,
        service_url_preset,
        extra,
        info_extra,
    ):
        if node_switch == 1:
            return (
                self._normalize_image_batch(image),
                json.dumps({"skipped": True, "reason": "node_switch"}, ensure_ascii=False),
            )

        selected_service_url = resolve_service_url_preset(
            service_url,
            service_url_preset,
            DWPOSE_SERVICE_URL_PRESETS,
        )
        final_service_url = _normalize_preprocess_url(selected_service_url, default_url, endpoint)
        image_tensors = self._expand_images(image)
        if not image_tensors:
            raise RuntimeError("未提供图片")
        if health_check:
            self._check_health(final_service_url, timeout, health_key)

        payload_extra = {
            "label": label,
            "resolution": resolution,
            "upscale_method": upscale_method,
            "extra": extra,
        }
        tasks = []
        for idx, image_tensor in enumerate(image_tensors):
            tasks.append((idx, self._single_request, (idx, image_tensor, final_service_url, payload_extra, timeout)))

        print(f"[{label}] 处理 {len(tasks)} 张图片，并发上限 {max_concurrency}")
        results = self._run_concurrent(tasks, max_concurrency, label)
        info = {
            "service_url": final_service_url,
            "resolution": int(resolution),
            "upscale_method": upscale_method,
            **info_extra,
            "items": [result["info"] for result in results],
        }
        return (
            torch.cat([result["image"] for result in results], dim=0),
            json.dumps(info, ensure_ascii=False),
        )


class ZhiYiLineArtPreprocessorNode(_ControlNetAuxApiBase):
    """知衣 LineArt 线稿预处理节点 - 纯 API 调用，不加载本地模型。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "service_url": ("STRING", {
                    "default": FD_CONTROLNET_AUX_LINEART_URL or DEFAULT_LINEART_URL,
                    "multiline": False,
                    "tooltip": "LineArt 线稿预处理接口地址，例如 http://host:8001/v1/lineart",
                }),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "coarse": ("BOOLEAN", {"default": False}),
                "upscale_method": (UPSCALE_METHODS, {"default": "INTER_CUBIC"}),
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
                "health_check": ("BOOLEAN", {"default": False}),
                "node_switch": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "service_url_preset": (service_url_preset_options(DWPOSE_SERVICE_URL_PRESETS), {
                    "default": CUSTOM_SERVICE_URL_PRESET,
                    "tooltip": "选择部署预设时忽略 service_url；选择自定义时使用 service_url 文本框",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "JSON")
    RETURN_NAMES = ("LINEART_IMAGE", "INFO")
    FUNCTION = "preprocess"
    CATEGORY = "知衣/ControlNet预处理"
    OUTPUT_NODE = False

    def preprocess(
        self,
        image,
        service_url,
        resolution,
        coarse,
        upscale_method,
        max_concurrency,
        timeout,
        health_check=False,
        node_switch=0,
        service_url_preset=None,
    ):
        return self._execute(
            image=image,
            service_url=service_url,
            endpoint="lineart",
            default_url=FD_CONTROLNET_AUX_LINEART_URL or DEFAULT_LINEART_URL,
            health_key="lineart",
            label="知衣LineArt线稿预处理",
            resolution=resolution,
            upscale_method=upscale_method,
            max_concurrency=max_concurrency,
            timeout=timeout,
            health_check=health_check,
            node_switch=node_switch,
            service_url_preset=service_url_preset,
            extra={"coarse": bool(coarse)},
            info_extra={"coarse": bool(coarse)},
        )


class ZhiYiDepthAnythingV2PreprocessorNode(_ControlNetAuxApiBase):
    """知衣 Depth Anything V2 深度图预处理节点 - 纯 API 调用，不加载本地模型。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "service_url": ("STRING", {
                    "default": FD_CONTROLNET_AUX_DEPTH_ANYTHING_V2_URL or DEFAULT_DEPTH_ANYTHING_V2_URL,
                    "multiline": False,
                    "tooltip": "Depth Anything V2 预处理接口地址，例如 http://host:8001/v1/depth-anything-v2",
                }),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "max_depth": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01,
                }),
                "upscale_method": (UPSCALE_METHODS, {"default": "INTER_CUBIC"}),
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
                "health_check": ("BOOLEAN", {"default": False}),
                "node_switch": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "service_url_preset": (service_url_preset_options(DWPOSE_SERVICE_URL_PRESETS), {
                    "default": CUSTOM_SERVICE_URL_PRESET,
                    "tooltip": "选择部署预设时忽略 service_url；选择自定义时使用 service_url 文本框",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "JSON")
    RETURN_NAMES = ("DEPTH_IMAGE", "INFO")
    FUNCTION = "preprocess"
    CATEGORY = "知衣/ControlNet预处理"
    OUTPUT_NODE = False

    def preprocess(
        self,
        image,
        service_url,
        resolution,
        max_depth,
        upscale_method,
        max_concurrency,
        timeout,
        health_check=False,
        node_switch=0,
        service_url_preset=None,
    ):
        return self._execute(
            image=image,
            service_url=service_url,
            endpoint="depth-anything-v2",
            default_url=FD_CONTROLNET_AUX_DEPTH_ANYTHING_V2_URL or DEFAULT_DEPTH_ANYTHING_V2_URL,
            health_key="depth_anything_v2",
            label="知衣DepthAnythingV2深度图预处理",
            resolution=resolution,
            upscale_method=upscale_method,
            max_concurrency=max_concurrency,
            timeout=timeout,
            health_check=health_check,
            node_switch=node_switch,
            service_url_preset=service_url_preset,
            extra={"max_depth": float(max_depth)},
            info_extra={"max_depth": float(max_depth)},
        )

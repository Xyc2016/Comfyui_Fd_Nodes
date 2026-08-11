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

from .config import CUSTOM_SERVICE_URL_PRESET, DWPOSE_SERVICE_URL_PRESETS, FD_DWPOSE_POSE_URL
from .utils.error_utils import ERROR_TIMEOUT, normalize_error_message
from .utils.logging_utils import configure_default_logging
from .utils.service_url import resolve_service_url_preset, service_url_preset_options

configure_default_logging()
logger = logging.getLogger(__name__)


DEFAULT_DWPOSE_POSE_URL = "http://model-api-dwpose-svc.online-server-gray:8001/v1/pose"
UPSCALE_METHODS = [
    "INTER_NEAREST",
    "INTER_LINEAR",
    "INTER_AREA",
    "INTER_CUBIC",
    "INTER_LANCZOS4",
]


def _normalize_pose_url(service_url: str) -> str:
    url = (service_url or FD_DWPOSE_POSE_URL or DEFAULT_DWPOSE_POSE_URL).strip()
    if not url:
        raise RuntimeError("未配置 DWPose 服务地址，请设置 FD_DWPOSE_POSE_URL 或在节点 service_url 中填写完整接口地址")
    if "://" not in url:
        url = f"http://{url}"

    parsed = urlparse(url.rstrip("/"))
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"DWPose 服务地址无效: {service_url}")

    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1/pose"
    elif not (path.endswith("/v1/pose") or path.endswith("/pose")):
        path = f"{path}/v1/pose"

    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _health_url(service_url: str) -> str:
    pose_url = _normalize_pose_url(service_url)
    parsed = urlparse(pose_url)
    path = parsed.path.rstrip("/")
    for suffix in ("/v1/pose", "/pose"):
        if path.endswith(suffix):
            base_path = path[:-len(suffix)].rstrip("/")
            health_path = f"{base_path}/health" if base_path else "/health"
            return urlunparse((parsed.scheme, parsed.netloc, health_path, "", "", ""))
    return urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))


class ZhiYiDWPoseDetectNode:
    """知衣 DWPose 姿态检测节点 - 纯 API 调用，不加载本地模型。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "service_url": ("STRING", {
                    "default": FD_DWPOSE_POSE_URL or DEFAULT_DWPOSE_POSE_URL,
                    "multiline": False,
                    "tooltip": "DWPose 姿态检测接口地址，例如 http://host:8001/v1/pose",
                }),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "detect_body": ("BOOLEAN", {"default": True}),
                "detect_hand": ("BOOLEAN", {"default": True}),
                "detect_face": ("BOOLEAN", {"default": False}),
                "xinsr_stick_scaling": ("BOOLEAN", {"default": False}),
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

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "JSON")
    RETURN_NAMES = ("POSE_IMAGE", "POSE_KEYPOINT", "INFO")
    FUNCTION = "detect_pose"
    CATEGORY = "知衣/姿态检测"
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
            raise RuntimeError(f"DWPose 响应缺少 {field_name}")
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
            raise RuntimeError(f"DWPose 响应 {field_name} 不是有效 PNG base64") from exc

    def _pil_to_image_tensor(self, image):
        arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32)).unsqueeze(0)

    def _build_payload(
        self,
        image_data_url,
        resolution,
        detect_body,
        detect_hand,
        detect_face,
        xinsr_stick_scaling,
        upscale_method,
        return_pose_image=True,
        return_openpose_json=True,
    ):
        if upscale_method not in UPSCALE_METHODS:
            raise RuntimeError(f"upscale_method 无效: {upscale_method}")
        return {
            "image": image_data_url,
            "resolution": int(resolution),
            "detect_body": bool(detect_body),
            "detect_hand": bool(detect_hand),
            "detect_face": bool(detect_face),
            "xinsr_stick_scaling": bool(xinsr_stick_scaling),
            "upscale_method": upscale_method,
            "return_pose_image": bool(return_pose_image),
            "return_openpose_json": bool(return_openpose_json),
            "image_format": "png_base64",
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

    def _post_pose(
        self,
        service_url,
        payload,
        timeout,
        require_pose_image=True,
    ):
        response = self._request_with_retry(
            "POST",
            service_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
            retry_count=1,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"DWPose API 请求失败: {response.status_code}\n{self._extract_response_error(response)}")
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"DWPose API 响应不是 JSON: {response.text[:500]}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"DWPose API 响应格式错误: 期望 dict，实际 {type(data).__name__}")
        if require_pose_image and not data.get("pose_image"):
            raise RuntimeError("DWPose API 响应缺少 pose_image")
        return data

    def _check_health(self, service_url, timeout):
        health_url = _health_url(service_url)
        response = self._request_with_retry("GET", health_url, timeout=timeout, retry_count=0)
        if response.status_code >= 400:
            raise RuntimeError(f"DWPose health check 失败: {response.status_code}\n{response.text[:500]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"DWPose health check 响应不是 JSON: {response.text[:500]}") from exc
        if data.get("status") != "ok" or data.get("loaded") is not True:
            raise RuntimeError(f"DWPose 服务未就绪: {json.dumps(data, ensure_ascii=False)[:1000]}")

    def _single_request(
        self,
        idx,
        image_tensor,
        service_url,
        resolution,
        detect_body,
        detect_hand,
        detect_face,
        xinsr_stick_scaling,
        upscale_method,
        timeout,
        return_pose_image=True,
        return_openpose_json=True,
    ):
        original_image = self._image_tensor_to_rgb_pil(image_tensor)
        image_data_url = self._tensor_to_png_data_url(image_tensor)
        payload = self._build_payload(
            image_data_url,
            resolution,
            detect_body,
            detect_hand,
            detect_face,
            xinsr_stick_scaling,
            upscale_method,
            return_pose_image,
            return_openpose_json,
        )
        logger.info("Calling DWPose API idx=%s service_url=%s resolution=%s", idx, service_url, resolution)
        result = self._post_pose(
            service_url,
            payload,
            timeout,
            require_pose_image=return_pose_image,
        )

        if return_pose_image and result.get("pose_image"):
            pose_image = self._decode_png_data_url(result["pose_image"], "pose_image")
            if pose_image.size != original_image.size:
                pose_image = pose_image.resize(original_image.size, Image.Resampling.BILINEAR)
            pose_image_tensor = self._pil_to_image_tensor(pose_image)
        else:
            pose_image_tensor = image_tensor

        openpose_json = result.get("openpose_json")
        if not isinstance(openpose_json, dict):
            openpose_json = {}
        if not return_openpose_json:
            openpose_json = []

        return {
            "pose_image": pose_image_tensor,
            "openpose_json": openpose_json,
            "info": {
                "index": idx,
                "request_id": result.get("request_id"),
                "people_count": result.get("people_count"),
                "image_size": result.get("image_size"),
                "inference_size": result.get("inference_size"),
                "model_name": result.get("model_name"),
                "device": result.get("device"),
                "inference_ms": result.get("inference_ms"),
                "total_ms": result.get("total_ms"),
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
                except requests.exceptions.Timeout as exc:
                    message = normalize_error_message(
                        exc,
                        category=ERROR_TIMEOUT,
                        fallback_detail="DWPose request timed out",
                    )
                    errors.append(f"[请求 {idx + 1}] {type(exc).__name__}: {message}")
                    print(f"[知衣DWPose姿态检测] 请求 {idx + 1} 失败: {message}")
                    traceback.print_exc()
                except Exception as exc:
                    message = normalize_error_message(exc)
                    errors.append(f"[请求 {idx + 1}] {type(exc).__name__}: {message}")
                    print(f"[知衣DWPose姿态检测] 请求 {idx + 1} 失败: {message}")
                    traceback.print_exc()
        if errors:
            raise RuntimeError("DWPose 姿态检测失败:\n" + "\n".join(errors))
        return results

    def detect_pose(
        self,
        image,
        service_url,
        resolution,
        detect_body,
        detect_hand,
        detect_face,
        xinsr_stick_scaling,
        upscale_method,
        max_concurrency,
        timeout,
        health_check=False,
        node_switch=0,
        service_url_preset=None,
    ):
        if node_switch == 1:
            return (
                self._normalize_image_batch(image),
                [],
                json.dumps({"skipped": True, "reason": "node_switch"}, ensure_ascii=False),
            )

        selected_service_url = resolve_service_url_preset(
            service_url,
            service_url_preset,
            DWPOSE_SERVICE_URL_PRESETS,
        )
        final_service_url = _normalize_pose_url(selected_service_url)
        image_tensors = self._expand_images(image)
        if not image_tensors:
            raise RuntimeError("未提供图片")
        if health_check:
            self._check_health(final_service_url, timeout)

        tasks = []
        for idx, image_tensor in enumerate(image_tensors):
            tasks.append((
                idx,
                self._single_request,
                (
                    idx,
                    image_tensor,
                    final_service_url,
                    resolution,
                    detect_body,
                    detect_hand,
                    detect_face,
                    xinsr_stick_scaling,
                    upscale_method,
                    timeout,
                ),
            ))

        print(f"[知衣DWPose姿态检测] 处理 {len(tasks)} 张图片，并发上限 {max_concurrency}")
        results = self._run_concurrent(tasks, max_concurrency)
        info = {
            "service_url": final_service_url,
            "resolution": int(resolution),
            "detect_body": bool(detect_body),
            "detect_hand": bool(detect_hand),
            "detect_face": bool(detect_face),
            "xinsr_stick_scaling": bool(xinsr_stick_scaling),
            "upscale_method": upscale_method,
            "items": [result["info"] for result in results],
        }

        return (
            torch.cat([result["pose_image"] for result in results], dim=0),
            [result["openpose_json"] for result in results],
            json.dumps(info, ensure_ascii=False),
        )

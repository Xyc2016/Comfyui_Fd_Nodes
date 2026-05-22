import io
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import oss2
import requests
import torch
from PIL import Image, ImageColor, ImageFilter

from .config import (
    FD_OSS_ACCESS_KEY_ID,
    FD_OSS_ACCESS_KEY_SECRET,
    FD_OSS_BUCKET_NAME,
    FD_OSS_ENDPOINT,
    FD_OSS_URL_PATH_PREFIX_REMOVE_BG,
    FD_OSS_URL_PREFIX,
    FD_REMOVE_BG_BY_MEITU_URL,
)
from .utils.common_util import bytes_calculate_hex_md5, bytesio_to_image_tensor
from .utils.error_utils import normalize_error_message
from .utils.logging_utils import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)


class ZhiYiRemoveBgByMeituNode:
    """知衣美图服装抠图节点 - 纯 API 调用，不加载本地模型。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "service_url": ("STRING", {
                    "default": FD_REMOVE_BG_BY_MEITU_URL or "",
                    "multiline": False,
                    "tooltip": "完整下游接口地址，例如 {image.detail.url}/image/remove_bg_by_meitu",
                }),
                "repaint_edge": ("BOOLEAN", {"default": True}),
                "edge_thickness": ("INT", {
                    "default": 40,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                }),
                "mask_blur": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                }),
                "mask_offset": ("INT", {
                    "default": 0,
                    "min": -64,
                    "max": 64,
                    "step": 1,
                }),
                "invert_output": ("BOOLEAN", {"default": False}),
                "background": (["Alpha", "Color"], {"default": "Alpha"}),
                "background_color": ("STRING", {"default": "#222222"}),
                "max_concurrency": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                }),
                "timeout": ("INT", {
                    "default": 120,
                    "min": 10,
                    "max": 600,
                    "step": 10,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "RED_EDGE_IMAGE")
    FUNCTION = "remove_bg"
    CATEGORY = "知衣/抠图"
    OUTPUT_NODE = False

    def __init__(self):
        self.bucket = None
        self._bucket_lock = threading.Lock()
        self.oss_url_prefix = FD_OSS_URL_PREFIX

    def _get_bucket(self):
        if self.bucket is not None:
            return self.bucket

        with self._bucket_lock:
            if self.bucket is not None:
                return self.bucket

            missing = [
                name
                for name, value in {
                    "FD_OSS_ACCESS_KEY_ID": FD_OSS_ACCESS_KEY_ID,
                    "FD_OSS_ACCESS_KEY_SECRET": FD_OSS_ACCESS_KEY_SECRET,
                    "FD_OSS_BUCKET_NAME": FD_OSS_BUCKET_NAME,
                    "FD_OSS_ENDPOINT": FD_OSS_ENDPOINT,
                    "FD_OSS_URL_PREFIX": FD_OSS_URL_PREFIX,
                }.items()
                if not value
            ]
            if missing:
                raise RuntimeError(f"未配置 OSS 上传参数: {', '.join(missing)}")

            auth = oss2.Auth(FD_OSS_ACCESS_KEY_ID, FD_OSS_ACCESS_KEY_SECRET)
            self.bucket = oss2.Bucket(
                auth=auth,
                bucket_name=FD_OSS_BUCKET_NAME,
                endpoint=FD_OSS_ENDPOINT,
                connect_timeout=30,
            )
            self.oss_url_prefix = FD_OSS_URL_PREFIX
            return self.bucket

    def _expand_images(self, images):
        if images is None:
            return []
        if images.ndim == 4:
            return [images[i:i + 1] for i in range(images.shape[0])]
        if images.ndim == 3:
            return [images.unsqueeze(0)]
        raise RuntimeError(f"图片 tensor 维度错误: {images.ndim}")

    def _tensor_to_png_bytes(self, image_tensor):
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]
        arr = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(arr[..., :3]).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def _upload_image(self, image_tensor):
        image_bytes = self._tensor_to_png_bytes(image_tensor)
        file_oss_path = f"{FD_OSS_URL_PATH_PREFIX_REMOVE_BG}/{bytes_calculate_hex_md5(image_bytes)}.png"
        self._get_bucket().put_object(file_oss_path, image_bytes)
        image_url = f"{self.oss_url_prefix}{file_oss_path}"
        print(f"[知衣美图服装抠图] upload {file_oss_path}")
        return image_url

    def _build_payload(self, image_url, repaint_edge, edge_thickness):
        return {
            "image_url": image_url,
            "repaint_edge": 1 if repaint_edge else 0,
            "edge_thickness": int(edge_thickness),
        }

    def _unwrap_response_data(self, result):
        if not isinstance(result, dict):
            raise RuntimeError(f"API 响应格式错误: 期望 dict，实际 {type(result).__name__}")
        data = result.get("data")
        if isinstance(data, dict):
            return data
        return result

    def _extract_error_message(self, data):
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            message = error.get("message")
            code = error.get("code")
            if message and code:
                return f"{code}: {message}"
            if message:
                return str(message)
            if code:
                return str(code)
        message = data.get("message") if isinstance(data, dict) else None
        if message:
            return str(message)
        return "API 返回失败"

    def _extract_urls(self, result):
        data = self._unwrap_response_data(result)
        if data.get("status") is not True:
            raise RuntimeError(self._extract_error_message(data))

        result_url = data.get("result_url") or data.get("resultUrl")
        main_mask_url = data.get("main_mask_url") or data.get("mainMaskUrl")
        red_edge_image_url = (
            data.get("red_edge_image_url")
            or data.get("redEdgeImageUrl")
            or data.get("RedEdgeImageUrl")
        )
        if not result_url:
            raise RuntimeError("API 响应缺少 result_url")
        if not main_mask_url:
            raise RuntimeError("API 响应缺少 main_mask_url")
        return {
            "result_url": result_url,
            "main_mask_url": main_mask_url,
            "red_edge_image_url": red_edge_image_url,
        }

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

    def _post_remove_bg(self, service_url, payload, timeout):
        response = self._request_with_retry(
            "POST",
            service_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
            retry_count=1,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"API 请求失败: {response.status_code}\n{response.text[:1000]}")
        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError(f"API 响应不是 JSON: {response.text[:500]}") from exc
        return self._extract_urls(result)

    def _download_bytes(self, url, timeout):
        response = self._request_with_retry("GET", url, timeout=timeout, retry_count=1)
        if response.status_code >= 400:
            raise RuntimeError(f"下载图片失败: {response.status_code}, url={url}")
        return response.content

    def _download_image_tensor(self, url, timeout, size=None):
        tensor = bytesio_to_image_tensor(BytesIO(self._download_bytes(url, timeout)), mode="RGB")
        if size is not None:
            tensor = self._resize_image_tensor(tensor, size)
        return tensor

    def _download_mask_image(self, url, timeout, size=None):
        image = Image.open(BytesIO(self._download_bytes(url, timeout))).convert("L")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        return image

    def _resize_image_tensor(self, image_tensor, size):
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]
        arr = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(arr[..., :3]).convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        image_array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_array).unsqueeze(0)

    def _normalize_filter_size(self, amount):
        radius = int(abs(amount))
        if radius <= 0:
            return 0
        return radius * 2 + 1

    def _process_mask_image(self, mask_image, mask_blur, mask_offset, invert_output):
        image = mask_image.convert("L")
        filter_size = self._normalize_filter_size(mask_offset)
        if filter_size:
            if mask_offset > 0:
                image = image.filter(ImageFilter.MaxFilter(filter_size))
            else:
                image = image.filter(ImageFilter.MinFilter(filter_size))
        if mask_blur > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(mask_blur)))

        mask_array = np.array(image).astype(np.float32) / 255.0
        if invert_output:
            mask_array = 1.0 - mask_array
        mask_array = np.clip(mask_array, 0.0, 1.0)
        return torch.from_numpy(mask_array).unsqueeze(0)

    def _mask_to_image_tensor(self, mask_tensor):
        mask = mask_tensor
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        image = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        return image.to(dtype=torch.float32)

    def _parse_hex_color(self, background_color):
        try:
            return ImageColor.getrgb((background_color or "#222222").strip())
        except ValueError as exc:
            raise RuntimeError(f"background_color 不是有效 hex 颜色: {background_color}") from exc

    def _compose_color_background(self, original_image_tensor, mask_tensor, background_color):
        if original_image_tensor.ndim == 4:
            original = original_image_tensor[0]
        else:
            original = original_image_tensor
        original_np = original.detach().cpu().numpy().astype(np.float32)
        mask_np = mask_tensor[0].detach().cpu().numpy().astype(np.float32)
        color = np.array(self._parse_hex_color(background_color), dtype=np.float32) / 255.0
        background = np.ones_like(original_np[..., :3], dtype=np.float32) * color
        result = original_np[..., :3] * mask_np[..., None] + background * (1.0 - mask_np[..., None])
        return torch.from_numpy(np.clip(result, 0.0, 1.0)).unsqueeze(0)

    def _single_request(
        self,
        idx,
        image_tensor,
        service_url,
        repaint_edge,
        edge_thickness,
        mask_blur,
        mask_offset,
        invert_output,
        background,
        background_color,
        timeout,
    ):
        image_url = self._upload_image(image_tensor)
        payload = self._build_payload(image_url, repaint_edge, edge_thickness)
        print(f"[知衣美图服装抠图] 请求 {idx + 1}: {payload}")
        logger.info("Calling remove-bg-by-meitu API with payload=%s", payload)

        urls = self._post_remove_bg(service_url, payload, timeout)
        if image_tensor.ndim == 4:
            height, width = image_tensor.shape[1], image_tensor.shape[2]
        else:
            height, width = image_tensor.shape[0], image_tensor.shape[1]
        size = (int(width), int(height))

        raw_mask_image = self._download_mask_image(urls["main_mask_url"], timeout, size=size)
        mask = self._process_mask_image(raw_mask_image, mask_blur, mask_offset, invert_output)
        mask_image = self._mask_to_image_tensor(mask)

        if background == "Color":
            result_image = self._compose_color_background(image_tensor, mask, background_color)
        else:
            result_image = self._download_image_tensor(urls["result_url"], timeout, size=size)

        red_edge_url = urls.get("red_edge_image_url")
        if red_edge_url:
            red_edge_image = self._download_image_tensor(red_edge_url, timeout, size=size)
        else:
            print(f"[知衣美图服装抠图] 请求 {idx + 1}: red_edge_image_url 缺失，使用 MASK_IMAGE 兜底")
            red_edge_image = mask_image

        return {
            "result_image": result_image,
            "mask": mask,
            "mask_image": mask_image,
            "red_edge_image": red_edge_image,
            "image_url": image_url,
            **urls,
        }

    def _run_concurrent(self, tasks, max_workers):
        results = [None] * len(tasks)
        log_lines = []
        last_error_message = None
        workers = min(len(tasks), max(1, int(max_workers)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fn, *args): idx
                for idx, fn, args in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                    log_lines.append(
                        f"[请求 {idx + 1}] 成功 input={result['image_url']} result={result['result_url']} mask={result['main_mask_url']}"
                    )
                except Exception as exc:
                    last_error_message = normalize_error_message(exc)
                    msg = f"[请求 {idx + 1}] 失败: {type(exc).__name__}: {last_error_message}"
                    log_lines.append(msg)
                    print(f"[知衣美图服装抠图] {msg}")
                    traceback.print_exc()
        return results, log_lines, last_error_message

    def remove_bg(
        self,
        images,
        service_url,
        repaint_edge,
        edge_thickness,
        mask_blur,
        mask_offset,
        invert_output,
        background,
        background_color,
        max_concurrency,
        timeout,
    ):
        service_url = (service_url or "").strip()
        if not service_url:
            raise RuntimeError("未配置美图抠图服务地址，请设置 FD_REMOVE_BG_BY_MEITU_URL 或在节点 service_url 中填写完整接口地址")

        image_tensors = self._expand_images(images)
        if not image_tensors:
            raise RuntimeError("未提供图片")

        tasks = []
        for idx, image_tensor in enumerate(image_tensors):
            tasks.append((
                idx,
                self._single_request,
                (
                    idx,
                    image_tensor,
                    service_url,
                    repaint_edge,
                    edge_thickness,
                    mask_blur,
                    mask_offset,
                    invert_output,
                    background,
                    background_color,
                    timeout,
                ),
            ))

        print(f"[知衣美图服装抠图] 并发处理 {len(tasks)} 张图片，并发上限 {max_concurrency}")
        results, log_lines, last_error_message = self._run_concurrent(tasks, max_concurrency)
        successful = [result for result in results if result is not None]
        if not successful:
            log_text = "\n".join(log_lines)
            raise RuntimeError(last_error_message or f"所有请求均失败，无图片返回\n{log_text}")

        if len(successful) < len(tasks):
            print(f"[知衣美图服装抠图] 部分成功 {len(successful)}/{len(tasks)}")
            print("\n".join(log_lines))
        else:
            print(f"[知衣美图服装抠图] 全部成功 {len(successful)}/{len(tasks)}")

        return (
            torch.cat([result["result_image"] for result in successful], dim=0),
            torch.cat([result["mask"] for result in successful], dim=0),
            torch.cat([result["mask_image"] for result in successful], dim=0),
            torch.cat([result["red_edge_image"] for result in successful], dim=0),
        )

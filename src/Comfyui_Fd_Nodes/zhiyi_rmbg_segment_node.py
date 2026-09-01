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
from PIL import Image, ImageColor

from .config import (
    CUSTOM_SERVICE_URL_PRESET,
    FD_BODY_SEGMENT_URL,
    FD_CLOTHES_SEGMENT_URL,
    FD_FASHION_SEGMENT_URL,
    FD_RMBG_URL,
    RMBG_SERVICE_URL_PRESETS,
)
from .utils.error_utils import ERROR_TIMEOUT, normalize_error_message
from .utils.logging_utils import configure_default_logging
from .utils.service_url import resolve_service_url_preset, service_url_preset_options

configure_default_logging()
logger = logging.getLogger(__name__)


DEFAULT_RMBG_URL = "http://10.1.0.230:8003/v1/rmbg"
DEFAULT_CLOTHES_SEGMENT_URL = "http://10.1.0.230:8003/v1/segment/clothes"
DEFAULT_FASHION_SEGMENT_URL = "http://10.1.0.230:8003/v1/segment/fashion"
DEFAULT_BODY_SEGMENT_URL = "http://10.1.0.230:8003/v1/segment/body"
BACKGROUND_MODES = ["Alpha", "Color"]

CLOTHES_CLASSES = [
    "Background",
    "Hat",
    "Hair",
    "Sunglasses",
    "Upper-clothes",
    "Skirt",
    "Pants",
    "Dress",
    "Belt",
    "Left-shoe",
    "Right-shoe",
    "Face",
    "Left-leg",
    "Right-leg",
    "Left-arm",
    "Right-arm",
    "Bag",
    "Scarf",
]

FASHION_CLASSES = [
    "unlabelled",
    "shirt, blouse",
    "top, t-shirt, sweatshirt",
    "sweater",
    "cardigan",
    "jacket",
    "vest",
    "pants",
    "shorts",
    "skirt",
    "coat",
    "dress",
    "jumpsuit",
    "cape",
    "glasses",
    "hat",
    "headband, head covering, hair accessory",
    "tie",
    "glove",
    "watch",
    "belt",
    "leg warmer",
    "tights, stockings",
    "sock",
    "shoe",
    "bag, wallet",
    "scarf",
    "umbrella",
    "hood",
    "collar",
    "lapel",
    "epaulette",
    "sleeve",
    "pocket",
    "neckline",
    "buckle",
    "zipper",
    "applique",
    "bead",
    "bow",
    "flower",
    "fringe",
    "ribbon",
    "rivet",
    "ruffle",
    "sequin",
    "tassel",
]

BODY_CLASSES = [
    "Hair",
    "Glasses",
    "Top-clothes",
    "Bottom-clothes",
    "Torso-skin",
    "Face",
    "Left-arm",
    "Right-arm",
    "Left-leg",
    "Right-leg",
    "Left-foot",
    "Right-foot",
]


def _normalize_rmbg_segment_url(service_url: str, default_url: str, endpoint: str) -> str:
    url = (service_url or default_url).strip()
    if not url:
        raise RuntimeError(f"未配置 RMBG/语义分割服务地址，请在节点 service_url 中填写 {endpoint} 接口地址")
    if "://" not in url:
        url = f"http://{url}"

    parsed = urlparse(url.rstrip("/"))
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"RMBG/语义分割服务地址无效: {service_url}")

    endpoint_path = f"/v1/{endpoint}"
    short_path = f"/{endpoint}"
    path = parsed.path.rstrip("/")
    if not path:
        path = endpoint_path
    elif path.endswith("/v1"):
        path = f"{path}/{endpoint}"
    elif not (path.endswith(endpoint_path) or path.endswith(short_path)):
        path = f"{path}{endpoint_path}"

    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _health_url(service_url: str) -> str:
    parsed = urlparse(service_url)
    path = parsed.path.rstrip("/")
    for suffix in (
        "/v1/rmbg",
        "/rmbg",
        "/v1/segment/clothes",
        "/segment/clothes",
        "/v1/segment/fashion",
        "/segment/fashion",
        "/v1/segment/body",
        "/segment/body",
    ):
        if path.endswith(suffix):
            base_path = path[:-len(suffix)].rstrip("/")
            health_path = f"{base_path}/health" if base_path else "/health"
            return urlunparse((parsed.scheme, parsed.netloc, health_path, "", "", ""))
    return urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))


class _RmbgSegmentApiBase:
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
            raise RuntimeError(f"RMBG/语义分割响应缺少 {field_name}")
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
            raise RuntimeError(f"RMBG/语义分割响应 {field_name} 不是有效 PNG base64") from exc

    def _parse_hex_color(self, background_color):
        try:
            return ImageColor.getrgb((background_color or "#222222").strip())
        except ValueError as exc:
            raise RuntimeError(f"background_color 不是有效 hex 颜色: {background_color}") from exc

    def _pil_to_image_tensor(self, image, alpha_background=(0, 0, 0)):
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, tuple(alpha_background) + (255,))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        arr = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32)).unsqueeze(0)

    def _mask_data_url_to_tensor(self, value, size):
        mask_image = self._decode_png_data_url(value, "mask").convert("L")
        if mask_image.size != size:
            mask_image = mask_image.resize(size, Image.Resampling.BILINEAR)
        mask_array = np.array(mask_image).astype(np.float32) / 255.0
        return torch.from_numpy(np.clip(mask_array, 0.0, 1.0).astype(np.float32)).unsqueeze(0)

    def _empty_mask_tensor(self, size):
        width, height = size
        return torch.zeros((1, height, width), dtype=torch.float32)

    def _mask_to_image_tensor(self, mask_tensor):
        mask = mask_tensor
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return mask.unsqueeze(-1).expand(-1, -1, -1, 3).to(dtype=torch.float32)

    def _compose_rgba_image(self, image_tensor, mask_tensor):
        """Create a straight-alpha RGBA tensor from an image and its mask."""
        image_batch = self._normalize_image_batch(image_tensor)
        if image_batch.shape[-1] < 3:
            raise RuntimeError(f"图片 tensor shape 错误: {tuple(image_batch.shape)}")
        rgb = image_batch[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)
        mask = mask_tensor
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[0] != rgb.shape[0] or mask.shape[1:3] != rgb.shape[1:3]:
            raise RuntimeError(
                f"遮罩 tensor shape 与图片不匹配: image={tuple(rgb.shape)}, mask={tuple(mask.shape)}"
            )
        alpha = mask.to(dtype=torch.float32).clamp(0.0, 1.0)
        rgba_rgb = rgb * (alpha.unsqueeze(-1) > 0).to(dtype=rgb.dtype)
        return torch.cat((rgba_rgb, alpha.unsqueeze(-1)), dim=-1)

    def _compose_result_image(self, original_image_tensor, mask_tensor, background, background_color):
        original = self._image_tensor_to_rgb_pil(original_image_tensor)
        original_np = np.array(original).astype(np.float32) / 255.0
        mask_np = mask_tensor[0].detach().cpu().numpy().astype(np.float32)

        if background == "Color":
            color = np.array(self._parse_hex_color(background_color), dtype=np.float32) / 255.0
            bg = np.ones_like(original_np, dtype=np.float32) * color
            result = original_np * mask_np[..., None] + bg * (1.0 - mask_np[..., None])
        else:
            result = original_np * mask_np[..., None]

        return torch.from_numpy(np.clip(result, 0.0, 1.0).astype(np.float32)).unsqueeze(0)

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

    def _post_process(self, service_url, payload, timeout, label):
        response = self._request_with_retry(
            "POST",
            service_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
            retry_count=1,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"{label} API 请求失败: {response.status_code}\n{self._extract_response_error(response)}")
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{label} API 响应不是 JSON: {response.text[:500]}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"{label} API 响应格式错误: 期望 dict，实际 {type(data).__name__}")
        if not data.get("processed_image") and not data.get("mask"):
            raise RuntimeError(f"{label} API 响应缺少 processed_image 和 mask，至少需要返回一个图像字段")
        return data

    def _check_health(self, service_url, timeout, health_key, label):
        health_url = _health_url(service_url)
        response = self._request_with_retry("GET", health_url, timeout=timeout, retry_count=0)
        if response.status_code >= 400:
            raise RuntimeError(f"{label} health check 失败: {response.status_code}\n{response.text[:500]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{label} health check 响应不是 JSON: {response.text[:500]}") from exc
        section = data.get(health_key)
        if not isinstance(section, dict):
            raise RuntimeError(f"{label} health check 缺少 {health_key} 信息: {json.dumps(data, ensure_ascii=False)[:1000]}")
        if section.get("loaded") is not True:
            raise RuntimeError(f"{label} 服务未就绪: {json.dumps(section, ensure_ascii=False)[:1000]}")

    def _parse_classes_text(self, classes, available_classes, allow_comma_split=False):
        if classes is None:
            return []
        if isinstance(classes, (list, tuple)):
            selected = [str(item).strip() for item in classes if str(item).strip()]
        else:
            text = str(classes).strip()
            if not text:
                return []
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                except ValueError as exc:
                    raise RuntimeError("classes JSON 数组格式错误") from exc
                if not isinstance(parsed, list):
                    raise RuntimeError("classes JSON 必须是字符串数组")
                selected = [str(item).strip() for item in parsed if str(item).strip()]
            else:
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if allow_comma_split and len(lines) == 1 and "," in lines[0]:
                    selected = [item.strip() for item in lines[0].split(",") if item.strip()]
                else:
                    selected = lines

        invalid = [class_name for class_name in selected if class_name not in available_classes]
        if invalid:
            supported = ", ".join(available_classes)
            raise RuntimeError(f"不支持的分割类别: {', '.join(invalid)}; 支持的类别: {supported}")
        return selected

    def _build_common_payload(
        self,
        image_data_url,
        process_res,
        mask_blur,
        mask_offset,
        invert_output,
        background,
        background_color,
        return_image,
        return_mask,
        return_mask_image,
    ):
        if background not in BACKGROUND_MODES:
            raise RuntimeError(f"background 无效: {background}")
        if background == "Color":
            self._parse_hex_color(background_color)
        return {
            "image": image_data_url,
            "process_res": int(process_res),
            "mask_blur": int(mask_blur),
            "mask_offset": int(mask_offset),
            "invert_output": bool(invert_output),
            "background": background,
            "background_color": background_color,
            "return_image": bool(return_image),
            "return_mask": bool(return_mask),
            "return_mask_image": bool(return_mask_image),
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
                        fallback_detail="RMBG/semantic segmentation request timed out",
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

    def _single_request(self, idx, image_tensor, service_url, payload_params, timeout):
        original_image = self._image_tensor_to_rgb_pil(image_tensor)
        payload = self._build_common_payload(
            self._tensor_to_png_data_url(image_tensor),
            payload_params["process_res"],
            payload_params["mask_blur"],
            payload_params["mask_offset"],
            payload_params["invert_output"],
            payload_params["background"],
            payload_params["background_color"],
            payload_params["return_image"],
            payload_params["return_mask"],
            payload_params["return_mask_image"],
        )
        payload.update(payload_params["extra"])

        label = payload_params["label"]
        logger.info("Calling %s API idx=%s service_url=%s", label, idx, service_url)
        result = self._post_process(service_url, payload, timeout, label)

        if result.get("mask"):
            mask = self._mask_data_url_to_tensor(result["mask"], original_image.size)
        else:
            mask = self._empty_mask_tensor(original_image.size)

        if result.get("processed_image"):
            processed_image = self._decode_png_data_url(result["processed_image"], "processed_image")
            if processed_image.size != original_image.size:
                processed_image = processed_image.resize(original_image.size, Image.Resampling.BILINEAR)
            alpha_background = (0, 0, 0)
            if payload_params["background"] == "Color":
                alpha_background = self._parse_hex_color(payload_params["background_color"])
            result_image = self._pil_to_image_tensor(processed_image, alpha_background=alpha_background)
        elif result.get("mask"):
            result_image = self._compose_result_image(
                image_tensor,
                mask,
                payload_params["background"],
                payload_params["background_color"],
            )
        else:
            result_image = image_tensor

        if result.get("mask_image"):
            mask_image = self._decode_png_data_url(result["mask_image"], "mask_image")
            if mask_image.size != original_image.size:
                mask_image = mask_image.resize(original_image.size, Image.Resampling.BILINEAR)
            mask_image_tensor = self._pil_to_image_tensor(mask_image)
        else:
            mask_image_tensor = self._mask_to_image_tensor(mask)

        return {
            "image": result_image,
            "mask": mask,
            "mask_image": mask_image_tensor,
            "info": {
                "index": idx,
                "request_id": result.get("request_id"),
                "image_size": result.get("image_size"),
                "inference_size": result.get("inference_size"),
                "output_size": result.get("output_size"),
                "classes": result.get("classes", []),
                "preprocessor": result.get("preprocessor"),
                "model_name": result.get("model_name"),
                "model_type": result.get("model_type"),
                "model_repo": result.get("model_repo"),
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
        process_res,
        mask_blur,
        mask_offset,
        invert_output,
        background,
        background_color,
        return_image,
        return_mask,
        return_mask_image,
        max_concurrency,
        timeout,
        health_check,
        node_switch,
        service_url_preset,
        extra,
        info_extra,
    ):
        if node_switch == 1:
            image_batch = self._normalize_image_batch(image)
            mask = torch.zeros((image_batch.shape[0], image_batch.shape[1], image_batch.shape[2]), dtype=torch.float32)
            return (
                image_batch,
                mask,
                self._mask_to_image_tensor(mask),
                json.dumps({"skipped": True, "reason": "node_switch"}, ensure_ascii=False),
                self._compose_rgba_image(image_batch, mask),
            )

        selected_service_url = resolve_service_url_preset(
            service_url,
            service_url_preset,
            RMBG_SERVICE_URL_PRESETS,
        )
        final_service_url = _normalize_rmbg_segment_url(selected_service_url, default_url, endpoint)
        image_tensors = self._expand_images(image)
        if not image_tensors:
            raise RuntimeError("未提供图片")
        if health_check:
            self._check_health(final_service_url, timeout, health_key, label)

        payload_params = {
            "label": label,
            "process_res": process_res,
            "mask_blur": mask_blur,
            "mask_offset": mask_offset,
            "invert_output": invert_output,
            "background": background,
            "background_color": background_color,
            "return_image": return_image,
            "return_mask": return_mask,
            "return_mask_image": return_mask_image,
            "extra": extra,
        }
        tasks = []
        for idx, image_tensor in enumerate(image_tensors):
            tasks.append((idx, self._single_request, (idx, image_tensor, final_service_url, payload_params, timeout)))

        print(f"[{label}] 处理 {len(tasks)} 张图片，并发上限 {max_concurrency}")
        results = self._run_concurrent(tasks, max_concurrency, label)
        info = {
            "service_url": final_service_url,
            "endpoint": f"/v1/{endpoint}",
            "process_res": int(process_res),
            "mask_blur": int(mask_blur),
            "mask_offset": int(mask_offset),
            "invert_output": bool(invert_output),
            "background": background,
            "background_color": background_color,
            "return_image": bool(return_image),
            "return_mask": bool(return_mask),
            "return_mask_image": bool(return_mask_image),
            **info_extra,
            "items": [result["info"] for result in results],
        }
        return (
            torch.cat([result["image"] for result in results], dim=0),
            torch.cat([result["mask"] for result in results], dim=0),
            torch.cat([result["mask_image"] for result in results], dim=0),
            json.dumps(info, ensure_ascii=False),
            torch.cat(
                [self._compose_rgba_image(image_tensors[idx], result["mask"]) for idx, result in enumerate(results)],
                dim=0,
            ),
        )


class ZhiYiRMBGNode(_RmbgSegmentApiBase):
    """知衣 RMBG 2.0 背景去除节点 - 纯 API 调用，不加载本地模型。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "service_url": ("STRING", {
                    "default": FD_RMBG_URL or DEFAULT_RMBG_URL,
                    "multiline": False,
                    "tooltip": "RMBG 2.0 背景去除接口地址，例如 http://host:8003/v1/rmbg",
                }),
                "process_res": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "refine_foreground": ("BOOLEAN", {"default": False}),
                "background": (BACKGROUND_MODES, {"default": "Alpha"}),
                "background_color": ("STRING", {"default": "#222222"}),
                "return_image": ("BOOLEAN", {"default": True}),
                "return_mask": ("BOOLEAN", {"default": True}),
                "return_mask_image": ("BOOLEAN", {"default": False}),
                "max_concurrency": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 1200, "step": 10}),
            },
            "optional": {
                "health_check": ("BOOLEAN", {"default": False}),
                "node_switch": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "service_url_preset": (service_url_preset_options(RMBG_SERVICE_URL_PRESETS), {
                    "default": CUSTOM_SERVICE_URL_PRESET,
                    "tooltip": "选择部署预设时忽略 service_url；选择自定义时使用 service_url 文本框",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "JSON", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "INFO", "RGBA_IMAGE")
    FUNCTION = "remove_background"
    CATEGORY = "知衣/抠图"
    OUTPUT_NODE = False

    def remove_background(
        self,
        image,
        service_url,
        process_res,
        sensitivity,
        mask_blur,
        mask_offset,
        invert_output,
        refine_foreground,
        background,
        background_color,
        return_image,
        return_mask,
        return_mask_image,
        max_concurrency,
        timeout,
        health_check=False,
        node_switch=0,
        service_url_preset=None,
    ):
        return self._execute(
            image=image,
            service_url=service_url,
            endpoint="rmbg",
            default_url=FD_RMBG_URL or DEFAULT_RMBG_URL,
            health_key="rmbg",
            label="知衣RMBG2.0背景去除",
            process_res=process_res,
            mask_blur=mask_blur,
            mask_offset=mask_offset,
            invert_output=invert_output,
            background=background,
            background_color=background_color,
            return_image=return_image,
            return_mask=return_mask,
            return_mask_image=return_mask_image,
            max_concurrency=max_concurrency,
            timeout=timeout,
            health_check=health_check,
            node_switch=node_switch,
            service_url_preset=service_url_preset,
            extra={"sensitivity": float(sensitivity), "refine_foreground": bool(refine_foreground)},
            info_extra={"sensitivity": float(sensitivity), "refine_foreground": bool(refine_foreground)},
        )


class ZhiYiClothesSegmentNode(_RmbgSegmentApiBase):
    """知衣衣物语义分割节点，支持 18 个衣物/配饰/身体区域类别。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "service_url": ("STRING", {
                    "default": FD_CLOTHES_SEGMENT_URL or DEFAULT_CLOTHES_SEGMENT_URL,
                    "multiline": False,
                    "tooltip": "衣物语义分割接口地址，例如 http://host:8003/v1/segment/clothes",
                }),
                "classes": ("STRING", {
                    "default": "Upper-clothes",
                    "multiline": True,
                    "tooltip": "每行一个类别，也支持 JSON 字符串数组；空值使用服务默认类别",
                }),
                "process_res": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "background": (BACKGROUND_MODES, {"default": "Alpha"}),
                "background_color": ("STRING", {"default": "#222222"}),
                "return_image": ("BOOLEAN", {"default": True}),
                "return_mask": ("BOOLEAN", {"default": True}),
                "return_mask_image": ("BOOLEAN", {"default": False}),
                "max_concurrency": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 1200, "step": 10}),
            },
            "optional": {
                "health_check": ("BOOLEAN", {"default": False}),
                "node_switch": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "service_url_preset": (service_url_preset_options(RMBG_SERVICE_URL_PRESETS), {
                    "default": CUSTOM_SERVICE_URL_PRESET,
                    "tooltip": "选择部署预设时忽略 service_url；选择自定义时使用 service_url 文本框",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "JSON", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "INFO", "RGBA_IMAGE")
    FUNCTION = "segment"
    CATEGORY = "知衣/语义分割"
    OUTPUT_NODE = False

    def segment(
        self,
        image,
        service_url,
        classes,
        process_res,
        mask_blur,
        mask_offset,
        invert_output,
        background,
        background_color,
        return_image,
        return_mask,
        return_mask_image,
        max_concurrency,
        timeout,
        health_check=False,
        node_switch=0,
        service_url_preset=None,
    ):
        selected_classes = self._parse_classes_text(classes, CLOTHES_CLASSES, allow_comma_split=True)
        return self._execute(
            image=image,
            service_url=service_url,
            endpoint="segment/clothes",
            default_url=FD_CLOTHES_SEGMENT_URL or DEFAULT_CLOTHES_SEGMENT_URL,
            health_key="clothes_segment",
            label="知衣衣物语义分割",
            process_res=process_res,
            mask_blur=mask_blur,
            mask_offset=mask_offset,
            invert_output=invert_output,
            background=background,
            background_color=background_color,
            return_image=return_image,
            return_mask=return_mask,
            return_mask_image=return_mask_image,
            max_concurrency=max_concurrency,
            timeout=timeout,
            health_check=health_check,
            node_switch=node_switch,
            service_url_preset=service_url_preset,
            extra={"classes": selected_classes},
            info_extra={"classes": selected_classes},
        )


class ZhiYiFashionSegmentNode(_RmbgSegmentApiBase):
    """知衣时尚单品分割节点，类别名中的逗号需按原样填写。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "service_url": ("STRING", {
                    "default": FD_FASHION_SEGMENT_URL or DEFAULT_FASHION_SEGMENT_URL,
                    "multiline": False,
                    "tooltip": "时尚单品分割接口地址，例如 http://host:8003/v1/segment/fashion",
                }),
                "classes": ("STRING", {
                    "default": "shirt, blouse",
                    "multiline": True,
                    "tooltip": "每行一个类别或填写 JSON 字符串数组；类别名里的逗号必须保留原样，不要用逗号分隔多个类别",
                }),
                "process_res": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "background": (BACKGROUND_MODES, {"default": "Alpha"}),
                "background_color": ("STRING", {"default": "#222222"}),
                "return_image": ("BOOLEAN", {"default": True}),
                "return_mask": ("BOOLEAN", {"default": True}),
                "return_mask_image": ("BOOLEAN", {"default": False}),
                "max_concurrency": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 1200, "step": 10}),
            },
            "optional": {
                "health_check": ("BOOLEAN", {"default": False}),
                "node_switch": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "service_url_preset": (service_url_preset_options(RMBG_SERVICE_URL_PRESETS), {
                    "default": CUSTOM_SERVICE_URL_PRESET,
                    "tooltip": "选择部署预设时忽略 service_url；选择自定义时使用 service_url 文本框",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "JSON", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "INFO", "RGBA_IMAGE")
    FUNCTION = "segment"
    CATEGORY = "知衣/语义分割"
    OUTPUT_NODE = False

    def segment(
        self,
        image,
        service_url,
        classes,
        process_res,
        mask_blur,
        mask_offset,
        invert_output,
        background,
        background_color,
        return_image,
        return_mask,
        return_mask_image,
        max_concurrency,
        timeout,
        health_check=False,
        node_switch=0,
        service_url_preset=None,
    ):
        selected_classes = self._parse_classes_text(classes, FASHION_CLASSES, allow_comma_split=False)
        return self._execute(
            image=image,
            service_url=service_url,
            endpoint="segment/fashion",
            default_url=FD_FASHION_SEGMENT_URL or DEFAULT_FASHION_SEGMENT_URL,
            health_key="fashion_segment",
            label="知衣时尚单品分割",
            process_res=process_res,
            mask_blur=mask_blur,
            mask_offset=mask_offset,
            invert_output=invert_output,
            background=background,
            background_color=background_color,
            return_image=return_image,
            return_mask=return_mask,
            return_mask_image=return_mask_image,
            max_concurrency=max_concurrency,
            timeout=timeout,
            health_check=health_check,
            node_switch=node_switch,
            service_url_preset=service_url_preset,
            extra={"classes": selected_classes},
            info_extra={"classes": selected_classes},
        )


class ZhiYiBodySegmentNode(_RmbgSegmentApiBase):
    """知衣身体部位分割节点；服务端会忽略 process_res，模型固定 512x512 输入。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "service_url": ("STRING", {
                    "default": FD_BODY_SEGMENT_URL or DEFAULT_BODY_SEGMENT_URL,
                    "multiline": False,
                    "tooltip": "身体部位分割接口地址，例如 http://host:8003/v1/segment/body",
                }),
                "classes": ("STRING", {
                    "default": "Face\nHair\nTop-clothes\nBottom-clothes",
                    "multiline": True,
                    "tooltip": "每行一个类别，也支持 JSON 字符串数组；空值使用服务默认类别",
                }),
                "process_res": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "此端点会忽略 process_res，ONNX 模型固定使用 512x512 输入",
                }),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "background": (BACKGROUND_MODES, {"default": "Alpha"}),
                "background_color": ("STRING", {"default": "#222222"}),
                "return_image": ("BOOLEAN", {"default": True}),
                "return_mask": ("BOOLEAN", {"default": True}),
                "return_mask_image": ("BOOLEAN", {"default": False}),
                "max_concurrency": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 1200, "step": 10}),
            },
            "optional": {
                "health_check": ("BOOLEAN", {"default": False}),
                "node_switch": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "service_url_preset": (service_url_preset_options(RMBG_SERVICE_URL_PRESETS), {
                    "default": CUSTOM_SERVICE_URL_PRESET,
                    "tooltip": "选择部署预设时忽略 service_url；选择自定义时使用 service_url 文本框",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "JSON", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "INFO", "RGBA_IMAGE")
    FUNCTION = "segment"
    CATEGORY = "知衣/语义分割"
    OUTPUT_NODE = False

    def segment(
        self,
        image,
        service_url,
        classes,
        process_res,
        mask_blur,
        mask_offset,
        invert_output,
        background,
        background_color,
        return_image,
        return_mask,
        return_mask_image,
        max_concurrency,
        timeout,
        health_check=False,
        node_switch=0,
        service_url_preset=None,
    ):
        selected_classes = self._parse_classes_text(classes, BODY_CLASSES, allow_comma_split=True)
        return self._execute(
            image=image,
            service_url=service_url,
            endpoint="segment/body",
            default_url=FD_BODY_SEGMENT_URL or DEFAULT_BODY_SEGMENT_URL,
            health_key="body_segment",
            label="知衣身体部位分割",
            process_res=process_res,
            mask_blur=mask_blur,
            mask_offset=mask_offset,
            invert_output=invert_output,
            background=background,
            background_color=background_color,
            return_image=return_image,
            return_mask=return_mask,
            return_mask_image=return_mask_image,
            max_concurrency=max_concurrency,
            timeout=timeout,
            health_check=health_check,
            node_switch=node_switch,
            service_url_preset=service_url_preset,
            extra={"classes": selected_classes},
            info_extra={"classes": selected_classes, "process_res_ignored": True},
        )

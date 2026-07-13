import io
import hashlib
import json
import logging
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image

from .config import (
    FD_AISTUDIO_PUBLISH_URL,
    FD_LITELLM_API_KEY,
    FD_LITELLM_BASE_URL,
    FD_OSS_URL_PATH_PREFIX_BEFORE_GEN,
)
from .utils.common_util import downscale_image_tensor
from .utils.error_utils import ERROR_TIMEOUT, normalize_error_message
from .utils.gpt_image_request import GptImageRequestMixin
from .utils.gpt_image_size import resolution_to_edit_size, resolve_gpt_image_size
from .utils.litellm_gemini_image import build_litellm_messages, call_litellm_gemini_image, tensor_to_base64
from .utils.logging_utils import configure_default_logging
from .utils.oss_client import upload_bytes_to_oss

configure_default_logging()
logger = logging.getLogger(__name__)


def _bytes_calculate_hex_md5(img_bytes, block_size=64 * 1024):
    md5 = hashlib.md5()
    for i in range(0, len(img_bytes), block_size):
        md5.update(img_bytes[i:i + block_size])
    return md5.hexdigest()


def _bytesio_to_image_tensor(image_bytesio):
    image = Image.open(image_bytesio).convert("RGB")
    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0)


class ZhiYiAiStudioImageComboNode(GptImageRequestMixin):
    """API 图生图测试 combo 节点 - 支持 AiStudio publish、LiteLLM GPT Image edits 与 Gemini image。"""

    MODELS = ["nano-banana-pro"]
    ASPECT_RATIOS = ["", "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"]
    IMAGE_SIZES = ["4K", "2K", "1080P", "720P"]
    QUALITIES = ["low", "medium", "high"]
    API_TYPES = ["auto", "gpt_image", "gemini_image", "aistudio_publish"]
    SEED_MODES = ["随机种子", "固定种子"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {
                    "default": cls.MODELS[0],
                    "tooltip": "模型名称。auto 模式下包含 gpt-image 走 GPT Image edits，包含 gemini 走 LiteLLM Gemini image，其他保持 AiStudio publish。",
                }),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": ""}),
                "image_size": (cls.IMAGE_SIZES, {"default": "4K"}),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }),
                "max_concurrency": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "同时发送请求的最大数量，避免触发API限流",
                }),
                "seed_mode": (cls.SEED_MODES, {"default": "随机种子"}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "step": 1,
                }),
            },
            "optional": {
                "out_request_id": ("STRING", {
                    "default": "",
                    "tooltip": "兼容旧节点输入；AiStudio publish 接口当前不传该字段",
                }),
                "combo_1": ("ZHIYI_COMBO",),
                "combo_2": ("ZHIYI_COMBO",),
                "combo_3": ("ZHIYI_COMBO",),
                "combo_4": ("ZHIYI_COMBO",),
                "combo_5": ("ZHIYI_COMBO",),
                "combo_6": ("ZHIYI_COMBO",),
                "combo_7": ("ZHIYI_COMBO",),
                "combo_8": ("ZHIYI_COMBO",),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "quality": (cls.QUALITIES, {
                    "default": "medium",
                    "tooltip": "GPT Image edits 质量参数，默认 medium。追加到 optional 末尾以兼容旧 workflow 的 widget 顺序；AiStudio publish 渠道会忽略该字段。",
                }),
                "target_url": ("STRING", {
                    "default": "",
                    "tooltip": "目标 URL。GPT/Gemini 可填 LiteLLM base URL 或完整 endpoint；AiStudio 可填 publish URL。留空使用环境变量。",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API Key。GPT/Gemini 为空时使用 FD_LITELLM_API_KEY；AiStudio publish 渠道会忽略该字段。",
                }),
                "api_type": (cls.API_TYPES, {
                    "default": "auto",
                    "tooltip": "调用类型。auto 根据模型名判断；也可强制选择 gpt_image、gemini_image 或 aistudio_publish。",
                }),
                "size_override": ("STRING", {
                    "default": "",
                    "tooltip": "可选。填精确像素尺寸 WIDTHxHEIGHT（如 1537x1025）时覆盖预设分辨率与宽高比，原样传给 image 服务。仅 GPT Image edits 路由生效，其他路由忽略该字段。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "log")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "generate"
    CATEGORY = "知衣/图生图"
    OUTPUT_NODE = False

    def _tensor_to_png_bytes(self, image_tensor):
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]
        arr = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    def _expand_images(self, combo_images):
        result = []
        for image in combo_images:
            if image is None:
                continue
            if image.ndim == 4:
                for i in range(image.shape[0]):
                    result.append(image[i:i + 1])
            elif image.ndim == 3:
                result.append(image.unsqueeze(0))
            else:
                raise RuntimeError(f"图片 tensor 维度错误: {image.ndim}")
        return result

    def _upload_image(self, image_tensor):
        image_bytes = self._tensor_to_png_bytes(image_tensor)
        file_oss_path = f"{FD_OSS_URL_PATH_PREFIX_BEFORE_GEN}/{_bytes_calculate_hex_md5(image_bytes)}.png"
        image_url = upload_bytes_to_oss(file_oss_path, image_bytes)
        print(f"upload {file_oss_path}")
        return image_url

    def _upload_images(self, image_tensors):
        urls = []
        for idx, image_tensor in enumerate(image_tensors):
            logger.info(
                "ZhiYiAiStudioImageCombo Image %s resolution: %s",
                idx,
                tuple(image_tensor.squeeze(0).shape),
            )
            urls.append(self._upload_image(image_tensor))
        return urls

    def _compose_prompt(self, prompt, system_prompt):
        prompt_text = (prompt or "").strip()
        system_text = (system_prompt or "").strip()
        if system_text and prompt_text:
            return f"{system_text}\n\n{prompt_text}"
        return system_text or prompt_text

    def _normalize_image_size(self, image_size):
        size_text = str(image_size or "").strip()
        if size_text.lower().endswith("k"):
            return f"{size_text[:-1]}K"
        return size_text

    def _normalize_model_name(self, model):
        return str(model or "").strip() or self.MODELS[0]

    def _resolve_api_type(self, api_type, model):
        normalized_api_type = str(api_type or "auto").strip() or "auto"
        if normalized_api_type != "auto":
            return normalized_api_type
        normalized_model = self._normalize_model_name(model).lower()
        if "gpt-image" in normalized_model:
            return "gpt_image"
        if "gemini" in normalized_model:
            return "gemini_image"
        return "aistudio_publish"

    def _should_use_gpt_litellm(self, model):
        return self._resolve_api_type("auto", model) == "gpt_image"

    def _normalize_litellm_base_url(self, target_url, endpoint_suffix):
        base_url = str(target_url or "").strip().rstrip("/")
        if base_url.endswith(endpoint_suffix):
            base_url = base_url[:-len(endpoint_suffix)].rstrip("/")
        return base_url

    def _build_gpt_size(self, aspect_ratio, image_size):
        normalized_size = self._normalize_image_size(image_size)
        if normalized_size == "720P":
            resolution = "720P"
        elif normalized_size == "1080P":
            resolution = "1080P"
        elif normalized_size in ("1K", "2K", "4K"):
            resolution = normalized_size
        else:
            resolution = "2K"
        return resolution_to_edit_size(resolution, aspect_ratio or "")

    def _normalize_quality(self, quality):
        normalized_quality = str(quality or "").strip().lower()
        if normalized_quality in self.QUALITIES:
            return normalized_quality
        return "medium"

    def _build_payload(self, prompt, image_urls, aspect_ratio, image_size):
        payload = {
            "prompt": prompt,
            "image": image_urls,
            "image_size": self._normalize_image_size(image_size),
        }
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        return {
            "type": "AiStudio",
            "payload": payload,
            "timeout": 300000,
        }

    def _summarize_request_for_log(self, body, model, seed, out_request_id):
        payload = body.get("payload", {})
        return {
            "type": body.get("type"),
            "model": model,
            "timeout": body.get("timeout"),
            "prompt": payload.get("prompt"),
            "image_count": len(payload.get("image", [])),
            "aspect_ratio": payload.get("aspect_ratio"),
            "image_size": payload.get("image_size"),
            "seed": seed,
            "out_request_id": out_request_id or None,
            "note": "model、seed、out_request_id 仅兼容旧节点 UI，AiStudio publish 接口当前不传这些字段",
        }

    def _extract_result_url(self, result):
        if result.get("success") is not True:
            message = result.get("message") or result.get("error") or result
            raise RuntimeError(f"API 返回失败: {message}")

        data = result.get("data")
        if not isinstance(data, dict):
            raise KeyError("响应缺少 data")

        result_url = data.get("url")
        if not result_url:
            raise KeyError("响应缺少 data.url")
        return result_url

    def _download_result_image(self, result_url):
        response = requests.get(result_url, timeout=300)
        response.raise_for_status()
        return _bytesio_to_image_tensor(BytesIO(response.content))

    def _build_gpt_multipart_files(self, images):
        multipart_files = []
        for idx, image in enumerate(images):
            scaled_image = downscale_image_tensor(image, total_pixels=4096 * 4096).squeeze(0)
            logger.info(
                "ZhiYiAiStudio GPT Image %s resolution: input=%s scaled=%s",
                idx,
                tuple(image.squeeze(0).shape),
                tuple(scaled_image.shape),
            )
            img_bytes = self._tensor_to_png_bytes(scaled_image)
            multipart_files.append(("image", (f"image_{idx}.png", img_bytes, "image/png")))
        return multipart_files

    def _single_request(self, publish_url, prompt, image_urls, aspect_ratio, image_size, model, seed, out_request_id=""):
        if not prompt or not prompt.strip():
            raise RuntimeError("prompt 不能为空")
        if not image_urls:
            raise RuntimeError("未提供图片 URL")

        body = self._build_payload(prompt.strip(), image_urls, aspect_ratio, image_size)
        log_payload = self._summarize_request_for_log(body, model, seed, out_request_id)
        print(
            "[知衣AiStudio图生图-combo] 调用 API 参数: "
            f"{json.dumps(log_payload, ensure_ascii=False)}"
        )
        logger.info("Calling ZhiYi AiStudio combo API with payload=%s", log_payload)

        response = None
        try:
            response = requests.post(
                url=publish_url,
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=600,
            )
            if not response.ok:
                raise RuntimeError(f"API 请求失败: {response.status_code}\n{response.text[:1000]}")

            result = response.json()
            logger.info(
                "ZhiYi AiStudio combo API response summary: %s",
                {
                    "status_code": response.status_code,
                    "taskId": result.get("taskId"),
                    "success": result.get("success"),
                    "result_url": (result.get("data") or {}).get("url") if isinstance(result.get("data"), dict) else None,
                },
            )
            result_url = self._extract_result_url(result)
            return self._download_result_image(result_url), result.get("taskId"), result_url
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                normalize_error_message(exc, category=ERROR_TIMEOUT, fallback_detail="request timed out")
            ) from exc
        except (KeyError, ValueError) as exc:
            response_text = response.text[:500] if response is not None else ""
            raise RuntimeError(normalize_error_message(f"解析响应失败: {exc}\n原始响应: {response_text}")) from exc
        except Exception as exc:
            raise RuntimeError(normalize_error_message(exc)) from exc

    def _single_gpt_request(
        self,
        base_url,
        api_key,
        model,
        prompt,
        images,
        aspect_ratio,
        image_size,
        quality,
        seed,
        out_request_id="",
        size_override="",
    ):
        if not prompt or not prompt.strip():
            raise RuntimeError("prompt 不能为空")
        if not images:
            raise RuntimeError("未提供图片")

        normalized_model = self._normalize_model_name(model)
        preset_size = self._build_gpt_size(aspect_ratio, image_size)
        size, _ = resolve_gpt_image_size(
            preset_size=preset_size,
            aspect_ratio="",
            size_override=size_override,
        )
        normalized_quality = self._normalize_quality(quality)
        data = {
            "model": normalized_model,
            "prompt": prompt.strip(),
            "size": size,
            "quality": normalized_quality,
        }
        if out_request_id:
            data["user"] = out_request_id

        multipart_files = self._build_gpt_multipart_files(images)
        log_payload = {
            "model": normalized_model,
            "prompt": prompt.strip(),
            "size": size,
            "quality": normalized_quality,
            "image_count": len(multipart_files),
            "seed": seed,
            "out_request_id": out_request_id or None,
        }
        print(
            "[知衣AiStudio图生图-combo] 调用 GPT Image API 参数: "
            f"{json.dumps(log_payload, ensure_ascii=False)}"
        )
        logger.info("Calling ZhiYi AiStudio GPT Image API with payload=%s", log_payload)

        try:
            image_bytesio, _, result_url = self._request_gpt_image_edit(
                base_url=base_url,
                api_key=api_key,
                data=data,
                multipart_files=multipart_files,
                batch_size=len(multipart_files),
                logger=logger,
            )
            return _bytesio_to_image_tensor(image_bytesio), None, result_url
        except Exception as exc:
            raise RuntimeError(normalize_error_message(exc)) from exc

    def _single_gemini_request(
        self,
        base_url,
        api_key,
        model,
        messages,
        aspect_ratio,
        image_size,
        seed,
        out_request_id="",
    ):
        normalized_model = self._normalize_model_name(model)
        log_payload = {
            "url": f"{base_url}/v1/chat/completions",
            "model": normalized_model,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "seed": seed,
            "out_request_id": out_request_id or None,
        }
        print(
            "[API图生图节点测试] 调用 Gemini Image API 参数: "
            f"{json.dumps(log_payload, ensure_ascii=False)}"
        )
        logger.info("Calling API image test Gemini Image API with payload=%s", log_payload)

        try:
            image = call_litellm_gemini_image(
                base_url=base_url,
                api_key=api_key,
                messages=messages,
                model=normalized_model,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                seed=seed,
                out_request_id=out_request_id,
            )
            return image, None, ""
        except Exception as exc:
            raise RuntimeError(normalize_error_message(exc)) from exc

    def _run_concurrent(self, tasks, max_workers, label="任务"):
        results = [None] * len(tasks)
        log_lines = []
        last_error_message = None
        workers = min(len(tasks), max(1, max_workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fn, *args): idx
                for idx, fn, args in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    image, task_id, result_url = future.result()
                    results[idx] = image
                    detail_parts = []
                    if task_id:
                        detail_parts.append(f"taskId={task_id}")
                    if result_url:
                        detail_parts.append(f"url={result_url}")
                    suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""
                    log_lines.append(f"[{label} {idx + 1}] 成功{suffix}")
                except Exception as exc:
                    last_error_message = normalize_error_message(exc)
                    msg = f"[{label} {idx + 1}] 失败: {type(exc).__name__}: {exc}"
                    log_lines.append(msg)
                    print(f"[知衣AiStudio图生图-combo] {msg}")
                    traceback.print_exc()
        return results, log_lines, last_error_message

    def generate(self, model, aspect_ratio, image_size, quality="medium",
                 target_url="", api_key="", api_type="auto",
                 batch_size=1, max_concurrency=16, seed_mode="随机种子", seed=0,
                 out_request_id="",
                 combo_1=None, combo_2=None, combo_3=None, combo_4=None,
                 combo_5=None, combo_6=None, combo_7=None, combo_8=None,
                 system_prompt="", size_override=""):
        normalized_model = self._normalize_model_name(model)
        resolved_api_type = self._resolve_api_type(api_type, normalized_model)
        use_gpt_litellm = resolved_api_type == "gpt_image"
        use_gemini_litellm = resolved_api_type == "gemini_image"
        normalized_target_url = str(target_url or "").strip()
        publish_url = normalized_target_url or FD_AISTUDIO_PUBLISH_URL
        litellm_base_url = ""
        litellm_api_key = str(api_key or "").strip() or FD_LITELLM_API_KEY or ""
        target_endpoint = publish_url

        if use_gpt_litellm:
            litellm_base_url = self._normalize_litellm_base_url(
                normalized_target_url or FD_LITELLM_BASE_URL,
                "/v1/images/edits",
            )
            target_endpoint = f"{litellm_base_url}/v1/images/edits" if litellm_base_url else ""
        elif use_gemini_litellm:
            litellm_base_url = self._normalize_litellm_base_url(
                normalized_target_url or FD_LITELLM_BASE_URL,
                "/v1/chat/completions",
            )
            target_endpoint = f"{litellm_base_url}/v1/chat/completions" if litellm_base_url else ""

        if use_gpt_litellm or use_gemini_litellm:
            if not litellm_base_url:
                raise RuntimeError("未配置 target_url/base_url，请填写 target_url 或设置环境变量 FD_LITELLM_BASE_URL")
            if not litellm_api_key:
                raise RuntimeError("未配置 api_key，请填写 api_key 或设置环境变量 FD_LITELLM_API_KEY")
        elif not publish_url:
            raise RuntimeError("未配置 AiStudio publish URL，请设置环境变量 FD_AISTUDIO_PUBLISH_URL")

        actual_seed = random.randint(0, 2147483647) if seed_mode == "随机种子" else seed

        combos = [c for c in [combo_1, combo_2, combo_3, combo_4, combo_5, combo_6, combo_7, combo_8] if c is not None]
        if not combos:
            raise RuntimeError("未提供任何组合输入，请连接至少一个 combo")

        tasks = []
        task_idx = 0
        pre_errors = []
        for c_idx, combo in enumerate(combos):
            try:
                if not isinstance(combo, dict):
                    raise RuntimeError(f"数据格式错误: 期望 dict，实际 {type(combo).__name__}")
                combo_images = combo.get("images", [])
                combo_prompts = combo.get("prompts") or [combo.get("prompt", "")]
                if not combo_images:
                    raise RuntimeError("无图片")

                expanded_images = self._expand_images(combo_images)
                if not expanded_images:
                    raise RuntimeError("无有效图片")

                if use_gpt_litellm:
                    image_urls = None
                    image_b64_list = None
                elif use_gemini_litellm:
                    image_urls = None
                    image_b64_list = [tensor_to_base64(image) for image in expanded_images]
                else:
                    image_urls = self._upload_images(expanded_images)
                    image_b64_list = None
                for p_idx, prompt in enumerate(combo_prompts):
                    if use_gemini_litellm:
                        messages = build_litellm_messages(prompt, image_b64_list, system_prompt)
                        combined_prompt = None
                    else:
                        combined_prompt = self._compose_prompt(prompt, system_prompt)
                    for b_idx in range(batch_size):
                        task_seed = (
                            actual_seed + task_idx
                            if seed_mode == "固定种子"
                            else random.randint(0, 2147483647)
                        )
                        logger.info(
                            "[知衣AiStudio图生图-combo] 准备请求 combo=%s prompt=%s batch=%s seed=%s",
                            c_idx + 1,
                            p_idx + 1,
                            b_idx + 1,
                            task_seed,
                        )
                        if use_gpt_litellm:
                            tasks.append((
                                task_idx,
                                self._single_gpt_request,
                                (
                                    litellm_base_url,
                                    litellm_api_key,
                                    normalized_model,
                                    combined_prompt,
                                    expanded_images,
                                    aspect_ratio,
                                    image_size,
                                    quality,
                                    task_seed,
                                    out_request_id,
                                    size_override,
                                ),
                            ))
                        elif use_gemini_litellm:
                            tasks.append((
                                task_idx,
                                self._single_gemini_request,
                                (
                                    litellm_base_url,
                                    litellm_api_key,
                                    normalized_model,
                                    messages,
                                    aspect_ratio,
                                    image_size,
                                    task_seed,
                                    out_request_id,
                                ),
                            ))
                        else:
                            tasks.append((
                                task_idx,
                                self._single_request,
                                (
                                    publish_url,
                                    combined_prompt,
                                    image_urls,
                                    aspect_ratio,
                                    image_size,
                                    normalized_model,
                                    task_seed,
                                    out_request_id,
                                ),
                            ))
                        task_idx += 1
            except Exception as exc:
                normalized_error = normalize_error_message(exc)
                msg = f"[combo_{c_idx + 1}] 预处理失败，跳过: {type(exc).__name__}: {normalized_error}"
                pre_errors.append(msg)
                print(f"[知衣AiStudio图生图-combo] {msg}")
                traceback.print_exc()

        if not tasks:
            if pre_errors:
                last_pre_error = pre_errors[-1].split(": ", 1)[-1]
                raise RuntimeError(last_pre_error)
            raise RuntimeError(normalize_error_message("所有组合均无有效图片"))

        print(f"[知衣AiStudio图生图-combo] 并发发送 {len(tasks)} 个请求（{len(combos)} 个组合 × batch_size {batch_size}），并发上限 {max_concurrency}")
        results, log_lines, last_error_message = self._run_concurrent(tasks, max_concurrency, label="请求")

        successful = [result for result in results if result is not None]
        header = f"总计: {len(successful)}/{len(tasks)} 成功, url={target_endpoint}"
        if pre_errors:
            header += f"\n预处理失败 {len(pre_errors)} 个: " + "; ".join(pre_errors)
        log_lines.insert(0, header)
        log_text = "\n".join(log_lines)

        if not successful:
            raise RuntimeError(
                last_error_message or normalize_error_message(f"所有请求均失败，无图片返回\n{log_text}")
            )

        print(f"[知衣AiStudio图生图-combo] 成功 {len(successful)}/{len(tasks)}")
        return (successful, actual_seed, log_text)

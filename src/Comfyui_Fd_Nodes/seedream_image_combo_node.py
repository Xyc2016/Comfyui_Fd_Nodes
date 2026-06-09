import io
import logging
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import oss2
import requests
from PIL import Image

from .config import (
    FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL,
    FD_LITELLM_API_KEY,
    FD_LITELLM_BASE_URL,
    FD_OSS_ACCESS_KEY_ID,
    FD_OSS_ACCESS_KEY_SECRET,
    FD_OSS_BUCKET_NAME,
    FD_OSS_ENDPOINT,
    FD_OSS_URL_PATH_PREFIX_BEFORE_GEN,
    FD_OSS_URL_PREFIX,
)
from .utils.common_util import bytes_calculate_hex_md5, bytesio_to_image_tensor
from .utils.error_utils import ERROR_TIMEOUT, normalize_error_message
from .utils.logging_utils import configure_default_logging
from .utils.webhook import webhook_send

configure_default_logging()
logger = logging.getLogger(__name__)


class FD_SeedreamImageComboNode:
    """Seedream 图生图 combo 节点 - 接收最多8个图片组合并发调用 Seedream API。"""

    MODELS = ["doubao-seedream-5.0-lite"]
    IMAGE_SIZES = ["2K", "3K"]
    OUTPUT_FORMATS = ["png", "jpg"]
    SEED_MODES = ["随机种子", "固定种子"]

    def __init__(self):
        self.bucket = None
        self.oss_url_prefix = FD_OSS_URL_PREFIX

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "size": (cls.IMAGE_SIZES, {"default": "2K"}),
                "output_format": (cls.OUTPUT_FORMATS, {"default": "png"}),
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
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "log")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "generate"
    CATEGORY = "image/generation"
    OUTPUT_NODE = False

    def _get_bucket(self):
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

    def _tensor_to_png_bytes(self, image_tensor):
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]
        arr = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

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
        file_oss_path = f"{FD_OSS_URL_PATH_PREFIX_BEFORE_GEN}/{bytes_calculate_hex_md5(image_bytes)}.png"
        self._get_bucket().put_object(file_oss_path, image_bytes)
        print(f"upload {file_oss_path}")
        return f"{self.oss_url_prefix}{file_oss_path}"

    def _upload_images(self, image_tensors):
        urls = []
        for idx, image_tensor in enumerate(image_tensors):
            logger.info(
                "FD_SeedreamImageCombo Image %s resolution: %s",
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

    def _build_body(self, model, prompt, image_urls, size, output_format):
        return {
            "model": model,
            "prompt": prompt,
            "sequential_image_generation": "disabled",
            "size": size,
            "output_format": output_format,
            "watermark": False,
            "image": image_urls,
        }

    def _summarize_request_for_log(self, body, seed):
        return {
            "model": body.get("model"),
            "prompt": body.get("prompt"),
            "sequential_image_generation": body.get("sequential_image_generation"),
            "size": body.get("size"),
            "output_format": body.get("output_format"),
            "watermark": body.get("watermark"),
            "image_count": len(body.get("image", [])),
            "seed": seed,
            "note": "seed 仅用于批量任务记录，Seedream 请求体当前不传 seed 字段",
        }

    def _extract_result_url(self, result):
        data = result.get("data")
        if not isinstance(data, list) or not data:
            raise KeyError("响应缺少 data[0]")
        first_item = data[0]
        if not isinstance(first_item, dict):
            raise KeyError("响应 data[0] 格式错误")
        result_url = first_item.get("url")
        if not result_url:
            raise KeyError("响应缺少 data[0].url")
        return result_url

    def _download_result_image(self, result_url):
        response = requests.get(result_url, timeout=300)
        response.raise_for_status()
        return bytesio_to_image_tensor(BytesIO(response.content))

    def _single_request(self, base_url, api_key, model, prompt, image_urls, size, output_format, seed):
        if not prompt or not prompt.strip():
            raise RuntimeError("prompt 不能为空")
        if not image_urls:
            raise RuntimeError("未提供图片 URL")

        body = self._build_body(model, prompt.strip(), image_urls, size, output_format)
        log_payload = self._summarize_request_for_log(body, seed)

        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
                webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {
                    "seedream_combo_request": body,
                })
            except Exception:
                pass

        logger.info("Calling Seedream combo API with payload=%s", log_payload)

        response = None
        try:
            response = requests.post(
                url=f"{base_url}/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=300,
            )
            if not response.ok:
                raise RuntimeError(f"API 请求失败: {response.status_code}\n{response.text[:1000]}")

            result = response.json()
            logger.info(
                "Seedream combo API response summary: %s",
                {
                    "status_code": response.status_code,
                    "data_count": len(result.get("data", [])) if isinstance(result.get("data"), list) else None,
                    "result_url": (
                        result.get("data", [{}])[0].get("url")
                        if isinstance(result.get("data"), list) and result.get("data") and isinstance(result.get("data")[0], dict)
                        else None
                    ),
                },
            )
            result_url = self._extract_result_url(result)

            if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
                try:
                    webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {
                        "seedream_combo_full": {
                            "request": body,
                            "response": result,
                        }
                    })
                except Exception:
                    pass

            return self._download_result_image(result_url), None, result_url
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                normalize_error_message(exc, category=ERROR_TIMEOUT, fallback_detail="request timed out")
            ) from exc
        except (KeyError, ValueError) as exc:
            response_text = response.text[:500] if response is not None else ""
            raise RuntimeError(normalize_error_message(f"解析响应失败: {exc}\n原始响应: {response_text}")) from exc
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
                    image, output_text, result_url = future.result()
                    results[idx] = image
                    detail_parts = []
                    if result_url:
                        detail_parts.append(f"url={result_url}")
                    if output_text:
                        detail_parts.append(f"text={output_text[:200]}")
                    suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""
                    log_lines.append(f"[{label} {idx + 1}] 成功{suffix}")
                except Exception as exc:
                    last_error_message = normalize_error_message(exc)
                    msg = f"[{label} {idx + 1}] 失败: {type(exc).__name__}: {exc}"
                    log_lines.append(msg)
                    print(f"[Seedream图生图-combo] {msg}")
                    traceback.print_exc()
        return results, log_lines, last_error_message

    def generate(self, model, size, output_format,
                 batch_size=1, max_concurrency=16, seed_mode="随机种子", seed=0,
                 combo_1=None, combo_2=None, combo_3=None, combo_4=None,
                 combo_5=None, combo_6=None, combo_7=None, combo_8=None,
                 system_prompt=""):
        final_base_url = (FD_LITELLM_BASE_URL or "").rstrip("/")
        final_api_key = FD_LITELLM_API_KEY or ""
        if not final_base_url or final_base_url == "https://your-api-base-url":
            raise RuntimeError("未配置 base_url，请在环境变量 FD_LITELLM_BASE_URL 中设置")
        if not final_api_key or final_api_key == "your-api-key":
            raise RuntimeError("未配置 api_key，请在环境变量 FD_LITELLM_API_KEY 中设置")

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

                image_urls = self._upload_images(expanded_images)
                for p_idx, prompt in enumerate(combo_prompts):
                    combined_prompt = self._compose_prompt(prompt, system_prompt)
                    for b_idx in range(batch_size):
                        task_seed = (
                            actual_seed + task_idx
                            if seed_mode == "固定种子"
                            else random.randint(0, 2147483647)
                        )
                        logger.info(
                            "[Seedream图生图-combo] 准备请求 combo=%s prompt=%s batch=%s seed=%s",
                            c_idx + 1,
                            p_idx + 1,
                            b_idx + 1,
                            task_seed,
                        )
                        tasks.append((
                            task_idx,
                            self._single_request,
                            (
                                final_base_url,
                                final_api_key,
                                model,
                                combined_prompt,
                                image_urls,
                                size,
                                output_format,
                                task_seed,
                            ),
                        ))
                        task_idx += 1
            except Exception as exc:
                normalized_error = normalize_error_message(exc)
                msg = f"[combo_{c_idx + 1}] 预处理失败，跳过: {type(exc).__name__}: {normalized_error}"
                pre_errors.append(msg)
                print(f"[Seedream图生图-combo] {msg}")
                traceback.print_exc()

        if not tasks:
            if pre_errors:
                last_pre_error = pre_errors[-1].split(": ", 1)[-1]
                raise RuntimeError(last_pre_error)
            raise RuntimeError(normalize_error_message("所有组合均无有效图片"))

        print(f"[Seedream图生图-combo] 并发发送 {len(tasks)} 个请求（{len(combos)} 个组合 × batch_size {batch_size}），并发上限 {max_concurrency}")
        results, log_lines, last_error_message = self._run_concurrent(tasks, max_concurrency, label="请求")

        successful = [result for result in results if result is not None]
        header = f"总计: {len(successful)}/{len(tasks)} 成功, url={final_base_url}/v1/images/generations"
        if pre_errors:
            header += f"\n预处理失败 {len(pre_errors)} 个: " + "; ".join(pre_errors)
        log_lines.insert(0, header)
        log_text = "\n".join(log_lines)

        if not successful:
            raise RuntimeError(
                last_error_message or normalize_error_message(f"所有请求均失败，无图片返回\n{log_text}")
            )

        print(f"[Seedream图生图-combo] 成功 {len(successful)}/{len(tasks)}")
        return (successful, actual_seed, log_text)

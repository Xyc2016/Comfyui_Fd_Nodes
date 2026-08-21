import io
import logging
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

from .config import FD_OSS_URL_PATH_PREFIX_BEFORE_GEN
from .utils.common_util import bytes_calculate_hex_md5, bytesio_to_image_tensor, downscale_image_tensor
from .utils.error_utils import normalize_error_message
from .utils.logging_utils import configure_default_logging
from .utils.oss_client import upload_bytes_to_oss
from .utils.seedream_image_client import SeedreamImageClient
from .utils.seedream_image_size import (
    SEEDREAM_ASPECT_RATIOS,
    SEEDREAM_IMAGE_SIZES,
)

configure_default_logging()
logger = logging.getLogger(__name__)


class FD_SeedreamImageComboNode:
    """Seedream 图生图 combo 节点 - 接收最多8个图片组合并发调用 Seedream API。"""

    MODELS = ["doubao-seedream-5.0-lite", "doubao-seedream-5.0-pro"]
    IMAGE_SIZES = SEEDREAM_IMAGE_SIZES
    ASPECT_RATIOS = SEEDREAM_ASPECT_RATIOS
    OUTPUT_FORMATS = ["png", "jpg"]
    SEED_MODES = ["随机种子", "固定种子"]

    def __init__(self):
        self._client = None

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
                "aspect_ratio": (cls.ASPECT_RATIOS, {
                    "default": "1:1",
                    "tooltip": "Output aspect ratio. Converted to a Seedream pixel size before request.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "log")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "generate"
    CATEGORY = "image/generation"
    OUTPUT_NODE = False

    def _get_client(self) -> SeedreamImageClient:
        if self._client is None:
            self._client = SeedreamImageClient(webhook_name="seedream_combo")
        return self._client

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
        image_url = upload_bytes_to_oss(file_oss_path, image_bytes)
        print(f"upload {file_oss_path}")
        return image_url

    def _upload_images(self, image_tensors):
        urls = []
        for idx, image_tensor in enumerate(image_tensors):
            scaled = downscale_image_tensor(image_tensor, total_pixels=4096 * 4096).squeeze(0)
            logger.info(
                "FD_SeedreamImageCombo Image %s resolution: %s",
                idx,
                tuple(scaled.shape),
            )
            urls.append(self._upload_image(scaled))
        return urls

    def _compose_prompt(self, prompt, system_prompt):
        prompt_text = (prompt or "").strip()
        system_text = (system_prompt or "").strip()
        if system_text and prompt_text:
            return f"{system_text}\n\n{prompt_text}"
        return system_text or prompt_text

    def _summarize_request_for_log(self, model, prompt, size, ratio, image_count, seed):
        return {
            "channel": model,
            "prompt": prompt,
            "size": size,
            "ratio": ratio,
            "image_count": image_count,
            "seed": seed,
            "note": "seed 仅用于批量任务记录，Seedream 请求体当前不传 seed 字段",
        }

    def _single_request(self, model, prompt, image_urls, size, seed, aspect_ratio="1:1"):
        if not prompt or not prompt.strip():
            raise RuntimeError("prompt 不能为空")
        if not image_urls:
            raise RuntimeError("未提供图片 URL")

        log_payload = self._summarize_request_for_log(model, prompt.strip(), size, aspect_ratio, len(image_urls), seed)
        logger.info("Calling Seedream combo API with payload=%s", log_payload)

        image_bytesio, result_url = self._get_client().edit_image_with_urls(
            image_urls=image_urls,
            prompt=prompt.strip(),
            model=model,
            size=size,
            ratio=aspect_ratio,
            resize=True,
        )
        image = bytesio_to_image_tensor(image_bytesio)
        return image, None, result_url

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
                 system_prompt="", aspect_ratio="1:1"):
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
                                model,
                                combined_prompt,
                                image_urls,
                                size,
                                task_seed,
                                aspect_ratio,
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
        edit_url = self._get_client().edit_url
        header = f"总计: {len(successful)}/{len(tasks)} 成功, url={edit_url}"
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

import io
import logging
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

from .config import (
    FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL,
    FD_LITELLM_API_KEY,
    FD_LITELLM_BASE_URL,
)
from .config_manager import load_config
from .utils.common_util import bytesio_to_image_tensor, downscale_image_tensor
from .utils.error_utils import normalize_error_message
from .utils.gpt_image_request import GptImageRequestMixin
from .utils.gpt_image_size import resolution_to_edit_size
from .utils.logging_utils import configure_default_logging
from .utils.webhook import webhook_send

configure_default_logging()
logger = logging.getLogger(__name__)


class FD_GPTImageComboNode(GptImageRequestMixin):
    """GPT 图生图 combo 节点 - 接收最多8个图片组合并发调用 GPT Image API。"""

    MODELS = ["gpt-image-2"]
    ASPECT_RATIOS = ["", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9"]
    IMAGE_SIZES = ["4K", "2K", "1K"]
    SEED_MODES = ["随机种子", "固定种子"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
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
                    "tooltip": "FD out_request_id，留空不传",
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
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "log")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "generate"
    CATEGORY = "GPT Image"
    OUTPUT_NODE = False

    def _compose_prompt(self, prompt, system_prompt):
        prompt_text = (prompt or "").strip()
        system_text = (system_prompt or "").strip()
        if system_text and prompt_text:
            return f"{system_text}\n\n{prompt_text}"
        return system_text or prompt_text

    def _tensor_to_png_bytes(self, image_tensor):
        arr = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _expand_images(self, combo_images):
        """展开 combo 内的 batch tensor 为单帧图片列表。"""
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

    def _build_gpt_size(self, aspect_ratio, image_size):
        resolution = {
            "720P": "1K",
            "1080P": "1K",
            "1K": "1K",
            "2K": "2K",
            "4K": "4K",
        }.get(image_size, "2K")
        return resolution_to_edit_size(resolution, aspect_ratio or "")

    def _single_request(
        self,
        base_url,
        api_key,
        model,
        prompt,
        images,
        aspect_ratio,
        image_size,
        out_request_id="",
    ):
        if not prompt or not prompt.strip():
            raise RuntimeError("prompt 不能为空")
        if not images:
            raise RuntimeError("未提供图片")

        size = self._build_gpt_size(aspect_ratio, image_size)
        data = {
            "model": model,
            "prompt": prompt.strip(),
            "size": size,
            "quality": "medium",
        }
        if out_request_id:
            data["user"] = out_request_id

        multipart_files = []
        for idx, image in enumerate(images):
            scaled_image = downscale_image_tensor(image, total_pixels=4096 * 4096).squeeze(0)
            logger.info(
                "FD_GPTImageCombo Image %s resolution: input=%s scaled=%s",
                idx,
                tuple(image.squeeze(0).shape),
                tuple(scaled_image.shape),
            )
            img_bytes = self._tensor_to_png_bytes(scaled_image)
            multipart_files.append(
                ("image", (f"image_{idx}.png", img_bytes, "image/png"))
            )

        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
                webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {
                    "gtp_image_request": {
                        "data": data,
                        "image_count": len(multipart_files),
                    }
                })
            except Exception:
                pass

        logger.info(
            "Calling GPT image combo API with data=%s image_count=%s",
            data,
            len(multipart_files),
        )
        image_bytesio, output_text, result_url = self._call_gpt_image_with_retry_policy(
            base_url=base_url,
            api_key=api_key,
            data=data,
            multipart_files=multipart_files,
            batch_size=len(multipart_files),
            logger=logger,
        )
        output_image = bytesio_to_image_tensor(image_bytesio)
        return output_image, output_text, result_url

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
                    print(f"[GPT图生图-combo] {msg}")
                    traceback.print_exc()
        return results, log_lines, last_error_message

    def generate(self, model, aspect_ratio, image_size,
                 batch_size=1, max_concurrency=16, seed_mode="随机种子", seed=0,
                 out_request_id="",
                 combo_1=None, combo_2=None, combo_3=None, combo_4=None,
                 combo_5=None, combo_6=None, combo_7=None, combo_8=None,
                 system_prompt=""):
        cfg = load_config()
        final_base_url = (FD_LITELLM_BASE_URL or cfg["base_url"] or "").rstrip("/")
        final_api_key = FD_LITELLM_API_KEY or cfg["api_key"]
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

                request_images = self._expand_images(combo_images)
                if not request_images:
                    raise RuntimeError("无有效图片")

                for p_idx, prompt in enumerate(combo_prompts):
                    combined_prompt = self._compose_prompt(prompt, system_prompt)
                    for b_idx in range(batch_size):
                        task_seed = (
                            actual_seed + task_idx
                            if seed_mode == "固定种子"
                            else random.randint(0, 2147483647)
                        )
                        logger.info(
                            "[GPT图生图-combo] 准备请求 combo=%s prompt=%s batch=%s seed=%s",
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
                                request_images,
                                aspect_ratio,
                                image_size,
                                out_request_id,
                            ),
                        ))
                        task_idx += 1
            except Exception as exc:
                normalized_error = normalize_error_message(exc)
                msg = f"[combo_{c_idx + 1}] 预处理失败，跳过: {type(exc).__name__}: {normalized_error}"
                pre_errors.append(msg)
                print(f"[GPT图生图-combo] {msg}")
                traceback.print_exc()

        if not tasks:
            if pre_errors:
                last_pre_error = pre_errors[-1].split(": ", 1)[-1]
                raise RuntimeError(last_pre_error)
            raise RuntimeError(normalize_error_message("所有组合均无有效图片"))

        print(f"[GPT图生图-combo] 并发发送 {len(tasks)} 个请求（{len(combos)} 个组合 × batch_size {batch_size}），并发上限 {max_concurrency}")
        results, log_lines, last_error_message = self._run_concurrent(tasks, max_concurrency, label="请求")

        successful = [result for result in results if result is not None]
        header = f"总计: {len(successful)}/{len(tasks)} 成功, url={final_base_url}/v1/images/edits"
        if pre_errors:
            header += f"\n预处理失败 {len(pre_errors)} 个: " + "; ".join(pre_errors)
        log_lines.insert(0, header)
        log_text = "\n".join(log_lines)

        if not successful:
            raise RuntimeError(
                last_error_message or normalize_error_message(f"所有请求均失败，无图片返回\n{log_text}")
            )

        print(f"[GPT图生图-combo] 成功 {len(successful)}/{len(tasks)}")
        return (successful, actual_seed, log_text)

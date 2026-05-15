import io
import logging
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image

from .config import (
    FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL,
    FD_LITELLM_API_KEY,
    FD_LITELLM_BASE_URL,
)
from .config_manager import load_config
from .old_gemini_api_node import GenImageServiceError
from .utils.common_util import bytesio_to_image_tensor, downscale_image_tensor
from .utils.gpt_image_size import resolution_to_edit_size
from .utils.gpt_image_request import GptImageRequestMixin
from .utils.logging_utils import configure_default_logging
from .utils.webhook import webhook_send

configure_default_logging()
logger = logging.getLogger(__name__)


class FD_GPTMultiImage(GptImageRequestMixin):
    """GPT 多图编辑节点，沿用知衣多图输入与并发方式，底层调用 GPT Image edits API。"""

    MODELS = ["gpt-image-2"]
    ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9"]
    IMAGE_SIZES = ["4K", "2K", "1K"]
    SEED_MODES = ["随机种子", "固定种子"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "将裤子变为老欧美风格的牛仔裤",
                    "multiline": True,
                }),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": "1:1"}),
                "image_size": (cls.IMAGE_SIZES, {"default": "4K"}),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
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
                "node_switch": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": "number",
                }),
                "out_request_id": ("STRING", {
                    "default": "default",
                    "tooltip": "FD out_request_id for generation",
                }),
                "prompt_list": ("LIST",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "seed")
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
        image_np = (image_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _build_gpt_size(self, aspect_ratio, image_size):
        resolution = {
            "720P": "1K",
            "1080P": "1K",
            "2K": "2K",
            "4K": "4K",
        }.get(image_size, "2K")
        return resolution_to_edit_size(resolution, aspect_ratio)

    def _run_concurrent(self, tasks, label="任务"):
        results = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {
                executor.submit(fn, *args): idx
                for idx, fn, args in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.warning(
                        "[GPT多图] %s 第 %s 个失败，已跳过: %s: %s",
                        label,
                        idx + 1,
                        type(exc).__name__,
                        exc,
                    )
                    traceback.print_exc()
        return results

    def _single_request(
        self,
        base_url,
        api_key,
        model,
        prompt,
        images,
        aspect_ratio,
        image_size,
        out_request_id="default",
    ):
        size = self._build_gpt_size(aspect_ratio, image_size)
        data = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "user": out_request_id,
            "quality": "medium",
        }

        multipart_files = []
        for idx, image in enumerate(images):
            scaled_image = downscale_image_tensor(image, total_pixels=4096 * 4096).squeeze(0)
            logger.info(
                "FD_GTPMultiImage Image %s resolution: input=%s scaled=%s",
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
            "Calling GPT multi-image API with data=%s image_count=%s",
            data,
            len(multipart_files),
        )

        try:
            image_bytesio, _, _ = self._call_gpt_image_with_retry_policy(
                base_url=base_url,
                api_key=api_key,
                data=data,
                multipart_files=multipart_files,
                batch_size=len(multipart_files),
                logger=logger,
            )
            output_image = bytesio_to_image_tensor(image_bytesio)
        except Exception as exc:
            traceback.print_exc()
            if isinstance(exc, GenImageServiceError):
                raise
            raise GenImageServiceError(f"UNEXPECTED_ERROR: {exc}") from exc

        return output_image

    def generate(
        self,
        image_1,
        prompt,
        model,
        aspect_ratio,
        image_size,
        batch_size=1,
        seed_mode="随机种子",
        seed=0,
        node_switch=0,
        out_request_id="default",
        prompt_list=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        system_prompt="",
    ):
        if node_switch == 1:
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, 0)

        cfg = load_config()
        base_url = (FD_LITELLM_BASE_URL or cfg["base_url"]).rstrip("/")
        api_key = FD_LITELLM_API_KEY or cfg["api_key"]

        actual_seed = random.randint(0, 2147483647) if seed_mode == "随机种子" else seed
        input_images = [
            image
            for image in [image_1, image_2, image_3, image_4, image_5, image_6]
            if image is not None
        ]
        request_images = [image[:1] for image in input_images]

        if isinstance(prompt_list, str):
            prompt_candidates = [prompt_list]
        else:
            prompt_candidates = prompt_list or []

        prompts = [item for item in prompt_candidates if isinstance(item, str) and item.strip()]
        if not prompts:
            prompts = [prompt]

        total = len(prompts) * batch_size
        tasks = []
        task_idx = 0
        for prompt_index, prompt_item in enumerate(prompts):
            combined_prompt = self._compose_prompt(prompt_item, system_prompt)
            for batch_index in range(batch_size):
                task_seed = (
                    actual_seed + prompt_index * batch_size + batch_index
                    if seed_mode == "固定种子"
                    else random.randint(0, 2147483647)
                )
                logger.info(
                    "[GPT多图] 准备请求 prompt_index=%s batch_index=%s seed=%s",
                    prompt_index,
                    batch_index,
                    task_seed,
                )
                tasks.append(
                    (
                        task_idx,
                        self._single_request,
                        (
                            base_url,
                            api_key,
                            model,
                            combined_prompt,
                            request_images,
                            aspect_ratio,
                            image_size,
                            out_request_id,
                        ),
                    )
                )
                task_idx += 1

        logger.info(
            "[GPT多图] 并发发送 %s 个请求（%s 条提示词 × batch_size %s）",
            total,
            len(prompts),
            batch_size,
        )
        results = self._run_concurrent(tasks, label="请求")

        successful = [result for result in results if result is not None]
        if not successful:
            raise RuntimeError("所有请求均失败，无图片返回")

        logger.info("[GPT多图] 成功 %s/%s", len(successful), total)
        return (torch.cat(successful, dim=0), actual_seed)

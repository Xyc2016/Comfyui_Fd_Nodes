import logging
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from .utils.error_utils import normalize_error_message
from .utils.gemini_service import GeminiImageServiceClient, compose_prompt
from .utils.litellm_gemini_image import (
    build_litellm_messages,
    call_litellm_gemini_image,
    should_use_litellm_gemini,
    tensor_to_base64,
)
from .utils.logging_utils import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)


class ZhiYiImageToImageNode:
    """知衣图生图节点 - 最多6张输入图片，batch_size 控制重复请求次数，支持提示词列表并发"""

    MODELS = [
        "google/gemini-3-pro-image-preview",
        "google/gemini-3.1-flash-image-preview",
        "google/gemini-2.5-flash-image-preview",
        "google/gemini-3-pro-image-preview-official",
        "gemini-3-pro-image-preview",
        "batch/gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
        "gemini-2.5-flash-image-preview",
        "gemini-3-pro-image-preview-official",
        "gemini-3-pro-image-preview-aistudio",
        "gemini-3-pro-image-preview-siphonlab",
    ]

    ASPECT_RATIOS = ["", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9"]
    IMAGE_SIZES = ["4K", "2K", "1080P", "720P"]
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
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": ""}),
                "image_size": (cls.IMAGE_SIZES, {"default": "2K"}),
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
    CATEGORY = "知衣/图生图"
    OUTPUT_NODE = False

    def __init__(self):
        self.gemini_client = GeminiImageServiceClient()

    def _single_request(self, image_url_list, prompt, model, aspect_ratio, image_size, seed, out_request_id="default"):
        image, _result_url, _message = self.gemini_client.call_with_image_urls(
            prompt=prompt,
            model=model,
            image_url_list=image_url_list,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            out_request_id=out_request_id,
        )
        return image

    def _single_litellm_request(self, messages, model, aspect_ratio, image_size, seed, out_request_id="default"):
        return call_litellm_gemini_image(
            messages=messages,
            model=model,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            seed=seed,
            out_request_id=out_request_id,
        )

    def _run_concurrent(self, tasks, label="任务"):
        """tasks: list of (idx, callable, args)，并发执行，返回 {idx: result}"""
        results = [None] * len(tasks)
        last_error_message = None
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {
                executor.submit(fn, *args): idx
                for idx, fn, args in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    last_error_message = normalize_error_message(e)
                    logger.warning(
                        "[知衣图生图] %s 第 %s 个失败，已跳过: %s: %s",
                        label,
                        idx + 1,
                        type(e).__name__,
                        e,
                    )
                    traceback.print_exc()
        return results, last_error_message

    def generate(self, image_1, prompt, model, aspect_ratio, image_size,
                 batch_size=1, seed_mode="随机种子", seed=0,
                 node_switch=0, out_request_id="default", prompt_list=None,
                 image_2=None, image_3=None, image_4=None,
                 image_5=None, image_6=None, system_prompt=""):
        if node_switch == 1:
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, 0)

        actual_seed = random.randint(0, 2147483647) if seed_mode == "随机种子" else seed

        image_tensors = [t for t in [image_1, image_2, image_3, image_4, image_5, image_6] if t is not None]
        use_litellm = should_use_litellm_gemini(model)
        if use_litellm:
            image_b64_list = [tensor_to_base64(t) for t in image_tensors]
            image_url_list = None
        else:
            image_url_list = self.gemini_client.upload_images(image_tensors)
            image_b64_list = None

        # 有效提示词列表；字符串输入按单条 prompt 处理，避免被逐字符拆分
        if isinstance(prompt_list, str):
            prompt_candidates = [prompt_list]
        else:
            prompt_candidates = prompt_list or []

        prompts = [p for p in prompt_candidates if isinstance(p, str) and p.strip()]
        if not prompts:
            prompts = [prompt]

        total = len(prompts) * batch_size
        tasks = []
        task_idx = 0
        for p_idx, p in enumerate(prompts):
            if use_litellm:
                messages = build_litellm_messages(p, image_b64_list, system_prompt)
            else:
                final_prompt = compose_prompt(p, system_prompt)
            for b_idx in range(batch_size):
                s = (actual_seed + p_idx * batch_size + b_idx) if seed_mode == "固定种子" else random.randint(0, 2147483647)
                if use_litellm:
                    tasks.append((task_idx, self._single_litellm_request, (messages, model, aspect_ratio or None, image_size, s, out_request_id)))
                else:
                    tasks.append((task_idx, self._single_request, (image_url_list, final_prompt, model, aspect_ratio or None, image_size, s, out_request_id)))
                task_idx += 1

        logger.info(
            "[知衣图生图] 并发发送 %s 个请求（%s 条提示词 × batch_size %s）",
            total,
            len(prompts),
            batch_size,
        )
        results, last_error_message = self._run_concurrent(tasks, label="请求")

        successful = [r for r in results if r is not None]
        if not successful:
            raise RuntimeError(
                last_error_message or normalize_error_message("所有请求均失败，无图片返回")
            )

        logger.info("[知衣图生图] 成功 %s/%s", len(successful), total)
        return (torch.cat(successful, dim=0), actual_seed)

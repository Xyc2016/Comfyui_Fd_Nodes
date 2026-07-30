import json
import logging
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils.error_utils import normalize_error_message
from .utils.gemini_service import (
    GeminiImageServiceClient,
    compose_prompt,
    is_batch_gemini_model,
    summarize_text,
    validate_gemini_batch_ids,
)
from .utils.litellm_gemini_image import (
    build_litellm_messages,
    call_litellm_gemini_image,
    should_use_litellm_gemini,
    tensor_to_base64,
)
from .utils.logging_utils import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)


class ZhiYiImageToImageComboNode:
    """知衣图生图节点 - 接收最多10个图片组合并发调用 API"""

    MODELS = [
        "google/gemini-3-pro-image-preview",
        "google/gemini-3-pro-image-preview-stable",
        "google/gemini-3-pro-image-preview-cheap",
        "google/gemini-3.1-flash-image-preview",
        "google/gemini-2.5-flash-image-preview",
        "google/gemini-3-pro-image-preview-official",
        "gemini-3-pro-image-preview",
        "gemini-3-pro-image-preview-stable",
        "gemini-3-pro-image-preview-cheap",
        "batch/gemini-3-pro-image-preview",
        "batch/gemini-3-pro-image-preview-stable",
        "batch/gemini-3-pro-image-preview-cheap",
        "gemini-3.1-flash-image-preview",
        "gemini-2.5-flash-image-preview",
        "gemini-3-pro-image-preview-official",
        "gemini-3-pro-image-preview-aistudio",
        "gemini-3-pro-image-preview-siphonlab",
        "gemini-3-pro-image-preview-vip",
        "google/gemini-3-pro-image-preview-vip",
        "batch/gemini-3-pro-image-preview-vip",
        "gemini-3-pro-image-preview-adobe",
    ]

    ASPECT_RATIOS = ["", "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"]
    IMAGE_SIZES = ["4K", "2K", "1080P", "720P"]
    SEED_MODES = ["随机种子", "固定种子"]

    def __init__(self):
        self.gemini_client = GeminiImageServiceClient()

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
                "combo_9": ("ZHIYI_COMBO",),
                "combo_10": ("ZHIYI_COMBO",),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "enable_color_bias_correction": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "仅直连 image-server /image/gemini_image 时启用全局偏红纠正",
                }),
                "color_bias_reference_image_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 63,
                    "step": 1,
                    "tooltip": "颜色参考图在每个组合最终 image_url_list 中的 0-based 索引",
                }),
                "batch_task_id": ("STRING", {"default": ""}),
                "batch_item_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "log")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "generate"
    CATEGORY = "知衣/图生图"
    OUTPUT_NODE = False

    def _expand_images(self, combo_images):
        """展开 batch tensor 为单帧列表"""
        result = []
        for t in combo_images:
            if t is None:
                continue
            if t.ndim == 4 and t.shape[0] > 1:
                for i in range(t.shape[0]):
                    result.append(t[i:i+1])
            else:
                result.append(t)
        return result

    def _single_request(self, image_url_list, prompt, model, aspect_ratio, image_size, seed, out_request_id="",
                        enable_color_bias_correction=False, color_bias_reference_image_index=0,
                        batch_task_id="", batch_item_id=""):
        log_payload = {
            "url": self.gemini_client.service_url,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "out_request_id": out_request_id or None,
            "seed": seed,
            "prompt_preview": summarize_text(prompt),
            "prompt_length": len(prompt or ""),
            "image_count": len(image_url_list),
        }
        print(
            "[知衣图生图-combo] 调用 API 参数: "
            f"{json.dumps(log_payload, ensure_ascii=False)}"
        )
        logger.info("Calling ZhiYi image-to-image combo API with payload=%s", log_payload)
        image, _result_url, _message = self.gemini_client.call_with_image_urls(
            prompt=prompt,
            model=model,
            image_url_list=image_url_list,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            out_request_id=out_request_id,
            enable_color_bias_correction=enable_color_bias_correction,
            color_bias_reference_image_index=color_bias_reference_image_index,
            batch_task_id=batch_task_id,
            batch_item_id=batch_item_id,
        )
        return image

    def _single_litellm_request(self, messages, model, aspect_ratio, image_size, seed, out_request_id=""):
        return call_litellm_gemini_image(
            messages=messages,
            model=model,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            seed=seed,
            out_request_id=out_request_id,
        )

    def _run_concurrent(self, tasks, max_workers, label="任务"):
        """tasks: list of (idx, callable, args)，并发执行，返回 (results_list, log_lines)"""
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
                    results[idx] = future.result()
                    log_lines.append(f"[{label} {idx + 1}] 成功")
                except Exception as e:
                    last_error_message = normalize_error_message(e)
                    msg = f"[{label} {idx + 1}] 失败: {type(e).__name__}: {e}"
                    log_lines.append(msg)
                    print(f"[知衣图生图] {msg}")
                    traceback.print_exc()
        return results, log_lines, last_error_message

    def generate(self, model, aspect_ratio, image_size,
                 batch_size=1, max_concurrency=16, seed_mode="随机种子", seed=0,
                 out_request_id="",
                 combo_1=None, combo_2=None, combo_3=None, combo_4=None,
                 combo_5=None, combo_6=None, combo_7=None, combo_8=None,
                 combo_9=None, combo_10=None,
                 system_prompt="", enable_color_bias_correction=False,
                 color_bias_reference_image_index=0,
                 batch_task_id="", batch_item_id=""):
        actual_seed = random.randint(0, 2147483647) if seed_mode == "随机种子" else seed
        validate_gemini_batch_ids(model, batch_task_id, batch_item_id)
        use_litellm = should_use_litellm_gemini(model)

        combos = [c for c in [combo_1, combo_2, combo_3, combo_4, combo_5, combo_6, combo_7, combo_8, combo_9, combo_10] if c is not None]
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
                if use_litellm:
                    image_b64_list = [tensor_to_base64(t) for t in expanded_images]
                    image_url_list = None
                else:
                    image_url_list = self.gemini_client.upload_images(expanded_images)
                    image_b64_list = None
                for p_idx, p in enumerate(combo_prompts):
                    if use_litellm:
                        messages = build_litellm_messages(p, image_b64_list, system_prompt)
                    else:
                        final_prompt = compose_prompt(p, system_prompt)
                    for b_idx in range(batch_size):
                        s = (actual_seed + task_idx) if seed_mode == "固定种子" else random.randint(0, 2147483647)
                        if use_litellm:
                            tasks.append((task_idx, self._single_litellm_request, (messages, model, aspect_ratio or None, image_size, s, out_request_id)))
                        else:
                            request_args = (
                                image_url_list,
                                final_prompt,
                                model,
                                aspect_ratio or None,
                                image_size,
                                s,
                                out_request_id,
                                enable_color_bias_correction is True,
                                (
                                    color_bias_reference_image_index
                                    if enable_color_bias_correction is True
                                    else 0
                                ),
                                batch_task_id,
                                batch_item_id,
                            )
                            tasks.append((task_idx, self._single_request, request_args))
                        task_idx += 1
            except Exception as e:
                normalized_error = normalize_error_message(e)
                msg = f"[combo_{c_idx + 1}] 预处理失败，跳过: {type(e).__name__}: {normalized_error}"
                pre_errors.append(msg)
                print(f"[知衣图生图] {msg}")
                traceback.print_exc()

        if not tasks:
            if pre_errors:
                last_pre_error = pre_errors[-1].split(": ", 1)[-1]
                raise RuntimeError(last_pre_error)
            raise RuntimeError(normalize_error_message("所有组合均无有效图片"))

        if is_batch_gemini_model(model) and len(tasks) > 1:
            tasks = [
                (idx, fn, args[:-1] + (f"{batch_item_id}-{idx}",))
                for idx, fn, args in tasks
            ]

        print(f"[知衣图生图] 并发发送 {len(tasks)} 个请求（{len(combos)} 个组合 × batch_size {batch_size}），并发上限 {max_concurrency}")
        results, log_lines, last_error_message = self._run_concurrent(tasks, max_concurrency, label="请求")

        successful = [r for r in results if r is not None]
        url = "FD_LITELLM_BASE_URL/v1/chat/completions" if use_litellm else self.gemini_client.service_url
        header = f"总计: {len(successful)}/{len(tasks)} 成功, url={url}"
        if pre_errors:
            header += f"\n预处理失败 {len(pre_errors)} 个: " + "; ".join(pre_errors)
        log_lines.insert(0, header)
        log_text = "\n".join(log_lines)

        if not successful:
            raise RuntimeError(
                last_error_message or normalize_error_message(f"所有请求均失败，无图片返回\n{log_text}")
            )

        print(f"[知衣图生图] 成功 {len(successful)}/{len(tasks)}")
        return (successful, actual_seed, log_text)

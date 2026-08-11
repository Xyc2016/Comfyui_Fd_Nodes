"""知衣 SAM2 bbox 抠图节点 V2：env 下拉切换灰度/开发/自定义环境，并补齐 node_switch 开关。"""

import json

import torch

from ..config import FD_SAM2_SEGMENT_URL, SAM2_SERVICE_ENVS
from ..sam2_segment_node import ZhiYiSAM2SegmentNode
from .env import resolve_service_env, service_env_options


class ZhiYiSAM2SegmentNodeV2(ZhiYiSAM2SegmentNode):
    """知衣 SAM2 bbox 抠图节点 V2 - 支持灰度/开发环境一键切换。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "bboxes": ("BBOXES",),
                "env": (service_env_options(), {
                    "default": "灰度环境",
                    "tooltip": "灰度/开发环境使用内置地址（忽略 service_url 与 FD_* 环境变量）；自定义使用 service_url",
                }),
                "service_url": ("STRING", {
                    "default": FD_SAM2_SEGMENT_URL or "",
                    "multiline": False,
                    "tooltip": "SAM2 分割接口地址，例如 http://host:8000/v1/segment；仅环境=自定义时使用",
                }),
                "dilate": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                }),
                "blur": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                }),
                "background": (["Alpha", "Color", "Original"], {"default": "Alpha"}),
                "background_color": ("STRING", {"default": "#222222"}),
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
                "invert_output": ("BOOLEAN", {"default": False}),
                "health_check": ("BOOLEAN", {"default": False}),
                "node_switch": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "JSON")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "INFO")
    FUNCTION = "segment"
    CATEGORY = "知衣/抠图"
    OUTPUT_NODE = False

    @staticmethod
    def _empty_mask_tensor(batch, height, width):
        return torch.zeros((batch, height, width), dtype=torch.float32)

    def _validate_background(self, background, background_color):
        if background not in ("Alpha", "Color", "Original"):
            raise RuntimeError(f"background 无效: {background}")
        if background == "Color":
            self._parse_hex_color(background_color)

    def segment(
        self,
        images,
        bboxes,
        env,
        service_url,
        dilate,
        blur,
        background,
        background_color,
        max_concurrency,
        timeout,
        invert_output=False,
        health_check=False,
        node_switch=0,
    ):
        if node_switch == 1:
            image_batch = self._expand_images(images)
            if not image_batch:
                raise RuntimeError("未提供图片")
            image_batch = torch.cat(image_batch, dim=0)
            batch, height, width = image_batch.shape[0], *image_batch.shape[-3:-1]
            mask = torch.zeros(
                (batch, height, width),
                dtype=image_batch.dtype,
                device=image_batch.device,
            )
            return (
                image_batch,
                mask,
                self._mask_to_image_tensor(mask),
                '{"skipped": true, "reason": "node_switch"}',
            )

        self._validate_background(background, background_color)

        resolved_base = resolve_service_env(env, SAM2_SERVICE_ENVS)
        if resolved_base:
            service_url = f"{resolved_base.rstrip('/')}/v1/segment"

        service_url = (service_url or FD_SAM2_SEGMENT_URL or "").strip()
        if not service_url:
            raise RuntimeError("未配置 SAM2 服务地址，请设置 FD_SAM2_SEGMENT_URL 或在节点 service_url 中填写完整接口地址")

        image_tensors = self._expand_images(images)
        if not image_tensors:
            raise RuntimeError("未提供图片")

        bbox_groups = self._normalize_bboxes_batch(bboxes, len(image_tensors))
        if health_check and any(bbox_groups):
            self._check_health(service_url, timeout)

        tasks = []
        for idx, image_tensor in enumerate(image_tensors):
            if bbox_groups[idx]:
                task = (
                    idx,
                    self._single_request,
                    (
                        idx,
                        image_tensor,
                        bbox_groups[idx],
                        service_url,
                        dilate,
                        blur,
                        background,
                        background_color,
                        invert_output,
                        timeout,
                    ),
                )
            else:
                task = (idx, self._empty_bbox_result, (idx, image_tensor))
            tasks.append(task)

        print(f"[知衣SAM2抠图] 处理 {len(tasks)} 张图片，并发上限 {max_concurrency}")
        results = self._run_concurrent(tasks, max_concurrency)
        info = {
            "service_url": service_url,
            "background": background,
            "dilate": int(dilate),
            "blur": int(blur),
            "env": env,
            "items": [result["info"] for result in results],
        }

        return (
            torch.cat([result["result_image"] for result in results], dim=0),
            torch.cat([result["mask"] for result in results], dim=0),
            torch.cat([result["mask_image"] for result in results], dim=0),
            json.dumps(info, ensure_ascii=False),
        )

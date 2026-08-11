"""知衣 DWPose/LineArt/Depth 外部 API 节点 V2：env 下拉切换灰度/开发/自定义环境。"""

import json

import torch

from ..config import (
    CUSTOM_SERVICE_URL_PRESET,
    DWPOSE_SERVICE_ENVS,
    FD_CONTROLNET_AUX_DEPTH_ANYTHING_V2_URL,
    FD_CONTROLNET_AUX_LINEART_URL,
    FD_DWPOSE_POSE_URL,
)
from ..zhiyi_controlnet_aux_node import (
    UPSCALE_METHODS,
    _ControlNetAuxApiBase,
)
from ..zhiyi_dwpose_detect_node import (
    DEFAULT_DWPOSE_POSE_URL,
    ZhiYiDWPoseDetectNode,
    _normalize_pose_url,
)
from .env import resolve_service_env, service_env_options


class ZhiYiDWPoseDetectNodeV2(ZhiYiDWPoseDetectNode):
    """知衣 DWPose 姿态检测节点 V2 - 支持灰度/开发环境一键切换与返回开关。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "env": (service_env_options(), {
                    "default": "灰度环境",
                    "tooltip": "灰度/开发环境使用内置地址（忽略 service_url 与 FD_* 环境变量）；自定义使用 service_url",
                }),
                "service_url": ("STRING", {
                    "default": FD_DWPOSE_POSE_URL or DEFAULT_DWPOSE_POSE_URL,
                    "multiline": False,
                    "tooltip": "DWPose 姿态检测接口地址，例如 http://host:8001/v1/pose；仅环境=自定义时使用",
                }),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "detect_body": ("BOOLEAN", {"default": True}),
                "detect_hand": ("BOOLEAN", {"default": True}),
                "detect_face": ("BOOLEAN", {"default": False}),
                "xinsr_stick_scaling": ("BOOLEAN", {"default": False}),
                "return_pose_image": ("BOOLEAN", {"default": True}),
                "return_openpose_json": ("BOOLEAN", {"default": True}),
                "upscale_method": (UPSCALE_METHODS, {"default": "INTER_CUBIC"}),
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
                "health_check": ("BOOLEAN", {"default": False}),
                "node_switch": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "JSON")
    RETURN_NAMES = ("POSE_IMAGE", "POSE_KEYPOINT", "INFO")
    FUNCTION = "detect_pose"
    CATEGORY = "知衣/姿态检测"
    OUTPUT_NODE = False

    def detect_pose(
        self,
        image,
        env,
        service_url,
        resolution,
        detect_body,
        detect_hand,
        detect_face,
        xinsr_stick_scaling,
        return_pose_image,
        return_openpose_json,
        upscale_method,
        max_concurrency,
        timeout,
        health_check=False,
        node_switch=0,
    ):
        if node_switch == 1:
            return (
                self._normalize_image_batch(image),
                [],
                json.dumps({"skipped": True, "reason": "node_switch"}, ensure_ascii=False),
            )

        resolved_base = resolve_service_env(env, DWPOSE_SERVICE_ENVS)
        selected_service_url = resolved_base or service_url
        final_service_url = _normalize_pose_url(selected_service_url)
        image_tensors = self._expand_images(image)
        if not image_tensors:
            raise RuntimeError("未提供图片")
        if health_check:
            self._check_health(final_service_url, timeout)

        tasks = []
        for idx, image_tensor in enumerate(image_tensors):
            tasks.append((
                idx,
                self._single_request,
                (
                    idx,
                    image_tensor,
                    final_service_url,
                    resolution,
                    detect_body,
                    detect_hand,
                    detect_face,
                    xinsr_stick_scaling,
                    upscale_method,
                    timeout,
                    return_pose_image,
                    return_openpose_json,
                ),
            ))

        print(f"[知衣DWPose姿态检测] 处理 {len(tasks)} 张图片，并发上限 {max_concurrency}")
        results = self._run_concurrent(tasks, max_concurrency)
        info = {
            "service_url": final_service_url,
            "resolution": int(resolution),
            "detect_body": bool(detect_body),
            "detect_hand": bool(detect_hand),
            "detect_face": bool(detect_face),
            "xinsr_stick_scaling": bool(xinsr_stick_scaling),
            "return_pose_image": bool(return_pose_image),
            "return_openpose_json": bool(return_openpose_json),
            "upscale_method": upscale_method,
            "items": [result["info"] for result in results],
        }

        return (
            torch.cat([result["pose_image"] for result in results], dim=0),
            [result["openpose_json"] for result in results],
            json.dumps(info, ensure_ascii=False),
        )


class ZhiYiLineArtPreprocessorNodeV2(_ControlNetAuxApiBase):
    """知衣 LineArt 线稿预处理节点 V2 - 支持灰度/开发环境一键切换与 return_image 开关。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "env": (service_env_options(), {
                    "default": "灰度环境",
                    "tooltip": "灰度/开发环境使用内置地址（忽略 service_url 与 FD_* 环境变量）；自定义使用 service_url",
                }),
                "service_url": ("STRING", {
                    "default": FD_CONTROLNET_AUX_LINEART_URL or "http://model-api-dwpose-svc.online-server-gray:8001/v1/lineart",
                    "multiline": False,
                    "tooltip": "LineArt 线稿预处理接口地址，例如 http://host:8001/v1/lineart；仅环境=自定义时使用",
                }),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "coarse": ("BOOLEAN", {"default": False}),
                "return_image": ("BOOLEAN", {"default": True}),
                "upscale_method": (UPSCALE_METHODS, {"default": "INTER_CUBIC"}),
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
                "health_check": ("BOOLEAN", {"default": False}),
                "node_switch": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "JSON")
    RETURN_NAMES = ("LINEART_IMAGE", "INFO")
    FUNCTION = "preprocess"
    CATEGORY = "知衣/ControlNet预处理"
    OUTPUT_NODE = False

    def preprocess(
        self,
        image,
        env,
        service_url,
        resolution,
        coarse,
        return_image,
        upscale_method,
        max_concurrency,
        timeout,
        health_check=False,
        node_switch=0,
    ):
        resolved_base = resolve_service_env(env, DWPOSE_SERVICE_ENVS)
        return self._execute(
            image=image,
            service_url=resolved_base or service_url,
            endpoint="lineart",
            default_url=FD_CONTROLNET_AUX_LINEART_URL or "http://model-api-dwpose-svc.online-server-gray:8001/v1/lineart",
            health_key="lineart",
            label="知衣LineArt线稿预处理",
            resolution=resolution,
            upscale_method=upscale_method,
            max_concurrency=max_concurrency,
            timeout=timeout,
            health_check=health_check,
            node_switch=node_switch,
            service_url_preset=CUSTOM_SERVICE_URL_PRESET,
            extra={"coarse": bool(coarse), "return_image": bool(return_image)},
            info_extra={"coarse": bool(coarse), "return_image": bool(return_image)},
        )


class ZhiYiDepthAnythingV2PreprocessorNodeV2(_ControlNetAuxApiBase):
    """知衣 Depth Anything V2 深度图预处理节点 V2 - 支持灰度/开发环境一键切换。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "env": (service_env_options(), {
                    "default": "灰度环境",
                    "tooltip": "灰度/开发环境使用内置地址（忽略 service_url 与 FD_* 环境变量）；自定义使用 service_url",
                }),
                "service_url": ("STRING", {
                    "default": FD_CONTROLNET_AUX_DEPTH_ANYTHING_V2_URL or "http://model-api-dwpose-svc.online-server-gray:8001/v1/depth-anything-v2",
                    "multiline": False,
                    "tooltip": "Depth Anything V2 预处理接口地址，例如 http://host:8001/v1/depth-anything-v2；仅环境=自定义时使用",
                }),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "max_depth": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01,
                }),
                "upscale_method": (UPSCALE_METHODS, {"default": "INTER_CUBIC"}),
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
                "health_check": ("BOOLEAN", {"default": False}),
                "node_switch": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "JSON")
    RETURN_NAMES = ("DEPTH_IMAGE", "INFO")
    FUNCTION = "preprocess"
    CATEGORY = "知衣/ControlNet预处理"
    OUTPUT_NODE = False

    def preprocess(
        self,
        image,
        env,
        service_url,
        resolution,
        max_depth,
        upscale_method,
        max_concurrency,
        timeout,
        health_check=False,
        node_switch=0,
    ):
        resolved_base = resolve_service_env(env, DWPOSE_SERVICE_ENVS)
        return self._execute(
            image=image,
            service_url=resolved_base or service_url,
            endpoint="depth-anything-v2",
            default_url=FD_CONTROLNET_AUX_DEPTH_ANYTHING_V2_URL or "http://model-api-dwpose-svc.online-server-gray:8001/v1/depth-anything-v2",
            health_key="depth_anything_v2",
            label="知衣DepthAnythingV2深度图预处理",
            resolution=resolution,
            upscale_method=upscale_method,
            max_concurrency=max_concurrency,
            timeout=timeout,
            health_check=health_check,
            node_switch=node_switch,
            service_url_preset=CUSTOM_SERVICE_URL_PRESET,
            extra={"max_depth": float(max_depth)},
            info_extra={"max_depth": float(max_depth)},
        )

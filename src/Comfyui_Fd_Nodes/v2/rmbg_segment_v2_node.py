"""知衣 RMBG/衣物/时尚/身体 外部 API 节点 V2：env 下拉切换灰度/开发/自定义环境。"""

from ..config import (
    CUSTOM_SERVICE_URL_PRESET,
    FD_BODY_SEGMENT_URL,
    FD_CLOTHES_SEGMENT_URL,
    FD_FASHION_SEGMENT_URL,
    FD_RMBG_URL,
    RMBG_SERVICE_ENVS,
)
from ..zhiyi_rmbg_segment_node import (
    BACKGROUND_MODES,
    BODY_CLASSES,
    CLOTHES_CLASSES,
    FASHION_CLASSES,
    _RmbgSegmentApiBase,
)
from .constants import BODY_CLASS_OPTIONS, CLOTHES_CLASS_OPTIONS, FASHION_CLASS_OPTIONS
from .env import resolve_service_env, service_env_options


class ZhiYiRMBGNodeV2(_RmbgSegmentApiBase):
    """知衣 RMBG 2.0 背景去除节点 V2 - 支持灰度/开发环境一键切换。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "env": (service_env_options(), {
                    "default": "dev",
                    "tooltip": "灰度/开发环境使用内置地址（忽略 service_url 与 FD_* 环境变量）；自定义使用 service_url",
                }),
                "service_url": ("STRING", {
                    "default": FD_RMBG_URL or "http://10.1.0.230:8003/v1/rmbg",
                    "multiline": False,
                    "tooltip": "RMBG 2.0 背景去除接口地址，例如 http://host:8003/v1/rmbg；仅环境=自定义时使用",
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
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "JSON")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "INFO")
    FUNCTION = "remove_background"
    CATEGORY = "知衣/抠图"
    OUTPUT_NODE = False

    def remove_background(
        self,
        image,
        env,
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
    ):
        resolved_base = resolve_service_env(env, RMBG_SERVICE_ENVS)
        return self._execute(
            image=image,
            service_url=resolved_base or service_url,
            endpoint="rmbg",
            default_url=FD_RMBG_URL or "http://10.1.0.230:8003/v1/rmbg",
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
            service_url_preset=CUSTOM_SERVICE_URL_PRESET,
            extra={"sensitivity": float(sensitivity), "refine_foreground": bool(refine_foreground)},
            info_extra={"sensitivity": float(sensitivity), "refine_foreground": bool(refine_foreground)},
        )


class ZhiYiClothesSegmentNodeV2(_RmbgSegmentApiBase):
    """知衣衣物语义分割节点 V2 - 支持 18 个类别多选下拉。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "env": (service_env_options(), {
                    "default": "dev",
                    "tooltip": "灰度/开发环境使用内置地址（忽略 service_url 与 FD_* 环境变量）；自定义使用 service_url",
                }),
                "service_url": ("STRING", {
                    "default": FD_CLOTHES_SEGMENT_URL or "http://10.1.0.230:8003/v1/segment/clothes",
                    "multiline": False,
                    "tooltip": "衣物语义分割接口地址，例如 http://host:8003/v1/segment/clothes；仅环境=自定义时使用",
                }),
                "classes": (CLOTHES_CLASS_OPTIONS, {
                    "default": ["Upper-clothes"],
                    "multiselect": True,
                    "multi_select": {"placeholder": "选择类别", "chip": True},
                    "tooltip": "多选类别；清空选择则使用服务默认类别",
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
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "JSON")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "INFO")
    FUNCTION = "segment"
    CATEGORY = "知衣/语义分割"
    OUTPUT_NODE = False

    def segment(
        self,
        image,
        env,
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
    ):
        selected_classes = self._parse_classes_text(classes, CLOTHES_CLASSES, allow_comma_split=True)
        resolved_base = resolve_service_env(env, RMBG_SERVICE_ENVS)
        return self._execute(
            image=image,
            service_url=resolved_base or service_url,
            endpoint="segment/clothes",
            default_url=FD_CLOTHES_SEGMENT_URL or "http://10.1.0.230:8003/v1/segment/clothes",
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
            service_url_preset=CUSTOM_SERVICE_URL_PRESET,
            extra={"classes": selected_classes},
            info_extra={"classes": selected_classes},
        )


class ZhiYiFashionSegmentNodeV2(_RmbgSegmentApiBase):
    """知衣时尚单品分割节点 V2 - 支持 47 个单品类别多选下拉，类别名中的逗号按原样提交。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "env": (service_env_options(), {
                    "default": "dev",
                    "tooltip": "灰度/开发环境使用内置地址（忽略 service_url 与 FD_* 环境变量）；自定义使用 service_url",
                }),
                "service_url": ("STRING", {
                    "default": FD_FASHION_SEGMENT_URL or "http://10.1.0.230:8003/v1/segment/fashion",
                    "multiline": False,
                    "tooltip": "时尚单品分割接口地址，例如 http://host:8003/v1/segment/fashion；仅环境=自定义时使用",
                }),
                "classes": (FASHION_CLASS_OPTIONS, {
                    "default": ["shirt, blouse"],
                    "multiselect": True,
                    "multi_select": {"placeholder": "选择类别", "chip": True},
                    "tooltip": "多选类别（类别名中的逗号需按原样保留）；清空选择则使用服务默认类别",
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
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "JSON")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "INFO")
    FUNCTION = "segment"
    CATEGORY = "知衣/语义分割"
    OUTPUT_NODE = False

    def segment(
        self,
        image,
        env,
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
    ):
        selected_classes = self._parse_classes_text(classes, FASHION_CLASSES, allow_comma_split=False)
        resolved_base = resolve_service_env(env, RMBG_SERVICE_ENVS)
        return self._execute(
            image=image,
            service_url=resolved_base or service_url,
            endpoint="segment/fashion",
            default_url=FD_FASHION_SEGMENT_URL or "http://10.1.0.230:8003/v1/segment/fashion",
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
            service_url_preset=CUSTOM_SERVICE_URL_PRESET,
            extra={"classes": selected_classes},
            info_extra={"classes": selected_classes},
        )


class ZhiYiBodySegmentNodeV2(_RmbgSegmentApiBase):
    """知衣身体部位分割节点 V2 - 支持 12 个部位类别多选下拉。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "env": (service_env_options(), {
                    "default": "dev",
                    "tooltip": "灰度/开发环境使用内置地址（忽略 service_url 与 FD_* 环境变量）；自定义使用 service_url",
                }),
                "service_url": ("STRING", {
                    "default": FD_BODY_SEGMENT_URL or "http://10.1.0.230:8003/v1/segment/body",
                    "multiline": False,
                    "tooltip": "身体部位分割接口地址，例如 http://host:8003/v1/segment/body；仅环境=自定义时使用",
                }),
                "classes": (BODY_CLASS_OPTIONS, {
                    "default": ["Face", "Hair", "Top-clothes", "Bottom-clothes"],
                    "multiselect": True,
                    "multi_select": {"placeholder": "选择类别", "chip": True},
                    "tooltip": "多选类别；清空选择则使用服务默认类别",
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
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "JSON")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "INFO")
    FUNCTION = "segment"
    CATEGORY = "知衣/语义分割"
    OUTPUT_NODE = False

    def segment(
        self,
        image,
        env,
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
    ):
        selected_classes = self._parse_classes_text(classes, BODY_CLASSES, allow_comma_split=True)
        resolved_base = resolve_service_env(env, RMBG_SERVICE_ENVS)
        return self._execute(
            image=image,
            service_url=resolved_base or service_url,
            endpoint="segment/body",
            default_url=FD_BODY_SEGMENT_URL or "http://10.1.0.230:8003/v1/segment/body",
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
            service_url_preset=CUSTOM_SERVICE_URL_PRESET,
            extra={"classes": selected_classes},
            info_extra={"classes": selected_classes, "process_res_ignored": True},
        )

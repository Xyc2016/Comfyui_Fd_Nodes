import io
import json
import logging
import os
import traceback
from datetime import datetime
from inspect import cleandoc
from io import BytesIO
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import numpy as np
import oss2
import requests
import torch
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from PIL import Image

from .config import (
    FD_FLUX2KLEIN_PASSWORD,
    FD_FLUX2KLEIN_URL,
    FD_FLUX2KLEIN_USERNAME,
    FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL,
    FD_LITELLM_API_KEY,
    FD_LITELLM_BASE_URL,
    FD_OSS_ACCESS_KEY_ID,
    FD_OSS_ACCESS_KEY_SECRET,
    FD_OSS_BUCKET_NAME,
    FD_OSS_ENDPOINT,
    FD_OSS_URL_PATH_PREFIX_BEFORE_GEN,
    FD_OSS_URL_PATH_PREFIX_FLUX,
    FD_OSS_URL_PREFIX,
    FD_Z_IMAGE_TURBO_PASSWORD,
    FD_Z_IMAGE_TURBO_URL,
    FD_Z_IMAGE_TURBO_USERNAME,
)
from .old_fd_nodes import FD_imgToText_Doubao, FD_Upload
from .old_gemini_api_node import FD_GeminiImage, GenImageServiceError
from .utils.common_util import (
    bytes_calculate_hex_md5,
    bytesio_to_image_tensor,
    downscale_image_tensor,
)
from .utils.webhook import webhook_send
from .gpt_image_node import FD_GTPImage
from .gpt_image_combo_node import FD_GPTImageComboNode
from .gpt_multi_image_node import FD_GPTMultiImage
from .prompt_nodes import EcommercePromptGenerator, PromptListSelector
from .zhiyi_text_node import ZhiYiTextGenNode
from .zhiyi_image_text_node import ZhiYiImageTextNode
from .zhiyi_image_text_combo_node import ZhiYiImageTextComboNode
from .zhiyi_image_to_image_node import ZhiYiImageToImageNode
from .toggle import NodeToggleByID
from .utils.logging_utils import configure_default_logging
from .zhiyi_image_combo_node import ZhiYiImageComboNode
from .zhiyi_image_to_image_combo_node import ZhiYiImageToImageComboNode
from .aistudio_image_combo_node import ZhiYiAiStudioImageComboNode
from .remove_bg_by_meitu_node import ZhiYiRemoveBgByMeituNode
from .zhiyi_qwen_detect_node import ZhiYiBBoxesToSAM2, ZhiYiQwenDetectNode
from .utils.gpt_image_size import resolution_to_edit_size

configure_default_logging()
logger = logging.getLogger(__name__)


FD_REMOVE_WATERMARK_SERVICE_URL = os.getenv("FD_REMOVE_WATERMARK_SERVICE_URL", "http://localhost:8000/v1/process")


def _resolution_to_edit_size(resolution: str, aspect_ratio: str) -> str:
    return resolution_to_edit_size(resolution, aspect_ratio)


class FD_RemoveWatermark:
    """
    Remove watermark from image using AI inpainting service
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "service_url": ("STRING", {"default": FD_REMOVE_WATERMARK_SERVICE_URL, "multiline": False}),
                "text_prompt": (
                    "STRING",
                    {"default": "bottom right watermark. bottom left watermark. top left watermark. top right watermark. corner logo. watermark text.", "multiline": True},
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                    },
                ),
                "text_threshold": (
                    "FLOAT",
                    {
                        "default": 0.35,
                    },
                ),
                "max_side": (
                    "INT",
                    {
                        "default": 3000,
                    },
                ),
                "mask_dilate_ksize": (
                    "INT",
                    {
                        "default": 9,
                    },
                ),
                "mask_dilate_iters": (
                    "INT",
                    {
                        "default": 2,
                    },
                ),
                "inpaint_method": (["lama"],),
                "enable_quality_check": ("BOOLEAN", {"default": True}),
                "fallback_to_original": ("BOOLEAN", {"default": True}),
                "jpeg_quality": (
                    "INT",
                    {
                        "default": 95,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_watermark"
    CATEGORY = "image/postprocessing"

    def remove_watermark(
        self,
        image: torch.Tensor,
        service_url: str,
        text_prompt: str,
        threshold: float,
        text_threshold: float,
        max_side: int,
        mask_dilate_ksize: int,
        mask_dilate_iters: int,
        inpaint_method: str,
        enable_quality_check: bool,
        fallback_to_original: bool,
        jpeg_quality: int,
    ) -> Tuple[torch.Tensor]:
        call_id = str(uuid4())[:8]
        print(
            f"FD_RemoveWatermark [{call_id}] Start processing. ",
            {
                "call_id": call_id,
                "image_shape": list(image.shape),
                "service_url": service_url,
                "text_prompt": text_prompt,
                "threshold": threshold,
                "text_threshold": text_threshold,
                "max_side": max_side,
                "mask_dilate_ksize": mask_dilate_ksize,
                "mask_dilate_iters": mask_dilate_iters,
                "inpaint_method": inpaint_method,
                "enable_quality_check": enable_quality_check,
                "fallback_to_original": fallback_to_original,
                "jpeg_quality": jpeg_quality,
            },
        )
        try:
            # Convert ComfyUI tensor to PIL Image (only process first image in batch)
            img_tensor = image[0]  # [H, W, C] in range [0, 1]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Convert to bytes
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # Prepare request
            files = {"image": ("image.png", img_bytes, "image/png")}
            data = {
                "text_prompt": text_prompt,
                "threshold": threshold,
                "text_threshold": text_threshold,
                "max_side": max_side,
                "mask_dilate_ksize": mask_dilate_ksize,
                "mask_dilate_iters": mask_dilate_iters,
                "inpaint_method": inpaint_method,
                "enable_quality_check": enable_quality_check,
                "fallback_to_original": fallback_to_original,
                "jpeg_quality": jpeg_quality,
            }

            print(f"FD_RemoveWatermark [{call_id}] Calling service at {service_url} ...")
            # Call service
            response = requests.post(service_url, files=files, data=data, timeout=60)
            response.raise_for_status()

            # Convert response back to ComfyUI tensor
            result_img = Image.open(io.BytesIO(response.content)).convert("RGB")
            result_np = np.array(result_img).astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_np).unsqueeze(0)  # [1, H, W, C]

            print(f"FD_RemoveWatermark [{call_id}] Successfully processed image.")
            return (result_tensor,)

        except Exception as e:
            # Return original image on any error
            print(f"FD_RemoveWatermark [{call_id}] error: {e}, returning original image")
            return (image,)


def fd_flux2_klein_send_webhook(service_url: str, flux2_klein_req_body: dict):
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    headers = {"Content-Type": "application/json"}
    data = {
        "msgtype": "text",
        "text": {
            "content": json.dumps(
                {"datetime": now_str, "service_url": service_url, "flux2_klein_req_body": flux2_klein_req_body},
                ensure_ascii=False,
                indent=4,
            )
        },
    }

    requests.post(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, headers=headers, json=data)


class FD_Flux2KleinGenImage(ComfyNodeABC):
    """
    Node to generate text and image responses from a Flux2KleinGen model.
    """
    def __init__(self):
        auth = oss2.Auth(FD_OSS_ACCESS_KEY_ID, FD_OSS_ACCESS_KEY_SECRET)
        self.bucket = oss2.Bucket(
            auth=auth,
            bucket_name=FD_OSS_BUCKET_NAME,
            endpoint=FD_OSS_ENDPOINT,
            connect_timeout=30
        )
        self.oss_url_prefix = FD_OSS_URL_PREFIX

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "out_request_id": (
                    IO.STRING,
                    {
                        "default": "unknown_request_id",
                        "tooltip": "FD out_request_id for generation",
                    },
                ),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for generation",
                    },
                ),
                "aspect_ratio": (
                    ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9"],
                    {
                        "default": "auto",
                        "tooltip": "Aspect ratio for generation",
                    },
                ),
            },
            "optional": {
                "images": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional image(s) to use as context for the model. To include multiple images, you can use the Batch Images node.",
                    },
                ),
                "service_url": (
                    IO.STRING,
                    {
                        "default": FD_FLUX2KLEIN_URL,
                        "tooltip": "Flux2Klein service URL",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2 ** 32 - 1,
                        "tooltip": "Random seed for generation.",
                    },
                ),
                "resolution": (
                    IO.COMBO,
                    {
                        "options": ["1K", "2K"],
                        "default": "2K",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE, )
    FUNCTION = "api_call"
    CATEGORY = "image/generation"
    DESCRIPTION = "Edit images synchronously via Flux2Klein API."
    API_NODE = True

    def api_call(self,
        out_request_id: str,
        prompt: str,
        aspect_ratio: str,
        images: Optional[IO.IMAGE] = None,
        service_url: str = FD_FLUX2KLEIN_URL,
        seed: int = 0,
        resolution: str = "2K",
        **kwargs,
    ):
        body = {
            "out_request_id": out_request_id,
            "prompt": prompt,
            "seed": seed,
            "ratio": aspect_ratio,
            "size": resolution,
        }
        if images is not None:
            batch_size = images.shape[0]
            image_url_list = []
            for i in range(batch_size):
                single_image = images[i : i + 1]
                original_image = single_image.squeeze()
                scaled_image = downscale_image_tensor(single_image, total_pixels=2048 * 2048).squeeze()
                logger.info(
                    "FD_Flux2KleinGenImage Image %s resolution: original=%s scaled=%s",
                    i,
                    tuple(original_image.shape),
                    tuple(scaled_image.shape),
                )
                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()
                file_oss_path = f"{FD_OSS_URL_PATH_PREFIX_FLUX}/{bytes_calculate_hex_md5(img_byte_arr)}.png"
                self.bucket.put_object(file_oss_path, img_byte_arr)
                print(f"upload {file_oss_path}")
                oss_file_url = f"{self.oss_url_prefix}{file_oss_path}"
                image_url_list.append(oss_file_url)
            body['images'] = image_url_list

        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
                print("Sending flux2_klein webhook message...")
                fd_flux2_klein_send_webhook(service_url, body)
            except Exception:
                pass

        logger.info(f"Calling Flux2Klein API with {body}")
        try:
            # example response json {'urls': ['https://zhiyi-image.oss-cn-hangzhou.aliyuncs.com//devops/comfyui/output/20260121/bed973ec3ccb31d49d43a31d9f535b65.png'], 'status': 'success', 'cost_time': 45.2}
            response = requests.post(service_url, auth=(FD_FLUX2KLEIN_USERNAME, FD_FLUX2KLEIN_PASSWORD), json=body)
            response.raise_for_status()
            if response.status_code != 200:
                raise Exception(f"Failed to call API: {response.content}")
            result = response.json()
            logger.info(f"Flux2Klein API response: {result}")
            result_url = result["urls"][0] # TODO: 暂时只支持1张图
        except Exception:
            traceback.print_exc()
            raise GenImageServiceError("TIMEOUT")
        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
                print("Sending flux2_klein webhook message...")
                webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {
                    "flux2_klein_full": {
                        "request": body,
                        "response": result,
                    }
                })
            except Exception:
                pass
        image_content = requests.get(result_url).content
        image_bytesio = BytesIO(image_content)
        output_image = bytesio_to_image_tensor(image_bytesio)
        return (output_image,)


class FD_ZImageTurboGenImage(ComfyNodeABC):
    """
    Node to generate text and image responses from a Z-Image-Turbo model.
    """

    def __init__(self):
        auth = oss2.Auth(FD_OSS_ACCESS_KEY_ID, FD_OSS_ACCESS_KEY_SECRET)
        self.bucket = oss2.Bucket(
            auth=auth,
            bucket_name=FD_OSS_BUCKET_NAME,
            endpoint=FD_OSS_ENDPOINT,
            connect_timeout=30,
        )
        self.oss_url_prefix = FD_OSS_URL_PREFIX

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "out_request_id": (
                    IO.STRING,
                    {
                        "default": "unknown_request_id",
                        "tooltip": "FD out_request_id for generation",
                    },
                ),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for generation",
                    },
                ),
                "aspect_ratio": (
                    ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9"],
                    {
                        "default": "auto",
                        "tooltip": "Aspect ratio for generation",
                    },
                ),
            },
            "optional": {
                "images": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional image(s) to edit with. To include multiple images, you can use the Batch Images node.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 42,
                        "min": 0,
                        "max": 2 ** 32 - 1,
                        "tooltip": "Random seed for generation.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "number",
                        "tooltip": "How strongly to edit the input image.",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "display": "number",
                        "tooltip": "Classifier-free guidance scale.",
                    },
                ),
                "num_inference_steps": (
                    IO.INT,
                    {
                        "default": 9,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Number of denoising steps.",
                    },
                ),
                "num_images": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "tooltip": "Number of output images.",
                    },
                ),
                "resolution": (
                    IO.COMBO,
                    {
                        "options": ["1K", "2K"],
                        "default": "1K",
                        "tooltip": "Output image size.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE, IO.STRING)
    FUNCTION = "api_call"
    CATEGORY = "image/generation"
    DESCRIPTION = "Edit images synchronously via Z-Image-Turbo API."
    API_NODE = True

    def api_call(
        self,
        out_request_id: str,
        prompt: str,
        aspect_ratio: str,
        images: Optional[IO.IMAGE] = None,
        seed: int = 42,
        strength: float = 0.25,
        guidance_scale: float = 0.0,
        num_inference_steps: int = 9,
        num_images: int = 1,
        resolution: str = "1K",
        **kwargs,
    ):
        body = {
            "out_request_id": out_request_id,
            "prompt": prompt,
            "seed": seed,
            "ratio": aspect_ratio,
            "size": resolution,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "num_images": num_images,
        }
        if images is not None:
            batch_size = images.shape[0]
            image_url_list = []
            for i in range(batch_size):
                single_image = images[i : i + 1]
                scaled_image = single_image.squeeze()
                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()
                file_oss_path = f"{FD_OSS_URL_PATH_PREFIX_FLUX}/{bytes_calculate_hex_md5(img_byte_arr)}.png"
                self.bucket.put_object(file_oss_path, img_byte_arr)
                print(f"upload {file_oss_path}")
                oss_file_url = f"{self.oss_url_prefix}{file_oss_path}"
                image_url_list.append(oss_file_url)
            body["images"] = image_url_list

        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
                print("Sending z_image_turbo webhook message...")
                webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {
                    "z_image_turbo_request": {
                        "service_url": FD_Z_IMAGE_TURBO_URL,
                        "request": body,
                    }
                })
            except Exception:
                pass

        logger.info(f"Calling Z-Image-Turbo API with {body}")
        try:
            response = requests.post(FD_Z_IMAGE_TURBO_URL, auth=(FD_Z_IMAGE_TURBO_USERNAME, FD_Z_IMAGE_TURBO_PASSWORD), json=body)
            response.raise_for_status()
            if response.status_code != 200:
                raise Exception(f"Failed to call API: {response.content}")
            result = response.json()
            logger.info(f"Z-Image-Turbo API response: {result}")
            result_url = result["urls"][0]
        except Exception:
            traceback.print_exc()
            raise

        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
                print("Sending z_image_turbo webhook message...")
                webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {
                    "z_image_turbo_full": {
                        "request": body,
                        "response": result,
                    }
                })
            except Exception:
                pass
        image_content = requests.get(result_url).content
        image_bytesio = BytesIO(image_content)
        output_image = bytesio_to_image_tensor(image_bytesio)
        return (output_image, result_url)


class FD_SeedreamImage(ComfyNodeABC):
    """
    Node to generate images using Seedream 5.0 Lite model.
    """
    def __init__(self):
        auth = oss2.Auth(FD_OSS_ACCESS_KEY_ID, FD_OSS_ACCESS_KEY_SECRET)
        self.bucket = oss2.Bucket(
            auth=auth,
            bucket_name=FD_OSS_BUCKET_NAME,
            endpoint=FD_OSS_ENDPOINT,
            connect_timeout=30
        )
        self.oss_url_prefix = FD_OSS_URL_PREFIX

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for generation",
                    },
                ),
                "model": (
                    ["doubao-seedream-5.0-lite"],
                    {
                        "default": "doubao-seedream-5.0-lite",
                        "tooltip": "Model to use for generation",
                    },
                ),
                "size": (
                    ["2K", "3K"],
                    {
                        "default": "2K",
                        "tooltip": "Output image size",
                    },
                ),
            },
            "optional": {
                "images": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional image(s) to use as context. Multiple images supported.",
                    },
                ),
                "output_format": (
                    ["png", "jpg"],
                    {
                        "default": "png",
                        "tooltip": "Output image format",
                    },
                ),
            },
        }

    RETURN_TYPES = (IO.IMAGE, )
    FUNCTION = "api_call"
    CATEGORY = "image/generation"
    DESCRIPTION = "Generate images using Seedream 5.0 Lite model."
    API_NODE = True

    def api_call(
        self,
        prompt: str,
        model: str,
        size: str,
        images: Optional[IO.IMAGE] = None,
        output_format: str = "png",
        **kwargs,
    ):
        body = {
            "model": model,
            "prompt": prompt,
            "sequential_image_generation": "disabled",
            "size": size,
            "output_format": output_format,
            "watermark": False,
        }

        # Upload images to OSS if provided
        if images is not None:
            batch_size = images.shape[0]
            image_url_list = []
            for i in range(batch_size):
                single_image = images[i : i + 1]
                # original_image = single_image.squeeze()
                # scaled_image = downscale_image_tensor(single_image, total_pixels=2048 * 2048).squeeze()
                # logger.info(
                #     "FD_SeedreamImage Image %s resolution: original=%s scaled=%s",
                #     i,
                #     tuple(original_image.shape),
                #     tuple(scaled_image.shape),
                # )
                scaled_image = single_image.squeeze() # 现在是测试阶段，先不要downscale TODO: 到时候考虑改一下
                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()
                file_oss_path = f"{FD_OSS_URL_PATH_PREFIX_BEFORE_GEN}/{bytes_calculate_hex_md5(img_byte_arr)}.png"
                self.bucket.put_object(file_oss_path, img_byte_arr)
                print(f"upload {file_oss_path}")
                oss_file_url = f"{self.oss_url_prefix}{file_oss_path}"
                image_url_list.append(oss_file_url)
            body['image'] = image_url_list

        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
                print("Sending seedream webhook message...")
                webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {
                    "seedream_request": body,
                })
            except Exception:
                pass

        logger.info(f"Calling Seedream API with {body}")

        try:
            # Call API
            headers = {
                "Authorization": f"Bearer {FD_LITELLM_API_KEY}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                url=f"{FD_LITELLM_BASE_URL}/v1/images/generations",
                headers=headers,
                json=body,
                timeout=300,
            )
            response.raise_for_status()

            if response.status_code != 200:
                raise Exception(f"Failed to call API: {response.content}")

            result = response.json()
            logger.info(f"Seedream API response: {result}")

            # Get result image URL
            result_url = result["data"][0]["url"]
        except Exception:
            traceback.print_exc()
            raise GenImageServiceError("TIMEOUT")

        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
                print("Sending seedream webhook message...")
                webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {
                    "seedream_full": {
                        "request": body,
                        "response": result,
                    }
                })
            except Exception:
                pass
        image_content = requests.get(result_url).content
        image_bytesio = BytesIO(image_content)
        output_image = bytesio_to_image_tensor(image_bytesio)

        return (output_image,)


class Example:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("Image", { "tooltip": "This is an image"}),
                "int_field": ("INT", {
                    "default": 0,
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "float_field": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number"}),
                "print_to_screen": (["enable", "disable"],),
                "string_field": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Hello World!"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "test"

    #OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Example"

    def test(self, image, string_field, int_field, float_field, print_to_screen):
        if print_to_screen == "enable":
            print(f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """)
        #do some processing on the image, in this example I just invert it
        image = 1.0 - image
        return (image,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FD_RemoveWatermark": FD_RemoveWatermark,
    "FD_Upload": FD_Upload,
    "FD_imgToText_Doubao": FD_imgToText_Doubao,
    "FD_GeminiImage": FD_GeminiImage,
    "FD_GTPImage": FD_GTPImage,
    "FD_GPTImageComboNode": FD_GPTImageComboNode,
    "FD_GPTMultiImage": FD_GPTMultiImage,
    "FD_Flux2KleinGenImage": FD_Flux2KleinGenImage,
    "FD_ZImageTurboGenImage": FD_ZImageTurboGenImage,
    "FD_SeedreamImage": FD_SeedreamImage,
    "MaoziEcommercePromptGenerator": EcommercePromptGenerator,
    "MaoziPromptListSelector": PromptListSelector,
    "ZhiYiImageTextNode": ZhiYiImageTextNode,
    "ZhiYiImageTextComboNode": ZhiYiImageTextComboNode,
    "ZhiYiImageToImageNode": ZhiYiImageToImageNode,
    "ZhiYiTextGenNode": ZhiYiTextGenNode,
    "NodeToggleByID": NodeToggleByID,
    "ZhiYiImageComboNode": ZhiYiImageComboNode,
    "ZhiYiImageToImageComboNode": ZhiYiImageToImageComboNode,
    "ZhiYiAiStudioImageComboNode": ZhiYiAiStudioImageComboNode,
    "ZhiYiRemoveBgByMeituNode": ZhiYiRemoveBgByMeituNode,
    "ZhiYiQwenDetectNode": ZhiYiQwenDetectNode,
    "ZhiYiBBoxesToSAM2": ZhiYiBBoxesToSAM2,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FD_RemoveWatermark": "Remove Watermark",
    "FD_Upload": "FD Upload to OSS",
    "FD_imgToText_Doubao": "FD Image to Text (Doubao)",
    "FD_GeminiImage": "FD Gemini Image",
    "FD_GTPImage": "FD GTP Image",
    "FD_GPTImageComboNode": "FD GPT Image Combo",
    "FD_GPTMultiImage": "FD GPT Multi Image",
    "FD_Flux2KleinGenImage": "FD Flux2Klein Gen Image",
    "FD_ZImageTurboGenImage": "FD Z-Image-Turbo Gen Image",
    "FD_SeedreamImage": "FD Seedream Image",
    "MaoziEcommercePromptGenerator": "猫子提示词节点-详情页生成器",
    "MaoziPromptListSelector": "猫子提示词节点-列表选择器",
    "ZhiYiImageTextNode": "知衣-图生文",
    "ZhiYiImageTextComboNode": "知衣-图生文-combo",
    "ZhiYiImageToImageNode": "知衣-图生图",
    "ZhiYiTextGenNode": "知衣-文生文",
    "NodeToggleByID": "节点开关 (按ID)",
    "ZhiYiImageComboNode": "fd-输入参数组合",
    "ZhiYiImageToImageComboNode": "知衣-图生图-combo",
    "ZhiYiAiStudioImageComboNode": "知衣-AiStudio图生图-combo",
    "ZhiYiRemoveBgByMeituNode": "知衣-美图服装抠图",
    "ZhiYiQwenDetectNode": "知衣-Qwen目标检测",
    "ZhiYiBBoxesToSAM2": "知衣-BBox转SAM2格式",
}

from datetime import datetime
import io
import json
import logging
import os
from inspect import cleandoc
from io import BytesIO
import random
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
    FD_OSS_ACCESS_KEY_ID,
    FD_OSS_ACCESS_KEY_SECRET,
    FD_OSS_BUCKET_NAME,
    FD_OSS_ENDPOINT,
    FD_OSS_URL_PATH_PREFIX_FLUX,
    FD_OSS_URL_PREFIX,
)
from .old_fd_nodes import FD_imgToText_Doubao, FD_Upload
from .old_gemini_api_node import FD_GeminiImage
from .utils.common_util import (
    bytes_calculate_hex_md5,
    bytesio_to_image_tensor,
    downscale_image_tensor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


FD_REMOVE_WATERMARK_SERVICE_URL = os.getenv("FD_REMOVE_WATERMARK_SERVICE_URL", "http://localhost:8000/v1/process")


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


def fd_flux2_klein_send_webhook(flux2_klein_req_body: dict):
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    headers = {"Content-Type": "application/json"}
    data = {"msgtype": "text", "text": {"content": json.dumps({"datetime": now_str, "flux2_klein_req_body": flux2_klein_req_body}, ensure_ascii=False, indent=4)}}

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
                "service_url": ("STRING", {"default": FD_FLUX2KLEIN_URL, "multiline": False}),
                "out_request_id": (
                    IO.STRING,
                    {
                        "default": "default",
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
                    ["auto", "1:1", "3:4", "9:16"],
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
        service_url: str,
        out_request_id: str,
        prompt: str,
        aspect_ratio: str,
        images: Optional[IO.IMAGE] = None,
        **kwargs,
    ):
        body = {
            "out_request_id": out_request_id,
            "prompt": prompt,
            "seed": random.getrandbits(28),
            "ratio": aspect_ratio,
        }
        if images is not None:
            batch_size = images.shape[0]
            image_url_list = []
            for i in range(batch_size):
                single_image = images[i : i + 1]
                scaled_image = downscale_image_tensor(single_image).squeeze()

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
                fd_flux2_klein_send_webhook(body)
            except Exception:
                pass

        logger.info(f"Calling Flux2Klein API with {body}")
        # example response json {'urls': ['https://zhiyi-image.oss-cn-hangzhou.aliyuncs.com//devops/comfyui/output/20260121/bed973ec3ccb31d49d43a31d9f535b65.png'], 'status': 'success', 'cost_time': 45.2}
        response = requests.post(service_url, auth=(FD_FLUX2KLEIN_USERNAME, FD_FLUX2KLEIN_PASSWORD), json=body)
        response.raise_for_status()
        if response.status_code != 200:
            raise Exception(f"Failed to call API: {response.content}")
        result = response.json()
        logger.info(f"Flux2Klein API response: {result}")
        result_url = result["urls"][0] # TODO: 暂时只支持1张图
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
    "FD_Flux2KleinGenImage": FD_Flux2KleinGenImage,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FD_RemoveWatermark": "Remove Watermark",
    "FD_Upload": "FD Upload to OSS",
    "FD_imgToText_Doubao": "FD Image to Text (Doubao)",
    "FD_GeminiImage": "FD Gemini Image",
    "FD_Flux2KleinGenImage": "FD Flux2Klein Gen Image",
}

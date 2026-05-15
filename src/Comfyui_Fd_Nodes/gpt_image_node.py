import logging
from io import BytesIO
from typing import Optional

import numpy as np
import torch
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from PIL import Image

from .config import (
    FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL,
    FD_LITELLM_API_KEY,
    FD_LITELLM_BASE_URL,
)
from .utils.common_util import bytesio_to_image_tensor, downscale_image_tensor
from .utils.gpt_image_size import resolution_to_edit_size
from .utils.gpt_image_request import GptImageRequestMixin
from .utils.logging_utils import configure_default_logging
from .utils.webhook import webhook_send

configure_default_logging()
logger = logging.getLogger(__name__)


def _resolution_to_edit_size(resolution: str, aspect_ratio: str) -> str:
    return resolution_to_edit_size(resolution, aspect_ratio)


def _image_tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    image_np = (image.numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


class FD_GTPImage(GptImageRequestMixin, ComfyNodeABC):
    """
    Node to edit images using GPT Image API.
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
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
                "model": (
                    IO.COMBO,
                    {
                        "tooltip": "The GPT image model to use for image edits.",
                        "options": ["gpt-image-2"],
                        "default": "gpt-image-2",
                    },
                ),
                "resolution": (
                    IO.COMBO,
                    {
                        "tooltip": "Output size preset for the image edit request.",
                        "options": ["1K", "2K", "4K"],
                        "default": "2K",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Reserved for compatibility with FD_GeminiImage input format.",
                    },
                ),
            },
            "optional": {
                "images": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Input image(s) to edit. Multiple images are sent as repeated multipart fields.",
                    },
                ),
                "files": (
                    "GEMINI_INPUT_FILES",
                    {
                        "default": None,
                        "tooltip": "Reserved for compatibility with FD_GeminiImage input format.",
                    },
                ),
                "aspect_ratio": (
                    IO.COMBO,
                    {
                        "default": "",
                        "options": ["", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9"],
                        "tooltip": "Optional aspect ratio for the edited image.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE, IO.STRING, IO.STRING)
    FUNCTION = "api_call"
    CATEGORY = "image/generation"
    DESCRIPTION = "Edit images synchronously via GPT Image API."
    API_NODE = True

    def api_call(
        self,
        out_request_id: str,
        prompt: str,
        model: str,
        resolution: Optional[str] = None,
        images: Optional[IO.IMAGE] = None,
        aspect_ratio: str = "",
        files=None,
        seed: int = 42,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        del files, seed, unique_id, kwargs

        if images is None:
            raise ValueError("FD_GTPImage requires at least one input image.")
        if not prompt or not prompt.strip():
            raise ValueError("FD_GTPImage requires a non-empty prompt.")

        size = _resolution_to_edit_size(resolution or "2K", aspect_ratio)
        data = {
            "model": model,
            "prompt": prompt.strip(),
            "size": size,
            "user": out_request_id,
            "quality": "medium",
        }

        multipart_files = []
        batch_size = images.shape[0]
        for i in range(batch_size):
            single_image = images[i : i + 1]
            scaled_image = downscale_image_tensor(single_image, total_pixels=4096 * 4096).squeeze()
            logger.info(
                "FD_GTPImage Image %s resolution: input=%s scaled=%s",
                i,
                tuple(single_image.squeeze().shape),
                tuple(scaled_image.shape),
            )
            img_bytes = _image_tensor_to_png_bytes(scaled_image)
            multipart_files.append(
                ("image", (f"image_{i}.png", img_bytes, "image/png"))
            )

        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
                print("Sending gtp_image webhook message...")
                webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {
                    "gtp_image_request": {
                        "data": data,
                        "image_count": batch_size,
                    }
                })
            except Exception:
                pass

        logger.info("Calling GPT Image API with data=%s image_count=%s", data, batch_size)

        image_bytesio, output_text, result_url = self._call_gpt_image_with_retry_policy(
            base_url=FD_LITELLM_BASE_URL,
            api_key=FD_LITELLM_API_KEY,
            data=data,
            multipart_files=multipart_files,
            batch_size=batch_size,
            logger=logger,
        )
        output_image = bytesio_to_image_tensor(image_bytesio)
        return (output_image, output_text, result_url)

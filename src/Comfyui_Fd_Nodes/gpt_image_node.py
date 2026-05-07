import base64
import logging
import traceback
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import requests
import torch
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from PIL import Image

from .config import (
    FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL,
    FD_LITELLM_API_KEY,
    FD_LITELLM_BASE_URL,
)
from .old_gemini_api_node import GenImageServiceError
from .utils.common_util import bytesio_to_image_tensor, downscale_image_tensor
from .utils.gpt_image_size import resolution_to_edit_size
from .utils.logging_utils import configure_default_logging
from .utils.webhook import webhook_send

configure_default_logging()
logger = logging.getLogger(__name__)


def _image_tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    image_np = (image.numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def _summarize_gpt_image_result(result: dict) -> dict:
    data_items = result.get("data", [])
    item_summaries = []
    for item in data_items:
        item_summaries.append(
            {
                "keys": sorted(key for key in item.keys() if key != "b64_json"),
                "has_b64_json": "b64_json" in item,
                "has_url": bool(item.get("url")),
            }
        )
    return {
        "keys": sorted(key for key in result.keys() if key != "data"),
        "data_count": len(data_items),
        "data_items": item_summaries,
    }


class FD_GTPImage(ComfyNodeABC):
    """
    Node to edit images using GPT Image API.
    """

    PRIMARY_MODEL_MAX_ATTEMPTS = 3
    AZURE_FALLBACK_MODEL = "gpt-image-2-azure"

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
                        "options": ["", "1:1", "3:4", "9:16"],
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

    def _decode_gpt_image_result(self, result: dict) -> tuple[BytesIO, str, str]:
        first_item = result["data"][0]
        result_url = first_item.get("url", "")

        if result_url:
            image_content = requests.get(result_url, timeout=300).content
            image_bytesio = BytesIO(image_content)
        elif first_item.get("b64_json"):
            image_bytesio = BytesIO(base64.b64decode(first_item["b64_json"]))
        else:
            raise ValueError(
                "GPT Image API returned no usable image payload: "
                f"{_summarize_gpt_image_result(result)}"
            )

        output_text = first_item.get("revised_prompt") or result.get("message", "")
        return image_bytesio, output_text, result_url

    def _request_gpt_image_edit(
        self,
        *,
        data: Dict[str, Any],
        multipart_files: list[tuple[str, tuple[str, bytes, str]]],
        batch_size: int,
    ) -> tuple[BytesIO, str, str]:
        result_url = ""
        try:
            headers = {
                "Authorization": f"Bearer {FD_LITELLM_API_KEY}",
            }
            response = requests.post(
                url=f"{FD_LITELLM_BASE_URL}/v1/images/edits",
                headers=headers,
                data=data,
                files=multipart_files,
                timeout=600,
            )
            response.raise_for_status()
            result = response.json()
            logger.info("GPT Image API response summary: %s", _summarize_gpt_image_result(result))
            image_bytesio, output_text, result_url = self._decode_gpt_image_result(result)
        except requests.exceptions.Timeout as exc:
            traceback.print_exc()
            raise GenImageServiceError("TIMEOUT") from exc
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else "unknown"
            response_text = response.text if response is not None else str(exc)
            # logger.error(
            #     "GPT Image API HTTP error status=%s response=%s model=%s",
            #     status_code,
            #     response_text,
            #     data.get("model"),
            # )
            raise GenImageServiceError(
                f"HTTP {status_code} from GPT Image API: {response_text}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            traceback.print_exc()
            raise GenImageServiceError(f"REQUEST_ERROR: {exc}") from exc
        except Exception as exc:
            traceback.print_exc()
            raise GenImageServiceError(f"UNEXPECTED_ERROR: {exc}") from exc

        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
                print("Sending gtp_image webhook message...")
                webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {
                    "gtp_image_full": {
                        "request": {
                            "data": data,
                            "image_count": batch_size,
                        },
                        "response": {
                            "result_url": result_url,
                            "data_keys": list(result["data"][0].keys()),
                        },
                    }
                })
            except Exception:
                pass

        return image_bytesio, output_text, result_url

    def _request_azure_gpt_image_generation(
        self,
        *,
        data: Dict[str, Any],
        multipart_files: list[tuple[str, tuple[str, bytes, str]]],
        batch_size: int,
    ) -> tuple[BytesIO, str, str]:
        result_url = ""
        form_data = {
            "model": self.AZURE_FALLBACK_MODEL,
            "prompt": data["prompt"],
            "n": 1,
            "size": data["size"],
        }
        try:
            headers = {
                "Authorization": f"Bearer {FD_LITELLM_API_KEY}",
            }
            response = requests.post(
                url=f"{FD_LITELLM_BASE_URL}/v1/images/edits",
                headers=headers,
                data=form_data,
                files=multipart_files,
                timeout=600,
            )
            response.raise_for_status()
            result = response.json()
            logger.info("GPT Azure fallback response summary: %s", _summarize_gpt_image_result(result))
            image_bytesio, output_text, result_url = self._decode_gpt_image_result(result)
        except requests.exceptions.Timeout as exc:
            traceback.print_exc()
            raise GenImageServiceError("TIMEOUT") from exc
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else "unknown"
            response_text = response.text if response is not None else str(exc)
            logger.error(
                "GPT Azure fallback HTTP error status=%s response=%s model=%s",
                status_code,
                response_text,
                form_data["model"],
            )
            raise GenImageServiceError(
                f"HTTP {status_code} from GPT Azure fallback: {response_text}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            traceback.print_exc()
            raise GenImageServiceError(f"REQUEST_ERROR: {exc}") from exc
        except Exception as exc:
            traceback.print_exc()
            raise GenImageServiceError(f"UNEXPECTED_ERROR: {exc}") from exc

        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
                print("Sending gtp_image webhook message...")
                webhook_send(FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL, {
                    "gtp_image_full": {
                        "request": {
                            "data": form_data,
                            "image_count": batch_size,
                        },
                        "response": {
                            "result_url": result_url,
                            "data_keys": list(result["data"][0].keys()),
                        },
                    }
                })
            except Exception:
                pass

        return image_bytesio, output_text, result_url

    def _call_gpt_image_with_retry_policy(
        self,
        *,
        data: Dict[str, Any],
        multipart_files: list[tuple[str, tuple[str, bytes, str]]],
        batch_size: int,
    ) -> tuple[BytesIO, str, str]:
        primary_model = data["model"]
        last_non_timeout_error: Optional[Exception] = None

        for attempt in range(1, self.PRIMARY_MODEL_MAX_ATTEMPTS + 1):
            attempt_data = dict(data, model=primary_model)
            logger.info(
                "Calling GPT Image API with model=%s attempt=%s/%s image_count=%s",
                primary_model,
                attempt,
                self.PRIMARY_MODEL_MAX_ATTEMPTS,
                batch_size,
            )
            try:
                return self._request_gpt_image_edit(
                    data=attempt_data,
                    multipart_files=multipart_files,
                    batch_size=batch_size,
                )
            except GenImageServiceError as exc:
                if str(exc) == "TIMEOUT":
                    logger.warning(
                        "GPT Image API timed out for model=%s on attempt=%s/%s, falling back to model=%s",
                        primary_model,
                        attempt,
                        self.PRIMARY_MODEL_MAX_ATTEMPTS,
                        self.AZURE_FALLBACK_MODEL,
                    )
                    break

                last_non_timeout_error = exc
                logger.warning(
                    "GPT Image API failed for model=%s on attempt=%s/%s with non-timeout error: %s",
                    primary_model,
                    attempt,
                    self.PRIMARY_MODEL_MAX_ATTEMPTS,
                    exc,
                )
        else:
            logger.warning(
                "GPT Image API exhausted retries for model=%s, falling back to model=%s",
                primary_model,
                self.AZURE_FALLBACK_MODEL,
            )

        fallback_data = dict(data, model=self.AZURE_FALLBACK_MODEL)
        logger.info(
            "Calling GPT Image API fallback with model=%s image_count=%s",
            self.AZURE_FALLBACK_MODEL,
            batch_size,
        )
        try:
            return self._request_azure_gpt_image_generation(
                data=fallback_data,
                multipart_files=multipart_files,
                batch_size=batch_size,
            )
        except GenImageServiceError as fallback_exc:
            if last_non_timeout_error is not None:
                logger.error(
                    "GPT Image API fallback model=%s also failed after primary error=%s: fallback_error=%s",
                    self.AZURE_FALLBACK_MODEL,
                    last_non_timeout_error,
                    fallback_exc,
                )
            raise

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

        size = resolution_to_edit_size(resolution or "2K", aspect_ratio)
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
            data=data,
            multipart_files=multipart_files,
            batch_size=batch_size,
        )
        output_image = bytesio_to_image_tensor(image_bytesio)
        return (output_image, output_text, result_url)

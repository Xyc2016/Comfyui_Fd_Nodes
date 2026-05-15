import base64
import traceback
from io import BytesIO
from typing import Any, Dict, Optional

import requests

from ..config import FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL
from ..old_gemini_api_node import GenImageServiceError
from .error_utils import ERROR_TIMEOUT, classify_error_message, normalize_error_message
from .webhook import webhook_send


def summarize_gpt_image_result(result: dict) -> dict:
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


class GptImageRequestMixin:
    PRIMARY_MODEL_MAX_ATTEMPTS = 3
    AZURE_FALLBACK_MODEL = "gpt-image-2-azure"

    def _build_gpt_image_form_data(
        self,
        *,
        model: str,
        data: Dict[str, Any],
        include_n: bool = False,
    ) -> Dict[str, Any]:
        form_data = {
            "model": model,
            "prompt": data["prompt"],
            "size": data["size"],
        }
        if data.get("user"):
            form_data["user"] = data["user"]
        if data.get("quality"):
            form_data["quality"] = data["quality"]
        if include_n:
            form_data["n"] = data.get("n", 1)
        return form_data

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
                f"{summarize_gpt_image_result(result)}"
            )

        output_text = first_item.get("revised_prompt") or result.get("message", "")
        return image_bytesio, output_text, result_url

    def _post_gpt_image_request(
        self,
        *,
        base_url: str,
        api_key: str,
        form_data: Dict[str, Any],
        multipart_files: list[tuple[str, tuple[str, bytes, str]]],
        batch_size: int,
        log_label: str,
        logger,
    ) -> tuple[BytesIO, str, str]:
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.post(
                url=f"{base_url.rstrip('/')}/v1/images/edits",
                headers=headers,
                data=form_data,
                files=multipart_files,
                timeout=600,
            )
            response.raise_for_status()
            result = response.json()
            logger.info("%s response summary: %s", log_label, summarize_gpt_image_result(result))
            image_bytesio, output_text, result_url = self._decode_gpt_image_result(result)
        except requests.exceptions.Timeout as exc:
            traceback.print_exc()
            raise GenImageServiceError(
                normalize_error_message(exc, category=ERROR_TIMEOUT, fallback_detail="request timed out")
            ) from exc
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else "unknown"
            response_text = response.text if response is not None else str(exc)
            raise GenImageServiceError(
                normalize_error_message(
                    f"HTTP {status_code} from {log_label}: {response_text}"
                )
            ) from exc
        except requests.exceptions.RequestException as exc:
            traceback.print_exc()
            raise GenImageServiceError(
                normalize_error_message(f"REQUEST_ERROR: {exc}")
            ) from exc
        except Exception as exc:
            traceback.print_exc()
            raise GenImageServiceError(
                normalize_error_message(f"UNEXPECTED_ERROR: {exc}")
            ) from exc

        if FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL:
            try:
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

    def _request_gpt_image_edit(
        self,
        *,
        base_url: str,
        api_key: str,
        data: Dict[str, Any],
        multipart_files: list[tuple[str, tuple[str, bytes, str]]],
        batch_size: int,
        logger,
    ) -> tuple[BytesIO, str, str]:
        form_data = self._build_gpt_image_form_data(
            model=data["model"],
            data=data,
        )
        return self._post_gpt_image_request(
            base_url=base_url,
            api_key=api_key,
            form_data=form_data,
            multipart_files=multipart_files,
            batch_size=batch_size,
            log_label="GPT Image API",
            logger=logger,
        )

    def _request_azure_gpt_image_generation(
        self,
        *,
        base_url: str,
        api_key: str,
        data: Dict[str, Any],
        multipart_files: list[tuple[str, tuple[str, bytes, str]]],
        batch_size: int,
        logger,
    ) -> tuple[BytesIO, str, str]:
        form_data = self._build_gpt_image_form_data(
            model=self.AZURE_FALLBACK_MODEL,
            data=data,
            include_n=True,
        )
        return self._post_gpt_image_request(
            base_url=base_url,
            api_key=api_key,
            form_data=form_data,
            multipart_files=multipart_files,
            batch_size=batch_size,
            log_label="GPT Azure fallback",
            logger=logger,
        )

    def _call_gpt_image_with_retry_policy(
        self,
        *,
        base_url: str,
        api_key: str,
        data: Dict[str, Any],
        multipart_files: list[tuple[str, tuple[str, bytes, str]]],
        batch_size: int,
        logger,
    ) -> tuple[BytesIO, str, str]:
        primary_model = data["model"]
        last_non_timeout_error: Optional[Exception] = None

        for attempt in range(1, self.PRIMARY_MODEL_MAX_ATTEMPTS + 1):
            logger.info(
                "Calling GPT Image API with model=%s attempt=%s/%s image_count=%s",
                primary_model,
                attempt,
                self.PRIMARY_MODEL_MAX_ATTEMPTS,
                batch_size,
            )
            try:
                return self._request_gpt_image_edit(
                    base_url=base_url,
                    api_key=api_key,
                    data=data,
                    multipart_files=multipart_files,
                    batch_size=batch_size,
                    logger=logger,
                )
            except GenImageServiceError as exc:
                if classify_error_message(exc) == ERROR_TIMEOUT:
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

        logger.info(
            "Calling GPT Image API fallback with model=%s image_count=%s",
            self.AZURE_FALLBACK_MODEL,
            batch_size,
        )
        try:
            return self._request_azure_gpt_image_generation(
                base_url=base_url,
                api_key=api_key,
                data=data,
                multipart_files=multipart_files,
                batch_size=batch_size,
                logger=logger,
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

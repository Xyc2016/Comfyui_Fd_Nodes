"""The API test node's two wdcloud max models: direct edits, same-model overlap only."""

import base64
import logging
import time
from collections.abc import Callable
from io import BytesIO
from queue import Empty, Queue
from threading import Thread
from typing import TypeVar, cast

import requests

from .error_utils import ERROR_NSFW, classify_error_message
from .url_utils import normalize_oss_url

MAX_MODELS = frozenset({
    "gpt-image-2.6-flare-max-wdcloud",
    "gpt-image-2.6-sunburst-max-wdcloud",
})
OVERLAP_WAIT_SECONDS = 200.0
TOTAL_TIMEOUT_SECONDS = 400.0
T = TypeVar("T")
logger = logging.getLogger(__name__)


def _is_retryable(error: Exception) -> bool:
    if classify_error_message(error) == ERROR_NSFW:
        return False
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return error.response.status_code in {408, 409, 425, 429} or error.response.status_code >= 500
    return True


def call_with_same_model_overlap(
    call: Callable[[float], T],
    *,
    wait_seconds: float = OVERLAP_WAIT_SECONDS,
    total_seconds: float = TOTAL_TIMEOUT_SECONDS,
) -> T:
    """Race at most two attempts; return without joining an unfinished HTTP request."""
    deadline = time.monotonic() + total_seconds
    second_start = time.monotonic() + wait_seconds
    results: Queue[tuple[int, T | Exception, bool]] = Queue()
    started = 0
    finished = 0

    def start() -> None:
        nonlocal started
        started += 1
        index = started

        def run() -> None:
            logger.info("GPT Image max attempt %s started", index)
            try:
                result = call(max(0.001, deadline - time.monotonic()))
            except Exception as error:
                results.put((index, error, False))
            else:
                results.put((index, result, True))

        Thread(target=run, name=f"gpt-image-max-{index}", daemon=True).start()

    start()
    while True:
        now = time.monotonic()
        if now >= deadline:
            raise requests.Timeout(f"GPT Image max exceeded its {total_seconds:g}-second total budget")
        if started == 1 and (now >= second_start or finished == 1):
            logger.info("GPT Image max starting attempt 2 on the same model; no fallback")
            start()
        wake_at = min(second_start, deadline) if started == 1 else deadline
        try:
            _, result, succeeded = results.get(timeout=max(0.001, wake_at - time.monotonic()))
        except Empty:
            continue
        finished += 1
        if succeeded:
            logger.info("GPT Image max attempt %s succeeded", _)
            return cast(T, result)
        assert isinstance(result, Exception)
        last_error = result
        if not _is_retryable(last_error):
            raise last_error
        if started == 2 and finished == 2:
            raise last_error


def request_max_image_edit(
    *,
    base_url: str,
    api_key: str,
    data: dict,
    multipart_files: list,
    timeout: float,
) -> tuple[BytesIO, str, str]:
    """Send the exact model/quality and decode OpenAI URL or base64 image data."""
    deadline = time.monotonic() + timeout
    logger.info("GPT Image max edits model=%s size=%s quality=%s image_count=%s timeout=%.1f user=%s",
                data.get("model"), data.get("size"), data.get("quality"), len(multipart_files), timeout, data.get("user", ""))
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/images/edits",
        headers={"Authorization": f"Bearer {api_key}"},
        data=data,
        files=multipart_files,
        timeout=(10, timeout),
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("error") or response.text[:500]
        except ValueError:
            detail = response.text[:500]
        raise requests.HTTPError(f"HTTP {response.status_code}: {detail}", response=response)
    result = response.json()
    if result.get("error"):
        error = result["error"]
        raise RuntimeError(str(error))
    items = result.get("data")
    item = items[0] if isinstance(items, list) and items else None
    if not isinstance(item, dict):
        raise ValueError("GPT Image max returned no image data")
    result_url = item.get("url") or ""
    if result_url:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise requests.Timeout("GPT Image max timed out before image download")
        image_response = requests.get(normalize_oss_url(result_url), timeout=(10, remaining))
        image_response.raise_for_status()
        image_bytes = image_response.content
    elif item.get("b64_json"):
        image_bytes = base64.b64decode(item["b64_json"], validate=True)
    else:
        raise ValueError("GPT Image max returned neither an image URL nor base64 data")
    # Validate before declaring the attempt successful so an invalid download cannot win the race.
    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as image:
        image.verify()
    return BytesIO(image_bytes), item.get("revised_prompt") or result.get("message", ""), result_url

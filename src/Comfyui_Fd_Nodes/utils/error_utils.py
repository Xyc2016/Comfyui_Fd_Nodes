from __future__ import annotations

from typing import Optional


ERROR_TIMEOUT = "TIMEOUT"
ERROR_NSFW = "NSFW"
ERROR_UNKNOWN = "UNKNOWN"
ERROR_CATEGORIES = {ERROR_TIMEOUT, ERROR_NSFW, ERROR_UNKNOWN}


def classify_error_message(message: object) -> str:
    text = str(message or "").strip()
    if not text:
        return ERROR_UNKNOWN

    prefix, _, _ = text.partition(":")
    if prefix in ERROR_CATEGORIES:
        return prefix

    normalized = text.lower()
    if (
        "timeout" in normalized
        or "timed out" in normalized
        or "time out" in normalized
        or "超时" in normalized
    ):
        return ERROR_TIMEOUT

    if (
        "nsfw" in normalized
        or "content_filter" in normalized
        or "content filter" in normalized
        or "safety" in normalized
        or "unsafe" in normalized
        or "内容安全" in normalized
        or "内容被过滤" in normalized
        or "敏感" in normalized
        or "违规" in normalized
    ):
        return ERROR_NSFW

    return ERROR_UNKNOWN


def normalize_error_message(
    message: object,
    *,
    category: Optional[str] = None,
    fallback_detail: Optional[str] = None,
) -> str:
    raw_text = str(message or "").strip()
    actual_category = category or classify_error_message(raw_text or fallback_detail or "")
    if actual_category not in ERROR_CATEGORIES:
        actual_category = ERROR_UNKNOWN

    if raw_text:
        prefix, sep, remainder = raw_text.partition(":")
        if prefix in ERROR_CATEGORIES and sep and remainder.strip():
            return f"{prefix}: {remainder.strip()}"
        if prefix in ERROR_CATEGORIES:
            detail = remainder.strip() or fallback_detail or raw_text
            return f"{prefix}: {detail}"
        return f"{actual_category}: {raw_text}"

    detail = (fallback_detail or actual_category).strip()
    return f"{actual_category}: {detail}"

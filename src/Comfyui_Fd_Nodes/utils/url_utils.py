import os
import re
from urllib.parse import urlsplit


def is_url_delivery_required() -> bool:
    """Only explicit deployment settings enable URL-only delivery.

    config.FD_OVERSEAS_GPU_GATEWAY has a domestic default, so it cannot be
    used to infer that this process is deployed overseas.
    """
    return bool(os.getenv("FD_OVERSEAS_GPU_GATEWAY", "").strip()) or (
        os.getenv("FD_IMAGE_DELIVERY", "").strip().lower() == "url"
    )


def normalize_oss_url(image_url: str) -> str:
    """Keep domestic VPC downloads; use the same object's public URL overseas.

    Do not map arbitrary object keys into FD_OSS_URL_PREFIX: that prefix is
    authoritative for uploads to the configured bucket, not for other buckets.
    CDN URLs and signed query strings must be retained as supplied.
    """
    if not is_url_delivery_required():
        return image_url.replace("oss-cn-hangzhou.aliyuncs.com", "oss-cn-hangzhou-internal.aliyuncs.com")

    parsed = urlsplit(image_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return image_url
    public_host = re.sub(
        r"(^|\.)(oss-[a-z0-9-]+)-internal\.aliyuncs\.com$",
        r"\1\2.aliyuncs.com",
        parsed.hostname,
    )
    if public_host == parsed.hostname:
        return image_url
    # Replace only the hostname, leaving path, query, fragment and port intact.
    netloc = parsed.netloc.replace(parsed.hostname, public_host, 1)
    return parsed._replace(netloc=netloc).geturl()

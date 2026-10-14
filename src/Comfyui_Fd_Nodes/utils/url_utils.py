def normalize_oss_url(image_url: str) -> str:
    """Use the VPC endpoint for Hangzhou OSS downloads."""
    return image_url.replace("oss-cn-hangzhou.aliyuncs.com", "oss-cn-hangzhou-internal.aliyuncs.com")

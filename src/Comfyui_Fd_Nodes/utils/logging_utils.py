import logging


DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _formatter_includes_asctime(formatter):
    if formatter is None:
        return False
    fmt = getattr(formatter, "_fmt", None)
    if not fmt:
        fmt = getattr(getattr(formatter, "_style", None), "_fmt", "")
    return "%(asctime)" in (fmt or "")


def configure_default_logging(level=logging.INFO):
    """Install or upgrade root logger formatting so timestamps are included by default."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_LOG_DATE_FORMAT,
        )
        return

    timestamp_formatter = logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_LOG_DATE_FORMAT,
    )
    for handler in root_logger.handlers:
        if not _formatter_includes_asctime(handler.formatter):
            handler.setFormatter(timestamp_formatter)

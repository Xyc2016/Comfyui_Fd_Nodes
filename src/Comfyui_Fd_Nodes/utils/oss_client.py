import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import oss2
import requests

from ..config import (
    FD_OSS_ACCESS_KEY_ID,
    FD_OSS_ACCESS_KEY_SECRET,
    FD_OSS_BUCKET_NAME,
    FD_OSS_ENDPOINT,
    FD_OSS_STS_KEY,
    FD_OSS_STS_TIMEOUT,
    FD_OSS_STS_URL,
    FD_OSS_URL_PREFIX,
)


AUTH_FALLBACK_ERROR_CODES = {
    "InvalidAccessKeyId",
    "AccessDenied",
    "SignatureDoesNotMatch",
    "InvalidSecurityToken",
    "SecurityTokenExpired",
}
STS_TOKEN_ERROR_CODES = {"InvalidSecurityToken", "SecurityTokenExpired"}
STS_REFRESH_MARGIN_SECONDS = 5 * 60
STS_DEFAULT_TTL_SECONDS = 50 * 60
DEFAULT_OSS_CONNECT_TIMEOUT = 30
DEFAULT_STS_TIMEOUT = 10.0


@dataclass
class _StsCredentials:
    access_key_id: str
    access_key_secret: str
    security_token: str
    expiration: Optional[str]
    expires_at: float


def _has_value(value: Optional[str]) -> bool:
    return bool(str(value or "").strip())


def _as_timeout(value: Optional[Any]) -> float:
    if value is None or value == "":
        return DEFAULT_STS_TIMEOUT
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("FD_OSS_STS_TIMEOUT 必须是数字") from exc


def _pick_value(source: dict[str, Any], names: tuple[str, ...]) -> Optional[str]:
    lower_keys = {str(key).lower(): key for key in source}
    for name in names:
        actual_key = lower_keys.get(name.lower())
        if actual_key is None:
            continue
        value = source.get(actual_key)
        if _has_value(value):
            return str(value)
    return None


class OssUploadClient:
    def __init__(
        self,
        *,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        bucket_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        url_prefix: Optional[str] = None,
        sts_url: Optional[str] = None,
        sts_key: Optional[str] = None,
        sts_timeout: Optional[Any] = None,
        connect_timeout: int = DEFAULT_OSS_CONNECT_TIMEOUT,
        auth_factory: Optional[Callable[..., Any]] = None,
        sts_auth_factory: Optional[Callable[..., Any]] = None,
        bucket_factory: Optional[Callable[..., Any]] = None,
        request_post: Optional[Callable[..., Any]] = None,
        time_func: Optional[Callable[[], float]] = None,
    ):
        self.access_key_id = FD_OSS_ACCESS_KEY_ID if access_key_id is None else access_key_id
        self.access_key_secret = FD_OSS_ACCESS_KEY_SECRET if access_key_secret is None else access_key_secret
        self.bucket_name = FD_OSS_BUCKET_NAME if bucket_name is None else bucket_name
        self.endpoint = FD_OSS_ENDPOINT if endpoint is None else endpoint
        self.url_prefix = FD_OSS_URL_PREFIX if url_prefix is None else url_prefix
        self.sts_url = FD_OSS_STS_URL if sts_url is None else sts_url
        self.sts_key = FD_OSS_STS_KEY if sts_key is None else sts_key
        self.sts_timeout = _as_timeout(FD_OSS_STS_TIMEOUT if sts_timeout is None else sts_timeout)
        self.connect_timeout = connect_timeout
        self._auth_factory = auth_factory or oss2.Auth
        self._sts_auth_factory = sts_auth_factory or oss2.StsAuth
        self._bucket_factory = bucket_factory or oss2.Bucket
        self._request_post = request_post or requests.post
        self._time = time_func or time.time
        self._ak_bucket = None
        self._sts_credentials: Optional[_StsCredentials] = None
        self._sts_bucket = None
        self._sts_lock = threading.Lock()

    def upload_bytes(self, object_path: str, data: bytes) -> str:
        self._ensure_upload_config()

        if self._has_ak_config():
            try:
                self._get_ak_bucket().put_object(object_path, data)
                return self._build_url(object_path)
            except Exception as exc:
                if self._has_sts_config() and self._is_auth_fallback_error(exc):
                    return self._upload_with_sts(object_path, data)
                raise

        return self._upload_with_sts(object_path, data)

    def _build_url(self, object_path: str) -> str:
        return f"{self.url_prefix}{object_path}"

    def _has_ak_config(self) -> bool:
        return _has_value(self.access_key_id) and _has_value(self.access_key_secret)

    def _has_any_ak_config(self) -> bool:
        return _has_value(self.access_key_id) or _has_value(self.access_key_secret)

    def _has_sts_config(self) -> bool:
        return _has_value(self.sts_url) and _has_value(self.sts_key)

    def _has_any_sts_config(self) -> bool:
        return _has_value(self.sts_url) or _has_value(self.sts_key)

    def _ensure_upload_config(self) -> None:
        missing = [
            name
            for name, value in {
                "FD_OSS_BUCKET_NAME": self.bucket_name,
                "FD_OSS_ENDPOINT": self.endpoint,
                "FD_OSS_URL_PREFIX": self.url_prefix,
            }.items()
            if not _has_value(value)
        ]

        if not self._has_ak_config() and not self._has_sts_config():
            if self._has_any_ak_config():
                auth_values = {
                    "FD_OSS_ACCESS_KEY_ID": self.access_key_id,
                    "FD_OSS_ACCESS_KEY_SECRET": self.access_key_secret,
                }
                missing.extend(name for name, value in auth_values.items() if not _has_value(value))
            elif self._has_any_sts_config():
                sts_values = {
                    "FD_OSS_STS_URL": self.sts_url,
                    "FD_OSS_STS_KEY": self.sts_key,
                }
                missing.extend(name for name, value in sts_values.items() if not _has_value(value))
            else:
                missing.append("FD_OSS_ACCESS_KEY_ID/FD_OSS_ACCESS_KEY_SECRET 或 FD_OSS_STS_URL/FD_OSS_STS_KEY")

        if missing:
            raise RuntimeError(f"未配置 OSS 上传参数: {', '.join(missing)}")

    def _get_ak_bucket(self):
        if self._ak_bucket is None:
            auth = self._auth_factory(self.access_key_id, self.access_key_secret)
            self._ak_bucket = self._make_bucket(auth)
        return self._ak_bucket

    def _make_bucket(self, auth):
        return self._bucket_factory(
            auth=auth,
            bucket_name=self.bucket_name,
            endpoint=self.endpoint,
            connect_timeout=self.connect_timeout,
        )

    def _upload_with_sts(self, object_path: str, data: bytes) -> str:
        try:
            self._get_sts_bucket().put_object(object_path, data)
            return self._build_url(object_path)
        except Exception as exc:
            if not self._is_sts_token_error(exc):
                raise
            self._get_sts_bucket(force_refresh=True).put_object(object_path, data)
            return self._build_url(object_path)

    def _get_sts_bucket(self, force_refresh: bool = False):
        with self._sts_lock:
            if not force_refresh and self._sts_bucket is not None and self._sts_credentials_are_fresh():
                return self._sts_bucket

            credentials = self._fetch_sts_credentials()
            auth = self._sts_auth_factory(
                credentials.access_key_id,
                credentials.access_key_secret,
                credentials.security_token,
            )
            self._sts_credentials = credentials
            self._sts_bucket = self._make_bucket(auth)
            return self._sts_bucket

    def _sts_credentials_are_fresh(self) -> bool:
        if self._sts_credentials is None:
            return False
        return self._sts_credentials.expires_at - self._time() > STS_REFRESH_MARGIN_SECONDS

    def _fetch_sts_credentials(self) -> _StsCredentials:
        response = self._request_post(
            self.sts_url,
            headers={
                "X-Zhiyi-STS-Key": self.sts_key,
                "Content-Type": "application/json",
            },
            json={},
            timeout=self.sts_timeout,
        )
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        elif getattr(response, "status_code", 200) >= 400:
            raise RuntimeError(f"STS 请求失败: {response.status_code}")

        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError("STS 响应不是合法 JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"STS 响应格式错误: 期望 dict，实际 {type(payload).__name__}")

        values = self._extract_credentials_payload(payload)
        missing = [
            display_name
            for key, display_name in (
                ("access_key_id", "AccessKeyId"),
                ("access_key_secret", "AccessKeySecret"),
                ("security_token", "SecurityToken"),
            )
            if not _has_value(values.get(key))
        ]
        if missing:
            quoted = ", ".join(f"`{name}`" for name in missing)
            raise RuntimeError(f"STS 响应缺少必要字段: {quoted}")

        expiration = values.get("expiration")
        return _StsCredentials(
            access_key_id=str(values["access_key_id"]),
            access_key_secret=str(values["access_key_secret"]),
            security_token=str(values["security_token"]),
            expiration=str(expiration) if _has_value(expiration) else None,
            expires_at=self._parse_expiration(expiration),
        )

    def _extract_credentials_payload(self, payload: dict[str, Any]) -> dict[str, Optional[str]]:
        candidates = [payload]
        for key in ("data", "Credentials", "credentials"):
            value = payload.get(key)
            if isinstance(value, dict):
                candidates.append(value)

        best_values: dict[str, Optional[str]] = {}
        best_count = -1
        for candidate in candidates:
            values = {
                "access_key_id": _pick_value(candidate, ("AccessKeyId", "accessKeyId", "access_key_id")),
                "access_key_secret": _pick_value(candidate, ("AccessKeySecret", "accessKeySecret", "access_key_secret")),
                "security_token": _pick_value(candidate, ("SecurityToken", "securityToken", "security_token")),
                "expiration": _pick_value(candidate, ("Expiration", "expiration", "expiresAt", "expires_at")),
            }
            count = sum(1 for key in ("access_key_id", "access_key_secret", "security_token") if _has_value(values.get(key)))
            if count > best_count:
                best_count = count
                best_values = values
            if count == 3:
                return values
        return best_values

    def _parse_expiration(self, expiration: Optional[Any]) -> float:
        if not _has_value(expiration):
            return self._time() + STS_DEFAULT_TTL_SECONDS

        if isinstance(expiration, (int, float)):
            return float(expiration)

        value = str(expiration).strip()
        try:
            if value.isdigit():
                return float(value)
            if value.endswith("Z"):
                value = f"{value[:-1]}+00:00"
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            return self._time() + STS_DEFAULT_TTL_SECONDS

    def _is_sts_token_error(self, exc: Exception) -> bool:
        code = self._extract_error_code(exc)
        if code in STS_TOKEN_ERROR_CODES:
            return True
        text = str(exc)
        return any(error_code in text for error_code in STS_TOKEN_ERROR_CODES)

    def _is_auth_fallback_error(self, exc: Exception) -> bool:
        status = self._extract_status(exc)
        code = self._extract_error_code(exc)
        if status == 403:
            return True
        if code in AUTH_FALLBACK_ERROR_CODES:
            return True
        text = str(exc)
        return any(error_code in text for error_code in AUTH_FALLBACK_ERROR_CODES)

    def _extract_status(self, exc: Exception) -> Optional[int]:
        for attr in ("status", "status_code"):
            value = getattr(exc, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        response = getattr(exc, "response", None)
        if response is not None:
            value = getattr(response, "status_code", None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        return None

    def _extract_error_code(self, exc: Exception) -> Optional[str]:
        for attr in ("code", "error_code"):
            value = getattr(exc, attr, None)
            if _has_value(value):
                return str(value)
        details = getattr(exc, "details", None)
        if isinstance(details, dict):
            value = details.get("Code") or details.get("code")
            if _has_value(value):
                return str(value)
        return None


_default_client: Optional[OssUploadClient] = None
_default_client_lock = threading.Lock()


def get_default_oss_client() -> OssUploadClient:
    global _default_client
    if _default_client is not None:
        return _default_client
    with _default_client_lock:
        if _default_client is None:
            _default_client = OssUploadClient()
        return _default_client


def upload_bytes_to_oss(object_path: str, data: bytes) -> str:
    return get_default_oss_client().upload_bytes(object_path, data)


def _reset_default_oss_client_for_tests() -> None:
    global _default_client
    with _default_client_lock:
        _default_client = None

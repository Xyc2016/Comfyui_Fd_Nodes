import pytest

from src.Comfyui_Fd_Nodes.utils.oss_client import OssUploadClient


class FakeAuth:
    def __init__(self, access_key_id, access_key_secret):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret


class FakeStsAuth:
    def __init__(self, access_key_id, access_key_secret, security_token):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.security_token = security_token


class FakeOssError(Exception):
    def __init__(self, code="", status=None):
        super().__init__(code or status)
        self.code = code
        self.status = status
        self.details = {"Code": code} if code else {}


class FakeBucket:
    def __init__(self, auth, endpoint, bucket_name, connect_timeout, put_behaviors=None):
        self.auth = auth
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.connect_timeout = connect_timeout
        self.put_behaviors = put_behaviors if put_behaviors is not None else []
        self.put_calls = []

    def put_object(self, object_path, data):
        self.put_calls.append((object_path, data))
        if self.put_behaviors:
            behavior = self.put_behaviors.pop(0)
            if isinstance(behavior, Exception):
                raise behavior
        return None


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def make_client(**kwargs):
    bucket_instances = []
    bucket_behaviors = list(kwargs.pop("bucket_behaviors", []))
    captured_sts_requests = []
    sts_payloads = list(kwargs.pop("sts_payloads", []))
    now = kwargs.pop("now", [1000.0])

    def bucket_factory(**bucket_kwargs):
        behaviors = bucket_behaviors.pop(0) if bucket_behaviors else []
        bucket = FakeBucket(**bucket_kwargs, put_behaviors=list(behaviors))
        bucket_instances.append(bucket)
        return bucket

    def request_post(url, headers, json, timeout):
        captured_sts_requests.append({
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        })
        return FakeResponse(sts_payloads.pop(0))

    client = OssUploadClient(
        access_key_id=kwargs.pop("access_key_id", "ak-id"),
        access_key_secret=kwargs.pop("access_key_secret", "ak-secret"),
        bucket_name=kwargs.pop("bucket_name", "bucket"),
        endpoint=kwargs.pop("endpoint", "https://oss-endpoint"),
        url_prefix=kwargs.pop("url_prefix", "https://cdn/"),
        sts_url=kwargs.pop("sts_url", "https://sts.example.com/token"),
        sts_key=kwargs.pop("sts_key", "configured-secret"),
        sts_timeout=kwargs.pop("sts_timeout", 3),
        auth_factory=kwargs.pop("auth_factory", FakeAuth),
        sts_auth_factory=kwargs.pop("sts_auth_factory", FakeStsAuth),
        bucket_factory=kwargs.pop("bucket_factory", bucket_factory),
        request_post=kwargs.pop("request_post", request_post),
        time_func=kwargs.pop("time_func", lambda: now[0]),
        **kwargs,
    )
    return client, bucket_instances, captured_sts_requests, now


def sts_payload(access_key_id="sts-id", access_key_secret="sts-secret", security_token="sts-token", expiration="2099-01-01T00:00:00Z"):
    return {
        "AccessKeyId": access_key_id,
        "AccessKeySecret": access_key_secret,
        "SecurityToken": security_token,
        "Expiration": expiration,
    }


def test_ak_upload_success_does_not_request_sts():
    client, buckets, sts_requests, _now = make_client(sts_payloads=[])

    url = client.upload_bytes("path/image.png", b"image")

    assert url == "https://cdn/path/image.png"
    assert len(buckets) == 1
    assert isinstance(buckets[0].auth, FakeAuth)
    assert buckets[0].put_calls == [("path/image.png", b"image")]
    assert sts_requests == []


def test_ak_invalid_access_key_falls_back_to_sts_and_retries_successfully():
    client, buckets, sts_requests, _now = make_client(
        bucket_behaviors=[
            [FakeOssError(code="InvalidAccessKeyId", status=403)],
            [],
        ],
        sts_payloads=[sts_payload()],
    )

    url = client.upload_bytes("path/image.png", b"image")

    assert url == "https://cdn/path/image.png"
    assert len(buckets) == 2
    assert isinstance(buckets[0].auth, FakeAuth)
    assert isinstance(buckets[1].auth, FakeStsAuth)
    assert buckets[1].auth.access_key_id == "sts-id"
    assert buckets[1].auth.access_key_secret == "sts-secret"
    assert buckets[1].auth.security_token == "sts-token"
    assert buckets[1].put_calls == [("path/image.png", b"image")]
    assert len(sts_requests) == 1
    assert sts_requests[0]["url"] == "https://sts.example.com/token"
    assert sts_requests[0]["headers"]["X-Zhiyi-STS-Key"] == "configured-secret"
    assert sts_requests[0]["headers"]["Content-Type"] == "application/json"
    assert sts_requests[0]["json"] == {}
    assert sts_requests[0]["timeout"] == 3


def test_no_ak_with_sts_config_uploads_directly_with_sts():
    client, buckets, sts_requests, _now = make_client(
        access_key_id=None,
        access_key_secret=None,
        sts_payloads=[{
            "data": {
                "accessKeyId": "sts-id",
                "accessKeySecret": "sts-secret",
                "securityToken": "sts-token",
                "expiration": "2099-01-01T00:00:00Z",
            }
        }],
    )

    url = client.upload_bytes("path/direct.png", b"direct")

    assert url == "https://cdn/path/direct.png"
    assert len(buckets) == 1
    assert isinstance(buckets[0].auth, FakeStsAuth)
    assert buckets[0].put_calls == [("path/direct.png", b"direct")]
    assert len(sts_requests) == 1


def test_non_auth_ak_error_does_not_request_sts_and_reraises():
    expected = FakeOssError(code="NoSuchBucket", status=404)
    client, _buckets, sts_requests, _now = make_client(
        bucket_behaviors=[[expected]],
        sts_payloads=[sts_payload()],
    )

    with pytest.raises(FakeOssError) as exc_info:
        client.upload_bytes("path/image.png", b"image")

    assert exc_info.value is expected
    assert sts_requests == []


def test_sts_credentials_are_cached_until_refresh_margin():
    client, buckets, sts_requests, _now = make_client(
        access_key_id=None,
        access_key_secret=None,
        sts_payloads=[sts_payload(expiration="2099-01-01T00:00:00Z")],
    )

    assert client.upload_bytes("path/one.png", b"one") == "https://cdn/path/one.png"
    assert client.upload_bytes("path/two.png", b"two") == "https://cdn/path/two.png"

    assert len(sts_requests) == 1
    assert len(buckets) == 1
    assert buckets[0].put_calls == [("path/one.png", b"one"), ("path/two.png", b"two")]


def test_sts_credentials_refresh_when_expiring_soon():
    now = [1000.0]
    client, buckets, sts_requests, _now = make_client(
        access_key_id=None,
        access_key_secret=None,
        now=now,
        sts_payloads=[
            sts_payload(access_key_id="sts-id-1", expiration=1250),
            sts_payload(access_key_id="sts-id-2", expiration=5000),
        ],
    )

    client.upload_bytes("path/one.png", b"one")
    now[0] = 1001.0
    client.upload_bytes("path/two.png", b"two")

    assert len(sts_requests) == 2
    assert len(buckets) == 2
    assert buckets[0].auth.access_key_id == "sts-id-1"
    assert buckets[1].auth.access_key_id == "sts-id-2"
    assert buckets[1].put_calls == [("path/two.png", b"two")]


def test_sts_upload_token_expired_forces_refresh_and_retries_once():
    client, buckets, sts_requests, _now = make_client(
        access_key_id=None,
        access_key_secret=None,
        bucket_behaviors=[
            [FakeOssError(code="SecurityTokenExpired", status=403)],
            [],
        ],
        sts_payloads=[
            {"Credentials": sts_payload(access_key_id="sts-id-1")},
            {"Credentials": sts_payload(access_key_id="sts-id-2")},
        ],
    )

    url = client.upload_bytes("path/retry.png", b"retry")

    assert url == "https://cdn/path/retry.png"
    assert len(sts_requests) == 2
    assert len(buckets) == 2
    assert buckets[0].auth.access_key_id == "sts-id-1"
    assert buckets[1].auth.access_key_id == "sts-id-2"
    assert buckets[1].put_calls == [("path/retry.png", b"retry")]


@pytest.mark.parametrize(
    ("payload", "missing_field"),
    [
        (
            {
                "AccessKeySecret": "sts-secret",
                "SecurityToken": "sts-token",
            },
            "AccessKeyId",
        ),
        (
            {
                "AccessKeyId": "sts-id",
                "SecurityToken": "sts-token",
            },
            "AccessKeySecret",
        ),
        (
            {
                "AccessKeyId": "sts-id",
                "AccessKeySecret": "sts-secret",
            },
            "SecurityToken",
        ),
    ],
)
def test_sts_response_missing_required_fields_raises_clear_error(payload, missing_field):
    client, _buckets, _sts_requests, _now = make_client(
        access_key_id=None,
        access_key_secret=None,
        sts_payloads=[payload],
    )

    with pytest.raises(RuntimeError, match=missing_field):
        client.upload_bytes("path/image.png", b"image")

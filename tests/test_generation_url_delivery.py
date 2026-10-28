"""Mock-only coverage for deployment URL policy and generation transports."""

from io import BytesIO
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from src.Comfyui_Fd_Nodes.utils.gemini_service import GeminiImageServiceClient
from src.Comfyui_Fd_Nodes.utils.gpt_image_edit_client import GptImageEditClient
from src.Comfyui_Fd_Nodes.utils.seedream_image_client import SeedreamImageClient
from src.Comfyui_Fd_Nodes.utils.url_utils import is_url_delivery_required, normalize_oss_url


@pytest.fixture(autouse=True)
def explicit_deployment(monkeypatch):
    monkeypatch.delenv("FD_OVERSEAS_GPU_GATEWAY", raising=False)
    monkeypatch.delenv("FD_IMAGE_DELIVERY", raising=False)
    monkeypatch.delenv("FD_GPT_IMAGE_EDIT_URL", raising=False)
    monkeypatch.delenv("FD_GEMINI_URL", raising=False)


@pytest.mark.parametrize(("source", "expected"), [
    (
        "https://bucket.oss-cn-hangzhou.aliyuncs.com//a%20b.png?Expires=1&Signature=a%2Bb%3D#image",
        "https://bucket.oss-cn-hangzhou-internal.aliyuncs.com//a%20b.png?Expires=1&Signature=a%2Bb%3D#image",
    ),
    (
        "http://bucket.oss-cn-hangzhou.aliyuncs.com/a.png",
        "http://bucket.oss-cn-hangzhou-internal.aliyuncs.com/a.png",
    ),
    (
        "https://bucket.oss-cn-hangzhou-internal.aliyuncs.com/a.png",
        "https://bucket.oss-cn-hangzhou-internal.aliyuncs.com/a.png",
    ),
    ("https://cdn.example.com/a.png?token=a%2Bb", "https://cdn.example.com/a.png?token=a%2Bb"),
])
def test_domestic_url_policy_is_unchanged(source, expected):
    # config.FD_OVERSEAS_GPU_GATEWAY has a nonempty default; it is not a gate.
    assert not is_url_delivery_required()
    assert normalize_oss_url(source) == expected


@pytest.mark.parametrize(("gateway", "delivery", "expected"), [
    ("", "", False), ("  ", "inline", False), ("", "base64", False),
    ("https://gpu.example.com", "", True), ("", " URL ", True),
])
def test_url_policy_uses_only_explicit_environment(monkeypatch, gateway, delivery, expected):
    monkeypatch.setenv("FD_OVERSEAS_GPU_GATEWAY", gateway)
    monkeypatch.setenv("FD_IMAGE_DELIVERY", delivery)
    assert is_url_delivery_required() is expected


@pytest.mark.parametrize("gate", ["FD_OVERSEAS_GPU_GATEWAY", "FD_IMAGE_DELIVERY"])
@pytest.mark.parametrize(("source", "expected"), [
    (
        "https://original.oss-cn-hangzhou-internal.aliyuncs.com:443//a%20b.png?Expires=1&Signature=a%2Bb%3D#image",
        "https://original.oss-cn-hangzhou.aliyuncs.com:443//a%20b.png?Expires=1&Signature=a%2Bb%3D#image",
    ),
    (
        "http://original.oss-cn-hangzhou.aliyuncs.com/a.png?token=x%2By",
        "http://original.oss-cn-hangzhou.aliyuncs.com/a.png?token=x%2By",
    ),
    (
        "https://overseas.oss-us-east-1-internal.aliyuncs.com/a.png?token=x%2By",
        "https://overseas.oss-us-east-1.aliyuncs.com/a.png?token=x%2By",
    ),
    (
        "https://cdn.example.com/a.png?source=oss-cn-hangzhou.aliyuncs.com",
        "https://cdn.example.com/a.png?source=oss-cn-hangzhou.aliyuncs.com",
    ),
    (
        "https://bucket.oss-cn-hangzhou-internal.aliyuncs.com.example.com/a.png",
        "https://bucket.oss-cn-hangzhou-internal.aliyuncs.com.example.com/a.png",
    ),
])
def test_overseas_urls_stay_public_in_original_bucket(monkeypatch, gate, source, expected):
    monkeypatch.setenv(gate, "url" if gate == "FD_IMAGE_DELIVERY" else "https://gpu.example.com")
    monkeypatch.setenv("FD_OSS_URL_PREFIX", "https://different-bucket-cdn.example.com/")
    assert normalize_oss_url(source) == expected
    assert normalize_oss_url(expected) == expected


@pytest.mark.parametrize("gate", ["FD_OVERSEAS_GPU_GATEWAY", "FD_IMAGE_DELIVERY"])
@pytest.mark.parametrize("entrypoint", ["public", "private"])
def test_strict_gpt_litellm_rejects_before_upload_or_multipart(monkeypatch, gate, entrypoint):
    monkeypatch.setenv(gate, "url" if gate == "FD_IMAGE_DELIVERY" else "https://gpu.example.com")
    upload, post, get, adapter, encode = (Mock() for _ in range(5))
    monkeypatch.setattr(
        "src.Comfyui_Fd_Nodes.utils.gpt_image_edit_client._LiteLLMAdapter._call_gpt_image_with_retry_policy", adapter,
    )
    client = GptImageEditClient(backend="litellm", oss_uploader=upload, request_post=post, request_get=get)
    monkeypatch.setattr(client, "_tensor_to_png_bytes", encode)
    kwargs = dict(
        image_tensors=[torch.zeros((1, 2, 2, 3))], prompt="edit", size="2K",
        model="gpt-image-2", aspect_ratio="", quality="medium", resize=True, out_request_id="",
    )
    call = client.edit_image if entrypoint == "public" else client._edit_via_litellm
    with pytest.raises(NotImplementedError, match=r"Unsupported:.*binary multipart"):
        call(**kwargs)
    for mock in (upload, post, get, adapter, encode):
        mock.assert_not_called()


def _response(result_url):
    image = BytesIO()
    Image.new("RGB", (2, 2), "red").save(image, format="PNG")
    response = Mock(status_code=200, ok=True, content=image.getvalue())
    response.json.return_value = {"status": True, "result_image_url": result_url}
    return response


@pytest.mark.parametrize("kind", ["gpt", "gpt_litellm", "gpt_litellm_url_channel", "seedream", "seedream_litellm", "gemini"])
def test_overseas_existing_generation_paths_use_urls(monkeypatch, kind):
    monkeypatch.setenv("FD_IMAGE_DELIVERY", "url")
    input_url = "https://configured-cdn.example.com/a.png?token=a%2Bb%3D"
    result_url = "http://original.oss-cn-hangzhou-internal.aliyuncs.com/result.png?Signature=a%2Bb%3D"
    public_result = result_url.replace("-internal", "")
    upload = Mock(return_value=input_url)
    response = _response(result_url)
    if kind == "seedream_litellm":
        response.json.return_value = {"data": [{"url": result_url}]}
    post, get = Mock(return_value=response), Mock(return_value=response)
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("requests.get", get)
    images = torch.zeros((1, 2, 2, 3))
    if kind.startswith("gpt"):
        client = GptImageEditClient(
            backend="litellm" if kind.startswith("gpt_litellm") else "image_generation",
            edit_url="https://service.example.com/image/edit", oss_uploader=upload,
        )
        client.edit_image(
            image_tensors=[images], prompt="edit", size="2K",
            model="gpt-image-2.5" if kind == "gpt_litellm_url_channel" else "gpt-image-2",
        )
    elif kind.startswith("seedream"):
        client = SeedreamImageClient(
            backend="litellm" if kind == "seedream_litellm" else "image_generation",
            edit_url="https://service.example.com/image/edit", oss_uploader=upload,
        )
        monkeypatch.setattr(client, "_notify_webhook", Mock())
        client.edit_image(image_tensors=images, prompt="edit", size="2K", model="doubao-seedream-5.0-pro")
    else:
        client = GeminiImageServiceClient(service_url="https://service.example.com/image/gemini_image", oss_uploader=upload)
        client.call(image_tensors=images, prompt="edit", model="gemini-3-pro-image-preview")
    upload.assert_called_once()
    post.assert_called_once()
    body = post.call_args.kwargs["json"]
    assert body["image" if kind == "seedream_litellm" else "image_url_list"] == [input_url]
    if kind.startswith("gpt"):
        assert body["channel"] == ("gpt-image-2.5" if kind == "gpt_litellm_url_channel" else "gpt-image-2")
    assert "files" not in post.call_args.kwargs
    assert "data" not in post.call_args.kwargs
    get.assert_called_once()
    assert get.call_args.args[0] == public_result


@pytest.mark.parametrize("source", ["gemini", "edit"])
def test_strict_gpt_uses_configured_url_service_without_changing_model(monkeypatch, source):
    from src.Comfyui_Fd_Nodes import config
    from src.Comfyui_Fd_Nodes.utils import gpt_image_edit_client

    monkeypatch.setenv("FD_IMAGE_DELIVERY", "url")
    if source == "gemini":
        gemini_url = "https://gateway.example.com/prefix/image/gemini_image"
        monkeypatch.setenv("FD_GEMINI_URL", gemini_url)
        monkeypatch.setattr(config, "FD_GEMINI_URL", gemini_url)
        edit_url = config._derive_gpt_image_edit_url()
        assert edit_url == "https://gateway.example.com/prefix/image/edit"
    else:
        edit_url = "https://edit.example.com/image/edit"
        monkeypatch.setenv("FD_GPT_IMAGE_EDIT_URL", edit_url)
    # Mirror config's existing import-time derivation without reloading shared modules.
    monkeypatch.setattr(gpt_image_edit_client, "FD_GPT_IMAGE_EDIT_URL", edit_url)
    client = GptImageEditClient(backend="litellm")
    url_call, binary_call = Mock(return_value="url-result"), Mock()
    monkeypatch.setattr(client, "_edit_via_image_generation", url_call)
    monkeypatch.setattr(gpt_image_edit_client._LiteLLMAdapter, "_call_gpt_image_with_retry_policy", binary_call)
    assert client.edit_image(image_tensors=[], prompt="edit", size="2K", model="gpt-image-2") == "url-result"
    assert client.edit_url == edit_url
    assert url_call.call_args.kwargs["model"] == "gpt-image-2"
    binary_call.assert_not_called()


@pytest.mark.parametrize("kind", ["gpt", "seedream", "seedream_litellm", "gemini"])
@pytest.mark.parametrize("fail_at", [1, 2])
def test_generation_upload_failure_prevents_all_model_requests(monkeypatch, kind, fail_at):
    monkeypatch.setenv("FD_IMAGE_DELIVERY", "url")
    upload = Mock(side_effect=["https://cdn.example.com/first.png"] * (fail_at - 1) + [RuntimeError("upload failed")])
    post, get = Mock(), Mock()
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("requests.get", get)
    images = torch.zeros((2, 2, 2, 3))
    with pytest.raises(RuntimeError, match="upload failed"):
        if kind == "gpt":
            client = GptImageEditClient(backend="image_generation", oss_uploader=upload)
            client.edit_image(image_tensors=[images[:1], images[1:]], prompt="edit", size="2K")
        elif kind.startswith("seedream"):
            client = SeedreamImageClient(backend="litellm" if kind == "seedream_litellm" else "image_generation", oss_uploader=upload)
            client.edit_image(image_tensors=images, prompt="edit", size="2K", model="doubao-seedream-5.0-pro")
        else:
            client = GeminiImageServiceClient(service_url="https://service.example.com/gemini", oss_uploader=upload)
            client.call(image_tensors=images, prompt="edit", model="gemini-3-pro-image-preview")
    assert upload.call_count == fail_at
    post.assert_not_called()
    get.assert_not_called()

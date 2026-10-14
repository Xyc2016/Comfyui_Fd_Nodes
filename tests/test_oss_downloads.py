from io import BytesIO

import pytest
import requests
from PIL import Image

from src.Comfyui_Fd_Nodes import nodes, old_gemini_api_node
from src.Comfyui_Fd_Nodes.aistudio_image_combo_node import ZhiYiAiStudioImageComboNode
from src.Comfyui_Fd_Nodes.remove_bg_by_meitu_node import ZhiYiRemoveBgByMeituNode
from src.Comfyui_Fd_Nodes.utils.gemini_service import GeminiImageServiceClient
from src.Comfyui_Fd_Nodes.utils.gpt_image_edit_client import GptImageEditClient
from src.Comfyui_Fd_Nodes.utils.gpt_image_request import GptImageRequestMixin
from src.Comfyui_Fd_Nodes.utils.seedream_image_client import SeedreamImageClient


@pytest.mark.parametrize("download", [
    "gemini", "old_gemini", "flux", "z_image", "gpt", "gpt_edit",
    "seedream", "seedream_litellm", "aistudio", "meitu",
])
@pytest.mark.parametrize(("result_url", "download_url"), [
    (
        "https://zhiyi-image.oss-cn-hangzhou.aliyuncs.com//xyc_dev/a%20b.png?Expires=1&Signature=a%2Bb%3D",
        "https://zhiyi-image.oss-cn-hangzhou-internal.aliyuncs.com//xyc_dev/a%20b.png?Expires=1&Signature=a%2Bb%3D",
    ),
    (
        "http://zhiyi-image.oss-cn-hangzhou.aliyuncs.com/xyc_dev/a.png",
        "http://zhiyi-image.oss-cn-hangzhou-internal.aliyuncs.com/xyc_dev/a.png",
    ),
    (
        "https://zhiyi-image.oss-cn-hangzhou-internal.aliyuncs.com/xyc_dev/a.png",
        "https://zhiyi-image.oss-cn-hangzhou-internal.aliyuncs.com/xyc_dev/a.png",
    ),
    (
        "https://example.com/xyc_dev/a.png?token=a%2Bb%3D",
        "https://example.com/xyc_dev/a.png?token=a%2Bb%3D",
    ),
])
def test_download_uses_internal_oss_and_preserves_business_url(
    monkeypatch: pytest.MonkeyPatch, download: str, result_url: str, download_url: str,
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), "red").save(image_bytes, format="PNG")
    get_urls = []

    class Response:
        status_code = 200
        ok = True
        content = image_bytes.getvalue()

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "status": True, "result_image_url": result_url, "message": "ok",
                "urls": [result_url], "data": [{"url": result_url}],
            }

    def fake_get(url: str, **kwargs: object) -> Response:
        get_urls.append(url)
        return Response()

    def fake_post(*args: object, **kwargs: object) -> Response:
        return Response()

    def fake_request(method: str, url: str, **kwargs: object) -> Response:
        assert method == "GET"
        return fake_get(url, **kwargs)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "request", fake_request)
    monkeypatch.setattr(nodes, "FD_GEN_IMAGE_NOTIFICATION_WEBHOOK_URL", None)
    monkeypatch.setattr(old_gemini_api_node, "FD_GEMINI_WEBHOOK_URL", None)
    params = {"prompt": "draw", "out_request_id": "xyc_dev", "aspect_ratio": "1:1"}

    if download == "gemini":
        _, url, _ = GeminiImageServiceClient(service_url="https://example.com/gemini").call_with_image_urls(
            prompt="draw", model="google/gemini-3-pro-image-preview", image_url_list=["https://example.com/input.png"],
        )
        assert url == result_url
    elif download == "old_gemini":
        _, _, url = old_gemini_api_node.FD_GeminiImage().api_call(model="google/gemini-3-pro-image-preview", **params)
        assert url == result_url
    elif download == "flux":
        nodes.FD_Flux2KleinGenImage().api_call(**params)
    elif download == "z_image":
        _, url = nodes.FD_ZImageTurboGenImage().api_call(**params)
        assert url == result_url
    elif download == "gpt":
        _, _, url = GptImageRequestMixin()._decode_gpt_image_result(Response().json())
        assert url == result_url
    elif download == "gpt_edit":
        assert GptImageEditClient()._download_result_image(result_url).getvalue() == Response.content
    elif download in ("seedream", "seedream_litellm"):
        client = SeedreamImageClient(
            backend="litellm" if download == "seedream_litellm" else "image_generation",
            generate_url="https://example.com/generate",
        )
        image, url = client.generate_image(prompt="draw", model="doubao-seedream-5.0-pro", size="2K")
        assert url == result_url
        assert image.getvalue() == Response.content
    elif download == "aistudio":
        ZhiYiAiStudioImageComboNode()._download_result_image(result_url)
    elif download == "meitu":
        assert ZhiYiRemoveBgByMeituNode()._download_bytes(result_url, 30) == Response.content

    assert get_urls == [download_url]

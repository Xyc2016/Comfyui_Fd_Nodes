import logging

import pytest

from src.Comfyui_Fd_Nodes.utils import gpt_image_request as gpt_image_request_module
from src.Comfyui_Fd_Nodes.utils.gpt_image_request import GptImageRequestMixin


class DummyGptImageClient(GptImageRequestMixin):
    pass


def test_gpt_image_retry_policy_does_not_use_azure_fallback(monkeypatch):
    client = DummyGptImageClient()
    posted_models = []

    class DummyResponse:
        status_code = 500
        text = "upstream failed"

        def raise_for_status(self):
            raise gpt_image_request_module.requests.exceptions.HTTPError(
                "500 Server Error",
                response=self,
            )

    def fake_post(url, headers, data, files, timeout):
        posted_models.append(data["model"])
        return DummyResponse()

    monkeypatch.setattr(gpt_image_request_module.requests, "post", fake_post)

    with pytest.raises(Exception, match="HTTP 500"):
        client._call_gpt_image_with_retry_policy(
            base_url="https://example.com",
            api_key="secret",
            data={
                "model": "gpt-image-2",
                "prompt": "edit image",
                "size": "1024x1024",
            },
            multipart_files=[],
            batch_size=1,
            logger=logging.getLogger(__name__),
        )

    assert posted_models == ["gpt-image-2", "gpt-image-2", "gpt-image-2"]
    assert "gpt-image-2-azure" not in posted_models

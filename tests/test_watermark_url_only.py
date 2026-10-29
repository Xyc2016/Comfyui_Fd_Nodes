import os
from unittest.mock import patch

import pytest
import torch

from src.Comfyui_Fd_Nodes.nodes import FD_RemoveWatermark


def test_watermark_url_only_upload_failure_does_not_send_or_return_original():
    with patch.dict(os.environ, {"FD_OVERSEAS_GPU_GATEWAY": "https://gateway.example/gpu"}):
        with patch("src.Comfyui_Fd_Nodes.utils.image_payload.tensor_to_image_payload", side_effect=RuntimeError("upload failed")):
            with patch("requests.post") as post:
                with pytest.raises(RuntimeError, match="URL request failed: upload failed"):
                    FD_RemoveWatermark().remove_watermark(
                        torch.zeros((1, 2, 2, 3)), "http://model/v1/process", "", 0.3, 0.35, 3000, 9, 2, "lama", True, True, 95
                    )
                post.assert_not_called()

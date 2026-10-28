import os
from unittest.mock import patch

import pytest

from Comfyui_Fd_Nodes.nodes import FD_RemoveWatermark


def test_watermark_url_only_fails_before_sending_or_returning_original():
    with patch.dict(os.environ, {"FD_OVERSEAS_GPU_GATEWAY": "https://gateway.example/gpu"}):
        with pytest.raises(NotImplementedError, match="URL-only"):
            FD_RemoveWatermark().remove_watermark(
                None, "http://model/v1/process", "", 0.3, 0.35, 3000, 9, 2, "lama", True, True, 95
            )

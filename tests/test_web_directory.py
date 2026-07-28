import importlib.util
from pathlib import Path


def test_web_directory_registers_body_segment_extension():
    project_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("comfyui_fd_nodes_root", project_root / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.WEB_DIRECTORY == "./src/Comfyui_Fd_Nodes/other"
    assert "WEB_DIRECTORY" in module.__all__
    web_directory = project_root / module.WEB_DIRECTORY
    assert web_directory.is_dir()
    assert (web_directory / "zhiyi_body_segment.js").is_file()

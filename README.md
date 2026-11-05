# comfyui-fd-nodes

Custom nodes for fd project

节点下载图片时，按 image2oss 的方式将 URL 中的 `oss-cn-hangzhou.aliyuncs.com` 替换为 `oss-cn-hangzhou-internal.aliyuncs.com`，无需环境变量，运行环境需能访问阿里云 VPC 内网。通常该改写仅用于本地下载；GPU 图像预处理节点的 `image_delivery=url` 模式也会使用内网 OSS 链接，供杭州模型服务下载。其他节点返回的图片 URL 和传给下游服务的 URL 保持原样。

## GPU 图像服务 URL 传图

SAM2、DWPose、LineArt、DepthAnythingV2、RMBG、衣物/时尚/身体分割的 V1/V2 节点提供可选参数 `image_delivery`：

- `base64`（默认）：继续将 IMAGE tensor 编码为 PNG data URL，旧工作流无需修改。
- `url`：将当前输入 tensor 编码为原尺寸 RGB PNG，上传 OSS 后将链接写入请求的 `image` 字段。适合大图经过海外公网网关的场景，不改变 bbox 坐标、推理参数或响应格式。

例如日志中的 `ZhiYiSAM2SegmentNodeV2`，保留原有 `env`、`service_url`、`dilate` 等配置，只增加 `"image_delivery": "url"`。`env` 控制模型服务地址，`image_delivery` 独立控制图片传输方式；切换“海外环境”不会自动启用 URL 模式。更新插件后需重启 ComfyUI，界面刷新后可在节点末尾选择该参数。

URL 模式沿用现有 OSS 上传配置：`FD_OSS_BUCKET_NAME`、`FD_OSS_ENDPOINT`、`FD_OSS_URL_PREFIX`，以及 AK 凭证 `FD_OSS_ACCESS_KEY_ID` / `FD_OSS_ACCESS_KEY_SECRET` 或 STS 配置 `FD_OSS_STS_URL` / `FD_OSS_STS_KEY`。可通过 `FD_OSS_URL_PATH_PREFIX_GPU_INPUT` 覆盖对象路径前缀，默认 `devops/comfyui/segment_img`；文件名为 PNG 内容的 MD5。每次执行仍会调用 OSS 上传，相同内容使用相同对象路径。

链接必须允许模型服务无需特殊请求头直接 GET，上传工具不会自动签名私有对象。杭州 OSS 域名会改为内网域名，因此模型服务需能访问杭州 OSS VPC 网络；其他 CDN/自定义域名保持原样。这里上传的是节点当前收到的图像，不会直接复用上游加载前的图片 URL，以保留裁剪/缩放等处理结果。

服务端图片上限为 20MB（可由服务端配置调整），URL 下载和推理仍需要时间，调用 timeout 建议至少 60 秒。URL 模式减少模型 API 的 JSON 请求体，不保证消除所有连接、下载或排队超时。422 `INVALID_IMAGE`、413 `PAYLOAD_TOO_LARGE` 直接透出错误，不自动退回 base64 或重试；OSS 上传失败也直接报错。

若模型服务开启鉴权，分别配置 `FD_SAM2_API_KEY` 和 `FD_DWPOSE_API_KEY`，节点会对推理与健康检查请求附加 `Authorization: Bearer <KEY>`。未配置时不附加鉴权头。

## 远程 RMBG 部署环境与传输

`ZhiYiRMBGNode` / `ZhiYiRMBGNodeV2` 使用 Comfy 进程环境变量路由，不再提供 `env`、`service_url`、`service_url_preset` 或三个 `return_*` 节点输入。旧 API JSON 中残留的这些字段会被忽略；不会覆盖部署配置。

| `FD_COMFY_RUNTIME_ENV` | 默认服务 | `image_encoding` 缺省值 |
|---|---|---|
| `dev`（未设置时） | `http://10.1.0.230:8003/v1/rmbg` | `png` |
| `domestic` | `http://model-api-dwpose-svc.online-server-gray:8001/v1/rmbg` | `png` |
| `overseas` | `FD_OVERSEAS_GPU_GATEWAY` 下的 `/model-api-dwpose/v1/rmbg` | `jpeg_q95` |

部署可用 `FD_RMBG_URL` 覆盖服务地址（包括生产依赖地址）；非法运行环境在加载配置时明确失败。修改部署变量需重启进程。国内集群与 102 保持无损 PNG，海外默认 JPEG Q95；节点末尾仍可用 `image_encoding=png/jpeg_q95` 手动指定，已保存的显式选择优先于缺省值。Q95 使用原尺寸 RGB、quality=95、4:2:0、optimize=true、非 progressive；有损压缩可能改变 mask 边界和背景残留。

- `refine_foreground=false`：只请求无损 8 位灰度 PNG mask，用压缩前原始 RGB 本地合成输出。Alpha 与旧回图解码一致：按软 mask 叠到黑底后输出三通道 RGB（含 8 位舍入），不再输出直通 RGBA；透明度仍通过独立 MASK 输出提供。Color 保持原背景色合成。
- `refine_foreground=true`：请求精修图片和 mask，使用服务端精修后的像素；Alpha 保留 RGBA，不用原图覆盖精修结果。服务端漏返图片时明确失败。
- MASK_IMAGE 始终本地从 mask 生成，不传输重复预览。

仅 mask 响应校验声明尺寸、原图尺寸、8 位灰度 PNG 编码与返回字段；推理尺寸 mask 按 BILINEAR 恢复原尺寸，异常明确失败。INFO 和调用日志记录编码、返回模式及实际服务地址，INFO 额外记录运行环境。

保留 `image_delivery=base64/url` 及既有海外 URL 传输约束；Q95 URL 上传 `.jpg`，Base64 使用 `data:image/jpeg;base64,`，服务端响应仍为 PNG。不再读取旧 `FD_RMBG_REMOTE_OPTIMIZATION` / `FD_OVERSEAS_RMBG_MASK_ONLY`。衣物/时尚/身体分割及本地 RMBG 不受本次契约调整影响。

> [!NOTE]
> This projected was created with a [cookiecutter](https://github.com/Comfy-Org/cookiecutter-comfy-extension) template. It helps you start writing custom nodes without worrying about the Python setup.

## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started).
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
1. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
1. Restart ComfyUI.

# Features

- A list of features

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd Comfyui_Fd_Nodes
pip install -e .[dev]
pre-commit install
```

The `-e` flag above will result in a "live" install, in the sense that any changes you make to your node extension will automatically be picked up the next time you run ComfyUI.

## Publish to Github

Install Github Desktop or follow these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for ssh.

1. Create a Github repository that matches the directory name. 
2. Push the files to Git
```
git add .
git commit -m "project scaffolding"
git push
``` 

## Writing custom nodes

An example custom node is located in [node.py](src/Comfyui_Fd_Nodes/nodes.py). To learn more, read the [docs](https://docs.comfy.org/essentials/custom_node_overview).


## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. It is recommended to unit test your custom node.

- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes

## Publishing to Registry

If you wish to share this custom node with others in the community, you can publish it to the registry. We've already auto-populated some fields in `pyproject.toml` under `tool.comfy`, but please double-check that they are correct.

You need to make an account on https://registry.comfy.org and create an API key token.

- [ ] Go to the [registry](https://registry.comfy.org). Login and create a publisher id (everything after the `@` sign on your registry profile). 
- [ ] Add the publisher id into the pyproject.toml file.
- [ ] Create an api key on the Registry for publishing from Github. [Instructions](https://docs.comfy.org/registry/publishing#create-an-api-key-for-publishing).
- [ ] Add it to your Github Repository Secrets as `REGISTRY_ACCESS_TOKEN`.

A Github action will run on every git push. You can also run the Github action manually. Full instructions [here](https://docs.comfy.org/registry/publishing). Join our [discord](https://discord.com/invite/comfyorg) if you have any questions!

# 图生文 Combo 模型

`知衣-图生文-combo`（`ZhiYiImageTextComboNode`）从 53.8.76 起可在末尾的
`model` 下拉框选择 `qwen3.7-plus`。从 53.8.77 起默认使用 `qwen3.7-plus`，未填写
`model` 的旧 API 工作流也使用该默认值；豆包仍可手动选择。Qwen 直接调用 `FD_LITELLM_BASE_URL` 的 Chat Completions，
不经过 `FD_DOUBAO_VISION_URL`；保留多图、system_prompt、temperature、max_tokens、
并发和重试设置。国内默认 base64，海外继续按部署规则上传 OSS 后传 URL。
使用前须确认所在 LiteLLM 网关和调用 Key 已提供 `qwen3.7-plus`。


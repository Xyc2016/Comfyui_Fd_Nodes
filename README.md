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

## 远程 RMBG 传输质量（单选）

`ZhiYiRMBGNode` / `ZhiYiRMBGNodeV2` 在末尾提供可选单选项 `image_encoding`：

- `png`（默认）：无损 PNG 推理输入；旧工作流缺少该字段时仍使用 PNG。
- `jpeg_q95`：原尺寸 RGB JPEG 推理副本，quality=95、4:2:0、optimize=true、非 progressive。原输入无论 PNG 或 JPEG 都统一重编码为 Q95。

质量选择与 `env` / `service_url` / `image_delivery` 独立。两种质量均仅请求无损 8 位灰度 PNG mask（`return_image=false`、`return_mask=true`、`return_mask_image=false`），客户端用压缩前原始 RGB 与 mask 合成 Alpha 透明图或 Color 背景换色图，并本地生成 MASK_IMAGE。保留旧 return 入参以兼容 JSON，RMBG 请求固定使用上述实际返回开关；衣物/时尚/身体分割仍沿用原返回参数，不新增质量选项。

不再读取 `FD_RMBG_REMOTE_OPTIMIZATION` 或 `FD_OVERSEAS_RMBG_MASK_ONLY`。刷新界面后即可在节点选择质量；无需通过环境变量切换或重启进程。INFO 和正式调用日志记录 `image_encoding`、`mask_only_response`，可直接确认 Q95 是否生效。

仅 mask 响应要求 `image_size` 与原图一致，灰度 PNG 的实际尺寸与 `output_size` 一致，且输出尺寸属于原图尺寸或声明的 `inference_size`。推理尺寸的 mask 按现有 BILINEAR 规则恢复原尺寸；缺失、损坏、错误尺寸、非灰度 PNG 或服务仍返回成图均明确报错。

选择 Q95 时，`image_delivery=url` 上传内容 MD5 命名的 `.jpg`，base64 模式使用 `data:image/jpeg;base64,`。服务端 `image_format=png_base64` 是响应格式，保持不变。JPEG 有损压缩可能改变分割边界和背景残留，可选择 PNG 比较效果。该选项不自动替换工作流中的本地 RMBG 或 ClothesSegment，款式上身 #313 保持现状。

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


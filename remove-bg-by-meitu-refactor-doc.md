# 服装退底接口 remove-bg-by-meitu 重构文档

## 一、接口概述

**业务含义**: 接收一张服装图片 URL，调用外部 AI 服务（美图）进行背景移除（抠图），返回处理后的图片、mask 和红边图片 URL。

**当前实现方式**: Spring Cloud OpenFeign 直连外部 AI 服务，Controller 层透传，无额外业务逻辑。

**重构目标**: 重新实现请求逻辑（具体目标待确认，以下文档供 Codex 理解现有代码结构）。

---

## 二、调用链路

```
HTTP 请求 (POST /image-combine/mt/remove-bg-by-meitu)
  → ImageCombineMTController.removeBgByMeitu()
    → ImageCombineMTServingClient.removeBgByMeitu()  [Feign 调用]
      → POST {image.detail.url}/image/remove_bg_by_meitu  [外部 AI 服务]
    ← ClothRemoveBGDTO
  ← ApiResult<ClothRemoveBGDTO>
```

---

## 三、涉及文件清单（绝对路径）

### 3.1 Controller 层

**文件**: `/Users/chenjunhong/Documents/javaProject/fashion-parent/app-business-clothingAI/src/main/java/com/zhiyi/fashion/clothingai/server/facade/http/controller/ImageCombineMTController.java`

- 第 74-78 行: `remove-bg-by-meitu` 接口入口
- 依赖注入: `ImageCombineMTServingClient`, `XinChenManager`, `ImageProcessStatisticsManager`
- 注意: 当前该接口**没有** XinChen 团队判断和统计埋点，但同文件中 `merge-forge-back` 接口（第 108-118 行）有这些逻辑，可作为参考

```java
// 第 74-78 行
@ApiOperation(value = "服装退底", httpMethod = "POST")
@PostMapping("/remove-bg-by-meitu")
public ApiResult<ClothRemoveBGDTO> removeBgByMeitu(HttpServletRequest request, @RequestBody ClothRemoveBGRequest clothRemoveBGRequest) {
    ClothRemoveBGDTO clothRemoveBGDTO = imageCombineMTServingClient.removeBgByMeitu(clothRemoveBGRequest);
    return ApiResult.success(clothRemoveBGDTO);
}
```

### 3.2 Feign Client 层（HTTP 客户端）

**文件**: `/Users/chenjunhong/Documents/javaProject/fashion-parent/app-business-clothingAI/src/main/java/com/zhiyi/fashion/clothingai/server/infra/api/ai/ImageCombineMTServingClient.java`

- 第 51-52 行: `removeBgByMeitu` 方法定义
- 第 15 行: `@FeignClient` 配置，基础 URL 为 `${image.detail.url}`
- 第 60-61 行: 还有一个 API 版本 `remove_bg_by_api`，接收相同请求/响应

```java
// 第 51-52 行
@PostMapping("/image/remove_bg_by_meitu")
ClothRemoveBGDTO removeBgByMeitu(@RequestBody ClothRemoveBGRequest request);
```

### 3.3 Feign 序列化配置

**文件**: `/Users/chenjunhong/Documents/javaProject/fashion-parent/app-business-clothingAI/src/main/java/com/zhiyi/fashion/clothingai/server/infra/api/ai/config/FeignSnakeCaseConfig.java`

- 功能: 将 Java camelCase 字段自动转为 snake_case 发送给下游服务
- 影响范围: 所有使用此 configuration 的 Feign Client

### 3.4 请求对象

**文件**: `/Users/chenjunhong/Documents/javaProject/fashion-parent/app-business-clothingAI/src/main/java/com/zhiyi/fashion/clothingai/server/infra/api/ai/request/ClothRemoveBGRequest.java`

```java
@Data
public class ClothRemoveBGRequest {
    @ApiModelProperty("图片链接")
    private String imageUrl;

    private Integer repaintEdge;

    @ApiModelProperty("红边框粗细，默认40")
    private Integer edgeThickness = 40;
}
```

经 FeignSnakeCaseConfig 转换后，实际发送给下游的 JSON 格式:
```json
{
  "image_url": "https://xxx/xxx.jpg",
  "repaint_edge": 1,
  "edge_thickness": 40
}
```

### 3.5 响应对象

**基础 DTO 文件**: `/Users/chenjunhong/Documents/javaProject/fashion-parent/app-business-clothingAI/src/main/java/com/zhiyi/fashion/clothingai/server/infra/api/ai/dto/ImageCombineBaseDTO.java`

```java
@Data
public class ImageCombineBaseDTO implements Serializable {
    private Boolean status;
    @JsonProperty("cost_time")
    private String costTime;
    @JsonProperty("server_name")
    private String serverName;
    private ImageCombineError error;

    @Data
    public static class ImageCombineError {
        private String code;
        private String message;
    }
}
```

**业务 DTO 文件**: `/Users/chenjunhong/Documents/javaProject/fashion-parent/app-business-clothingAI/src/main/java/com/zhiyi/fashion/clothingai/server/infra/api/ai/dto/ClothRemoveBGDTO.java`

```java
@Data
public class ClothRemoveBGDTO extends ImageCombineBaseDTO implements Serializable {
    private String resultUrl;
    private String mainMaskUrl;
    private String RedEdgeImageUrl;  // 注意: 这里是大写 R 开头，snake_case 转换后为 red_edge_image_url
}
```

下游返回的 JSON 格式:
```json
{
  "status": true,
  "cost_time": "1.23s",
  "server_name": "xxx",
  "error": null,
  "result_url": "https://xxx/result.jpg",
  "main_mask_url": "https://xxx/mask.png",
  "red_edge_image_url": "https://xxx/edge.png"
}
```

### 3.6 可参考的辅助类（同模块已有逻辑）

**XinChen 团队处理**: `/Users/chenjunhong/Documents/javaProject/fashion-parent/app-business-clothingAI/src/main/java/com/zhiyi/fashion/clothingai/server/domain/manager/XinChenManager.java`

**图片处理统计**: `/Users/chenjunhong/Documents/javaProject/fashion-parent/app-business-clothingAI/src/main/java/com/zhiyi/fashion/clothingai/server/domain/manager/ImageProcessStatisticsManager.java`

---

## 四、同文件中可参考的完整业务逻辑（merge-forge-back）

`ImageCombineMTController.java` 第 106-118 行的 `merge-forge-back` 接口包含了完整的业务处理模式，包含:
1. Feign 调用
2. XinChen 团队图片转存判断
3. 统计埋点

```java
@ApiOperation(value = "背景融合接口", httpMethod = "POST")
@PostMapping("/merge-forge-back")
public ApiResult<MergeForgeBackDTO> mergeForgeBack(HttpServletRequest request, @RequestBody MergeForgeBackRequest mergeForgeBackRequest) {
    MergeForgeBackDTO mainMaskDTO = imageCombineMTServingClient.mergeForgeBack(mergeForgeBackRequest);
    RequestContext requestContext = RequestHelper.formContext(request);
    if (xinChenManager.isXinChenTeam(requestContext.getTeamId())) {
        mainMaskDTO.setImageUrl(xinChenManager.copy2XinChenImage(mainMaskDTO.getImageUrl()));
        mainMaskDTO.setMainMaskUrl(xinChenManager.copy2XinChenImage(mainMaskDTO.getMainMaskUrl()));
        mainMaskDTO.setResultUrl(xinChenManager.copy2XinChenImage(mainMaskDTO.getResultUrl()));
    }
    imageProcessStatisticsManager.saveImageProcessStatistics(mergeForgeBackRequest.getImageUrl(), 5);
    return ApiResult.success(mainMaskDTO);
}
```

---

## 五、项目包结构

```
com.zhiyi.fashion.clothingai.server
├── facade.http.controller     → Controller 层
│   └── ImageCombineMTController.java
├── infra.api.ai               → 外部 API 调用层
│   ├── ImageCombineMTServingClient.java   (Feign Client 接口)
│   ├── config/
│   │   └── FeignSnakeCaseConfig.java      (snake_case 序列化配置)
│   ├── request/
│   │   └── ClothRemoveBGRequest.java      (请求体)
│   └── dto/
│       ├── ImageCombineBaseDTO.java        (基础响应)
│       └── ClothRemoveBGDTO.java           (业务响应)
└── domain.manager             → 业务逻辑层
    ├── XinChenManager.java
    └── ImageProcessStatisticsManager.java
```

---

## 六、关键配置

- **外部服务基础 URL**: `${image.detail.url}` — 在 Apollo 或环境变量中配置
- **Feign 超时**: 在 `bootstrap.yml` 中配置
- **命名策略**: camelCase → snake_case（双向，请求和响应都自动转换）

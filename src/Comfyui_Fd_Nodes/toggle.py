class NodeToggleByID:
    """
    根据节点 ID 标签批量开启或关闭工作流中的节点。
    node_switch: 0=关闭目标节点，1=开启目标节点
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "node_switch": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": "number",
                }),
                "node_ids": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "节点ID列表，由前端动态管理",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "toggle_nodes"
    CATEGORY = "utils/node_control"
    OUTPUT_NODE = True

    def toggle_nodes(self, node_switch, node_ids, prompt=None, unique_id=None):
        ids = [nid.strip() for nid in node_ids.split(",") if nid.strip()]

        if not ids:
            return ("未指定任何节点 ID",)

        enable = node_switch == 1
        action = "开启" if enable else "关闭"

        if not prompt:
            return (f"将对 {len(ids)} 个节点执行{action}操作",)

        status_parts = []
        for node_id in ids:
            if node_id in prompt:
                class_type = prompt[node_id].get("class_type", "未知")
                status_parts.append(f"节点 {node_id} ({class_type}): {action}")
            else:
                status_parts.append(f"节点 {node_id}: 未找到")

        return ("\n".join(status_parts),)


NODE_CLASS_MAPPINGS = {
    "NodeToggleByID": NodeToggleByID,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NodeToggleByID": "节点开关 (按ID)",
}

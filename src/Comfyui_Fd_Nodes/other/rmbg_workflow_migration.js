import { app } from "../../scripts/app.js";

// LiteGraph stores widgets positionally. Remove retired fields before loading
// an old workflow so process_res/refine/background/encoding cannot shift.
app.registerExtension({
    name: "ZhiYiRMBG.DeploymentInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["ZhiYiRMBGNode", "ZhiYiRMBGNodeV2"].includes(nodeData.name)) return;
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            const values = info.widgets_values;
            if (Array.isArray(values) && typeof values[0] === "string") {
                const v2 = nodeData.name === "ZhiYiRMBGNodeV2";
                const indices = v2
                    ? [2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18]
                    : [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 17, 18];
                info = { ...info, widgets_values: indices.map((index, i) => (
                    values[index] ?? this.widgets[i].value
                )) };
            }
            return configure.call(this, info);
        };
    },
});

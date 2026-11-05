import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ZhiYiImageTextCombo.Base64Only",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ZhiYiImageTextComboNode") return;
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            const values = info.widgets_values;
            // Retired image_delivery occupied index 6, immediately before model.
            if (Array.isArray(values) && ["base64", "url"].includes(values[6])) {
                const migrated = values.slice();
                migrated.splice(6, 1);
                info = { ...info, widgets_values: migrated };
            }
            return configure.call(this, info);
        };
    },
});

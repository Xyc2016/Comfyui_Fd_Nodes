import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

let extension;
const app = { registerExtension(value) { extension = value; } };
const context = vm.createContext({});
const mockApp = new vm.SyntheticModule(["app"], function () {
    this.setExport("app", app);
}, { context });
const source = await readFile(new URL(
    "../src/Comfyui_Fd_Nodes/other/image_text_combo_workflow_migration.js", import.meta.url,
), "utf8");
const module = new vm.SourceTextModule(source, { context });
await module.link(() => mockApp);
await module.evaluate();

class Combo {
    configure(info) { this.info = info; }
}
await extension.beforeRegisterNodeDef(Combo, { name: "ZhiYiImageTextComboNode" });
const prefix = [8, 0.7, 2048, 1, 0, "system prompt"];
for (const delivery of ["url", "base64"]) {
    for (const model of ["qwen3.7-plus", "doubao-seed-2.0-mini"]) {
        const original = [...prefix, delivery, model];
        const node = new Combo();
        node.configure({ widgets_values: original });
        assert.deepEqual(Array.from(node.info.widgets_values), [...prefix, model]);
        assert.deepEqual(original, [...prefix, delivery, model]);
        node.configure(node.info);
        assert.deepEqual(Array.from(node.info.widgets_values), [...prefix, model]);
    }
}
const beforeModelSelector = new Combo();
beforeModelSelector.configure({ widgets_values: [...prefix, "url"] });
assert.deepEqual(Array.from(beforeModelSelector.info.widgets_values), prefix);
const current = new Combo();
current.configure({ widgets_values: [...prefix, "qwen3.7-plus"] });
assert.deepEqual(Array.from(current.info.widgets_values), [...prefix, "qwen3.7-plus"]);
class Other { configure() {} }
const otherConfigure = Other.prototype.configure;
await extension.beforeRegisterNodeDef(Other, { name: "ZhiYiRMBGNode" });
assert.equal(Other.prototype.configure, otherConfigure);
console.log("Workflow widget migration passed");

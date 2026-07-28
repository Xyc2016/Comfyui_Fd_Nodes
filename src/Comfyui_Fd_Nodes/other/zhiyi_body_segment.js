import { app } from "../../scripts/app.js";

const CUSTOM_PRESET = "自定义（使用 service_url）";
const KNOWN_SERVICE_URL = "http://10.1.0.230:8003";
const BODY_CLASSES = [
    "Hair",
    "Glasses",
    "Top-clothes",
    "Bottom-clothes",
    "Torso-skin",
    "Face",
    "Left-arm",
    "Right-arm",
    "Left-leg",
    "Right-leg",
    "Left-foot",
    "Right-foot",
];

function hideWidget(widget) {
    if (!widget || widget._zhiyiHidden) return;
    widget._zhiyiHidden = true;
    widget._zhiyiComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
}

function parseClasses(value) {
    const text = String(value ?? "").trim();
    if (!text) return { classes: [], recognized: true };

    let parsed;
    if (text.startsWith("[")) {
        try {
            parsed = JSON.parse(text);
        } catch {
            return { classes: [], recognized: false };
        }
        if (!Array.isArray(parsed)) return { classes: [], recognized: false };
        parsed = parsed.map((item) => String(item).trim()).filter(Boolean);
    } else {
        const lines = text.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
        parsed = lines.length === 1 && lines[0].includes(",")
            ? lines[0].split(",").map((item) => item.trim()).filter(Boolean)
            : lines;
    }

    return {
        classes: parsed,
        recognized: parsed.every((item) => BODY_CLASSES.includes(item)),
    };
}

function setNodeDirty(node) {
    node.setSize(node.computeSize());
    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "ZhiYiBodySegment.UI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ZhiYiBodySegmentNode") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const serviceWidget = this.widgets?.find((widget) => widget.name === "service_url");
            const classesWidget = this.widgets?.find((widget) => widget.name === "classes");
            const presetWidget = this.widgets?.find((widget) => widget.name === "service_url_preset");
            if (!serviceWidget || !classesWidget || !presetWidget) return;

            hideWidget(serviceWidget);
            hideWidget(classesWidget);
            hideWidget(presetWidget);

            this._zhiyiAddressWidget = this.addWidget("combo", "服务地址", "10.1.0.230", (value) => {
                if (this._zhiyiSyncing) return;
                if (value === "10.1.0.230") {
                    serviceWidget.value = KNOWN_SERVICE_URL;
                    presetWidget.value = "10.1.0.230";
                } else {
                    presetWidget.value = CUSTOM_PRESET;
                }
                this._zhiyiCustomUrlWidget.hidden = value !== "自定义地址";
                setNodeDirty(this);
            }, {
                values: ["10.1.0.230", "自定义地址"],
                serialize: false,
            });

            this._zhiyiCustomUrlWidget = this.addWidget("text", "自定义地址", "", (value) => {
                if (this._zhiyiSyncing) return;
                serviceWidget.value = value;
                presetWidget.value = CUSTOM_PRESET;
            }, { serialize: false });
            this._zhiyiCustomUrlWidget.computeSize = () => this._zhiyiCustomUrlWidget.hidden ? [0, -4] : [0, 20];

            this._zhiyiClassWidgets = BODY_CLASSES.map((className) => this.addWidget(
                "toggle",
                className,
                false,
                () => {
                    if (this._zhiyiSyncing) return;
                    classesWidget.value = JSON.stringify(
                        this._zhiyiClassWidgets
                            .filter((widget) => widget.value)
                            .map((widget) => widget.name),
                    );
                    this._zhiyiClassesNotice.value = "";
                },
                { serialize: false },
            ));

            this._zhiyiClassesNotice = this.addWidget("text", "类别提示", "", () => {}, {
                readonly: true,
                serialize: false,
            });
            this._zhiyiClassesNotice.computeSize = () => this._zhiyiClassesNotice.value ? [0, 20] : [0, -4];

            this._syncZhiYiBodySegmentUi();
        };

        nodeType.prototype._syncZhiYiBodySegmentUi = function () {
            const serviceWidget = this.widgets?.find((widget) => widget.name === "service_url");
            const classesWidget = this.widgets?.find((widget) => widget.name === "classes");
            const presetWidget = this.widgets?.find((widget) => widget.name === "service_url_preset");
            if (!serviceWidget || !classesWidget || !presetWidget || !this._zhiyiAddressWidget) return;

            this._zhiyiSyncing = true;
            const preset = presetWidget.value;
            const serviceUrl = String(serviceWidget.value ?? "");
            const knownPreset = preset && preset !== CUSTOM_PRESET;
            const address = knownPreset || serviceUrl === KNOWN_SERVICE_URL || serviceUrl.startsWith(`${KNOWN_SERVICE_URL}/`)
                ? "10.1.0.230"
                : "自定义地址";
            this._zhiyiAddressWidget.value = address;
            this._zhiyiCustomUrlWidget.value = serviceUrl;
            this._zhiyiCustomUrlWidget.hidden = address !== "自定义地址";

            const parsed = parseClasses(classesWidget.value);
            this._zhiyiClassWidgets.forEach((widget) => {
                widget.value = parsed.recognized && parsed.classes.includes(widget.name);
            });
            this._zhiyiClassesNotice.value = parsed.recognized ? "" : "包含自定义或无法识别的类别，原值已保留";
            this._zhiyiSyncing = false;
            setNodeDirty(this);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            this._syncZhiYiBodySegmentUi?.();
        };
    },
});

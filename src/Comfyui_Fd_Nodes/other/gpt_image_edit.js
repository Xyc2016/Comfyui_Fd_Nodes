import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "GPTImageEdit.UI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GPTImageEdit") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            // ── colour the node header ──────────────────────────────────
            this.color = "#1a3a2a";
            this.bgcolor = "#0f2018";

            // ── keep track of how many image slots are visible ──────────
            this._visibleImageSlots = 0;

            // ── "Add Image" button ──────────────────────────────────────
            this.addWidget("button", "＋ Add Input Image", null, () => {
                if (this._visibleImageSlots >= 4) {
                    alert("Maximum 4 input images supported.");
                    return;
                }
                this._visibleImageSlots++;
                this._refreshImageInputs();
            }, { serialize: false });

            // ── "Remove Image" button ───────────────────────────────────
            this.addWidget("button", "－ Remove Input Image", null, () => {
                if (this._visibleImageSlots <= 0) return;
                // disconnect the slot before hiding
                const slotName = `image${this._visibleImageSlots}`;
                const slotIdx = this.findInputSlot(slotName);
                if (slotIdx !== -1 && this.inputs[slotIdx].link != null) {
                    app.graph.removeLink(this.inputs[slotIdx].link);
                }
                this._visibleImageSlots--;
                this._refreshImageInputs();
            }, { serialize: false });

            // ── batch_size indicator label ──────────────────────────────
            this._batchLabel = this.addWidget("text", "batch info", "", () => {}, {
                serialize: false,
                readonly: true,
            });
            this._batchLabel.computeSize = () => [0, 20];

            this._refreshImageInputs();
            this._updateBatchLabel();

            // watch batch_size widget changes
            const batchWidget = this.widgets?.find(w => w.name === "batch_size");
            if (batchWidget) {
                const origCallback = batchWidget.callback;
                batchWidget.callback = (v) => {
                    origCallback?.call(batchWidget, v);
                    this._updateBatchLabel();
                };
            }
        };

        // ── helpers ─────────────────────────────────────────────────────
        nodeType.prototype._refreshImageInputs = function () {
            const names = ["image1", "image2", "image3", "image4"];
            names.forEach((name, i) => {
                const slotIdx = this.findInputSlot(name);
                const shouldExist = i < this._visibleImageSlots;

                if (shouldExist && slotIdx === -1) {
                    this.addInput(name, "IMAGE");
                } else if (!shouldExist && slotIdx !== -1) {
                    // remove link first
                    if (this.inputs[slotIdx].link != null) {
                        app.graph.removeLink(this.inputs[slotIdx].link);
                    }
                    this.removeInput(slotIdx);
                }
            });
            this.setSize(this.computeSize());
            app.graph.setDirtyCanvas(true, true);
        };

        nodeType.prototype._updateBatchLabel = function () {
            const batchWidget = this.widgets?.find(w => w.name === "batch_size");
            const n = batchWidget ? batchWidget.value : 1;
            if (this._batchLabel) {
                this._batchLabel.value = `Will generate ${n} image${n > 1 ? "s" : ""} per run`;
            }
        };

        // ── serialise visible slot count so it survives save/load ───────
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
            onSerialize?.apply(this, arguments);
            data.visibleImageSlots = this._visibleImageSlots;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            onConfigure?.apply(this, arguments);
            if (data.visibleImageSlots !== undefined) {
                this._visibleImageSlots = data.visibleImageSlots;
                this._refreshImageInputs();
            }
            this._updateBatchLabel();
        };
    },
});

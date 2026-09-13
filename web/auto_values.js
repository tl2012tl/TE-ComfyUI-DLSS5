import { app } from "../../scripts/app.js";

const nodeTypes = new Set([
    "TE_DLSS5_VideoEnhancer",
    "TE_DLSS5_FrameInterpolator",
]);

function showAutoValues(node) {
    for (const widget of node.widgets ?? []) {
        if (!["frame_count", "output_fps"].includes(widget.name)) continue;
        if (widget._teAutoDisplay) continue;
        widget._teAutoDisplay = true;
        widget.label = `${widget.name} (0 = auto)`;
        const draw = widget.draw;
        if (typeof draw !== "function") continue;
        // Format only the canvas text. The stored value, numeric editor and
        // workflow serialization remain numeric, including the auto sentinel 0.
        widget.draw = function (ctx, ...args) {
            if (Number(this.value) !== 0) return draw.call(this, ctx, ...args);
            const fillText = ctx.fillText;
            ctx.fillText = function (text, ...textArgs) {
                const display = /^0(?:\.0+)?$/.test(String(text)) ? "auto" : text;
                return fillText.call(this, display, ...textArgs);
            };
            try {
                return draw.call(this, ctx, ...args);
            } finally {
                ctx.fillText = fillText;
            }
        };
    }
}

app.registerExtension({
    name: "TE.DLSS5.AutoValues",
    nodeCreated(node) {
        if (nodeTypes.has(node.comfyClass)) showAutoValues(node);
    },
    loadedGraphNode(node) {
        if (nodeTypes.has(node.comfyClass)) showAutoValues(node);
    },
});

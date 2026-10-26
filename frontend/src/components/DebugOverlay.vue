<script setup>
import { computed, reactive, ref } from "vue";

// Draws what the scan found on top of the photo (the straightened one, when it was taken at an
// angle). By default only a box around each card's name, with the card it was identified as
// written inside, so a glance shows whether each name was read right. The other layers show the
// working: the model's boxes, close-up areas, OpenCV outlines, tiles, and what was dropped.
const props = defineProps({
  url: String,
  width: Number,
  height: Number,
  detections: { type: Array, default: () => [] },
  duplicates: { type: Array, default: () => [] },
  outlines: { type: Array, default: () => [] },
  tiles: { type: Array, default: () => [] },
  filename: { type: String, default: "scan" },
});

const LAYERS = [
  { id: "name", label: "Names", color: "#ffd23f" },
  { id: "unsure", label: "Unsure names", color: "#ff8a3d" },
  { id: "card", label: "Cards & piles", color: "#ffd23f", dash: true },
  { id: "zoom", label: "Close-up area re-read", color: "#38d6ff", dash: true },
  { id: "model", label: "Model's box before lining up", color: "#e879f9", dash: true },
  { id: "outline", label: "Card outlines (OpenCV)", color: "#a3e635", dash: true },
  { id: "tile", label: "Tiles read separately", color: "#94a3b8", dash: true },
  { id: "ignored", label: "Ignored cut-outs", color: "#ff5c5c", dash: true },
  { id: "duplicate", label: "Dropped duplicates", color: "#b8b8c8", dash: true },
];
const SHOWN = ["name", "unsure"];
const on = reactive(Object.fromEntries(LAYERS.map((l) => [l.id, SHOWN.includes(l.id)])));
const style = Object.fromEntries(LAYERS.map((l) => [l.id, l]));

const boxPoly = (b) => [[b[0], b[1]], [b[0] + b[2], b[1]], [b[0] + b[2], b[1] + b[3]], [b[0], b[1] + b[3]]];

const shapes = computed(() => {
  const out = [];
  for (const d of props.detections) {
    const ignored = d.match_method === "unconfirmed";
    const unsure = d.raw?.unsure || d.match_score < 85 || !d.card;
    const name = d.card || d.read_as;
    const qty = d.qty > 1 ? ` ×${d.qty}` : "";
    if (ignored) {
      if (d.bbox) out.push({ layer: "ignored", poly: boxPoly(d.bbox), label: `ignored: ${d.read_as}`, above: true });
      continue;
    }
    const title = d.raw?.title;
    if (title?.quad) {
      out.push({ layer: unsure ? "unsure" : "name", poly: title.quad, label: name + qty, inside: true, dash: !title.exact });
    } else if (d.bbox) {
      out.push({ layer: unsure ? "unsure" : "name", poly: boxPoly(d.bbox), label: name + qty, above: true, dash: true });
    }
    if (d.bbox) out.push({ layer: "card", poly: boxPoly(d.bbox), label: "" });
    if (d.raw?.zoom_region) out.push({ layer: "zoom", poly: boxPoly(d.raw.zoom_region), label: "" });
    if (d.raw?.model_box) out.push({ layer: "model", poly: boxPoly(d.raw.model_box), label: "" });
  }
  for (const box of props.outlines) out.push({ layer: "outline", poly: boxPoly(box), label: "" });
  for (const box of props.tiles) out.push({ layer: "tile", poly: boxPoly(box), label: "" });
  for (const d of props.duplicates) {
    if (d.box) out.push({ layer: "duplicate", poly: boxPoly(d.box), label: `dropped: ${d.name} ×${d.count}`, above: true });
  }
  return out;
});
const visible = computed(() => shapes.value.filter((s) => on[s.layer]));
const counts = computed(() => Object.fromEntries(LAYERS.map((l) => [l.id, shapes.value.filter((s) => s.layer === l.id).length])));
const stroke = computed(() => Math.max(2, Math.round(props.width / 500)));
const font = computed(() => Math.max(12, Math.round(props.width / 70)));

const points = (poly) => poly.map((p) => p.join(",")).join(" ");
// A label written inside a name box fits its height; others go just above the box.
function label(s) {
  const [tl, , , bl] = s.poly;
  const h = Math.hypot(bl[0] - tl[0], bl[1] - tl[1]);
  const size = s.inside ? Math.max(9, Math.min(font.value, h * 0.62)) : font.value;
  const angle = (Math.atan2(s.poly[1][1] - tl[1], s.poly[1][0] - tl[0]) * 180) / Math.PI;
  const x = tl[0] + (s.inside ? size * 0.3 : 0);
  const y = s.inside ? tl[1] + h / 2 + size * 0.35 : tl[1] - size * 0.35;
  return { x, y, size, angle };
}
const dashOf = (s) => (style[s.layer].dash || s.dash ? `${stroke.value * 4} ${stroke.value * 3}` : null);

const saving = ref(false);
async function save() {
  saving.value = true;
  try {
    const img = new Image();
    img.src = props.url;
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = props.width;
    canvas.height = props.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, props.width, props.height);
    ctx.lineJoin = "round";
    for (const s of visible.value) {
      const st = style[s.layer];
      ctx.strokeStyle = st.color;
      ctx.lineWidth = stroke.value;
      ctx.setLineDash(dashOf(s) ? [stroke.value * 4, stroke.value * 3] : []);
      ctx.beginPath();
      s.poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.stroke();
      if (!s.label) continue;
      const l = label(s);
      ctx.save();
      ctx.translate(l.x, l.y);
      ctx.rotate((l.angle * Math.PI) / 180);
      ctx.font = `600 ${l.size}px system-ui, sans-serif`;
      ctx.setLineDash([]);
      ctx.lineWidth = Math.max(3, l.size / 4);
      ctx.strokeStyle = "rgba(0,0,0,0.85)";
      ctx.strokeText(s.label, 0, 0);
      ctx.fillStyle = st.color;
      ctx.fillText(s.label, 0, 0);
      ctx.restore();
    }
    const blob = await new Promise((r) => canvas.toBlob(r, "image/png"));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${props.filename.replace(/\.[^.]+$/, "")}-debug.png`;
    a.click();
    URL.revokeObjectURL(a.href);
  } finally {
    saving.value = false;
  }
}
</script>

<template>
  <div class="overlay">
    <div class="legend">
      <label v-for="l in LAYERS" :key="l.id" class="check" :class="{ dim: !counts[l.id] }">
        <input type="checkbox" v-model="on[l.id]" :disabled="!counts[l.id]" />
        <i :style="{ borderColor: l.color, borderStyle: l.dash ? 'dashed' : 'solid' }"></i>
        {{ l.label }} <span class="sub num">{{ counts[l.id] }}</span>
      </label>
      <button class="btn small" :disabled="saving" @click="save">{{ saving ? "Saving…" : "Save image" }}</button>
    </div>
    <p class="sub small">Dashed name boxes are approximate (the card wasn't outlined, so the name is where the model's box puts it).</p>
    <div class="stage">
      <img :src="url" :alt="`Photo with detections`" />
      <svg :viewBox="`0 0 ${width} ${height}`" preserveAspectRatio="none" aria-hidden="true">
        <g v-for="(s, i) in visible" :key="i">
          <polygon :points="points(s.poly)" fill="none" :stroke="style[s.layer].color" :stroke-width="stroke"
                   :stroke-dasharray="dashOf(s)" />
          <text v-if="s.label" :transform="`translate(${label(s).x} ${label(s).y}) rotate(${label(s).angle})`"
                :font-size="label(s).size" :fill="style[s.layer].color" stroke="rgba(0,0,0,0.85)"
                :stroke-width="Math.max(3, label(s).size / 4)" stroke-linejoin="round" paint-order="stroke" font-weight="600">
            {{ s.label }}
          </text>
        </g>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.overlay { display: flex; flex-direction: column; gap: 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; font-size: 0.85rem; }
.legend .check { display: inline-flex; flex-direction: row; align-items: center; gap: 6px; white-space: nowrap; }
.legend .check input { margin: 0; width: 14px; height: 14px; }
.legend .dim { opacity: 0.45; }
.legend i { display: inline-block; width: 18px; height: 12px; border-width: 2px; border-radius: 2px; }
.stage { position: relative; line-height: 0; border-radius: 8px; overflow: hidden; border: 1px solid var(--line); }
.stage img { width: 100%; height: auto; display: block; }
.stage svg { position: absolute; inset: 0; width: 100%; height: 100%; font-family: system-ui, sans-serif; }
</style>

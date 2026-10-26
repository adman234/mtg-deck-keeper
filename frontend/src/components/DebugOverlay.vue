<script setup>
import { computed, reactive, ref } from "vue";

// Draws what the scan found on top of the photo: the model's card/pile boxes (after lining them
// up with the card outlines OpenCV found), the close-up areas that were re-read, the OpenCV
// cut-outs, and what was ignored or dropped.
const props = defineProps({
  url: String,
  width: Number,
  height: Number,
  detections: { type: Array, default: () => [] },
  duplicates: { type: Array, default: () => [] },
  outlines: { type: Array, default: () => [] },
  filename: { type: String, default: "scan" },
});

const LAYERS = [
  { id: "found", label: "Cards & piles (model)", color: "#ffd23f" },
  { id: "unsure", label: "Unsure name", color: "#ff8a3d" },
  { id: "zoom", label: "Close-up area re-read", color: "#38d6ff", dash: true },
  { id: "model", label: "Model's box before lining up", color: "#e879f9", dash: true },
  { id: "outline", label: "Card outlines (OpenCV)", color: "#a3e635", dash: true },
  { id: "cutout", label: "Cut-outs (pre-pass)", color: "#4ade80" },
  { id: "ignored", label: "Ignored cut-outs", color: "#ff5c5c", dash: true },
  { id: "duplicate", label: "Dropped duplicates", color: "#b8b8c8", dash: true },
];
const HIDDEN = ["zoom", "model", "outline"];
const on = reactive(Object.fromEntries(LAYERS.map((l) => [l.id, !HIDDEN.includes(l.id)])));
const style = Object.fromEntries(LAYERS.map((l) => [l.id, l]));

const shapes = computed(() => {
  const out = [];
  for (const d of props.detections) {
    if (d.source === "photo") {
      if (d.bbox) {
        const unsure = d.raw?.unsure || d.match_score < 85 || !d.card;
        const label = `${d.card || d.read_as} ×${d.qty}`;
        out.push({ layer: unsure ? "unsure" : "found", box: d.bbox, label, sub: d.card && d.card !== d.read_as ? `read “${d.read_as}”` : "" });
      }
      if (d.raw?.zoom_region) out.push({ layer: "zoom", box: d.raw.zoom_region, label: "" });
      if (d.raw?.model_box) out.push({ layer: "model", box: d.raw.model_box, label: "" });
    } else if (d.bbox) {
      const ignored = d.match_method === "unconfirmed";
      out.push({ layer: ignored ? "ignored" : "cutout", box: d.bbox, label: `${ignored ? "ignored: " : ""}${d.read_as}`, below: 1 });
    }
  }
  for (const box of props.outlines) out.push({ layer: "outline", box, label: "" });
  for (const d of props.duplicates) {
    if (d.box) out.push({ layer: "duplicate", box: d.box, label: `dropped: ${d.name} ×${d.count}`, below: 2 });
  }
  return out;
});
const visible = computed(() => shapes.value.filter((s) => on[s.layer]));
const counts = computed(() => Object.fromEntries(LAYERS.map((l) => [l.id, shapes.value.filter((s) => s.layer === l.id).length])));
const stroke = computed(() => Math.max(2, Math.round(props.width / 400)));
const font = computed(() => Math.max(12, Math.round(props.width / 70)));

function labelY(s) {
  // Labels go above the model's boxes and below cut-outs (a second line for dropped duplicates),
  // so labels of boxes on the same spot don't land on top of each other.
  return s.below ? s.box[1] + s.box[3] + font.value * (0.1 + 1.15 * s.below) : s.box[1] - font.value * 0.4;
}

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
    ctx.font = `600 ${font.value}px system-ui, sans-serif`;
    ctx.lineJoin = "round";
    for (const s of visible.value) {
      const st = style[s.layer];
      ctx.strokeStyle = st.color;
      ctx.lineWidth = stroke.value;
      ctx.setLineDash(st.dash ? [stroke.value * 4, stroke.value * 3] : []);
      ctx.strokeRect(...s.box);
      if (!s.label) continue;
      const y = labelY(s);
      ctx.setLineDash([]);
      ctx.lineWidth = Math.max(3, font.value / 4);
      ctx.strokeStyle = "rgba(0,0,0,0.85)";
      ctx.strokeText(s.label, s.box[0], y);
      ctx.fillStyle = st.color;
      ctx.fillText(s.label, s.box[0], y);
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
    <div class="stage">
      <img :src="url" :alt="`Photo with detections`" />
      <svg :viewBox="`0 0 ${width} ${height}`" preserveAspectRatio="none" aria-hidden="true">
        <g v-for="(s, i) in visible" :key="i">
          <rect :x="s.box[0]" :y="s.box[1]" :width="s.box[2]" :height="s.box[3]" fill="none"
                :stroke="style[s.layer].color" :stroke-width="stroke"
                :stroke-dasharray="style[s.layer].dash ? `${stroke * 4} ${stroke * 3}` : null" />
          <text v-if="s.label" :x="s.box[0]" :y="labelY(s)" :font-size="font" :fill="style[s.layer].color"
                stroke="rgba(0,0,0,0.85)" :stroke-width="Math.max(3, font / 4)" stroke-linejoin="round"
                paint-order="stroke" font-weight="600">
            {{ s.label }}<tspan v-if="s.sub" :font-size="font * 0.8" fill="#fff"> {{ s.sub }}</tspan>
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

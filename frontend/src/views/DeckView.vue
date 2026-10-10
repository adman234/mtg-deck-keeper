<script setup>
import { computed, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { api, BOARDS, FORMATS } from "../api";
import CardSearch from "../components/CardSearch.vue";

const props = defineProps({ id: String });
const router = useRouter();
const deck = ref(null);
const error = ref("");
const editing = ref(false);
const draft = ref([]);
const meta = ref({ name: "", format: "", notes: "" });
const exportFmt = ref("arena");
const exportText = ref("");
const copied = ref(false);
const confirmDelete = ref(false);
const preview = ref(null);

const TYPES = ["Creature", "Planeswalker", "Battle", "Instant", "Sorcery", "Artifact", "Enchantment", "Land"];

async function load() {
  try {
    deck.value = await api(`/api/decks/${props.id}`);
    meta.value = { name: deck.value.name, format: deck.value.format, notes: deck.value.notes };
  } catch (e) {
    error.value = e.message;
  }
}
watch(() => props.id, load, { immediate: true });

function typeOf(card) {
  const t = card.type_line || "";
  const front = t.split("//")[0];
  return TYPES.find((x) => front.includes(x)) || (t ? "Other" : "Unknown type");
}

const groups = computed(() => {
  if (!deck.value) return [];
  const out = [];
  for (const b of BOARDS) {
    const cards = deck.value.cards.filter((c) => c.board === b.value);
    if (!cards.length) continue;
    const byType = {};
    cards.forEach((c) => (byType[b.value === "main" ? typeOf(c.card) : b.label] ||= []).push(c));
    const order = [...TYPES, "Other", "Unknown type", b.label];
    const sections = Object.entries(byType).sort((x, y) => order.indexOf(x[0]) - order.indexOf(y[0]))
      .map(([title, list]) => ({ title, list, count: list.reduce((a, c) => a + c.qty, 0) }));
    out.push({ board: b, sections, count: cards.reduce((a, c) => a + c.qty, 0) });
  }
  return out;
});

const curve = computed(() => {
  const bins = Array(8).fill(0);
  (deck.value?.cards || []).filter((c) => c.board === "main" && c.card.cmc != null && !(c.card.type_line || "").includes("Land"))
    .forEach((c) => (bins[Math.min(7, Math.floor(c.card.cmc))] += c.qty));
  const max = Math.max(1, ...bins);
  return bins.map((n, i) => ({ label: i === 7 ? "7+" : String(i), n, h: (n / max) * 100 }));
});
const hasCurve = computed(() => curve.value.some((b) => b.n));

function startEdit() {
  draft.value = deck.value.cards.map((c) => ({ card: c.card, qty: c.qty, board: c.board }));
  editing.value = true;
}
function addCard(card) {
  const existing = draft.value.find((d) => d.card.id === card.id && d.board === "main");
  if (existing) existing.qty++;
  else draft.value.push({ card: { id: card.id, name: card.name }, qty: 1, board: "main" });
}
async function saveEdit() {
  error.value = "";
  try {
    await api(`/api/decks/${props.id}`, { method: "PATCH", body: meta.value });
    deck.value = await api(`/api/decks/${props.id}/cards`, { method: "PUT",
      body: draft.value.map((d) => ({ card_id: d.card.id, qty: d.qty, board: d.board })) });
    editing.value = false;
  } catch (e) {
    error.value = e.message;
  }
}

async function loadExport() {
  const res = await fetch(`/api/decks/${props.id}/export?format=${exportFmt.value}`, { credentials: "same-origin" });
  exportText.value = await res.text();
}
watch([exportFmt, () => deck.value?.updated_at], () => deck.value && loadExport());

async function copy() {
  try {
    await navigator.clipboard.writeText(exportText.value);
    copied.value = true;
    setTimeout(() => (copied.value = false), 1500);
  } catch {
    document.getElementById("export-text")?.select();
  }
}
function download() {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([exportText.value], { type: "text/plain" }));
  a.download = `${deck.value.name.replace(/[^\w\- ]+/g, "")}.txt`;
  a.click();
  URL.revokeObjectURL(a.href);
}
async function del() {
  await api(`/api/decks/${props.id}`, { method: "DELETE" });
  router.push("/");
}
</script>

<template>
  <div class="page" v-if="deck">
    <div class="page-head">
      <div>
        <p class="sub"><RouterLink to="/">Decks</RouterLink></p>
        <h1>{{ deck.name }}</h1>
        <p class="sub row" style="gap:8px">
          <span class="num">{{ deck.counts.main + deck.counts.commander }} cards</span>
          <span v-if="deck.counts.side" class="num">· {{ deck.counts.side }} sideboard</span>
          <span v-if="deck.format" class="pill neutral">{{ deck.format }}</span>
        </p>
      </div>
      <template v-if="!editing">
        <button class="btn" @click="startEdit">Edit</button>
        <button v-if="!confirmDelete" class="btn danger" @click="confirmDelete = true">Delete</button>
        <span v-else class="row"><span class="sub">Delete this deck?</span><button class="btn danger" @click="del">Delete</button><button class="btn" @click="confirmDelete = false">Keep</button></span>
      </template>
    </div>
    <div v-if="error" class="alert">{{ error }}</div>

    <form v-if="editing" class="panel" @submit.prevent="saveEdit">
      <div class="row" style="align-items:flex-end">
        <label class="field" style="flex:2;min-width:200px">Name<input id="edit-name" v-model="meta.name" type="text" required /></label>
        <label class="field" style="flex:1;min-width:140px">Format<select id="edit-format" v-model="meta.format"><option v-for="f in FORMATS" :key="f" :value="f">{{ f || "Not set" }}</option></select></label>
      </div>
      <label class="field">Notes<textarea id="edit-notes" v-model="meta.notes"></textarea></label>
      <div class="table-wrap">
        <table class="list">
          <thead><tr><th style="width:90px">Qty</th><th>Card</th><th style="width:150px">Board</th><th></th></tr></thead>
          <tbody>
            <tr v-for="(d, i) in draft" :key="d.card.id + d.board + i">
              <td><input v-model.number="d.qty" type="number" min="0" max="999" :aria-label="`Quantity of ${d.card.name}`" /></td>
              <td>{{ d.card.name }}</td>
              <td><select v-model="d.board" aria-label="Board"><option v-for="b in BOARDS" :key="b.value" :value="b.value">{{ b.label }}</option></select></td>
              <td><button type="button" class="btn small danger" @click="draft.splice(i, 1)">Remove</button></td>
            </tr>
          </tbody>
        </table>
      </div>
      <CardSearch placeholder="Add a card…" @pick="addCard" />
      <div class="row"><button class="btn primary">Save changes</button><button type="button" class="btn" @click="editing = false">Cancel</button></div>
    </form>

    <div v-else class="layout">
      <div class="lists">
        <p v-if="deck.notes" class="panel notes">{{ deck.notes }}</p>
        <section v-for="g in groups" :key="g.board.value" class="panel">
          <h2>{{ g.board.label }} <span class="sub num">{{ g.count }}</span></h2>
          <div class="cols">
            <div v-for="s in g.sections" :key="s.title" class="section">
              <h3 v-if="g.board.value === 'main' && !(g.sections.length === 1 && s.title === 'Unknown type')">{{ s.title }} <span class="sub num">{{ s.count }}</span></h3>
              <ul>
                <li v-for="c in s.list" :key="c.card.id" @mouseenter="preview = c.card.image_uri" @focusin="preview = c.card.image_uri">
                  <span class="num qty">{{ c.qty }}</span>
                  <a v-if="c.card.scryfall_uri" :href="c.card.scryfall_uri" target="_blank" rel="noopener">{{ c.card.name }}</a>
                  <span v-else>{{ c.card.name }}</span>
                  <span class="cost sub">{{ c.card.mana_cost }}</span>
                </li>
              </ul>
            </div>
          </div>
        </section>
      </div>
      <aside class="side">
        <div class="panel preview-panel">
          <img v-if="preview" :src="preview" alt="Card preview" class="preview" />
          <p v-else class="sub">Hover a card to preview it.</p>
        </div>
        <div v-if="hasCurve" class="panel">
          <h3>Mana curve</h3>
          <div class="curve" role="img" :aria-label="'Mana curve: ' + curve.map((b) => `${b.label}: ${b.n}`).join(', ')">
            <div v-for="b in curve" :key="b.label" class="bar">
              <span class="num n">{{ b.n || "" }}</span>
              <div :style="{ height: b.h + '%' }"></div>
              <span class="num">{{ b.label }}</span>
            </div>
          </div>
        </div>
        <div class="panel">
          <div class="row" style="justify-content:space-between">
            <h3>Export</h3>
            <select id="export-format" v-model="exportFmt" style="width:auto">
              <option value="arena">MTG Arena</option>
              <option value="text">Plain text (Moxfield, Archidekt)</option>
            </select>
          </div>
          <textarea id="export-text" readonly :value="exportText" rows="8" class="num" style="font-size:0.82rem"></textarea>
          <div class="row"><button class="btn small" @click="copy">{{ copied ? "Copied" : "Copy" }}</button><button class="btn small" @click="download">Download .txt</button></div>
        </div>
        <div v-for="s in deck.scans" :key="s.id" class="panel">
          <h3>From <RouterLink :to="`/scans/${s.id}`">{{ s.name || `scan #${s.id}` }}</RouterLink></h3>
          <div class="thumbs"><img v-for="p in s.photos" :key="p" :src="p" alt="Scan photo" loading="lazy" /></div>
        </div>
      </aside>
    </div>
  </div>
  <div v-else-if="error" class="alert">{{ error }}</div>
</template>

<style scoped>
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 20px; align-items: start; }
.lists { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.side { display: flex; flex-direction: column; gap: 16px; position: sticky; top: 72px; }
.cols { columns: 2 220px; column-gap: 28px; }
.section { break-inside: avoid; margin-bottom: 14px; }
.section h3 { margin-bottom: 4px; }
ul { list-style: none; margin: 0; padding: 0; }
li { display: grid; grid-template-columns: 2ch minmax(0, 1fr) auto; gap: 10px; padding: 3px 0; align-items: baseline; }
li a { color: var(--ink); text-decoration: none; }
li a:hover { color: var(--accent); text-decoration: underline; }
.qty { text-align: right; color: var(--muted); }
.cost { font-size: 0.78rem; font-family: var(--mono); }
.notes { white-space: pre-wrap; }
.preview-panel { align-items: center; min-height: 120px; justify-content: center; }
.preview { border-radius: 4.75% / 3.5%; width: 100%; }
.curve { display: grid; grid-template-columns: repeat(8, 1fr); gap: 4px; height: 110px; align-items: end; }
.bar { display: flex; flex-direction: column; align-items: center; gap: 3px; height: 100%; justify-content: flex-end; font-size: 0.75rem; color: var(--muted); }
.bar div { width: 100%; background: var(--accent); border-radius: 3px 3px 0 0; min-height: 1px; }
.thumbs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.thumbs img { aspect-ratio: 1; object-fit: cover; border-radius: 4px; }
@media (max-width: 860px) {
  .layout { grid-template-columns: 1fr; }
  .side { position: static; }
  .preview-panel { display: none; }
}
</style>

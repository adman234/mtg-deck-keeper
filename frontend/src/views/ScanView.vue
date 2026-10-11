<script setup>
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { api, BOARDS, FLAG_TEXT, FORMATS, QTY_FLAGS, fmtDate, methodText, pct } from "../api";
import CardSearch from "../components/CardSearch.vue";

const props = defineProps({ id: String });
const router = useRouter();
const scan = ref(null);
const error = ref("");
const onlyFlagged = ref(false);
const changing = ref(null); // line id whose card is being changed
const zoom = ref(null);
const deckName = ref("");
const deckFormat = ref("");
const saving = ref(false);
const confirmDelete = ref(false);
const addQty = ref(1);
const addBoard = ref("main");
let timer;

async function load() {
  try {
    scan.value = await api(`/api/scans/${props.id}`);
    if (!deckName.value) deckName.value = scan.value.name || `Scan #${scan.value.id}`;
  } catch (e) {
    error.value = e.message;
  }
  clearTimeout(timer);
  if (scan.value && ["queued", "processing"].includes(scan.value.status)) timer = setTimeout(load, 2000);
}
watch(() => props.id, load, { immediate: true });
onBeforeUnmount(() => clearTimeout(timer));

async function act(fn) {
  error.value = "";
  try {
    const res = await fn();
    if (res && res.lines) scan.value = res;
  } catch (e) {
    error.value = e.message;
  }
}
const patch = (line, body) => act(() => api(`/api/scans/${props.id}/lines/${line.id}`, { method: "PATCH", body }));
const removeLine = (line) => act(() => api(`/api/scans/${props.id}/lines/${line.id}`, { method: "DELETE" }));
const acceptAll = () => act(() => api(`/api/scans/${props.id}/accept-all`, { method: "POST" }));
function pickCard(line, card) {
  changing.value = null;
  patch(line, { card_id: card.id });
}
function setQty(line, qty) {
  qty = Math.max(1, Math.min(999, parseInt(qty, 10) || 1));
  if (qty !== line.qty) patch(line, { qty });
}
function addCard(card) {
  act(() => api(`/api/scans/${props.id}/lines`, { method: "POST", body: { card_id: card.id, qty: addQty.value, board: addBoard.value } }));
  addQty.value = 1;
}
async function rerun(prepass) {
  await act(() => api(`/api/scans/${props.id}/rerun`, { method: "POST", body: { prepass } }));
  load();
}
async function del() {
  await act(() => api(`/api/scans/${props.id}`, { method: "DELETE" }));
  router.push("/scans");
}
async function save() {
  saving.value = true;
  error.value = "";
  try {
    const { deck_id } = await api(`/api/scans/${props.id}/save`, { method: "POST", body: { name: deckName.value, format: deckFormat.value } });
    router.push(`/decks/${deck_id}`);
  } catch (e) {
    error.value = e.message;
  } finally {
    saving.value = false;
  }
}

const lines = computed(() => scan.value?.lines || []);
const shown = computed(() => (onlyFlagged.value ? lines.value.filter((l) => l.needs_review) : lines.value));
const flagged = computed(() => lines.value.filter((l) => l.needs_review).length);
const unmatched = computed(() => lines.value.filter((l) => !l.card).length);
const totals = computed(() => {
  const t = { main: 0, side: 0, commander: 0, maybe: 0 };
  lines.value.forEach((l) => (t[l.board] += l.qty));
  return t;
});
const progress = computed(() => (scan.value?.total_images ? scan.value.processed_images / scan.value.total_images : 0));
const how = (l) => [...new Set(l.sources.map((s) => methodText(s.match_method)).filter(Boolean))].join(" · ");
const checkQty = (l) => !l.accepted && l.flags.some((f) => QTY_FLAGS.includes(f));
const pillFlags = (l) => l.flags.filter((f) => !QTY_FLAGS.includes(f));
const confClass = (l) => (!l.card ? "bad" : l.confidence >= scan.value.threshold ? "ok" : "warn");
const failedPhotos = computed(() => (scan.value?.images || [])
  .map((img, i) => ({ n: i + 1, error: img.photo_error }))
  .filter((p) => p.error));
const crops = (l) => l.sources.filter((s) => s.crop_url);
const sourceText = (l) => {
  const photos = new Set(l.sources.map((s) => s.image_id));
  const idx = scan.value.images.map((i) => i.id);
  return [...photos].map((p) => idx.indexOf(p) + 1).sort((a, b) => a - b).map((n) => `photo ${n}`).join(", ");
};
</script>

<template>
  <div class="page" v-if="scan">
    <div class="page-head">
      <div>
        <p class="sub"><RouterLink to="/scans">Scans</RouterLink> / #{{ scan.id }}</p>
        <h1>{{ scan.name || `Scan #${scan.id}` }}</h1>
        <p class="sub">{{ scan.total_images }} photo{{ scan.total_images === 1 ? "" : "s" }} · {{ scan.model || "default model" }} · pre-pass {{ scan.prepass ? "on" : "off" }} · {{ fmtDate(scan.created_at) }}</p>
      </div>
      <template v-if="!['queued', 'processing'].includes(scan.status)">
        <details class="rerun">
          <summary class="btn small">Read again</summary>
          <div class="menu">
            <button class="btn small" @click="rerun(true)">With card pre-pass</button>
            <button class="btn small" @click="rerun(false)">Whole photos only</button>
            <p class="sub" style="font-size:0.8rem">Replaces the current review list.</p>
          </div>
        </details>
        <button v-if="!confirmDelete" class="btn small danger" @click="confirmDelete = true">Delete</button>
        <span v-else class="row"><span class="sub">Delete scan and photos?</span>
          <button class="btn small danger" @click="del">Delete</button><button class="btn small" @click="confirmDelete = false">Keep</button></span>
      </template>
    </div>

    <div v-if="error" class="alert">{{ error }}</div>

    <div v-if="['queued', 'processing'].includes(scan.status)" class="panel">
      <h2>{{ scan.status === "queued" ? "Waiting for the vision model…" : scan.step || "Reading photos…" }}</h2>
      <div class="progress"><div :style="{ width: Math.max(4, progress * 100) + '%' }"></div></div>
      <p class="sub">The first photo can take a while if llama-swap has to load the model. You can leave this page; the scan keeps running.</p>
    </div>

    <div v-if="scan.status === 'failed'" class="alert">
      <strong>The scan failed.</strong> {{ scan.error }}
    </div>

    <template v-if="['review', 'saved'].includes(scan.status)">
      <div v-if="failedPhotos.length" class="alert warn">
        The whole-photo read failed for {{ failedPhotos.map((p) => `photo ${p.n}`).join(", ") }}, so cards the
        cut-outs missed aren't counted there. Press <b>Read again</b> to retry.
        <span class="small">({{ failedPhotos[0].error }})</span>
      </div>
      <div v-if="scan.status === 'saved' && scan.deck_id" class="alert ok">Saved as a deck. <RouterLink :to="`/decks/${scan.deck_id}`">Open deck</RouterLink></div>

      <div class="summary">
        <div class="tally">
          <span><b class="num">{{ totals.main }}</b> main</span>
          <span v-if="totals.side"><b class="num">{{ totals.side }}</b> side</span>
          <span v-if="totals.commander"><b class="num">{{ totals.commander }}</b> commander</span>
          <span v-if="totals.maybe"><b class="num">{{ totals.maybe }}</b> maybe</span>
          <span :class="flagged ? 'pill warn' : 'pill ok'">{{ flagged ? `${flagged} to review` : "All checked" }}</span>
        </div>
        <div class="row">
          <label class="check" style="align-items:center"><input id="only-flagged" v-model="onlyFlagged" type="checkbox" />Only flagged</label>
          <button class="btn small" :disabled="!flagged || flagged === unmatched" @click="acceptAll">Accept all matched</button>
        </div>
      </div>

      <div class="panel" style="padding:0">
        <div v-if="!shown.length" class="empty">{{ onlyFlagged ? "Nothing left to review." : "The model didn't find any cards. Try reading again or add cards below." }}</div>
        <div v-for="l in shown" :key="l.id" class="line" :class="{ review: l.needs_review }">
          <button v-if="crops(l).length" class="thumb" @click="zoom = crops(l)[0].crop_url" :aria-label="`Show crop for ${l.raw_name}`">
            <img :src="crops(l)[0].crop_url" alt="" loading="lazy" />
            <span v-if="crops(l).length > 1" class="more num">×{{ crops(l).length }}</span>
          </button>
          <button v-else-if="l.card?.image_uri" class="thumb ref" @click="zoom = l.card.image_uri"
                  :aria-label="`Show ${l.card.name}`" title="Card image from Scryfall (no cut-out from your photo)">
            <img :src="l.card.image_uri" alt="" loading="lazy" />
          </button>
          <div v-else class="thumb none" aria-hidden="true"></div>

          <div class="qty" :class="{ check: checkQty(l) }">
            <div class="qty-controls">
              <button class="btn small" @click="setQty(l, l.qty - 1)" :disabled="l.qty <= 1" aria-label="One fewer">−</button>
              <input type="number" min="1" max="999" :value="l.qty" @change="setQty(l, $event.target.value)" :aria-label="`Quantity of ${l.card?.name || l.raw_name}`" />
              <button class="btn small" @click="setQty(l, l.qty + 1)" aria-label="One more">+</button>
            </div>
            <span v-if="checkQty(l)" class="qty-note">Check the quantity</span>
          </div>

          <div class="name">
            <template v-if="changing !== l.id">
              <div class="row" style="gap:8px">
                <strong v-if="l.card">{{ l.card.name }}</strong>
                <strong v-else class="unknown">“{{ l.raw_name }}”</strong>
                <button class="link" @click="changing = l.id">Change</button>
              </div>
              <p class="sub small">
                <span v-if="l.card && l.raw_name.toLowerCase() !== l.card.name.toLowerCase()">Read as “{{ l.raw_name }}” · </span>
                <span v-if="l.sources.length">{{ sourceText(l) }}</span>
                <span v-else>Added by you</span>
              </p>
              <p v-if="l.card && how(l)" class="sub small how">{{ how(l) }}</p>
              <div v-if="pillFlags(l).length && !l.accepted" class="row" style="gap:6px">
                <span v-for="f in pillFlags(l)" :key="f" class="pill" :class="f === 'unmatched' ? 'bad' : 'warn'">{{ FLAG_TEXT[f] || f }}</span>
              </div>
              <div v-if="l.needs_review && l.suggestions.length" class="row suggest">
                <span class="sub small">Did you mean</span>
                <button v-for="s in l.suggestions" :key="s.card_id" class="btn small" @click="pickCard(l, { id: s.card_id })">{{ s.name }}</button>
              </div>
            </template>
            <div v-else class="row" style="flex-wrap:nowrap">
              <CardSearch :initial="l.raw_name" autofocus style="flex:1" @pick="pickCard(l, $event)" @cancel="changing = null" />
              <button class="btn small" @click="changing = null">Cancel</button>
            </div>
          </div>

          <span v-if="l.user_edited" class="pill conf ok" title="You picked or adjusted this line">Edited</span>
          <span v-else class="pill conf" :class="confClass(l)" title="Model confidence × name match">{{ l.card ? pct(l.confidence) : "?" }}</span>

          <select :value="l.board" @change="patch(l, { board: $event.target.value })" aria-label="Board">
            <option v-for="b in BOARDS" :key="b.value" :value="b.value">{{ b.label }}</option>
          </select>

          <div class="actions">
            <button v-if="l.needs_review && l.card" class="btn small primary" @click="patch(l, { accepted: true })">Looks right</button>
            <button class="btn small danger" @click="removeLine(l)" :aria-label="`Remove ${l.card?.name || l.raw_name}`">Remove</button>
          </div>
        </div>

        <div class="line add">
          <div class="thumb none" aria-hidden="true"></div>
          <div class="qty"><input type="number" min="1" max="999" v-model.number="addQty" aria-label="Quantity to add" /></div>
          <div class="name"><CardSearch placeholder="Add a card the model missed…" @pick="addCard" /></div>
          <span></span>
          <select v-model="addBoard" aria-label="Board for new card">
            <option v-for="b in BOARDS" :key="b.value" :value="b.value">{{ b.label }}</option>
          </select>
          <span></span>
        </div>
      </div>

      <form class="panel" @submit.prevent="save">
        <h2>{{ scan.status === "saved" ? "Save another copy" : "Save deck" }}</h2>
        <div class="row" style="align-items:flex-end">
          <label class="field" style="flex:2;min-width:200px">Deck name<input id="deck-name" v-model="deckName" type="text" required maxlength="200" /></label>
          <label class="field" style="flex:1;min-width:140px">Format
            <select id="deck-format" v-model="deckFormat"><option v-for="f in FORMATS" :key="f" :value="f">{{ f || "Not set" }}</option></select>
          </label>
          <button class="btn primary" :disabled="saving || unmatched > 0 || !lines.length">Save deck ({{ totals.main + totals.commander }} cards)</button>
        </div>
        <p v-if="unmatched" class="sub">{{ unmatched }} line{{ unmatched === 1 ? " has" : "s have" }} no card yet. Pick a card or remove {{ unmatched === 1 ? "it" : "them" }} to save.</p>
        <p v-else-if="flagged" class="sub">{{ flagged }} flagged line{{ flagged === 1 ? " is" : "s are" }} still unchecked. You can save anyway and edit the deck later.</p>
      </form>

      <section class="page" style="gap:10px">
        <h2>Photos</h2>
        <div class="photos">
          <figure v-for="(img, i) in scan.images" :key="img.id">
            <button class="photo" @click="zoom = img.url"><img :src="img.url" :alt="`Photo ${i + 1}`" loading="lazy" /></button>
            <figcaption class="sub small">
              Photo {{ i + 1 }}
              <span v-if="scan.prepass"> · {{ img.crops_found }} cut out</span>
              <span v-if="img.photo_cards != null"> · {{ img.photo_cards }} listed from whole photo</span>
              <span v-if="img.photo_error" class="warn-text"> · whole-photo read failed</span>
              <span v-if="img.unconfirmed_crops"> · {{ img.unconfirmed_crops }} unreadable cut-out{{ img.unconfirmed_crops === 1 ? "" : "s" }} ignored</span>
            </figcaption>
          </figure>
        </div>
      </section>
    </template>

    <div v-if="zoom" class="zoom" @click="zoom = null" @keydown.esc="zoom = null" tabindex="-1">
      <img :src="zoom" alt="Enlarged image" />
    </div>
  </div>
  <div v-else-if="error" class="alert">{{ error }}</div>
</template>

<style scoped>
.small { font-size: 0.82rem; }
.summary { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; align-items: center; }
.tally { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
.line { display: grid; grid-template-columns: 52px 128px minmax(0, 1fr) 56px 120px auto; gap: 12px; align-items: center;
  padding: 10px 14px; border-bottom: 1px solid var(--line); }
.line.review { background: color-mix(in srgb, var(--warn-soft) 45%, transparent); box-shadow: inset 3px 0 0 var(--warn); }
.line.add { border-bottom: none; }
.thumb { width: 52px; height: 72px; padding: 0; border: 1px solid var(--line); border-radius: 4px; overflow: hidden; background: var(--surface-2); cursor: zoom-in; position: relative; }
.thumb.none { cursor: default; border-style: dashed; opacity: 0.5; }
.thumb.ref { opacity: 0.75; border-style: dashed; }
.warn-text { color: var(--warn); font-weight: 600; }
.thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.more { position: absolute; right: 2px; bottom: 2px; background: var(--ink); color: var(--bg); font-size: 0.68rem; padding: 0 4px; border-radius: 3px; }
.qty { display: flex; flex-direction: column; gap: 3px; align-items: flex-start; }
.qty-controls { display: flex; gap: 4px; align-items: center; }
.qty input { width: 52px; text-align: center; padding: 5px 4px; }
.qty.check input { border: 2px solid var(--warn); background: var(--warn-soft); font-weight: 700; }
.qty-note { font-size: 0.75rem; font-weight: 600; color: var(--warn); }
.name { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.unknown { color: var(--bad); }
.how { color: var(--accent); }
.suggest { gap: 6px; }
.conf { justify-self: start; }
.actions { display: flex; gap: 6px; justify-content: flex-end; flex-wrap: wrap; }
.rerun { position: relative; }
.rerun summary { list-style: none; }
.rerun summary::-webkit-details-marker { display: none; }
.rerun .menu { position: absolute; right: 0; top: 110%; z-index: 5; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 10px; display: flex; flex-direction: column; gap: 6px; width: 220px; box-shadow: 0 8px 24px rgb(0 0 0 / 0.15); }
.photos { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.photos figure { margin: 0; display: flex; flex-direction: column; gap: 4px; }
.photo { padding: 0; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; cursor: zoom-in; background: none; }
.photo img { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; }
.zoom { position: fixed; inset: 0; z-index: 50; background: rgb(0 0 0 / 0.8); display: flex; align-items: center; justify-content: center; padding: 20px; cursor: zoom-out; }
.zoom img { max-height: 100%; max-width: 100%; border-radius: 8px; }
@media (max-width: 760px) {
  .line { grid-template-columns: 52px minmax(0, 1fr) auto; grid-template-areas: "t n c" "t q b" "a a a"; }
  .thumb { grid-area: t; } .name { grid-area: n; } .conf { grid-area: c; } .qty { grid-area: q; } select { grid-area: b; } .actions { grid-area: a; justify-content: flex-start; }
  .line.add > span { display: none; }
}
</style>

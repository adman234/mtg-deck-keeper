<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { api, session } from "../api";

const router = useRouter();
const files = ref([]);
const name = ref("");
const prepass = ref(true);
const layout = ref("piles");
const model = ref("");
const error = ref("");
const busy = ref(false);
const dragging = ref(false);
const settings = ref(null);

onMounted(async () => {
  try {
    if (session.user?.is_admin) {
      const s = await api("/api/admin/settings");
      settings.value = Object.fromEntries(s.settings.map((x) => [x.key, x]));
      prepass.value = settings.value["vision.prepass_default"].value;
    }
  } catch { /* non-admins just get defaults */ }
});
onBeforeUnmount(() => files.value.forEach((f) => URL.revokeObjectURL(f.url)));

function add(list) {
  for (const file of list) {
    if (!file.type.startsWith("image/") && !/\.(heic|heif)$/i.test(file.name)) continue;
    files.value.push({ file, url: URL.createObjectURL(file), id: crypto.randomUUID?.() || Math.random() });
  }
}
function remove(i) {
  URL.revokeObjectURL(files.value[i].url);
  files.value.splice(i, 1);
}
function drop(e) {
  dragging.value = false;
  add(e.dataTransfer.files);
}
const totalMb = computed(() => (files.value.reduce((a, f) => a + f.file.size, 0) / 1048576).toFixed(1));
const visionMissing = computed(() => settings.value && !settings.value["vision.model"].value);

async function submit() {
  busy.value = true;
  error.value = "";
  const form = new FormData();
  files.value.forEach((f) => form.append("files", f.file, f.file.name));
  form.append("name", name.value);
  form.append("prepass", prepass.value ? "true" : "false");
  form.append("layout", layout.value);
  form.append("model", model.value);
  try {
    const job = await api("/api/scans", { method: "POST", form });
    router.push(`/scans/${job.id}`);
  } catch (e) {
    error.value = e.message;
    busy.value = false;
  }
}
</script>

<template>
  <div class="page">
    <div class="page-head"><div><h1>New scan</h1><p class="sub">Upload photos of one deck. Each card should appear in only one photo.</p></div></div>
    <div v-if="visionMissing" class="alert warn">No vision model is set yet. <RouterLink to="/settings/vision">Choose one in Settings</RouterLink> first.</div>
    <div v-if="error" class="alert">{{ error }}</div>

    <label class="drop" :class="{ dragging }" @dragover.prevent="dragging = true" @dragleave="dragging = false" @drop.prevent="drop">
      <input id="scan-files" type="file" accept="image/*" multiple @change="add($event.target.files); $event.target.value = ''" />
      <strong>Drop photos here or tap to choose</strong>
      <span class="sub">On a phone this opens the camera. Up to 30 photos, 30 MB each.</span>
    </label>

    <div v-if="files.length" class="thumbs">
      <figure v-for="(f, i) in files" :key="f.id">
        <img :src="f.url" :alt="f.file.name" />
        <button class="btn small" type="button" @click="remove(i)" :aria-label="`Remove ${f.file.name}`">Remove</button>
      </figure>
    </div>

    <form class="panel" @submit.prevent="submit">
      <label class="field">Scan name <small>optional, becomes the default deck name</small>
        <input id="scan-name" v-model="name" type="text" placeholder="e.g. Mono-Red Aggro" maxlength="200" />
      </label>
      <fieldset class="layout">
        <legend>How are the cards laid out?</legend>
        <label class="check">
          <input id="layout-piles" v-model="layout" type="radio" value="piles" />
          <span>In piles of the same card
            <small>Copies stacked or fanned so only the front card's name shows. The model names the front card and counts the cards behind it.</small>
          </span>
        </label>
        <label class="check">
          <input id="layout-spread" v-model="layout" type="radio" value="spread" />
          <span>Spread out
            <small>Every card's name is visible.</small>
          </span>
        </label>
      </fieldset>
      <label class="check">
        <input id="scan-prepass" v-model="prepass" type="checkbox" />
        <span>Find individual cards first (pre-pass)
          <small>Cuts each card out of the photo before reading it. More accurate when cards are laid out flat and apart. The whole photo is always read as well, so fanned cards are still caught.</small>
        </span>
      </label>
      <div class="row">
        <button class="btn primary" :disabled="!files.length || busy">{{ busy ? "Uploading…" : `Scan ${files.length || ""} photo${files.length === 1 ? "" : "s"}` }}</button>
        <span v-if="files.length" class="sub num">{{ totalMb }} MB</span>
      </div>
    </form>

    <details class="panel tips">
      <summary><strong>Tips for better results</strong></summary>
      <ul>
        <li>Piles are fine: keep the front card's name fully visible and fan the cards behind slightly so each edge shows.</li>
        <li>Lay cards on a plain, contrasting surface with a little space between piles.</li>
        <li>Shoot from straight above in even light; avoid glare on foils and sleeves.</li>
        <li>Make sure every title line is readable. 9 to 20 cards per photo works well.</li>
        <li>Count big stacks of basic lands yourself and fix the number on the review screen.</li>
      </ul>
    </details>
  </div>
</template>

<style scoped>
.drop { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 40px 20px; border: 2px dashed var(--line);
  border-radius: var(--radius); background: var(--surface); cursor: pointer; text-align: center; }
.drop.dragging, .drop:hover { border-color: var(--accent); background: var(--accent-soft); }
.drop input { position: absolute; width: 1px; height: 1px; opacity: 0; }
.drop:focus-within { outline: 2px solid var(--accent); }
.thumbs { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; }
.thumbs figure { margin: 0; display: flex; flex-direction: column; gap: 6px; }
.thumbs img { aspect-ratio: 4 / 3; object-fit: cover; border-radius: 8px; border: 1px solid var(--line); width: 100%; }
.tips ul { margin: 8px 0 0; padding-left: 1.2em; color: var(--muted); display: flex; flex-direction: column; gap: 4px; }
.tips summary { cursor: pointer; }
.layout { border: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 8px; }
.layout legend { font-weight: 500; font-size: 0.9rem; margin-bottom: 6px; }
</style>

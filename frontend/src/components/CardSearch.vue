<script setup>
import { onBeforeUnmount, ref, watch } from "vue";
import { api } from "../api";

const props = defineProps({ placeholder: { type: String, default: "Search card names…" }, autofocus: Boolean, initial: String });
const emit = defineEmits(["pick", "cancel"]);
const q = ref(props.initial || "");
const results = ref([]);
const active = ref(0);
let timer;

watch(q, (v) => {
  clearTimeout(timer);
  if (!v.trim()) return (results.value = []);
  timer = setTimeout(async () => {
    try {
      results.value = await api("/api/cards/search?q=" + encodeURIComponent(v.trim()));
      active.value = 0;
    } catch {
      results.value = [];
    }
  }, 150);
}, { immediate: true });
onBeforeUnmount(() => clearTimeout(timer));

function key(e) {
  if (e.key === "ArrowDown") { active.value = Math.min(active.value + 1, results.value.length - 1); e.preventDefault(); }
  else if (e.key === "ArrowUp") { active.value = Math.max(active.value - 1, 0); e.preventDefault(); }
  else if (e.key === "Enter" && results.value[active.value]) { pick(results.value[active.value]); e.preventDefault(); }
  else if (e.key === "Escape") emit("cancel");
}
function pick(r) {
  emit("pick", r);
  q.value = "";
  results.value = [];
}
</script>

<template>
  <div class="cardsearch">
    <input v-model="q" type="search" :placeholder="placeholder" :autofocus="autofocus" @keydown="key" aria-label="Search cards" />
    <ul v-if="results.length" role="listbox">
      <li v-for="(r, i) in results" :key="r.id" :class="{ active: i === active }" role="option"
          @mousedown.prevent="pick(r)" @mouseenter="active = i">{{ r.name }}</li>
    </ul>
  </div>
</template>

<style scoped>
.cardsearch { position: relative; min-width: 0; }
ul { position: absolute; z-index: 20; left: 0; right: 0; top: calc(100% + 4px); margin: 0; padding: 4px; list-style: none;
  background: var(--surface); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 8px 24px rgb(0 0 0 / 0.18); max-height: 280px; overflow-y: auto; }
li { padding: 7px 10px; border-radius: 6px; cursor: pointer; }
li.active { background: var(--accent-soft); }
</style>

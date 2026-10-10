<script setup>
import { computed, onMounted, ref } from "vue";
import { api, fmtDate } from "../api";

const decks = ref(null);
const filter = ref("");
const error = ref("");

onMounted(async () => {
  try { decks.value = await api("/api/decks"); } catch (e) { error.value = e.message; }
});
const shown = computed(() => (decks.value || []).filter((d) => d.name.toLowerCase().includes(filter.value.toLowerCase())));
const total = (c) => c.main + c.commander;
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div><h1>Decks</h1><p class="sub" v-if="decks">{{ decks.length }} saved</p></div>
      <input v-if="decks && decks.length > 6" id="deck-filter" v-model="filter" type="search" placeholder="Filter decks" style="max-width:240px" />
      <RouterLink to="/scan" class="btn primary">Scan a deck</RouterLink>
    </div>
    <div v-if="error" class="alert">{{ error }}</div>
    <div v-if="decks && !decks.length" class="panel empty">
      <h2>No decks yet</h2>
      <p>Lay your cards out on a table, take a few photos, and the vision model will build the list for you to check.</p>
      <RouterLink to="/scan" class="btn primary">Scan your first deck</RouterLink>
    </div>
    <div class="grid-cards">
      <RouterLink v-for="d in shown" :key="d.id" :to="`/decks/${d.id}`" class="deck">
        <div class="art" :style="d.cover_image ? { backgroundImage: `url(${d.cover_image})` } : {}"></div>
        <div class="body">
          <h3>{{ d.name }}</h3>
          <div class="row sub">
            <span class="num">{{ total(d.counts) }} cards</span>
            <span v-if="d.counts.side" class="num">+{{ d.counts.side }} side</span>
            <span v-if="d.format" class="pill neutral">{{ d.format }}</span>
          </div>
          <p class="sub" style="font-size:0.8rem">Updated {{ fmtDate(d.updated_at) }}</p>
        </div>
      </RouterLink>
    </div>
  </div>
</template>

<style scoped>
.deck { display: flex; flex-direction: column; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; color: var(--ink); text-decoration: none; }
.deck:hover { border-color: var(--accent); }
.art { aspect-ratio: 16 / 7; background: linear-gradient(135deg, var(--accent-soft), var(--surface-2)); background-size: cover; background-position: center 22%; }
.body { padding: 12px 14px; display: flex; flex-direction: column; gap: 6px; }
</style>

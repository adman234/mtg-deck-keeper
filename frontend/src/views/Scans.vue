<script setup>
import { onMounted, ref } from "vue";
import { api, fmtDate } from "../api";

const scans = ref(null);
const error = ref("");
const STATUS = { queued: ["neutral", "Queued"], processing: ["warn", "Reading"], review: ["warn", "Needs review"],
  saved: ["ok", "Saved"], failed: ["bad", "Failed"] };

onMounted(async () => {
  try { scans.value = await api("/api/scans"); } catch (e) { error.value = e.message; }
});
</script>

<template>
  <div class="page">
    <div class="page-head"><div><h1>Scans</h1><p class="sub">Every photo upload, newest first.</p></div>
      <RouterLink to="/scan" class="btn primary">New scan</RouterLink></div>
    <div v-if="error" class="alert">{{ error }}</div>
    <div v-if="scans && !scans.length" class="panel empty"><p>No scans yet.</p></div>
    <div v-if="scans && scans.length" class="panel table-wrap" style="padding:0">
      <table class="list">
        <thead><tr><th>Scan</th><th>Status</th><th>Photos</th><th>Model</th><th>Started</th></tr></thead>
        <tbody>
          <tr v-for="s in scans" :key="s.id">
            <td><RouterLink :to="`/scans/${s.id}`">{{ s.name || `Scan #${s.id}` }}</RouterLink></td>
            <td><span class="pill" :class="STATUS[s.status][0]">{{ STATUS[s.status][1] }}</span></td>
            <td class="num">{{ s.total_images }}</td>
            <td class="sub">{{ s.model || "—" }}</td>
            <td class="sub">{{ fmtDate(s.created_at) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

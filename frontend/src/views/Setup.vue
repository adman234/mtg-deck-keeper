<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
import { api, loadStatus } from "../api";

const router = useRouter();
const form = ref({ username: "", email: "", password: "", full_name: "" });
const error = ref("");
const busy = ref(false);

async function submit() {
  busy.value = true;
  error.value = "";
  try {
    await api("/api/auth/setup", { method: "POST", body: form.value });
    await loadStatus();
    router.replace("/settings/vision");
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="page">
    <div class="row"><img src="/favicon.svg" alt="" width="40" height="40" /><h1>Welcome</h1></div>
    <p class="sub">Create the first account. It will be an admin, so you can set up the vision model and single sign-on next.</p>
    <div v-if="error" class="alert">{{ error }}</div>
    <form class="panel" @submit.prevent="submit">
      <label class="field">Username<input id="setup-user" v-model="form.username" type="text" required minlength="2" autocomplete="username" /></label>
      <label class="field">Email<input id="setup-email" v-model="form.email" type="email" required autocomplete="email" /></label>
      <label class="field">Name <small>optional</small><input id="setup-name" v-model="form.full_name" type="text" /></label>
      <label class="field">Password <small>at least 8 characters</small><input id="setup-pass" v-model="form.password" type="password" required minlength="8" autocomplete="new-password" /></label>
      <button class="btn primary" :disabled="busy">Create admin account</button>
    </form>
  </div>
</template>

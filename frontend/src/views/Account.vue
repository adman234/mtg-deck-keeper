<script setup>
import { ref } from "vue";
import { api, session } from "../api";

const current = ref("");
const next = ref("");
const msg = ref("");
const error = ref("");

async function submit() {
  msg.value = error.value = "";
  try {
    await api("/api/auth/me/password", { method: "PUT", body: { current_password: current.value, new_password: next.value } });
    msg.value = "Password saved.";
    session.user.has_password = true;
    current.value = next.value = "";
  } catch (e) {
    error.value = e.message;
  }
}
</script>

<template>
  <div class="page" style="max-width:520px">
    <div class="page-head"><div><h1>Account</h1>
      <p class="sub">{{ session.user.username }} · {{ session.user.email }} · {{ session.user.auth_method === "oidc" ? "single sign-on" : "local account" }}{{ session.user.is_admin ? " · admin" : "" }}</p></div></div>
    <div v-if="msg" class="alert ok">{{ msg }}</div>
    <div v-if="error" class="alert">{{ error }}</div>
    <form class="panel" @submit.prevent="submit">
      <h2>{{ session.user.has_password ? "Change password" : "Set a local password" }}</h2>
      <p v-if="!session.user.has_password" class="sub">Optional. Lets you log in with your username if single sign-on is unavailable.</p>
      <label v-else class="field">Current password<input id="pw-current" v-model="current" type="password" autocomplete="current-password" required /></label>
      <label class="field">New password <small>at least 8 characters</small><input id="pw-new" v-model="next" type="password" autocomplete="new-password" minlength="8" required /></label>
      <button class="btn primary">Save password</button>
    </form>
  </div>
</template>

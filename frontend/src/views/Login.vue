<script setup>
import { onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api, loadStatus, session } from "../api";

const route = useRoute();
const router = useRouter();
const username = ref("");
const password = ref("");
const remember = ref(false);
const error = ref(route.query.error || "");
const busy = ref(false);

onMounted(async () => {
  const s = await loadStatus();
  if (s.user) return router.replace(route.query.next || "/");
  if (s.oidc_auto_redirect && !route.query.direct && !route.query.error) {
    window.location.assign("/api/auth/oidc/login");
  }
});

async function submit() {
  busy.value = true;
  error.value = "";
  try {
    await api("/api/auth/login", { method: "POST", body: { username: username.value, password: password.value, remember: remember.value } });
    await loadStatus();
    router.replace(route.query.next || "/");
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="page">
    <div class="row"><img src="/favicon.svg" alt="" width="40" height="40" /><h1>Deck Keeper</h1></div>
    <div v-if="error" class="alert">{{ error }}</div>
    <div class="panel" v-if="session.status">
      <a v-if="session.status.oidc_enabled" class="btn primary" href="/api/auth/oidc/login">
        Sign in with {{ session.status.oidc_provider_name }}
      </a>
      <form v-if="session.status.password_login" class="page" style="gap:12px" @submit.prevent="submit">
        <label class="field">Username or email<input id="login-user" v-model="username" type="text" autocomplete="username" required /></label>
        <label class="field">Password<input id="login-pass" v-model="password" type="password" autocomplete="current-password" required /></label>
        <label class="check"><input id="login-remember" v-model="remember" type="checkbox" />Keep me signed in for 30 days</label>
        <button class="btn" :class="{ primary: !session.status.oidc_enabled }" :disabled="busy">Log in</button>
      </form>
    </div>
  </div>
</template>

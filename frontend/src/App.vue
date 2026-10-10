<script setup>
import { computed } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api, session } from "./api";

const route = useRoute();
const router = useRouter();
const bare = computed(() => route.meta.public);

async function logout() {
  await api("/api/auth/logout", { method: "POST" });
  session.user = null;
  session.status = null;
  router.push("/login");
}
</script>

<template>
  <div class="shell" :class="{ bare }">
    <header v-if="!bare && session.user" class="topbar">
      <RouterLink to="/" class="brand">
        <img src="/favicon.svg" alt="" width="26" height="26" />
        <span>Deck Keeper</span>
      </RouterLink>
      <nav>
        <RouterLink to="/" exact-active-class="active">Decks</RouterLink>
        <RouterLink to="/scans" active-class="active">Scans</RouterLink>
        <RouterLink v-if="session.user.is_admin" to="/settings" active-class="active">Settings</RouterLink>
      </nav>
      <div class="spacer"></div>
      <RouterLink to="/scan" class="btn primary small">New scan</RouterLink>
      <details class="usermenu">
        <summary>{{ session.user.username }}</summary>
        <div class="menu">
          <RouterLink to="/account">Account</RouterLink>
          <button class="link" @click="logout">Log out</button>
        </div>
      </details>
    </header>
    <main>
      <RouterView />
    </main>
  </div>
</template>

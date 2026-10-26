<script setup>
import { computed, onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api, session } from "./api";

const route = useRoute();
const router = useRouter();
const bare = computed(() => route.meta.public);

// Which build is running, so it's easy to tell whether an update has been pulled.
const build = ref(null);
onMounted(async () => {
  try {
    build.value = await api("/api/health");
  } catch {
    build.value = null;
  }
});
const builtText = computed(() => {
  const d = build.value?.built ? new Date(build.value.built) : null;
  return d && !isNaN(d) ? d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "";
});

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
      <a v-if="build" class="build" :title="`Version ${build.version}, build ${build.build}${builtText ? ', committed ' + builtText : ''}`"
         :href="build.build !== 'dev' ? `https://github.com/adman234/mtg-deck-keeper/commit/${build.build}` : undefined"
         target="_blank" rel="noopener">
        {{ build.build }}<span v-if="builtText"> · {{ builtText }}</span>
      </a>
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

<script setup>
import { computed, ref, watch } from "vue";
import { api, fmtDate, session } from "../api";
import SettingField from "../components/SettingField.vue";

const props = defineProps({ tab: String });
const TABS = [
  { id: "vision", label: "Vision model", groups: ["vision"] },
  { id: "login", label: "Login & OIDC", groups: ["auth", "oidc"] },
  { id: "cards", label: "Card database", groups: ["cards"] },
  { id: "users", label: "Users", groups: [] },
];
const current = computed(() => TABS.find((t) => t.id === props.tab) || TABS[0]);

const specs = ref([]);
const values = ref({});
const redirectUri = ref("");
const msg = ref("");
const error = ref("");
const busy = ref(false);
const test = ref(null);
const models = ref([]);

async function load() {
  const data = await api("/api/admin/settings");
  specs.value = data.settings;
  redirectUri.value = data.oidc_redirect_uri;
  values.value = Object.fromEntries(data.settings.map((s) => [s.key, s.secret ? "" : s.value]));
}
load().catch((e) => (error.value = e.message));
watch(() => props.tab, () => { msg.value = error.value = ""; test.value = null; });

const shown = (group) => specs.value.filter((s) => s.group === group);
const envCount = computed(() => specs.value.filter((s) => s.locked && current.value.groups.includes(s.group)).length);

async function save(clearKey) {
  busy.value = true;
  msg.value = error.value = "";
  const out = {};
  for (const s of specs.value) {
    if (s.locked || !current.value.groups.includes(s.group)) continue;
    if (s.secret) {
      if (s.key === clearKey) out[s.key] = "";
      else if (values.value[s.key]) out[s.key] = values.value[s.key];
      else out[s.key] = null;
    } else out[s.key] = values.value[s.key];
  }
  try {
    const data = await api("/api/admin/settings", { method: "PUT", body: { values: out } });
    specs.value = data.settings;
    values.value = Object.fromEntries(data.settings.map((s) => [s.key, s.secret ? "" : s.value]));
    msg.value = "Saved.";
    if (current.value.id === "login") session.status = null;
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}

async function testVision() {
  test.value = { busy: true };
  const res = await api("/api/admin/test-vision", { method: "POST",
    body: { base_url: values.value["vision.base_url"] || null, api_key: values.value["vision.api_key"] || null } });
  test.value = res;
  models.value = res.models || [];
}
async function testOidc() {
  test.value = { busy: true };
  test.value = await api("/api/admin/test-oidc", { method: "POST", body: { configuration_url: values.value["oidc.configuration_url"] || null } });
}
async function copy(text) {
  try { await navigator.clipboard.writeText(text); msg.value = "Copied."; } catch { /* select manually */ }
}

// Card database tab
const cardDb = ref(null);
const syncing = ref(false);
async function loadCards() { cardDb.value = await api("/api/admin/cards"); }
async function syncCards() {
  syncing.value = true;
  error.value = "";
  try { await api("/api/admin/cards/sync", { method: "POST" }); await loadCards(); }
  catch (e) { error.value = e.message; }
  finally { syncing.value = false; }
}

// Users tab
const users = ref([]);
const newUser = ref({ username: "", email: "", password: "", is_admin: false });
const confirmDel = ref(null);
const resetFor = ref(null);
const resetPw = ref("");
async function loadUsers() { users.value = await api("/api/users"); }
async function userAct(fn, okMsg) {
  error.value = msg.value = "";
  try { await fn(); await loadUsers(); if (okMsg) msg.value = okMsg; } catch (e) { error.value = e.message; }
}
const addUser = () => userAct(async () => {
  await api("/api/users", { method: "POST", body: newUser.value });
  newUser.value = { username: "", email: "", password: "", is_admin: false };
}, "User added.");
const toggleAdmin = (u) => userAct(() => api(`/api/users/${u.id}`, { method: "PATCH", body: { is_admin: !u.is_admin } }));
const delUser = (u) => userAct(() => api(`/api/users/${u.id}`, { method: "DELETE" }), `${u.username} deleted.`);
const resetPassword = (u) => userAct(async () => {
  await api(`/api/users/${u.id}`, { method: "PATCH", body: { password: resetPw.value } });
  resetFor.value = null; resetPw.value = "";
}, "Password reset.");

watch(() => current.value.id, (id) => {
  if (id === "cards") loadCards().catch((e) => (error.value = e.message));
  if (id === "users") loadUsers().catch((e) => (error.value = e.message));
}, { immediate: true });
</script>

<template>
  <div class="page">
    <div class="page-head"><div><h1>Settings</h1>
      <p class="sub">Anything set with an environment variable is shown locked here. Everything else can be changed here.</p></div></div>
    <nav class="tabs">
      <RouterLink v-for="t in TABS" :key="t.id" :to="`/settings/${t.id}`" :class="{ active: current.id === t.id }">{{ t.label }}</RouterLink>
    </nav>
    <div v-if="msg" class="alert ok">{{ msg }}</div>
    <div v-if="error" class="alert">{{ error }}</div>
    <div v-if="envCount" class="alert info">{{ envCount }} setting{{ envCount === 1 ? " is" : "s are" }} on this tab come from environment variables. Remove the variable from the container to manage it here instead.</div>

    <form v-if="current.id === 'vision'" class="panel" @submit.prevent="save()">
      <p class="sub">Any OpenAI-compatible server works. For llama-swap, the model is the key under <code>models:</code> in its config.yaml (for example <code>daily</code>), not the display name.</p>
      <template v-for="s in shown('vision')" :key="s.key">
        <SettingField :spec="s" v-model="values[s.key]" :options="s.key === 'vision.model' ? models : []" @clear="save(s.key)" />
      </template>
      <div class="row">
        <button class="btn primary" :disabled="busy">Save</button>
        <button type="button" class="btn" @click="testVision">Test connection &amp; list models</button>
      </div>
      <div v-if="test && !test.busy" class="alert" :class="test.ok ? 'ok' : ''">
        <template v-if="test.ok">Connected. {{ test.models.length }} model{{ test.models.length === 1 ? "" : "s" }} found<span v-if="test.models.length">: {{ test.models.join(", ") }}</span>. Pick one in the Model field.</template>
        <template v-else>{{ test.error }}</template>
      </div>
    </form>

    <template v-if="current.id === 'login'">
      <form class="panel" @submit.prevent="save()">
        <h2>Password login</h2>
        <SettingField v-for="s in shown('auth')" :key="s.key" :spec="s" v-model="values[s.key]" />
        <h2>OpenID Connect</h2>
        <p class="sub">Works with Authentik, Authelia, Keycloak, Pocket ID, Zitadel, Google and others. The settings match Mealie's OIDC_* variables.</p>
        <label class="field">Redirect URI <small>register this with your provider</small>
          <div class="row" style="flex-wrap:nowrap"><input id="oidc-redirect" type="text" readonly :value="redirectUri" /><button type="button" class="btn small" @click="copy(redirectUri)">Copy</button></div>
          <small>Wrong host or http instead of https? Set the BASE_URL environment variable to your public URL.</small>
        </label>
        <template v-for="s in shown('oidc')" :key="s.key">
          <SettingField :spec="s" v-model="values[s.key]" @clear="save(s.key)" />
        </template>
        <div class="row">
          <button class="btn primary" :disabled="busy">Save</button>
          <button type="button" class="btn" @click="testOidc">Test configuration URL</button>
        </div>
        <div v-if="test && !test.busy" class="alert" :class="test.ok ? 'ok' : ''">
          <template v-if="test.ok">Found issuer {{ test.issuer }}.</template>
          <template v-else>{{ test.error }}</template>
        </div>
      </form>
    </template>

    <template v-if="current.id === 'cards'">
      <div class="panel">
        <h2>Scryfall card names</h2>
        <p class="sub">Photos are matched against this list, so a misread like "Lightning Bolf" still becomes Lightning Bolt. Each day the app downloads Scryfall's card-name catalog (about 1 MB) and adds only names it hasn't seen.</p>
        <div v-if="cardDb" class="row" style="gap:24px">
          <div><div class="sub">Cards known</div><div class="num" style="font-size:1.4rem">{{ cardDb.count.toLocaleString() }}</div></div>
          <div v-if="cardDb.last_sync"><div class="sub">Last checked</div><div>{{ fmtDate(cardDb.last_sync.synced_at) }}</div></div>
          <div v-if="cardDb.last_sync"><div class="sub">New last time</div><div class="num">{{ cardDb.last_sync.added }}</div></div>
        </div>
        <p v-if="cardDb?.last_sync?.added_sample?.length" class="sub">Latest additions: {{ cardDb.last_sync.added_sample.slice(0, 10).join(", ") }}</p>
        <div><button class="btn" :disabled="syncing" @click="syncCards">{{ syncing ? "Checking…" : "Check for new cards now" }}</button></div>
      </div>
      <form class="panel" @submit.prevent="save()">
        <h2>Schedule</h2>
        <SettingField v-for="s in shown('cards')" :key="s.key" :spec="s" v-model="values[s.key]" />
        <div><button class="btn primary" :disabled="busy">Save</button></div>
      </form>
    </template>

    <template v-if="current.id === 'users'">
      <div class="panel table-wrap" style="padding:0">
        <table class="list">
          <thead><tr><th>User</th><th>Email</th><th>Login</th><th>Admin</th><th></th></tr></thead>
          <tbody>
            <tr v-for="u in users" :key="u.id">
              <td>{{ u.username }}<div class="sub" v-if="u.full_name">{{ u.full_name }}</div></td>
              <td>{{ u.email }}</td>
              <td><span class="pill neutral">{{ u.auth_method === "oidc" ? "OIDC" : "Password" }}</span></td>
              <td><input type="checkbox" :checked="u.is_admin" :disabled="u.id === session.user.id" @change="toggleAdmin(u)" :aria-label="`Admin: ${u.username}`" /></td>
              <td>
                <div class="row" style="justify-content:flex-end" v-if="u.id !== session.user.id">
                  <template v-if="resetFor === u.id">
                    <input v-model="resetPw" type="password" placeholder="New password" minlength="8" style="width:160px" aria-label="New password" />
                    <button class="btn small" :disabled="resetPw.length < 8" @click="resetPassword(u)">Set</button>
                    <button class="btn small" @click="resetFor = null">Cancel</button>
                  </template>
                  <button v-else class="btn small" @click="resetFor = u.id; resetPw = ''">Reset password</button>
                  <button v-if="confirmDel !== u.id" class="btn small danger" @click="confirmDel = u.id">Delete</button>
                  <template v-else><button class="btn small danger" @click="delUser(u)">Delete {{ u.username }} and their decks</button><button class="btn small" @click="confirmDel = null">Keep</button></template>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <form class="panel" @submit.prevent="addUser">
        <h2>Add a user</h2>
        <p class="sub">With OIDC sign-up turned on, people are created automatically the first time they sign in.</p>
        <div class="row" style="align-items:flex-end">
          <label class="field" style="flex:1;min-width:150px">Username<input id="nu-name" v-model="newUser.username" type="text" required minlength="2" /></label>
          <label class="field" style="flex:1;min-width:180px">Email<input id="nu-email" v-model="newUser.email" type="email" required /></label>
          <label class="field" style="flex:1;min-width:150px">Password<input id="nu-pass" v-model="newUser.password" type="password" required minlength="8" autocomplete="new-password" /></label>
        </div>
        <label class="check"><input id="nu-admin" v-model="newUser.is_admin" type="checkbox" />Admin</label>
        <div><button class="btn primary">Add user</button></div>
      </form>
    </template>
  </div>
</template>

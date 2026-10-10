import { createRouter, createWebHistory } from "vue-router";
import { loadStatus, session } from "./api";

const routes = [
  { path: "/login", component: () => import("./views/Login.vue"), meta: { public: true } },
  { path: "/setup", component: () => import("./views/Setup.vue"), meta: { public: true } },
  { path: "/", component: () => import("./views/Decks.vue") },
  { path: "/decks/:id", component: () => import("./views/DeckView.vue"), props: true },
  { path: "/scan", component: () => import("./views/NewScan.vue") },
  { path: "/scans", component: () => import("./views/Scans.vue") },
  { path: "/scans/:id", component: () => import("./views/ScanView.vue"), props: true },
  { path: "/settings/:tab?", component: () => import("./views/Settings.vue"), props: true, meta: { admin: true } },
  { path: "/account", component: () => import("./views/Account.vue") },
  { path: "/:pathMatch(.*)*", redirect: "/" },
];

const router = createRouter({ history: createWebHistory(), routes });

router.beforeEach(async (to) => {
  if (!session.status) await loadStatus();
  if (session.status.setup_required) return to.path === "/setup" ? true : "/setup";
  if (to.path === "/setup") return "/";
  if (to.meta.public) return true;
  if (!session.user) return { path: "/login", query: to.fullPath !== "/" ? { next: to.fullPath } : {} };
  if (to.meta.admin && !session.user.is_admin) return "/";
  return true;
});

export default router;

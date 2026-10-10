import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  build: { outDir: "../backend/deck_keeper/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://localhost:9925" } },
});

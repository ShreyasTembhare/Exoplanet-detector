import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Listen on every interface so a localhost port forward can attach.
    // The page you open is still http://localhost:5173.
    host: true,
    port: 5173,
    strictPort: true,
    allowedHosts: true,
    hmr: {
      host: "localhost",
      clientPort: 5173,
    },
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "../tool/static",
    emptyOutDir: true,
  },
});

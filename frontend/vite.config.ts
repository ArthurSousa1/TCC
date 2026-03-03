import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const apiUrl = process.env.VITE_API_URL || "http://127.0.0.1:8000";
  const isDev = mode === "development";
  
  return {
    define: {
      'import.meta.env.VITE_API_URL': JSON.stringify(apiUrl),
    },
    server: {
      host: "0.0.0.0",
      port: 8080,
      hmr: isDev ? {
        host: "127.0.0.1",
        port: 8080,
        protocol: "ws"
      } : false,
      proxy: {
        "/api": {
          target: "http://api:8000",
          changeOrigin: true,
          secure: false,
        },
      },
    },
    plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
  };
});

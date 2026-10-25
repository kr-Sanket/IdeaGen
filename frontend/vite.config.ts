import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The dev server proxies API calls to the FastAPI backend. Override the target with
// IDEAGEN_API_PROXY_TARGET (e.g. to point a second frontend at a backend on another port).
// The Vite process itself must be allowed to open local connections to this address; when it is
// not (e.g. inside a sandbox), proxied requests fail with "connect EPERM" and the UI shows API Offline.
const apiTarget = process.env.IDEAGEN_API_PROXY_TARGET || 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
      '/health': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})

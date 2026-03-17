import { defineConfig, type Plugin, type ViteDevServer } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { buildDevApiFallbackResponse, shouldBypassApiProxy } from './src/app/devApiFallback'

const DEFAULT_PROXY_TARGET = "http://127.0.0.1:8095";

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function resolveProxyTarget(): string {
  return trimTrailingSlash(String(process.env.AURA_DASHBOARD_PROXY_TARGET ?? DEFAULT_PROXY_TARGET).trim() || DEFAULT_PROXY_TARGET);
}

function createBackendHealthProbe(proxyTarget: string) {
  let lastCheckedAt = 0;
  let lastHealthy = false;

  return async function isBackendHealthy(): Promise<boolean> {
    const now = Date.now();
    if (now - lastCheckedAt < 1200) {
      return lastHealthy;
    }
    lastCheckedAt = now;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 400);
    try {
      const response = await fetch(`${proxyTarget}/api/bootstrap`, {
        method: "GET",
        signal: controller.signal,
      });
      lastHealthy = response.ok;
      return lastHealthy;
    } catch {
      lastHealthy = false;
      return false;
    } finally {
      clearTimeout(timeoutId);
    }
  };
}

function auraDashboardDevApiFallback(): Plugin {
  const proxyTarget = resolveProxyTarget();
  const isBackendHealthy = createBackendHealthProbe(proxyTarget);

  return {
    name: "aura-dashboard-dev-api-fallback",
    apply: "serve" as const,
    configureServer(server: ViteDevServer) {
      server.middlewares.use(async (req, res, next) => {
        const requestUrl = req.url ?? "/";
        if (!shouldBypassApiProxy(requestUrl)) {
          return next();
        }
        if (await isBackendHealthy()) {
          return next();
        }
        const fallback = buildDevApiFallbackResponse({
          apiBaseUrl: proxyTarget,
          method: req.method,
          requestUrl,
        });
        res.statusCode = fallback.status;
        for (const [headerName, headerValue] of Object.entries(fallback.headers)) {
          res.setHeader(headerName, headerValue);
        }
        if (fallback.keepAlive === true) {
          res.write(fallback.body);
          const keepAliveTimer = setInterval(() => {
            if (res.writableEnded) {
              clearInterval(keepAliveTimer);
              return;
            }
            res.write(": keepalive\n\n");
          }, 10000);
          req.on("close", () => clearInterval(keepAliveTimer));
          return;
        }
        res.end(fallback.body);
      });
    },
  };
}

export default defineConfig({
  plugins: [
    // The React and Tailwind plugins are both required for Make, even if
    // Tailwind is not being actively used – do not remove them
    react(),
    tailwindcss(),
    auraDashboardDevApiFallback(),
  ],
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: resolveProxyTarget(),
        changeOrigin: false,
      },
    },
  },

  // File types to support raw imports. Never add .css, .tsx, or .ts files to this.
  assetsInclude: ['**/*.svg', '**/*.csv'],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./vitest.setup.ts",
  },
})

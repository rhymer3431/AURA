import { defineConfig, type Plugin, type ViteDevServer } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { buildDevApiFallbackResponse, shouldBypassApiProxy } from './src/app/devApiFallback'
import {
  createBackendHealthProbe,
  resolveBackendHealthCacheMs,
  resolveBackendHealthGraceMs,
  resolveBackendHealthTimeoutMs,
  resolveProxyTarget,
} from './scripts/devBackendProxy'

const DEFAULT_DEV_PORT = 5173;

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function resolveDevServerPort(): number {
  const parsed = Number.parseInt(String(process.env.AURA_DASHBOARD_DEV_PORT ?? DEFAULT_DEV_PORT), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_DEV_PORT;
}

function auraDashboardDevApiFallback(): Plugin {
  const proxyTarget = resolveProxyTarget();
  const backendHealth = createBackendHealthProbe(proxyTarget, {
    timeoutMs: resolveBackendHealthTimeoutMs(),
    cacheMs: resolveBackendHealthCacheMs(),
    graceMs: resolveBackendHealthGraceMs(),
  });

  return {
    name: "aura-dashboard-dev-api-fallback",
    apply: "serve" as const,
    configureServer(server: ViteDevServer) {
      server.config.logger.info(
        `[AURA_DASHBOARD] dev proxy target ${trimTrailingSlash(proxyTarget)} ` +
          `(timeout=${resolveBackendHealthTimeoutMs()}ms cache=${resolveBackendHealthCacheMs()}ms grace=${resolveBackendHealthGraceMs()}ms)`,
      );
      server.middlewares.use(async (req, res, next) => {
        const requestUrl = req.url ?? "/";
        if (!shouldBypassApiProxy(requestUrl)) {
          return next();
        }
        if (await backendHealth.shouldProxyToBackend()) {
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
    port: resolveDevServerPort(),
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

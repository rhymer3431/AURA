import { spawn } from "node:child_process";
import net from "node:net";
import process from "node:process";

const DEFAULT_DEV_HOST = "127.0.0.1";
const DEFAULT_DEV_PORT = 5173;
const MAX_PORT_SCAN = 25;

function parsePreferredPort(value) {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_DEV_PORT;
  }
  return parsed;
}

function canListen(host, port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.unref();
    server.once("error", () => resolve(false));
    server.listen(port, host, () => {
      server.close(() => resolve(true));
    });
  });
}

async function findAvailablePort(host, preferredPort) {
  for (let offset = 0; offset < MAX_PORT_SCAN; offset += 1) {
    const candidate = preferredPort + offset;
    if (await canListen(host, candidate)) {
      return candidate;
    }
  }
  throw new Error(
    `Unable to find an available AURA dashboard dev port in range ${preferredPort}-${preferredPort + MAX_PORT_SCAN - 1}.`,
  );
}

async function main() {
  const host = String(process.env.AURA_DASHBOARD_DEV_HOST ?? DEFAULT_DEV_HOST).trim() || DEFAULT_DEV_HOST;
  const preferredPort = parsePreferredPort(process.env.AURA_DASHBOARD_DEV_PORT);
  const selectedPort = await findAvailablePort(host, preferredPort);
  const devUrl = `http://${host}:${selectedPort}`;
  const tauriConfigOverride = {
    build: {
      beforeDevCommand: `npm run dev -- --host ${host} --port ${selectedPort}`,
      devUrl,
    },
  };

  process.stdout.write(`[AURA_DASHBOARD] Tauri dev URL ${devUrl}\n`);
  if (selectedPort !== preferredPort) {
    process.stdout.write(
      `[AURA_DASHBOARD] Port ${preferredPort} is already in use; falling back to ${selectedPort}.\n`,
    );
  }

  const npmExecPath = String(process.env.npm_execpath ?? "").trim();
  const npmNodeExecPath = String(process.env.npm_node_execpath ?? process.execPath).trim();
  const fallbackTauriBin = process.platform === "win32" ? "npx.cmd" : "npx";
  const child = npmExecPath !== ""
    ? spawn(
        npmNodeExecPath,
        [
          npmExecPath,
          "exec",
          "--",
          "tauri",
          "dev",
          "--config",
          JSON.stringify(tauriConfigOverride),
          ...process.argv.slice(2),
        ],
        {
          stdio: "inherit",
          env: {
            ...process.env,
            AURA_DASHBOARD_DEV_HOST: host,
            AURA_DASHBOARD_DEV_PORT: String(selectedPort),
          },
        },
      )
    : spawn(
        fallbackTauriBin,
        ["tauri", "dev", "--config", JSON.stringify(tauriConfigOverride), ...process.argv.slice(2)],
        {
          stdio: "inherit",
          env: {
            ...process.env,
            AURA_DASHBOARD_DEV_HOST: host,
            AURA_DASHBOARD_DEV_PORT: String(selectedPort),
          },
        },
      );

  child.on("exit", (code, signal) => {
    if (signal !== null) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 1);
  });
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`${message}\n`);
  process.exit(1);
});

import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const mode = process.argv[2] === "start" ? "start" : "dev";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const concurrentlyScript = path.join(
  repoRoot,
  "node_modules",
  "concurrently",
  "dist",
  "bin",
  "concurrently.js"
);
const PORT_SCAN_LIMIT = 20;
const POSIX_TEMP_DIR = "/tmp";
const MIN_NODE_MAJOR = 24;

function parsePort(rawValue, fallback, label) {
  const parsed = Number(rawValue ?? fallback);
  if (!Number.isInteger(parsed) || parsed <= 0 || parsed > 65535) {
    throw new Error(`${label} must be a valid TCP port. Received: ${rawValue}`);
  }
  return parsed;
}

function getNodeMajorVersion(version) {
  const match = /^v?(\d+)\./.exec(version);
  return match ? Number(match[1]) : NaN;
}

function toPublicHost(host) {
  return host === "0.0.0.0" ? "127.0.0.1" : host;
}

function envIsSet(name) {
  return process.env[name] !== undefined && process.env[name] !== "";
}

function shellQuote(value) {
  const escaped = String(value).replace(/"/g, '\\"');
  if (process.platform === "win32") {
    return `"${escaped}"`;
  }
  return `'${String(value).replace(/'/g, "'\\''")}'`;
}

function resolvePythonCommand() {
  const configured = (process.env.PYTHON || "").trim();
  if (configured) {
    return configured;
  }

  const candidates =
    process.platform === "win32"
      ? [
          path.join(repoRoot, ".venv", "Scripts", "python.exe"),
          path.join(path.dirname(repoRoot), ".venv", "Scripts", "python.exe")
        ]
      : [
          path.join(repoRoot, ".venv", "bin", "python"),
          path.join(path.dirname(repoRoot), ".venv", "bin", "python")
        ];

  const localPython = candidates.find((candidate) => fs.existsSync(candidate));
  if (localPython) {
    return localPython;
  }

  return process.platform === "win32" ? "python" : "python3";
}

function normalizeChildTempEnv(env) {
  if (process.platform === "win32") {
    return env;
  }

  const tempCandidates = [env.TMPDIR, env.TMP, env.TEMP].filter(Boolean);
  const hasMountedWindowsTemp = tempCandidates.some((value) =>
    value.startsWith("/mnt/")
  );

  if (!hasMountedWindowsTemp) {
    return env;
  }

  return {
    ...env,
    TMPDIR: env.TMPDIR && !env.TMPDIR.startsWith("/mnt/") ? env.TMPDIR : POSIX_TEMP_DIR,
    TMP: env.TMP && !env.TMP.startsWith("/mnt/") ? env.TMP : POSIX_TEMP_DIR,
    TEMP: env.TEMP && !env.TEMP.startsWith("/mnt/") ? env.TEMP : POSIX_TEMP_DIR
  };
}

function checkPortAvailability(host, port) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();

    server.once("error", (error) => {
      server.close(() => reject(error));
    });

    server.listen({ host, port, exclusive: true }, () => {
      server.close((closeError) => {
        if (closeError) {
          reject(closeError);
          return;
        }
        resolve();
      });
    });
  });
}

async function findAvailablePort(host, startPort) {
  const lastPort = Math.min(65535, startPort + PORT_SCAN_LIMIT);

  for (let port = startPort; port <= lastPort; port += 1) {
    try {
      await checkPortAvailability(host, port);
      return port;
    } catch {
      // Keep scanning nearby ports.
    }
  }

  return null;
}

async function endpointIsHealthy(url) {
  try {
    const response = await fetch(url, {
      signal: AbortSignal.timeout(2000)
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function main() {
  const nodeMajor = getNodeMajorVersion(process.version);
  if (!Number.isFinite(nodeMajor) || nodeMajor < MIN_NODE_MAJOR) {
    console.error(
      `Dashboard startup requires Node ${MIN_NODE_MAJOR}+ because the API server uses node:sqlite. ` +
        `Current runtime is ${process.version}.`
    );
    console.error(
      "Use a Node 24+ shell before running `npm run dev`, or run this launcher with a Node 24 binary."
    );
    process.exit(1);
  }

  if (!fs.existsSync(concurrentlyScript)) {
    console.error("Missing local dashboard dependencies. Run `npm install` first.");
    process.exit(1);
  }

  const bridgeHost = process.env.DASHBOARD_BRIDGE_HOST || "127.0.0.1";
  let bridgePort = parsePort(
    process.env.DASHBOARD_BRIDGE_PORT,
    8101,
    "DASHBOARD_BRIDGE_PORT"
  );
  const serverHost = process.env.DASHBOARD_SERVER_HOST || "127.0.0.1";
  let serverPort = parsePort(
    process.env.DASHBOARD_SERVER_PORT,
    4000,
    "DASHBOARD_SERVER_PORT"
  );
  let webPort = parsePort(
    process.env.DASHBOARD_WEB_PORT,
    3000,
    "DASHBOARD_WEB_PORT"
  );

  const services = [
    {
      name: "Python bridge",
      host: bridgeHost,
      port: bridgePort,
      envVar: "DASHBOARD_BRIDGE_PORT",
      configured: envIsSet("DASHBOARD_BRIDGE_PORT")
    },
    {
      name: "Fastify API",
      host: serverHost,
      port: serverPort,
      envVar: "DASHBOARD_SERVER_PORT",
      configured: envIsSet("DASHBOARD_SERVER_PORT")
    },
    {
      name: "Next.js web app",
      host: "0.0.0.0",
      port: webPort,
      envVar: "DASHBOARD_WEB_PORT",
      configured: envIsSet("DASHBOARD_WEB_PORT")
    }
  ];

  const portFailures = [];
  for (const service of services) {
    try {
      // Fail early with a readable message before any child process starts.
      await checkPortAvailability(service.host, service.port);
    } catch (error) {
      portFailures.push({ service, error });
    }
  }

  if (portFailures.length > 0) {
    const preferredBridgeUrl = `http://${toPublicHost(bridgeHost)}:${bridgePort}`;
    const preferredApiUrl = `http://${toPublicHost(serverHost)}:${serverPort}`;
    const preferredWebUrl = `http://127.0.0.1:${webPort}`;
    const existingStackHealth = [
      {
        name: "Python bridge",
        url: `${preferredBridgeUrl}/health`,
        healthy: await endpointIsHealthy(`${preferredBridgeUrl}/health`)
      },
      {
        name: "Fastify API",
        url: `${preferredApiUrl}/api/dashboard/overview`,
        healthy: await endpointIsHealthy(`${preferredApiUrl}/api/dashboard/overview`)
      },
      {
        name: "Next.js web app",
        url: preferredWebUrl,
        healthy: await endpointIsHealthy(preferredWebUrl)
      }
    ];

    if (existingStackHealth.every((check) => check.healthy)) {
      console.log("Dashboard stack is already running.");
      console.log(`- Web app: ${preferredWebUrl}`);
      console.log(`- Fastify API: ${preferredApiUrl}`);
      console.log(`- Python bridge: ${preferredBridgeUrl}`);
      process.exitCode = 0;
      return;
    }

    const unresolvedFailures = [];
    for (const failure of portFailures) {
      const { service } = failure;
      if (service.configured) {
        unresolvedFailures.push(failure);
        continue;
      }

      const fallbackPort = await findAvailablePort(service.host, service.port + 1);
      if (fallbackPort === null) {
        unresolvedFailures.push(failure);
        continue;
      }

      console.warn(
        `${service.name}: ${service.host}:${service.port} is unavailable; ` +
          `using ${service.envVar}=${fallbackPort} for this run.`
      );
      service.port = fallbackPort;
    }

    if (unresolvedFailures.length === 0) {
      bridgePort = services[0].port;
      serverPort = services[1].port;
      webPort = services[2].port;
    } else {
      console.error("Dashboard startup check failed:");
      console.error("Health checks did not find a complete running dashboard stack:");
      for (const check of existingStackHealth) {
        console.error(`- ${check.name}: ${check.healthy ? "healthy" : "not responding"} (${check.url})`);
      }
      for (const { service, error } of unresolvedFailures) {
        const detail = error?.code || error?.message || "unknown error";
        console.error(
          `- ${service.name}: ${service.host}:${service.port} is unavailable (${detail}). ` +
            `Set ${service.envVar} to another port or stop the process already using it.`
        );
      }
      process.exit(1);
    }
  }

  const localBridgeUrl = `http://${toPublicHost(bridgeHost)}:${bridgePort}`;
  const dashboardApiUrl = `http://${toPublicHost(serverHost)}:${serverPort}`;
  const dashboardWebUrl = `http://127.0.0.1:${webPort}`;
  const analysisBridgeUrl =
    process.env.ANALYSIS_BRIDGE_URL || localBridgeUrl;
  const dashboardDbPath =
    process.env.DB_PATH || path.join(repoRoot, "trading_system.db");
  const pythonCommand = resolvePythonCommand();

  console.log(`Dashboard Python: ${pythonCommand}`);

  const bridgeCommand = [
    shellQuote(pythonCommand),
    "-m",
    "uvicorn",
    "python_bridge.app.main:app",
    "--host",
    bridgeHost,
    "--port",
    String(bridgePort),
    ...(mode === "dev"
      ? [
          "--reload",
          "--reload-dir",
          "python_bridge",
          "--reload-dir",
          "src"
        ]
      : [])
  ].join(" ");

  const concurrentlyArgs = [
    "-k",
    "-n",
    "bridge,server,web",
    "-c",
    "green,blue,magenta",
    bridgeCommand,
    `${shellQuote(process.execPath)} scripts/wait-for-url.mjs ${shellQuote(`${analysisBridgeUrl}/health`)} && npm run ${mode} --workspace server`,
    `npm run ${mode} --workspace web`
  ];

  const child = spawn(process.execPath, [concurrentlyScript, ...concurrentlyArgs], {
    cwd: repoRoot,
    stdio: "inherit",
    env: normalizeChildTempEnv({
      ...process.env,
      DASHBOARD_BRIDGE_HOST: bridgeHost,
      DASHBOARD_BRIDGE_PORT: String(bridgePort),
      DASHBOARD_SERVER_HOST: serverHost,
      DASHBOARD_SERVER_PORT: String(serverPort),
      DASHBOARD_WEB_PORT: String(webPort),
      ANALYSIS_BRIDGE_URL: analysisBridgeUrl,
      DB_PATH: dashboardDbPath,
      DASHBOARD_API_URL:
        process.env.DASHBOARD_API_URL || dashboardApiUrl,
      NEXT_PUBLIC_DASHBOARD_API_URL:
        process.env.NEXT_PUBLIC_DASHBOARD_API_URL || dashboardApiUrl,
      PORT: String(webPort)
    })
  });

  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.on(signal, () => {
      if (!child.killed) {
        child.kill(signal);
      }
    });
  }

  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 1);
  });
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

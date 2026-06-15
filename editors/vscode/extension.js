const vscode = require("vscode");
const fs = require("fs");
const path = require("path");
const {
  LanguageClient,
  RevealOutputChannelOn,
  State,
  TransportKind,
} = require("vscode-languageclient/node");

let client;
let clientReady;
let extensionContext;
let outputChannel;
let statusBarItem;

function activate(context) {
  extensionContext = context;
  outputChannel = vscode.window.createOutputChannel("Ora Language Server", {
    log: true,
  });
  context.subscriptions.push(outputChannel);

  statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    0
  );
  statusBarItem.name = "Ora";
  statusBarItem.command = "ora.restartServer";
  context.subscriptions.push(statusBarItem);

  context.subscriptions.push(
    vscode.commands.registerCommand("ora.restartServer", async () => {
      const restarted = await restartClient();
      if (restarted) {
        vscode.window.showInformationMessage("Ora language server restarted.");
      } else {
        vscode.window.showWarningMessage("Ora language server did not start.");
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ora.showCacheStats", async () => {
      await showCacheStats();
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ora.openServerOutput", () => {
      outputChannel.show();
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ora.verify", () => {
      vscode.window.showInformationMessage(
        "Ora verification results are shown through diagnostics and code lenses; direct editor execution is not exposed yet."
      );
    })
  );

  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (!event.affectsConfiguration("ora.lsp")) return;
      vscode.window
        .showInformationMessage(
          "Ora language server settings changed.",
          "Restart server"
        )
        .then((selection) => {
          if (selection === "Restart server") {
            restartClient();
          }
        });
    })
  );

  startClient(context);
}

function startClient(context) {
  const config = vscode.workspace.getConfiguration("ora");
  if (config.get("lsp.enabled") === false) {
    setStatus("disabled");
    outputChannel.info("Ora language server disabled by ora.lsp.enabled.");
    return;
  }

  const server = resolveServerCommand(context, config);
  outputChannel.info(
    `Starting Ora language server: ${server.command} ${server.args.join(" ")}`
  );

  const serverOptions = {
    command: server.command,
    args: server.args,
    options: server.options,
    transport: TransportKind.stdio,
  };

  const clientOptions = {
    documentSelector: [
      { scheme: "file", language: "ora" },
      { scheme: "untitled", language: "ora" },
    ],
    synchronize: {
      configurationSection: "ora",
      fileEvents: vscode.workspace.createFileSystemWatcher("**/*.ora"),
    },
    outputChannel,
    traceOutputChannel: outputChannel,
    revealOutputChannelOn: RevealOutputChannelOn.Never,
  };

  client = new LanguageClient(
    "ora-lsp",
    "Ora Language Server",
    serverOptions,
    clientOptions
  );

  context.subscriptions.push(
    client.onDidChangeState((event) => {
      switch (event.newState) {
        case State.Starting:
          setStatus("starting");
          break;
        case State.Running:
          setStatus("running");
          break;
        case State.Stopped:
          if (client) setStatus("stopped");
          break;
      }
    })
  );

  setStatus("starting");

  clientReady = client
    .start()
    .then(() => setStatus("running"))
    .catch((err) => {
      setStatus("error");
      outputChannel.error(`Failed to start ora-lsp: ${formatError(err)}`);
      vscode.window.showErrorMessage(
        `Ora language server failed to start: ${formatError(err)}`
      );
    });

  context.subscriptions.push(client);
}

async function restartClient() {
  setStatus("starting");
  if (client) {
    const oldClient = client;
    client = undefined;
    clientReady = undefined;
    await oldClient.stop().catch((err) => {
      outputChannel.warn(`Failed to stop previous ora-lsp: ${formatError(err)}`);
    });
  }

  startClient(extensionContext);
  if (clientReady) {
    await clientReady;
  }
  return client?.state === State.Running;
}

async function showCacheStats() {
  if (!client) {
    startClient(extensionContext);
  }
  if (!clientReady) {
    vscode.window.showWarningMessage("Ora language server is not running.");
    return;
  }

  await clientReady;
  if (!client || client.state !== State.Running) {
    vscode.window.showWarningMessage("Ora language server is not running.");
    return;
  }

  const stats = await client.sendRequest("workspace/executeCommand", {
    command: "ora.cacheStats",
    arguments: [],
  });

  outputChannel.info("Ora cache stats:");
  outputChannel.info(JSON.stringify(stats, null, 2));
  outputChannel.show();

  const openDocuments = numberStat(stats, "openDocuments");
  const coldDocuments = numberStat(stats, "coldDocuments");
  const trackedCacheBytes = numberStat(stats, "trackedCacheBytes");
  vscode.window.showInformationMessage(
    `Ora cache stats: ${openDocuments} open docs, ${coldDocuments} cold docs, ${formatBytes(trackedCacheBytes)} tracked cache.`
  );
}

function resolveServerCommand(context, config) {
  const cwd = workspaceCwd() || context.extensionPath;
  const configuredPath = stringSetting(config, "lsp.path");
  const envPath = process.env.ORA_LSP_PATH;
  const preferWorkspaceBinary = config.get("lsp.preferWorkspaceBinary") !== false;
  const command =
    normalizeCommand(configuredPath, cwd) ||
    normalizeCommand(envPath, cwd) ||
    (preferWorkspaceBinary ? workspaceBinaryPath(context) : undefined) ||
    "ora-lsp";

  return {
    command,
    args: arraySetting(config, "lsp.args"),
    options: {
      cwd,
      env: {
        ...process.env,
        ...objectSetting(config, "lsp.env"),
      },
    },
  };
}

function workspaceBinaryPath(context) {
  const binaryName = process.platform === "win32" ? "ora-lsp.exe" : "ora-lsp";
  const candidates = [];

  for (const folder of vscode.workspace.workspaceFolders || []) {
    candidates.push(path.join(folder.uri.fsPath, "zig-out", "bin", binaryName));
  }

  candidates.push(path.join(context.extensionPath, "..", "..", "zig-out", "bin", binaryName));

  return candidates.find((candidate) => fileExists(candidate));
}

function workspaceCwd() {
  const folder = vscode.workspace.workspaceFolders?.[0];
  return folder?.uri?.fsPath;
}

function normalizeCommand(value, cwd) {
  if (!value || typeof value !== "string" || value.trim() === "") {
    return undefined;
  }
  const trimmed = value.trim();
  if (path.isAbsolute(trimmed)) return trimmed;
  if (trimmed.startsWith(".") || trimmed.includes("/") || trimmed.includes("\\")) {
    return path.resolve(cwd, trimmed);
  }
  return trimmed;
}

function arraySetting(config, name) {
  const value = config.get(name);
  return Array.isArray(value) ? value.filter((item) => typeof item === "string") : [];
}

function objectSetting(config, name) {
  const value = config.get(name);
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};

  const result = {};
  for (const [key, entry] of Object.entries(value)) {
    if (typeof entry === "string") result[key] = entry;
  }
  return result;
}

function stringSetting(config, name) {
  const value = config.get(name);
  return typeof value === "string" ? value : "";
}

function fileExists(candidate) {
  try {
    return fs.statSync(candidate).isFile();
  } catch (_) {
    return false;
  }
}

function numberStat(stats, key) {
  const value = stats?.[key];
  return typeof value === "number" ? value : 0;
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
}

function formatError(err) {
  if (!err) return "unknown error";
  if (err.message) return err.message;
  return String(err);
}

function setStatus(state) {
  switch (state) {
    case "starting":
      statusBarItem.text = "$(loading~spin) Ora";
      statusBarItem.tooltip = "Ora Language Server starting…";
      statusBarItem.show();
      break;
    case "running":
      statusBarItem.text = "$(check) Ora";
      statusBarItem.tooltip = "Ora Language Server running (click to restart)";
      statusBarItem.show();
      break;
    case "error":
      statusBarItem.text = "$(error) Ora";
      statusBarItem.tooltip =
        "Ora Language Server failed to start (click to retry)";
      statusBarItem.show();
      break;
    case "stopped":
      statusBarItem.text = "$(circle-slash) Ora";
      statusBarItem.tooltip = "Ora Language Server stopped (click to restart)";
      statusBarItem.show();
      break;
    case "disabled":
      statusBarItem.text = "$(circle-slash) Ora";
      statusBarItem.tooltip = "Ora Language Server disabled";
      statusBarItem.show();
      break;
  }
}

function deactivate() {
  if (client) {
    return client.stop();
  }
}

module.exports = { activate, deactivate };

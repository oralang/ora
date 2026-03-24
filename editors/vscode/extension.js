const vscode = require("vscode");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

let client;
let outputChannel;
let statusBarItem;

function activate(context) {
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
      if (client) {
        setStatus("starting");
        await client.restart();
        setStatus("running");
        vscode.window.showInformationMessage("Ora language server restarted.");
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ora.verify", () => {
      vscode.window.showInformationMessage(
        "Ora: Formal verification from the editor is not yet available."
      );
    })
  );

  startClient(context);
}

function startClient(context) {
  const config = vscode.workspace.getConfiguration("ora");
  const lspPath = config.get("lsp.path") || "ora-lsp";

  const serverOptions = {
    command: lspPath,
    transport: TransportKind.stdio,
  };

  const clientOptions = {
    documentSelector: [{ scheme: "file", language: "ora" }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher("**/*.ora"),
    },
    outputChannel,
  };

  client = new LanguageClient(
    "ora-lsp",
    "Ora Language Server",
    serverOptions,
    clientOptions
  );

  setStatus("starting");

  client
    .start()
    .then(() => setStatus("running"))
    .catch((err) => {
      setStatus("error");
      outputChannel.error(`Failed to start ora-lsp: ${err.message}`);
    });

  context.subscriptions.push(client);
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
  }
}

function deactivate() {
  if (client) {
    return client.stop();
  }
}

module.exports = { activate, deactivate };

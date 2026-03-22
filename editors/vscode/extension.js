const vscode = require("vscode");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

let client;

function activate(context) {
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
  };

  client = new LanguageClient("ora-lsp", "Ora Language Server", serverOptions, clientOptions);
  client.start();
}

function deactivate() {
  if (client) {
    return client.stop();
  }
}

module.exports = { activate, deactivate };

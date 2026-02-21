# tree-sitter-ora

Tree-sitter grammar for Ora (`.ora`) source files.

## Status

Initial bootstrap grammar intended for editor integration and incremental improvement.

- Source of truth for compiler semantics remains Ora's handwritten parser.
- This grammar is for editor-facing parsing (highlighting, structure, and future LSP plumbing).

## Prerequisites

- Node.js + npm
- `tree-sitter-cli` (via npm dev dependency)

## Quick start

```bash
cd tree-sitter-ora
pnpm install
pnpm run generate
pnpm test
```

Equivalent with npm:

```bash
cd tree-sitter-ora
npm install
npm run generate
npm test
```

## Repository-wide test script

From repo root:

```bash
./scripts/test-tree-sitter.sh
```

## Neovim usage

Add parser config:

```lua
local parser_config = require("nvim-treesitter.parsers").get_parser_configs()

parser_config.ora = {
  install_info = {
    url = "/Users/logic/Ora/Ora/tree-sitter-ora",
    files = { "src/parser.c" },
    generate_requires_npm = true,
    requires_generate_from_grammar = true,
  },
  filetype = "ora",
}

vim.filetype.add({ extension = { ora = "ora" } })

require("nvim-treesitter.configs").setup({
  highlight = { enable = true },
})
```

Then run in Neovim:

```vim
:TSInstall ora
```

## Project layout

- `grammar.js`: grammar definition
- `tree-sitter.json`: parser metadata
- `queries/highlights.scm`: highlight captures
- `test/corpus/*.txt`: corpus tests
- `src/`: generated parser artifacts (`parser.c`, `node-types.json`) after `tree-sitter generate`

## Troubleshooting (pnpm v10)

If `pnpm run generate` fails with missing `tree-sitter-cli/tree-sitter` (ENOENT), dependency build scripts were skipped.

Run:

```bash
cd tree-sitter-ora
pnpm approve-builds tree-sitter-cli
pnpm rebuild tree-sitter-cli
pnpm run generate
```

If needed:

```bash
pnpm install --ignore-scripts=false
```

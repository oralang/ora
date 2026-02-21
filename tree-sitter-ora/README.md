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
local parsers = require("nvim-treesitter.parsers")

parsers.ora = {
  install_info = {
    url = "/Users/logic/Ora/Ora/tree-sitter-ora",
    files = { "src/parser.c" },
    generate_requires_npm = true,
    requires_generate_from_grammar = true,
  },
  filetype = "ora",
}

vim.filetype.add({ extension = { ora = "ora" } })

require("nvim-treesitter").setup({})
require("nvim-treesitter").install({ "ora" })
```

Then run in Neovim:

```vim
:TSInstall ora
```

## Highlight coverage

`queries/highlights.scm` includes captures for:

- Core language: contracts, structs, enums, errors, functions, operators.
- Regions: `storage`, `memory`, `tstore`, `calldata` with dedicated variable captures.
- Formal verification: `requires`, `ensures`, `invariant`, `decreases`, quantifiers.
- Comptime: comptime keyword, comptime parameters, and comptime functions.
- Builtins: `std.*` namespaces (`constants`, `msg`, `transaction`, `block`) and intrinsic calls (`@addWithOverflow`, `@lock`, etc.).
- Ghost and verification constructs.

Useful custom links in Neovim:

```lua
vim.api.nvim_set_hl(0, "@function.comptime.ora", { bold = true })
vim.api.nvim_set_hl(0, "@module.builtin.ora", { italic = true })
vim.api.nvim_set_hl(0, "@function.builtin.ora", { bold = true })
vim.api.nvim_set_hl(0, "@constant.builtin.ora", { bold = true })
vim.api.nvim_set_hl(0, "@keyword.fv.ora", { italic = true })
vim.api.nvim_set_hl(0, "@keyword.ghost.ora", { italic = true })
```

## Shareable Neovim Highlight File

Full example file:

- `examples/nvim/ora_highlights.lua`

Usage from your `init.lua`:

```lua
local ora_hl = dofile("/absolute/path/to/tree-sitter-ora/examples/nvim/ora_highlights.lua")
ora_hl.autocmd()
```

Optional palette override:

```lua
local ora_hl = dofile("/absolute/path/to/tree-sitter-ora/examples/nvim/ora_highlights.lua")
ora_hl.autocmd({
  palette = {
    red = "#ff6b6b",
    cyan = "#63e6be",
  },
})
```

Notes:

- By default, highlights are scoped to `.ora` captures only (`@capture.ora`).
- Pass `{ global = true }` to also set non-language-scoped captures (affects other filetypes).

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

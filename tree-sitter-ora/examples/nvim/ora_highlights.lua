local M = {}

M.default_palette = {
  red = "#E67E80",
  orange = "#E69875",
  yellow = "#DBBC7F",
  green = "#A7C080",
  aqua = "#83C092",
  cyan = "#7FBBB3",
  blue = "#7F9FB3",
  purple = "#D699B6",
  fg = "#D3C6AA",
}

local function merge_palette(overrides)
  if type(vim) == "table" and vim.tbl_extend then
    return vim.tbl_extend("force", M.default_palette, overrides or {})
  end

  local out = {}
  for k, v in pairs(M.default_palette) do
    out[k] = v
  end
  for k, v in pairs(overrides or {}) do
    out[k] = v
  end
  return out
end

local function set_capture(group, spec, opts)
  vim.api.nvim_set_hl(0, group .. ".ora", spec)
  if opts.global then
    vim.api.nvim_set_hl(0, group, spec)
  end
end

function M.setup(opts)
  opts = opts or {}
  local c = merge_palette(opts.palette)

  -- Core language polish
  set_capture("@type", { fg = c.blue }, opts)
  set_capture("@type.builtin", { fg = c.aqua }, opts)
  set_capture("@function", { fg = c.blue }, opts)
  set_capture("@function.call", { fg = c.blue }, opts)
  set_capture("@property", { fg = c.fg }, opts)
  set_capture("@label", { fg = c.aqua, bold = true }, opts)
  set_capture("@string.special", { fg = c.orange }, opts)

  -- Comptime
  set_capture("@keyword.comptime", { fg = c.orange, bold = true }, opts)
  set_capture("@function.comptime", { fg = c.red, bold = true }, opts)
  set_capture("@variable.parameter.comptime", { fg = c.yellow, italic = true }, opts)

  -- Builtins and intrinsics
  set_capture("@module.builtin", { fg = c.cyan, italic = true }, opts)
  set_capture("@function.builtin", { fg = c.green, bold = true }, opts)
  set_capture("@constant.builtin", { fg = c.aqua, bold = true }, opts)
  set_capture("@variable.builtin", { fg = c.cyan }, opts)
  set_capture("@keyword.directive", { fg = c.red, bold = true }, opts)

  -- Formal verification and ghosts
  set_capture("@keyword.fv", { fg = c.green, italic = true }, opts)
  set_capture("@keyword.ghost", { fg = c.purple, italic = true }, opts)
  set_capture("@function.ghost", { fg = c.purple }, opts)
  set_capture("@variable.ghost", { fg = c.purple }, opts)

  -- Region-specific cues
  set_capture("@keyword.storage", { fg = c.yellow, bold = true }, opts)
  set_capture("@variable.storage", { fg = c.yellow }, opts)

  set_capture("@keyword.memory", { fg = c.green, bold = true }, opts)
  set_capture("@variable.memory", { fg = c.green }, opts)

  set_capture("@keyword.tstore", { fg = c.red, bold = true }, opts)
  set_capture("@variable.tstore", { fg = c.red }, opts)

  set_capture("@keyword.calldata", { fg = c.cyan, bold = true }, opts)
  set_capture("@variable.calldata", { fg = c.cyan }, opts)
end

function M.autocmd(opts)
  local group = vim.api.nvim_create_augroup("OraHighlights", { clear = true })
  vim.api.nvim_create_autocmd("ColorScheme", {
    group = group,
    callback = function()
      M.setup(opts)
    end,
  })
  M.setup(opts)
end

return M

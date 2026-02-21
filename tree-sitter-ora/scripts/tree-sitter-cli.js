#!/usr/bin/env node

const fs = require('node:fs')
const path = require('node:path')
const { spawn } = require('node:child_process')

function fail(msg) {
  process.stderr.write(`${msg}\n`)
  process.exit(1)
}

let pkgRoot
try {
  const localRoot = path.resolve(__dirname, '..')
  const pkgJsonPath = require.resolve('tree-sitter-cli/package.json', {
    paths: [localRoot, process.cwd()],
  })
  pkgRoot = path.dirname(pkgJsonPath)
} catch {
  fail(
    [
      'error: tree-sitter-cli is not installed.',
      'hint: run one of:',
      '  pnpm install',
      '  npm install',
    ].join('\n'),
  )
}

const cliPath = path.join(pkgRoot, 'cli.js')
const exePath = path.join(pkgRoot, process.platform === 'win32' ? 'tree-sitter.exe' : 'tree-sitter')

if (!fs.existsSync(exePath)) {
  fail(
    [
      'error: tree-sitter-cli binary is missing.',
      'This usually means dependency build scripts were not executed.',
      '',
      'For pnpm v10, run:',
      '  pnpm approve-builds tree-sitter-cli',
      '  pnpm rebuild tree-sitter-cli',
      '',
      'Or reinstall with scripts enabled:',
      '  pnpm install --ignore-scripts=false',
      '',
      'If you are offline, reconnect and rerun the commands above.',
    ].join('\n'),
  )
}

const args = process.argv.slice(2)
const child = spawn(process.execPath, [cliPath, ...args], {
  stdio: 'inherit',
  env: process.env,
})

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal)
    return
  }
  process.exit(code ?? 1)
})

child.on('error', (err) => {
  fail(`error: failed to launch tree-sitter-cli: ${err.message}`)
})

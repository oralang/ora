# Native Sanitizer Builds

Ora's normal build uses the installed `vendor/mlir` libraries. For native
MLIR/OraToSIR reliability work, the compiler can also build the project-owned
SIR and Ora dialect libraries with Clang sanitizers while reusing the normal
MLIR dependency.

## Run

```sh
zig build test-mlir -Dnative-sanitize=address --summary all
```

Supported values:

- `none` (default)
- `address` or `asan`
- `undefined` or `ubsan`
- `address,undefined` or `asan-ubsan`

Sanitized dialect artifacts are installed beside the normal MLIR prefix:

- `vendor/mlir-asan`
- `vendor/mlir-ubsan`
- `vendor/mlir-asan-ubsan`

The `ora` executable links those prefixes before `vendor/mlir/lib`, so
`MLIROraDialectC` and `MLIRSIRDialect` come from the sanitizer build while
`MLIR-C` and the rest of MLIR still come from the normal install.

## Scope

This is intentionally a native-dialect reliability path, not a full LLVM/MLIR
sanitizer rebuild. It covers Ora-owned C++ lowering code, including OraToSIR,
without making every sanitizer run rebuild LLVM.

If the bug appears to live inside MLIR itself, rebuild MLIR separately with the
same sanitizer flags and use that as `vendor/mlir`.

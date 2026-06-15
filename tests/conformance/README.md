# Ora conformance corpus

`zig build test-conformance` runs every `*.spec.toml` file in this directory.
Every `*.ora` file must have a matching sidecar or be listed in `SKIP`; an
orphan source is a test failure.

The current corpus ports the static storage-smoke cases from
`scripts/bytecode_storage_equivalence_smoke.sh`: scalar storage, address and
integer map keys, nested mappings, structs under maps, and packed bitfields.

The sidecar ABI encoder accepts static ABI words (`uintN`/`intN`, `address`,
`bool`, `bytes1` through `bytes32`) plus dynamic ABI head/tail encoding for
arrays and tuples. The live corpus includes `uint256[]` calldata; tuple-array
encoding is unit-tested in the harness, and mixed dynamic tuple calldata is
covered by the dispatcher corpus.

Still open from the shell smoke matrix:

- storage slice struct fixtures
  (`storage_slice_struct.ora` and `storage_slice_dynamic_struct.ora` are
  explicit in `SKIP`);
- multi-contract deploy/call scenarios;
- raw-slot parity for fixed-bytes map keys and negative signed narrow map keys.

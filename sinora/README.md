# Sinora

Sinora is Ora's owned Zig implementation of the SIR backend migration path.

This directory is intentionally separate from the old Rust Plank subprocess
path. The first goal was a small, honest backend foundation that could ingest
the SIR text Ora already emits, diagnose malformed IR without panics, and grow
into a dual-backend comparison and bytecode-emission harness. It now owns the
Plank SIR backend surface Ora depends on: parsing/rendering, legality,
analyses, SSA/critical-edge transforms, optimization passes, debug/release
codegen, and bytecode/gas comparison harnesses.

Current slice:

- SIR text data model for functions, blocks, instructions, terminators, and
  data segments.
- Line-oriented parser for Ora-emitted SIR text.
- Normalized renderer for parsed SIR.
- Structural legality checks for blocks, values, terminator targets, and
  duplicate definitions.
- Plank-style pass framework with cached analyses and invalidation masks:
  legalizer, reachability, predecessors, reverse post-order, dominators,
  dominance frontiers, def-use, local liveness, allocation liveness, function
  effects, basic-block ownership, and CFG in/out bundling.
- Plank-style transforms for critical-edge splitting and sealed-block SSA
  construction, including pre-SSA function-entry regularization.
- Plank-style SIR optimization passes: SCCP, copy propagation, unused-operation
  elimination, defragmenter, and switch peephole. The pass manager also exposes
  the same `s/c/u/d/l` optimization-string ordering surface for tests and
  future production pipeline decisions.
- Opcode metadata for the Plank SIR operation surface, including memory-width
  mnemonics such as `mload256`/`mstore256`.
- Internal-call target and signature-shape checks inferred from SIR function
  entry inputs and `iret` block outputs.
- A small bytecode assembler and conservative debug emitter for one function,
  including minimal-width label patching, Plank-style worklist block emission,
  jumps, branches, switches, and block input/output transfer slots.
- Debug bump allocation for `malloc`, `mallocany`, `freeptr`, `salloc`, and
  `sallocany` inside the one-function emitter.
- Reachable debug `icall`/`iret` lowering with return-destination slots for
  non-recursive internal call graphs.
- Debug deployment offset patching for `runtime_start_offset`,
  `init_end_offset`, and `runtime_length` when emitting the `init` root with a
  `main` runtime function.
- SIR `data` segment parsing/rendering/legality plus debug `data_offset`
  lowering with appended code bytes.
- Plank-compatible debug static memory layout, runtime-relative code offsets,
  and generic memory load/store lowering for bytecode parity on supported
  debug slices.
- Plank-compatible effect model and effect-aware release scheduling over
  memory, returndata, accounts, persistent storage, transient storage,
  revert/terminate, allocation, and logs.
- Narrow release deployment codegen for the standard `init` runtime copier and
  the no-arg `__ora_user_init` constructor wrapper with empty or straight-line
  storage-writing init bodies, the one-argument constructor ABI tail decoder
  that stores the decoded value through `__ora_user_init`, plus a narrow
  entry-branch-to-terminal-blocks
  runtime shape and the generated default-only selector dispatcher shape,
  including selector cases that jump directly to terminal blocks, emitted
  through a small stack state emitter. Selector cases can also lower no-arg
  internal calls that return two values, including a strict constant-guard
  callee branch shape with one `iret` side and one halt side, a constant
  guard/fallback return shape with a custom-error edge, and three- or four-word
  aggregate constructor returns, plus nested aggregate storage reads that
  project a malloc-backed ABI word return through Plank-style
  post-free-pointer scheduler scratch. Selector entries can reserve free memory
  with a `codesize + 0x20`, `codesize + 0x40`, or `codesize + 0x60`
  dynamic offset for malloc-backed ABI return payloads. They can also
  lower no-arg void internal
  calls that return an empty ABI payload as
  `return 0 0`, plus a strict one-argument void public case with calldata
  length guard, `calldataload 0x4`, and an empty ABI return, and a strict
  two-argument void public case with calldata length guard, `calldataload 0x4`,
  `calldataload 0x24`, and an empty ABI return, plus a strict three-argument
  raw-word, bool-guarded, masked-value public case with distinct ABI decode
  reverts for empty ABI returns and one malloc-backed word return from the
  masked third argument. Selector cases can also lower
  one-argument internal calls that return one malloc-backed ABI
  word from the argument, a computed ABI word when the public argument is
  unused by the callee, a computed ABI word over the public argument and
  constants, a masked bit-not style word that XORs the masked argument with
  its ABI mask, a strict ABI-validated single-key storage mapping read that
  hashes `key || slot` and returns the loaded word, or signed one-argument
  widening returns that use `signextend`
  decode guards and `shl`/`sar` sign recovery, plus strict two-argument
  internal calls that return one malloc-backed ABI word from the first
  argument directly or from a single binary operation over both arguments,
  including strict masked ABI decode guard cases with `const`/`large_const`
  masks that share the generated ABI decode revert edge, and pre-masked
  `and`/`or`/`xor`/`div`/`mod`/`shl`/`shr` callees that mask both inputs before the
  operation and mask the result, plus a strict two-argument mapping copy callee
  that reads one `key || slot`, writes the loaded word to another `key || slot`,
  and returns the copied word, plus strict booleanized `lt`/`gt` and
  inclusive `lt|eq`/`gt|eq` comparison-return callees for masked narrow integer
  arguments and unmasked `u256` arguments, first-argument masked two-argument
  void calls into storage/transient-storage side-effect callees, and strict
  `eq`/`ne` return callees
  for unmasked `u256`, masked narrow integer/address, and ABI-validated bool
  arguments. Signed narrow ABI decode guards using `signextend` are supported
  for two-argument wrapping `add`/`sub`/`mul` and signed-value `shl`/`shr`
  return callees that mask and sign-extend the returned word, plus signed
  `slt`/`sgt` and inclusive `slt|eq`/`sgt|eq` comparison-return callees for
  unmasked `i256` and sign-recovered narrow arguments. Multiple shared ABI
  decode revert blocks reserve distinct static return words while preserving
  Plank's emitted order. One-argument lower-bound public guards also cover a
  strict branch-join payload proof that carries either the original argument or
  the bound constant into a shared ABI word return block, plus a strict
  checked-add proof callee with max-value guard, constant custom-error revert,
  and malloc-backed ABI word return, a checked-add branch that writes either a
  constant or the checked sum through a memory join before returning, and a
  strict one-argument bounded
  multiplication proof callee that reuses a range proof to return the product
  or a constant custom error.
  No-argument return callees also cover a strict counted loop that carries
  index/payload state through a loop header and returns the final payload word,
  and one-argument return callees cover strict range/while counted loops that
  carry index plus accumulator state into a malloc-backed ABI word return.
  No-argument void callees also cover strict checked storage increment and
  decrement shapes with Plank-style custom-error reverts.
  Void callees may be single-block functions or
  source-ordered empty jump chains ending in `iret`, and one-argument void
  callees may be no-op or side-effecting storage/transient-store functions,
  including arithmetic that leaves dead stack values before `iret`, a strict
  checked storage add with input/storage/add overflow guards and Plank-style
  custom-error reverts, a strict owner-guarded masked-address `sstore` update
  that writes only when `caller` matches the stored owner, plus a
  strict switch-style storage write ladder, with or without an upfront range
  guard, that joins through a shared `iret`.
  The subset covers numeric constants,
  supported fixed EVM opcodes from the SIR opcode table,
  byte-aligned `mloadN`/`mstoreN`, dynamic
  `malloc`/`mallocany`, static `salloc`/`sallocany`, `freeptr`, `data_offset`,
  `copy`, `noop`, bounded scheduler-style stack spills for deep locals that do
  not need runtime dynamic allocation, plus selector no-arg straight-line
  callees that reserve Plank-style scratch after the free-pointer slot before
  malloc-backed returns, and halt terminators `return`, `revert`, `stop`,
  `invalid`, and `selfdestruct`. This subset is byte-for-byte compared
  against Rust Plank and rejects unsupported shapes instead of approximating the
  full release scheduler.
- Small CLI for parsing/checking SIR files and comparing acceptance with the
  pinned Rust Plank SIR oracle.

Explicit non-goals for this slice:

- No Plank frontend.
- No default backend integration.

Production pipeline decision for this port:

- Ora already emits block-parameter SSA SIR, so the release bytecode path does
  not run `SSATransform` by default. The pass is still owned and tested here so
  non-Ora/pre-SSA SIR can be normalized deliberately when needed.
- The release bytecode path does run critical-edge splitting before scheduling,
  because Plank's release scheduler expects normalized CFG value transfers.
- The release bytecode path uses the effect-aware op graph, not the old
  source-order-only scheduler.
- SCCP/copy-prop/DCE/defragment/switch-peephole are ported and selectable
  through the pass manager, but they are not enabled automatically in Ora's
  production codegen until a separate metrics/semantics review decides the
  optimization sequence.

Run locally:

```sh
cd sinora
zig build test
zig build
zig build test-fixtures-release
zig build test-corpus-debug
zig build test-corpus-release
zig build test-corpus
zig build run -- compare-corpus --release --details ../artifacts
zig build run -- ../artifacts/smoke/sir/smoke.sir
zig build run -- compare ../artifacts/smoke/sir/smoke.sir
zig build run -- compare --debug --json ../artifacts/smoke/sir/smoke.sir
zig build run -- compare --release --json fixtures/init_runtime_offsets.sir
zig build run -- compare --release --json fixtures/revert_runtime.sir
zig build run -- compare --release --json fixtures/invalid_runtime.sir
zig build run -- compare --release --json fixtures/selfdestruct_runtime.sir
zig build run -- compare --release --json fixtures/data_offset_return.sir
zig build run -- compare --release --json fixtures/mallocany_runtime_return.sir
zig build run -- compare --release --json fixtures/salloc_runtime_return.sir
zig build run -- compare --release --json fixtures/sallocany_runtime_return.sir
zig build run -- compare --release --json fixtures/sallocany_two_runtime_return.sir
zig build run -- compare --release --json fixtures/static_dynamic_runtime_return.sir
zig build run -- compare --release --json fixtures/copy_noop_runtime_return.sir
zig build run -- compare --release --json fixtures/mcopy_runtime_return.sir
zig build run -- compare --release --json fixtures/flat_runtime_return.sir
zig build run -- compare --release --json fixtures/sub_runtime_return.sir
zig build run -- compare --release --json fixtures/mload_runtime_return.sir
zig build run -- compare --release --json fixtures/narrow_memory_runtime_return.sir
zig build run -- compare --release --json fixtures/narrow_memory16_runtime_return.sir
zig build run -- compare --release --json fixtures/branch_terminal_runtime_return.sir
zig build run -- compare --release --json fixtures/branch_terminal_zero_live_return.sir
zig build run -- compare --release --json fixtures/deep_stack_spill_runtime_return.sir
zig build run -- compare --release --json fixtures/constructor_no_arg_user_init_empty.sir
zig build run -- compare --release --json fixtures/constructor_no_arg_user_init_storage.sir
zig build run -- compare --release --json fixtures/constructor_one_arg_user_init_storage.sir
zig build run -- compare --release --json fixtures/default_dispatcher_runtime_revert.sir
zig build run -- compare --release --json fixtures/selector_case_terminal_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_codesize20_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_branch_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_branch_jump_bool_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_branch_jump_word_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_runtime_caller_guard_void_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_void_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_void_jump_chain_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_void_branch_join_tstore.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_void_tload_guard_tstore_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_void_compact_constant_branch_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_aggregate_pair_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_aggregate_triple_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_nested_aggregate_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_storage_tag_switch_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_constant_guard_chain_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_constant_guard_chain_distinct_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_guard_fallback_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_interleaved_error_guard_chain_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_compact_interleaved_error_guard_chain_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_builtins_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_error_union_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_dynamic_abi_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_internal_word_discard_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_counted_loop_return.sir
zig build run -- compare --release --json fixtures/selector_no_arg_icall_checked_storage_update_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_counted_loop_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_void_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_void_noop_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_void_storage_tstore_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_void_constant_branch_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_void_constant_branch_sstore_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_void_lock_guard_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_void_owner_guard_sstore_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_void_switch_sstore_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_void_guarded_switch_sstore_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_void_direct_bounded_proof_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_void_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_void_noop_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_first_masked_void_storage_tstore_return.sir
zig build run -- compare --release --json fixtures/selector_mixed_icall_log_return.sir
zig build run -- compare --release --json fixtures/selector_three_arg_first_two_masked_icall_log_return.sir
zig build run -- compare --release --json fixtures/selector_three_arg_raw_bool_masked_void_return.sir
zig build run -- compare --release --json fixtures/selector_three_arg_raw_bool_masked_void_storage_return.sir
zig build run -- compare --release --json fixtures/selector_three_arg_named_payload_storage_return.sir
zig build run -- compare --release --json fixtures/selector_three_arg_named_payload_field_return.sir
zig build run -- compare --release --json fixtures/selector_three_arg_raw_bool_masked_word_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_word_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_nested_identity_chain_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_computed_word_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_arg_computed_word_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_arg_bool_comparison_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_masked_bit_not_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_masked_void_ignored_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_storage_read_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_return_internal_void_call.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_masked_internal_storage_reads_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_static_memory_internal_word_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_static_memory_struct_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_static_memory_array_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_runtime_guard_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_memory_switch_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_lower_bound_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_lower_bound_branch_join_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_bounded_void_proof_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_bounded_void_strict_proof_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_bounded_void_multiple_proof_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_checked_add_proof_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_checked_add_memory_join_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_checked_storage_add_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_bounded_mul_proof_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_address_nonzero_proof_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_bool_guard_memory_join_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_signed_negate_overflow_return.sir
zig build run -- compare --release --json fixtures/selector_one_arg_icall_signed_widening_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_word_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_first_arg_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_aggregate_pair_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_input_select_branch_jump_return.sir
zig build run -- compare --release --json fixtures/selector_masked_void_supported_paths_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_signed_overflow_add_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_signed_overflow_sub_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_signed_mul_overflow_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_percent_overflow_proof_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_masked_word_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_mapping_copy_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_nested_generic_helper_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_first_bool_guard_second_word_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_mixed_masked_unmasked_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_masked_void_noop_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_second_bool_guard_void_noop_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_selected_payload_branch_join_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_selected_stateful_payload_branch_join_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_selected_static_pointer_branch_join_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_all_width_add_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_all_width_sub_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_all_width_mul_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_all_width_shift_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_all_width_bitwise_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_all_width_div_mod_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_all_width_comparison_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_eq_ne_all_types_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_signed_wrapping_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_signed_comparison_return.sir
zig build run -- compare --release --json fixtures/selector_two_arg_icall_word_many_ops_return.sir
zig build run -- compare --release --json fixtures/selector_mixed_icall_math_storage_return.sir
zig build run -- emit-release fixtures/deep_stack_spill_runtime_return.sir
zig build run -- compare-corpus --debug --json ../artifacts
zig build run -- compare-corpus --release --json ../artifacts
zig build run -- emit-debug fixtures/flat_return.sir main
zig build run -- emit-debug fixtures/cfg_branch_return.sir main
zig build run -- emit-debug fixtures/alloc_return.sir main
zig build run -- emit-debug fixtures/icall_return.sir main
zig build run -- emit-debug fixtures/data_offset_return.sir main
zig build run -- emit-debug fixtures/init_runtime_offsets.sir init
```

`compare` runs Sinora parse/legality checks and the pinned Rust Plank SIR oracle
against the same SIR text. The command defaults to Rust Plank release mode and
text output. Release mode emits and compares full deployment bytecode through
the generic Plank-port backend, which is byte-for-byte equal to Rust Plank on
the entire artifact corpus (586/586). With `--debug`, Sinora emits the debug
deployment package and reports `both-accept-bytecode-equal`,
`bytecode-mismatch`, or `zig-codegen-rejected`. Use `--json` for stable harness
output.

`compare-corpus` recursively compares every `.sir` file under a directory and
exits non-zero if any file is not byte-equal to Rust Plank. The gate is ARMED
for every mode: only `both-accept-bytecode-equal` passes, so a codegen gap
(`both-accept-codegen-pending`) or a wrong bytecode (`bytecode-mismatch`) fails
the gate and catches regressions. The `fixtures/` directory is intentionally not
a corpus because some fixtures are single-root `main` tests that Rust Plank's CLI
rejects without an `init` entry point, while the release deployment fixtures are
focused bytecode parity tests.

The named build step `test-fixtures-release` checks the focused release
deployment fixtures for byte-for-byte parity with Rust Plank. The named build
steps `test-corpus-debug`, `test-corpus-release`, and `test-corpus` run the
artifact corpus gates against `../artifacts` for repeatable CI/local checks.
Before adding new release shapes, use `docs/release-pending-inventory.md` to
pick recurring pending families rather than one-off corpus files.

`emit-debug` is intentionally narrow. It emits hex bytecode for one root
using supported straight-line operations, intra-function CFG, and debug bump
allocation, plus reachable non-recursive internal calls. Selecting `init` emits
a debug deployment package with patched runtime offsets and appended `data`
segments. It reports unsupported SIR instead of trying to fake recursive calls,
source maps, shared init/runtime helper re-emission, release-quality
init/runtime packaging, or release scheduling.

`emit-release` emits release deployment hex through the generic Plank-port
backend (byte-equal to Rust Plank across the artifact corpus) and reports
unsupported SIR instead of falling back to Rust Plank. `emit-release-generic` is
a retained alias for the same backend.

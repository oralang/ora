What this means for the Lean model (start here, not syntax)

  1. Region → a 5-element enum (stack, memory, storage, transient, calldata); keep Provenance as a second enum, not folded in.
  2. Integers: model as a finite enum of the 13 (cleanest for the kernel) or (signed, bits) with a well-formedness predicate IsBuiltinInt restricting to the 13. The finite-enum is
  simpler and matches BuiltinTypeId.
  3. address ≠ u160 — distinct constructors even though both are 160-bit.
  4. fixed_bytes → len : Fin 32 (1..32) or {n // 1 ≤ n ≤ 32}.
  5. Recursive Ty carries: array/slice/map/tuple/struct/error_union/refinement/function as the recursive constructors (*const Type ⇒ Lean recursion).
  6. Exclude from the well-formed universe (or mark as a separate "elaboration state"): unknown, named. Keep never, void. Decide whether comptime_integer lives in the runtime type
  universe or a separate comptime universe (I'd put it in a ComptimeTy and require lowering — mirrors the compiler).
  7. Two layers worth separating: a BaseTy (the scalars A) and the recursive Ty (B–H over BaseTy), so refinements/aggregates compose over a clean primitive core.

  So the natural first Lean modules are: Ora/Types/Region.lean (regions + provenance), Ora/Types/Prim.lean (the 13 ints + bool + address + bytesN + string/bytes/void),
  Ora/Types/Ty.lean (the recursive universe B–H), and Ora/Types/WF.lean (well-formedness: which TypeKinds are admissible, integer-width restriction, fixed-bytes bound, no
  unknown/named). Syntax/typing/semantics layer on top later
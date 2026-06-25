/-
GENERATED — DATA ONLY.  Do NOT edit by hand and do NOT add any `theorem`,
`lemma`, `example`, `axiom`, `sorry`, `instance`, `macro`, attribute, or extra
`import` to this file. It contains only `def … := <literal>` declaration rows
emitted from the compiler. The TRUSTED checks live in `Ora/SyncDeclEnv.lean`.

Regenerate with `scripts/check-formal-declenv-sync.sh`. Source:
src/formal/emit_declenv_snapshot.zig, src/types/semantic.zig.
-/

namespace Ora.Generated

def compilerDeclKinds : List (String × String) :=
  [("Point", "struct_"),
   ("Color", "enum_"),
   ("Flags", "bitfield"),
   ("Vault", "contract"),
   ("Token", "resource_domain"),
   ("Digest", "resource_domain"),
   ("Buffer", "resource_domain")]

def compilerResourceCarriers : List (String × String × Option String) :=
  [("Token", "integer", some "u256"),
   ("Digest", "fixed_bytes", some "bytes32"),
   ("Buffer", "slice", none)]

end Ora.Generated

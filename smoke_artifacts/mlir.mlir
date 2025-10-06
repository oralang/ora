"builtin.module"() ({
  "ora.contract"() ({
    "ora.storage"() ({
      %15 = "ora.global"() {init = 0 : i256, sym_name = "counter"} : () -> i256
      %16 = "ora.global"() {init = false, sym_name = "status"} : () -> i1
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "init"}> ({
      %13 = "arith.constant"() <{value = 42 : i256}> : () -> i256
      "ora.sstore"(%13) {global = "counter"} : (i256) -> ()
      %14 = "arith.constant"() <{value = true}> : () -> i1
      "ora.sstore"(%14) {global = "status"} : (i1) -> ()
      "func.return"() : () -> ()
    }) {ora.init = true, ora.visibility = "private"} : () -> ()
    "func.func"() <{function_type = () -> i256, sym_name = "increment"}> ({
      %9 = "ora.sload"() {global = "counter"} : () -> i256
      %10 = "arith.constant"() <{value = 1 : i256}> : () -> i256
      %11 = "arith.addi"(%9, %10) <{overflowFlags = #arith.overflow<none>}> : (i256, i256) -> i256
      "ora.sstore"(%11) {global = "counter"} : (i256) -> ()
      %12 = "ora.sload"() {global = "counter"} : () -> i256
      "func.return"(%12) : (i256) -> ()
    }) {ora.visibility = "private"} : () -> ()
    "func.func"() <{function_type = () -> i1, sym_name = "checkStatus"}> ({
      %5 = "ora.sload"() {global = "status"} : () -> i1
      %6 = "arith.constant"() <{value = true}> : () -> i1
      %7 = "arith.constant"() <{value = false}> : () -> i1
      %8 = "scf.if"(%5) ({
        "scf.yield"(%6) : (i1) -> ()
      }, {
        "scf.yield"(%7) : (i1) -> ()
      }) : (i1) -> i1
      "func.return"(%8) : (i1) -> ()
    }) {ora.visibility = "private"} : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "reset"}> ({
      %0 = "ora.sload"() {global = "counter"} : () -> i256
      %1 = "arith.constant"() <{value = 100 : i256}> : () -> i256
      %2 = "arith.cmpi"(%0, %1) <{predicate = 8 : i64}> : (i256, i256) -> i1
      %3 = "arith.constant"() <{value = 0 : i256}> : () -> i256
      %4 = "arith.constant"() <{value = false}> : () -> i1
      "scf.if"(%2) ({
        "ora.sstore"(%3) {global = "counter"} : (i256) -> ()
        "ora.sstore"(%4) {global = "status"} : (i1) -> ()
        "scf.yield"() : () -> ()
      }, {
        "scf.yield"() : () -> ()
      }) : (i1) -> ()
      "func.return"() : () -> ()
    }) {ora.visibility = "private"} : () -> ()
  }, {
  ^bb0:
  }) {ora.contract_decl = true, sym_name = "SimpleContract"} : () -> ()
}) : () -> ()

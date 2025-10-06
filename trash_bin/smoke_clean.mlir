"builtin.module"() ({
  "ora.contract"() ({
    %0 = "ora.global"() {init = 0 : i256, sym_name = "counter"} : () -> i256
    %1 = "ora.global"() {init = false, sym_name = "status"} : () -> i1
    "func.func"() <{function_type = () -> (), sym_name = "init"}> ({
      %15 = "arith.constant"() <{value = 42 : i256}> : () -> i256
      "ora.sstore"(%15) {global = "counter"} : (i256) -> ()
      %16 = "arith.constant"() <{value = true}> : () -> i1
      "ora.sstore"(%16) {global = "status"} : (i1) -> ()
      "func.return"() : () -> ()
    }) {ora.init = true, ora.visibility = "private"} : () -> ()
    "func.func"() <{function_type = () -> i256, sym_name = "increment"}> ({
      %11 = "ora.sload"() {global = "counter"} : () -> i256
      %12 = "arith.constant"() <{value = 1 : i256}> : () -> i256
      %13 = "arith.addi"(%11, %12) <{overflowFlags = #arith.overflow<none>}> : (i256, i256) -> i256
      "ora.sstore"(%13) {global = "counter"} : (i256) -> ()
      %14 = "ora.sload"() {global = "counter"} : () -> i256
      "func.return"(%14) : (i256) -> ()
    }) {ora.visibility = "private"} : () -> ()
    "func.func"() <{function_type = () -> i1, sym_name = "checkStatus"}> ({
      %7 = "ora.sload"() {global = "status"} : () -> i1
      %8 = "arith.constant"() <{value = true}> : () -> i1
      %9 = "arith.constant"() <{value = false}> : () -> i1
      %10 = "scf.if"(%7) ({
        "scf.yield"(%8) : (i1) -> ()
      }, {
        "scf.yield"(%9) : (i1) -> ()
      }) : (i1) -> i1
      "func.return"(%10) : (i1) -> ()
    }) {ora.visibility = "private"} : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "reset"}> ({
      %2 = "ora.sload"() {global = "counter"} : () -> i256
      %3 = "arith.constant"() <{value = 100 : i256}> : () -> i256
      %4 = "arith.cmpi"(%2, %3) <{predicate = 8 : i64}> : (i256, i256) -> i1
      %5 = "arith.constant"() <{value = 0 : i256}> : () -> i256
      %6 = "arith.constant"() <{value = false}> : () -> i1
      "scf.if"(%4) ({
        "ora.sstore"(%5) {global = "counter"} : (i256) -> ()
        "ora.sstore"(%6) {global = "status"} : (i1) -> ()
        "scf.yield"() : () -> ()
      }, {
        "scf.yield"() : () -> ()
      }) : (i1) -> ()
      "func.return"() : () -> ()
    }) {ora.visibility = "private"} : () -> ()
  }) {ora.contract_decl = true, sym_name = "SimpleContract"} : () -> ()
}) : () -> ()

module {
  ora.contract @ArithmeticTest {
    ora.global "counter" : i256 {gas_cost = 0 : i64} loc("arithmetic_test.ora":3:17)

    func.func @init() attributes {ora.abi_params = [], ora.init = true, ora.selector = "0xe1c7392a", ora.visibility = "pub"} {
      %c0_i256 = arith.constant {gas_cost = 0 : i64} 0 : i256 loc("arithmetic_test.ora":6:19)
      ora.sstore %c0_i256, "counter" : i256 {gas_cost = 20000 : i64} loc("arithmetic_test.ora":6:9)
      ora.return {gas_cost = 8 : i64} loc("arithmetic_test.ora":5:12)
    } loc("arithmetic_test.ora":5:12)

    func.func @increment() attributes {ora.abi_params = [], ora.contract_level = "full", ora.formal = true, ora.requires_count = 1 : i32, ora.selector = "0xd09de08a", ora.verification = true, ora.verification_context = "function_contract", ora.visibility = "pub"} {
      %c1_i256 = arith.constant {gas_cost = 0 : i64} 1 : i256 loc("arithmetic_test.ora":12:29)
      %c_neg1_i256 = arith.constant {gas_cost = 0 : i64} -1 : i256 loc("arithmetic_test.ora":10:42)
      %counter = ora.sload "counter" : i256 {gas_cost = 2100 : i64} loc("arithmetic_test.ora":10:18)

      %0 = arith.cmpi ult, %counter, %c_neg1_i256 : i256 loc("arithmetic_test.ora":10:26)
      cf.assert %0, "Precondition 0 failed" {ora.formal = true, ora.precondition_index = 0 : i32, ora.requires = true, ora.verification = true, ora.verification_context = "function_precondition"} loc("arithmetic_test.ora":10:26)
      %counter_0 = ora.sload "counter" : i256 {gas_cost = 2100 : i64} loc("arithmetic_test.ora":12:19)
      %1 = arith.addi %counter_0, %c1_i256 {gas_cost = 3 : i64} : i256 loc("arithmetic_test.ora":12:27)

      %2 = arith.cmpi uge, %1, %counter_0 : i256 loc("arithmetic_test.ora":12:27)
      cf.assert %2, "checked addition overflow" loc("arithmetic_test.ora":12:27)
      ora.sstore %1, "counter" : i256 {gas_cost = 20000 : i64} loc("arithmetic_test.ora":12:9)
      ora.return {gas_cost = 8 : i64} loc("arithmetic_test.ora":9:12)
    } loc("arithmetic_test.ora":9:12)

    func.func @getCounter() -> (i256 {ora.type = i256}) attributes {ora.abi_params = [], ora.abi_return = "uint256", ora.selector = "0x8ada066e", ora.visibility = "pub"} {
      %counter = ora.sload "counter" : i256 {gas_cost = 2100 : i64} loc("arithmetic_test.ora":16:16)
      ora.return %counter : i256 {gas_cost = 8 : i64} loc("arithmetic_test.ora":16:23)
    } loc("arithmetic_test.ora":15:12)

    func.func @multiply(%arg0: i256 {ora.type = i256} loc("arithmetic_test.ora":19:21), %arg1: i256 {ora.type = i256} loc("arithmetic_test.ora":19:30)) -> (i256 {ora.type = i256}) attributes {ora.abi_params = ["uint256", "uint256"], ora.abi_return = "uint256", ora.contract_level = "full", ora.formal = true, ora.requires_count = 2 : i32, ora.selector = "0x165c4a16", ora.verification = true, ora.verification_context = "function_contract", ora.visibility = "pub"} {
      %true = arith.constant {gas_cost = 0 : i64} true loc("arithmetic_test.ora":23:18)
      %c0_i256 = arith.constant {gas_cost = 0 : i64} 0 : i256 loc("arithmetic_test.ora":23:18)

      %c340282366920938463463374607431768211455_i256 = arith.constant 340282366920938463463374607431768211455 : i256 loc(unknown)
      %0 = arith.cmpi ule, %arg0, %c340282366920938463463374607431768211455_i256 : i256 loc("arithmetic_test.ora":20:20)
      cf.assert %0, "Precondition 0 failed" {ora.formal = true, ora.precondition_index = 0 : i32, ora.requires = true, ora.verification = true, ora.verification_context = "function_precondition"} loc("arithmetic_test.ora":20:20)

      %1 = arith.cmpi ule, %arg1, %c340282366920938463463374607431768211455_i256 : i256 loc("arithmetic_test.ora":21:20)
      cf.assert %1, "Precondition 1 failed" {ora.formal = true, ora.precondition_index = 1 : i32, ora.requires = true, ora.verification = true, ora.verification_context = "function_precondition"} loc("arithmetic_test.ora":21:20)
      %2 = arith.muli %arg0, %arg1 {gas_cost = 5 : i64} : i256 loc("arithmetic_test.ora":23:18)

      %3 = arith.cmpi ne, %arg1, %c0_i256 : i256 loc("arithmetic_test.ora":23:18)
      %4 = arith.divui %2, %arg1 {gas_cost = 5 : i64, ora.guard_internal = "true"} : i256 loc("arithmetic_test.ora":23:18)

      %5 = arith.cmpi ne, %4, %arg0 : i256 loc("arithmetic_test.ora":23:18)
      %6 = arith.andi %5, %3 : i1 loc("arithmetic_test.ora":23:18)
      %7 = arith.xori %6, %true : i1 loc("arithmetic_test.ora":23:18)
      cf.assert %7, "checked multiplication overflow" loc("arithmetic_test.ora":23:18)
      ora.return %2 : i256 {gas_cost = 8 : i64} loc("arithmetic_test.ora":23:21)
    } loc("arithmetic_test.ora":19:12)

    func.func @add(%arg0: i256 {ora.type = i256} loc("arithmetic_test.ora":26:16), %arg1: i256 {ora.type = i256} loc("arithmetic_test.ora":26:25)) -> (i256 {ora.type = i256}) attributes {ora.abi_params = ["uint256", "uint256"], ora.abi_return = "uint256", ora.contract_level = "full", ora.formal = true, ora.requires_count = 1 : i32, ora.selector = "0x771602f7", ora.verification = true, ora.verification_context = "function_contract", ora.visibility = "pub"} {
      %c_neg1_i256 = arith.constant {gas_cost = 0 : i64} -1 : i256 loc("arithmetic_test.ora":27:37)
      %0 = arith.subi %c_neg1_i256, %arg1 {gas_cost = 3 : i64} : i256 loc("arithmetic_test.ora":27:46)

      %1 = arith.cmpi ule, %arg1, %c_neg1_i256 : i256 loc("arithmetic_test.ora":27:46)
      cf.assert %1, "checked subtraction overflow" loc("arithmetic_test.ora":27:46)
      %2 = arith.cmpi ule, %arg0, %0 : i256 loc("arithmetic_test.ora":27:20)
      cf.assert %2, "Precondition 0 failed" {ora.formal = true, ora.precondition_index = 0 : i32, ora.requires = true, ora.verification = true, ora.verification_context = "function_precondition"} loc("arithmetic_test.ora":27:20)
      %3 = arith.addi %arg0, %arg1 {gas_cost = 3 : i64} : i256 loc("arithmetic_test.ora":29:18)

      %4 = arith.cmpi uge, %3, %arg0 : i256 loc("arithmetic_test.ora":29:18)
      cf.assert %4, "checked addition overflow" loc("arithmetic_test.ora":29:18)
      ora.return %3 : i256 {gas_cost = 8 : i64} loc("arithmetic_test.ora":29:21)
    } loc("arithmetic_test.ora":26:12)

    func.func @subtract(%arg0: i256 {ora.type = i256} loc("arithmetic_test.ora":32:21), %arg1: i256 {ora.type = i256} loc("arithmetic_test.ora":32:30)) -> (i256 {ora.type = i256}) attributes {ora.abi_params = ["uint256", "uint256"], ora.abi_return = "uint256", ora.contract_level = "full", ora.formal = true, ora.requires_count = 1 : i32, ora.selector = "0x3ef5e445", ora.verification = true, ora.verification_context = "function_contract", ora.visibility = "pub"} {

      %0 = arith.cmpi uge, %arg0, %arg1 : i256 loc("arithmetic_test.ora":33:20)
      cf.assert %0, "Precondition 0 failed" {ora.formal = true, ora.precondition_index = 0 : i32, ora.requires = true, ora.verification = true, ora.verification_context = "function_precondition"} loc("arithmetic_test.ora":33:20)
      %1 = arith.subi %arg0, %arg1 {gas_cost = 3 : i64} : i256 loc("arithmetic_test.ora":35:18)

      %2 = arith.cmpi uge, %arg0, %arg1 : i256 loc("arithmetic_test.ora":35:18)
      cf.assert %2, "checked subtraction overflow" loc("arithmetic_test.ora":35:18)
      ora.return %1 : i256 {gas_cost = 8 : i64} loc("arithmetic_test.ora":35:21)
    } loc("arithmetic_test.ora":32:12)

    func.func @divide(%arg0: i256 {ora.type = i256} loc("arithmetic_test.ora":38:19), %arg1: i256 {ora.type = i256} loc("arithmetic_test.ora":38:28)) -> (i256 {ora.type = i256}) attributes {ora.abi_params = ["uint256", "uint256"], ora.abi_return = "uint256", ora.contract_level = "full", ora.formal = true, ora.requires_count = 1 : i32, ora.selector = "0xf88e9fbf", ora.verification = true, ora.verification_context = "function_contract", ora.visibility = "pub"} {
      %c0_i256 = arith.constant {gas_cost = 0 : i64} 0 : i256 loc(unknown)

      %0 = arith.cmpi ne, %arg1, %c0_i256 : i256 loc("arithmetic_test.ora":39:20)
      cf.assert %0, "Precondition 0 failed" {ora.formal = true, ora.precondition_index = 0 : i32, ora.requires = true, ora.verification = true, ora.verification_context = "function_precondition"} loc("arithmetic_test.ora":39:20)
      %1 = arith.divui %arg0, %arg1 {gas_cost = 5 : i64} : i256 loc("arithmetic_test.ora":41:18)

      %2 = arith.cmpi ne, %arg1, %c0_i256 : i256 loc("arithmetic_test.ora":41:18)
      cf.assert %2, "division by zero" loc("arithmetic_test.ora":41:18)
      ora.return %1 : i256 {gas_cost = 8 : i64} loc("arithmetic_test.ora":41:21)
    } loc("arithmetic_test.ora":38:12)

    func.func @modulo(%arg0: i256 {ora.type = i256} loc("arithmetic_test.ora":44:19), %arg1: i256 {ora.type = i256} loc("arithmetic_test.ora":44:28)) -> (i256 {ora.type = i256}) attributes {ora.abi_params = ["uint256", "uint256"], ora.abi_return = "uint256", ora.contract_level = "full", ora.formal = true, ora.requires_count = 1 : i32, ora.selector = "0xbaaf073d", ora.verification = true, ora.verification_context = "function_contract", ora.visibility = "pub"} {
      %c0_i256 = arith.constant {gas_cost = 0 : i64} 0 : i256 loc(unknown)

      %0 = arith.cmpi ne, %arg1, %c0_i256 : i256 loc("arithmetic_test.ora":45:20)
      cf.assert %0, "Precondition 0 failed" {ora.formal = true, ora.precondition_index = 0 : i32, ora.requires = true, ora.verification = true, ora.verification_context = "function_precondition"} loc("arithmetic_test.ora":45:20)
      %1 = arith.remui %arg0, %arg1 {gas_cost = 5 : i64} : i256 loc("arithmetic_test.ora":47:18)

      %2 = arith.cmpi ne, %arg1, %c0_i256 : i256 loc("arithmetic_test.ora":47:18)
      cf.assert %2, "division by zero" loc("arithmetic_test.ora":47:18)
      ora.return %1 : i256 {gas_cost = 8 : i64} loc("arithmetic_test.ora":47:21)
    } loc("arithmetic_test.ora":44:12)
  } {gas_cost = 0 : i64, ora.contract_decl = true, sym_name = "ArithmeticTest"} loc("arithmetic_test.ora":1:10)
} loc("arithmetic_test.ora":1:10)

module attributes {ora.global_slots = {counter = 0 : ui64}} {

  func.func @init() attributes {ora.abi_params = [], ora.init = true, ora.selector = "0xe1c7392a", ora.visibility = "pub"} {

    %c0 = sir.const 0 : !sir.u256 loc("arithmetic_test.ora":6:19)
    %0 = sir.bitcast %c0 : !sir.u256 : i256 loc("arithmetic_test.ora":6:19)
    %slot_counter = sir.const 0 : !sir.u256 loc("arithmetic_test.ora":6:9)
    %1 = sir.bitcast %0 : i256 : !sir.u256 loc("arithmetic_test.ora":6:9)
    sir.sstore %slot_counter : !sir.u256, %1 : !sir.u256 loc("arithmetic_test.ora":6:9)

    sir.iret  loc("arithmetic_test.ora":5:12)
  } loc("arithmetic_test.ora":5:12)

  func.func @increment() attributes {ora.abi_params = [], ora.contract_level = "full", ora.formal = true, ora.requires_count = 1 : i32, ora.selector = "0xd09de08a", ora.verification = true, ora.verification_context = "function_contract", ora.visibility = "pub"} {

    %0 = sir.const -1 : !sir.u256 loc("arithmetic_test.ora":10:42)
    %1 = sir.bitcast %0 : !sir.u256 : i256 loc("arithmetic_test.ora":10:42)
    %c1 = sir.const 1 : !sir.u256 loc("arithmetic_test.ora":12:29)
    %2 = sir.bitcast %c1 : !sir.u256 : i256 loc("arithmetic_test.ora":12:29)
    %slot_counter = sir.const 0 : !sir.u256 loc("arithmetic_test.ora":10:18)
    %3 = sir.sload %slot_counter : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":10:18)
    %4 = sir.bitcast %3 : !sir.u256 {sir.result_name_0 = "value"} : i256 loc("arithmetic_test.ora":10:18)

    %5 = sir.bitcast %4 : i256 : !sir.u256 loc("arithmetic_test.ora":10:26)
    %6 = sir.bitcast %1 : i256 : !sir.u256 loc("arithmetic_test.ora":10:26)
    %7 = sir.lt %5 : !sir.u256, %6 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":10:26)
    %8 = sir.iszero %7 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":10:26)
    %9 = sir.iszero %8 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":10:26)
    sir.cond_br %9 : !sir.u256, ^bb2, ^bb1 loc("arithmetic_test.ora":10:26)

  ^bb1:  // pred: ^bb0

    sir.invalid loc("arithmetic_test.ora":10:26)

  ^bb2:  // pred: ^bb0

    %10 = sir.sload %slot_counter : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":12:19)
    %11 = sir.bitcast %10 : !sir.u256 {sir.result_name_0 = "value"} : i256 loc("arithmetic_test.ora":12:19)

    %12 = sir.bitcast %11 : i256 : !sir.u256 loc("arithmetic_test.ora":12:27)
    %13 = sir.bitcast %2 : i256 : !sir.u256 loc("arithmetic_test.ora":12:27)
    %14 = sir.add %12 : !sir.u256, %13 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":12:27)
    %15 = sir.bitcast %14 : !sir.u256 : i256 loc("arithmetic_test.ora":12:27)
    %16 = sir.bitcast %15 : i256 : !sir.u256 loc("arithmetic_test.ora":12:27)
    %17 = sir.bitcast %11 : i256 : !sir.u256 loc("arithmetic_test.ora":12:27)
    %18 = sir.gt %16 : !sir.u256, %17 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":12:27)
    %19 = sir.eq %16 : !sir.u256, %17 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":12:27)
    %20 = sir.or %18 : !sir.u256, %19 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":12:27)
    %21 = sir.iszero %20 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":12:27)
    %22 = sir.iszero %21 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":12:27)
    sir.cond_br %22 : !sir.u256, ^bb4, ^bb3 loc("arithmetic_test.ora":12:27)

  ^bb3:  // pred: ^bb2

    sir.invalid loc("arithmetic_test.ora":12:27)

  ^bb4:  // pred: ^bb2

    %23 = sir.bitcast %15 : i256 : !sir.u256 loc("arithmetic_test.ora":12:9)
    sir.sstore %slot_counter : !sir.u256, %23 : !sir.u256 loc("arithmetic_test.ora":12:9)

    sir.iret  loc("arithmetic_test.ora":9:12)
  } loc("arithmetic_test.ora":9:12)

  func.func @getCounter() -> (i256 {ora.type = i256}) attributes {ora.abi_params = [], ora.abi_return = "uint256", ora.selector = "0x8ada066e", ora.visibility = "pub"} {

    %slot_counter = sir.const 0 : !sir.u256 loc("arithmetic_test.ora":16:16)
    %0 = sir.sload %slot_counter : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":16:16)
    %1 = sir.bitcast %0 : !sir.u256 {sir.result_name_0 = "value"} : i256 loc("arithmetic_test.ora":16:16)

    %2 = sir.bitcast %1 : i256 : !sir.u256 loc("arithmetic_test.ora":16:23)
    %3 = sir.const 32 : !sir.u256 loc("arithmetic_test.ora":16:23)
    %4 = sir.malloc %3 : !sir.u256 : !sir.ptr<1> loc("arithmetic_test.ora":16:23)
    sir.store %4 : !sir.ptr<1>, %2 : !sir.u256 loc("arithmetic_test.ora":16:23)

    sir.return %4 : !sir.ptr<1>, %3 : !sir.u256 loc("arithmetic_test.ora":16:23)
  } loc("arithmetic_test.ora":15:12)

  func.func @multiply(%arg0: i256 {ora.type = i256} loc("arithmetic_test.ora":19:21), %arg1: i256 {ora.type = i256} loc("arithmetic_test.ora":19:30)) -> (i256 {ora.type = i256}) attributes {ora.abi_params = ["uint256", "uint256"], ora.abi_return = "uint256", ora.contract_level = "full", ora.formal = true, ora.requires_count = 2 : i32, ora.selector = "0x165c4a16", ora.verification = true, ora.verification_context = "function_contract", ora.visibility = "pub"} {

    %c1 = sir.const 1 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %c0 = sir.const 0 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %0 = sir.bitcast %c0 : !sir.u256 : i256 loc("arithmetic_test.ora":23:18)
    %1 = sir.const 340282366920938463463374607431768211455 : !sir.u256 loc(unknown)
    %2 = sir.bitcast %1 : !sir.u256 : i256 loc(unknown)
    %3 = sir.bitcast %arg0 : i256 : !sir.u256 loc("arithmetic_test.ora":20:20)
    %4 = sir.bitcast %2 : i256 : !sir.u256 loc("arithmetic_test.ora":20:20)
    %5 = sir.lt %3 : !sir.u256, %4 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":20:20)
    %6 = sir.eq %3 : !sir.u256, %4 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":20:20)
    %7 = sir.or %5 : !sir.u256, %6 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":20:20)
    %8 = sir.iszero %7 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":20:20)
    %9 = sir.iszero %8 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":20:20)
    sir.cond_br %9 : !sir.u256, ^bb2, ^bb1 loc("arithmetic_test.ora":20:20)

  ^bb1:  // pred: ^bb0

    sir.invalid loc("arithmetic_test.ora":20:20)

  ^bb2:  // pred: ^bb0

    %10 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":21:20)
    %11 = sir.bitcast %2 : i256 : !sir.u256 loc("arithmetic_test.ora":21:20)
    %12 = sir.lt %10 : !sir.u256, %11 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":21:20)
    %13 = sir.eq %10 : !sir.u256, %11 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":21:20)
    %14 = sir.or %12 : !sir.u256, %13 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":21:20)
    %15 = sir.iszero %14 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":21:20)
    %16 = sir.iszero %15 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":21:20)
    sir.cond_br %16 : !sir.u256, ^bb4, ^bb3 loc("arithmetic_test.ora":21:20)

  ^bb3:  // pred: ^bb2

    sir.invalid loc("arithmetic_test.ora":21:20)

  ^bb4:  // pred: ^bb2

    %17 = sir.bitcast %arg0 : i256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %18 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %19 = sir.mul %17 : !sir.u256, %18 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %20 = sir.bitcast %19 : !sir.u256 : i256 loc("arithmetic_test.ora":23:18)
    %21 = sir.const 1 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %22 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %23 = sir.bitcast %0 : i256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %24 = sir.eq %22 : !sir.u256, %23 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %25 = sir.xor %24 : !sir.u256, %21 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %26 = sir.bitcast %20 : i256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %27 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %28 = sir.div %26 : !sir.u256, %27 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %29 = sir.bitcast %28 : !sir.u256 : i256 loc("arithmetic_test.ora":23:18)
    %30 = sir.bitcast %29 : i256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %31 = sir.bitcast %arg0 : i256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %32 = sir.eq %30 : !sir.u256, %31 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %33 = sir.xor %32 : !sir.u256, %21 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %34 = sir.and %33 : !sir.u256, %25 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %35 = sir.xor %34 : !sir.u256, %c1 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %36 = sir.iszero %35 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    %37 = sir.iszero %36 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":23:18)
    sir.cond_br %37 : !sir.u256, ^bb6, ^bb5 loc("arithmetic_test.ora":23:18)

  ^bb5:  // pred: ^bb4

    sir.invalid loc("arithmetic_test.ora":23:18)

  ^bb6:  // pred: ^bb4

    %38 = sir.bitcast %20 : i256 : !sir.u256 loc("arithmetic_test.ora":23:21)
    %39 = sir.const 32 : !sir.u256 loc("arithmetic_test.ora":23:21)
    %40 = sir.malloc %39 : !sir.u256 : !sir.ptr<1> loc("arithmetic_test.ora":23:21)
    sir.store %40 : !sir.ptr<1>, %38 : !sir.u256 loc("arithmetic_test.ora":23:21)

    sir.return %40 : !sir.ptr<1>, %39 : !sir.u256 loc("arithmetic_test.ora":23:21)
  } loc("arithmetic_test.ora":19:12)

  func.func @add(%arg0: i256 {ora.type = i256} loc("arithmetic_test.ora":26:16), %arg1: i256 {ora.type = i256} loc("arithmetic_test.ora":26:25)) -> (i256 {ora.type = i256}) attributes {ora.abi_params = ["uint256", "uint256"], ora.abi_return = "uint256", ora.contract_level = "full", ora.formal = true, ora.requires_count = 1 : i32, ora.selector = "0x771602f7", ora.verification = true, ora.verification_context = "function_contract", ora.visibility = "pub"} {

    %0 = sir.const -1 : !sir.u256 loc("arithmetic_test.ora":27:37)
    %1 = sir.bitcast %0 : !sir.u256 : i256 loc("arithmetic_test.ora":27:37)
    %2 = sir.bitcast %1 : i256 : !sir.u256 loc("arithmetic_test.ora":27:46)
    %3 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":27:46)
    %4 = sir.sub %2 : !sir.u256, %3 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":27:46)
    %5 = sir.bitcast %4 : !sir.u256 : i256 loc("arithmetic_test.ora":27:46)
    %6 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":27:46)
    %7 = sir.bitcast %1 : i256 : !sir.u256 loc("arithmetic_test.ora":27:46)
    %8 = sir.lt %6 : !sir.u256, %7 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":27:46)
    %9 = sir.eq %6 : !sir.u256, %7 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":27:46)
    %10 = sir.or %8 : !sir.u256, %9 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":27:46)
    %11 = sir.iszero %10 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":27:46)
    %12 = sir.iszero %11 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":27:46)
    sir.cond_br %12 : !sir.u256, ^bb2, ^bb1 loc("arithmetic_test.ora":27:46)

  ^bb1:  // pred: ^bb0

    sir.invalid loc("arithmetic_test.ora":27:46)

  ^bb2:  // pred: ^bb0

    %13 = sir.bitcast %arg0 : i256 : !sir.u256 loc("arithmetic_test.ora":27:20)
    %14 = sir.bitcast %5 : i256 : !sir.u256 loc("arithmetic_test.ora":27:20)
    %15 = sir.lt %13 : !sir.u256, %14 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":27:20)
    %16 = sir.eq %13 : !sir.u256, %14 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":27:20)
    %17 = sir.or %15 : !sir.u256, %16 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":27:20)
    %18 = sir.iszero %17 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":27:20)
    %19 = sir.iszero %18 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":27:20)
    sir.cond_br %19 : !sir.u256, ^bb4, ^bb3 loc("arithmetic_test.ora":27:20)

  ^bb3:  // pred: ^bb2

    sir.invalid loc("arithmetic_test.ora":27:20)

  ^bb4:  // pred: ^bb2

    %20 = sir.bitcast %arg0 : i256 : !sir.u256 loc("arithmetic_test.ora":29:18)
    %21 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":29:18)
    %22 = sir.add %20 : !sir.u256, %21 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":29:18)
    %23 = sir.bitcast %22 : !sir.u256 : i256 loc("arithmetic_test.ora":29:18)
    %24 = sir.bitcast %23 : i256 : !sir.u256 loc("arithmetic_test.ora":29:18)
    %25 = sir.bitcast %arg0 : i256 : !sir.u256 loc("arithmetic_test.ora":29:18)
    %26 = sir.gt %24 : !sir.u256, %25 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":29:18)
    %27 = sir.eq %24 : !sir.u256, %25 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":29:18)
    %28 = sir.or %26 : !sir.u256, %27 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":29:18)
    %29 = sir.iszero %28 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":29:18)
    %30 = sir.iszero %29 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":29:18)
    sir.cond_br %30 : !sir.u256, ^bb6, ^bb5 loc("arithmetic_test.ora":29:18)

  ^bb5:  // pred: ^bb4

    sir.invalid loc("arithmetic_test.ora":29:18)

  ^bb6:  // pred: ^bb4

    %31 = sir.bitcast %23 : i256 : !sir.u256 loc("arithmetic_test.ora":29:21)
    %32 = sir.const 32 : !sir.u256 loc("arithmetic_test.ora":29:21)
    %33 = sir.malloc %32 : !sir.u256 : !sir.ptr<1> loc("arithmetic_test.ora":29:21)
    sir.store %33 : !sir.ptr<1>, %31 : !sir.u256 loc("arithmetic_test.ora":29:21)

    sir.return %33 : !sir.ptr<1>, %32 : !sir.u256 loc("arithmetic_test.ora":29:21)
  } loc("arithmetic_test.ora":26:12)

  func.func @subtract(%arg0: i256 {ora.type = i256} loc("arithmetic_test.ora":32:21), %arg1: i256 {ora.type = i256} loc("arithmetic_test.ora":32:30)) -> (i256 {ora.type = i256}) attributes {ora.abi_params = ["uint256", "uint256"], ora.abi_return = "uint256", ora.contract_level = "full", ora.formal = true, ora.requires_count = 1 : i32, ora.selector = "0x3ef5e445", ora.verification = true, ora.verification_context = "function_contract", ora.visibility = "pub"} {

    %0 = sir.bitcast %arg0 : i256 : !sir.u256 loc("arithmetic_test.ora":33:20)
    %1 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":33:20)
    %2 = sir.gt %0 : !sir.u256, %1 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":33:20)
    %3 = sir.eq %0 : !sir.u256, %1 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":33:20)
    %4 = sir.or %2 : !sir.u256, %3 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":33:20)
    %5 = sir.iszero %4 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":33:20)
    %6 = sir.iszero %5 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":33:20)
    sir.cond_br %6 : !sir.u256, ^bb2, ^bb1 loc("arithmetic_test.ora":33:20)

  ^bb1:  // pred: ^bb0

    sir.invalid loc("arithmetic_test.ora":33:20)

  ^bb2:  // pred: ^bb0

    %7 = sir.bitcast %arg0 : i256 : !sir.u256 loc("arithmetic_test.ora":35:18)
    %8 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":35:18)
    %9 = sir.sub %7 : !sir.u256, %8 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":35:18)
    %10 = sir.bitcast %9 : !sir.u256 : i256 loc("arithmetic_test.ora":35:18)
    %11 = sir.bitcast %arg0 : i256 : !sir.u256 loc("arithmetic_test.ora":35:18)
    %12 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":35:18)
    %13 = sir.gt %11 : !sir.u256, %12 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":35:18)
    %14 = sir.eq %11 : !sir.u256, %12 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":35:18)
    %15 = sir.or %13 : !sir.u256, %14 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":35:18)
    %16 = sir.iszero %15 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":35:18)
    %17 = sir.iszero %16 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":35:18)
    sir.cond_br %17 : !sir.u256, ^bb4, ^bb3 loc("arithmetic_test.ora":35:18)

  ^bb3:  // pred: ^bb2

    sir.invalid loc("arithmetic_test.ora":35:18)

  ^bb4:  // pred: ^bb2

    %18 = sir.bitcast %10 : i256 : !sir.u256 loc("arithmetic_test.ora":35:21)
    %19 = sir.const 32 : !sir.u256 loc("arithmetic_test.ora":35:21)
    %20 = sir.malloc %19 : !sir.u256 : !sir.ptr<1> loc("arithmetic_test.ora":35:21)
    sir.store %20 : !sir.ptr<1>, %18 : !sir.u256 loc("arithmetic_test.ora":35:21)

    sir.return %20 : !sir.ptr<1>, %19 : !sir.u256 loc("arithmetic_test.ora":35:21)
  } loc("arithmetic_test.ora":32:12)

  func.func @divide(%arg0: i256 {ora.type = i256} loc("arithmetic_test.ora":38:19), %arg1: i256 {ora.type = i256} loc("arithmetic_test.ora":38:28)) -> (i256 {ora.type = i256}) attributes {ora.abi_params = ["uint256", "uint256"], ora.abi_return = "uint256", ora.contract_level = "full", ora.formal = true, ora.requires_count = 1 : i32, ora.selector = "0xf88e9fbf", ora.verification = true, ora.verification_context = "function_contract", ora.visibility = "pub"} {

    %c0 = sir.const 0 : !sir.u256 loc(unknown)
    %0 = sir.bitcast %c0 : !sir.u256 : i256 loc(unknown)
    %1 = sir.const 1 : !sir.u256 loc("arithmetic_test.ora":39:20)
    %2 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":39:20)
    %3 = sir.bitcast %0 : i256 : !sir.u256 loc("arithmetic_test.ora":39:20)
    %4 = sir.eq %2 : !sir.u256, %3 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":39:20)
    %5 = sir.xor %4 : !sir.u256, %1 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":39:20)
    %6 = sir.iszero %5 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":39:20)
    %7 = sir.iszero %6 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":39:20)
    sir.cond_br %7 : !sir.u256, ^bb2, ^bb1 loc("arithmetic_test.ora":39:20)

  ^bb1:  // pred: ^bb0

    sir.invalid loc("arithmetic_test.ora":39:20)

  ^bb2:  // pred: ^bb0

    %8 = sir.bitcast %arg0 : i256 : !sir.u256 loc("arithmetic_test.ora":41:18)
    %9 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":41:18)
    %10 = sir.div %8 : !sir.u256, %9 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":41:18)
    %11 = sir.bitcast %10 : !sir.u256 : i256 loc("arithmetic_test.ora":41:18)
    %12 = sir.const 1 : !sir.u256 loc("arithmetic_test.ora":41:18)
    %13 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":41:18)
    %14 = sir.bitcast %0 : i256 : !sir.u256 loc("arithmetic_test.ora":41:18)
    %15 = sir.eq %13 : !sir.u256, %14 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":41:18)
    %16 = sir.xor %15 : !sir.u256, %12 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":41:18)
    %17 = sir.iszero %16 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":41:18)
    %18 = sir.iszero %17 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":41:18)
    sir.cond_br %18 : !sir.u256, ^bb4, ^bb3 loc("arithmetic_test.ora":41:18)

  ^bb3:  // pred: ^bb2

    sir.invalid loc("arithmetic_test.ora":41:18)

  ^bb4:  // pred: ^bb2

    %19 = sir.bitcast %11 : i256 : !sir.u256 loc("arithmetic_test.ora":41:21)
    %20 = sir.const 32 : !sir.u256 loc("arithmetic_test.ora":41:21)
    %21 = sir.malloc %20 : !sir.u256 : !sir.ptr<1> loc("arithmetic_test.ora":41:21)
    sir.store %21 : !sir.ptr<1>, %19 : !sir.u256 loc("arithmetic_test.ora":41:21)

    sir.return %21 : !sir.ptr<1>, %20 : !sir.u256 loc("arithmetic_test.ora":41:21)
  } loc("arithmetic_test.ora":38:12)

  func.func @modulo(%arg0: i256 {ora.type = i256} loc("arithmetic_test.ora":44:19), %arg1: i256 {ora.type = i256} loc("arithmetic_test.ora":44:28)) -> (i256 {ora.type = i256}) attributes {ora.abi_params = ["uint256", "uint256"], ora.abi_return = "uint256", ora.contract_level = "full", ora.formal = true, ora.requires_count = 1 : i32, ora.selector = "0xbaaf073d", ora.verification = true, ora.verification_context = "function_contract", ora.visibility = "pub"} {

    %c0 = sir.const 0 : !sir.u256 loc(unknown)
    %0 = sir.bitcast %c0 : !sir.u256 : i256 loc(unknown)
    %1 = sir.const 1 : !sir.u256 loc("arithmetic_test.ora":45:20)
    %2 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":45:20)
    %3 = sir.bitcast %0 : i256 : !sir.u256 loc("arithmetic_test.ora":45:20)
    %4 = sir.eq %2 : !sir.u256, %3 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":45:20)
    %5 = sir.xor %4 : !sir.u256, %1 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":45:20)
    %6 = sir.iszero %5 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":45:20)
    %7 = sir.iszero %6 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":45:20)
    sir.cond_br %7 : !sir.u256, ^bb2, ^bb1 loc("arithmetic_test.ora":45:20)

  ^bb1:  // pred: ^bb0

    sir.invalid loc("arithmetic_test.ora":45:20)

  ^bb2:  // pred: ^bb0

    %8 = sir.bitcast %arg0 : i256 : !sir.u256 loc("arithmetic_test.ora":47:18)
    %9 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":47:18)
    %10 = sir.mod %8 : !sir.u256, %9 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":47:18)
    %11 = sir.bitcast %10 : !sir.u256 : i256 loc("arithmetic_test.ora":47:18)
    %12 = sir.const 1 : !sir.u256 loc("arithmetic_test.ora":47:18)
    %13 = sir.bitcast %arg1 : i256 : !sir.u256 loc("arithmetic_test.ora":47:18)
    %14 = sir.bitcast %0 : i256 : !sir.u256 loc("arithmetic_test.ora":47:18)
    %15 = sir.eq %13 : !sir.u256, %14 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":47:18)
    %16 = sir.xor %15 : !sir.u256, %12 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":47:18)
    %17 = sir.iszero %16 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":47:18)
    %18 = sir.iszero %17 : !sir.u256 : !sir.u256 loc("arithmetic_test.ora":47:18)
    sir.cond_br %18 : !sir.u256, ^bb4, ^bb3 loc("arithmetic_test.ora":47:18)

  ^bb3:  // pred: ^bb2

    sir.invalid loc("arithmetic_test.ora":47:18)

  ^bb4:  // pred: ^bb2

    %19 = sir.bitcast %11 : i256 : !sir.u256 loc("arithmetic_test.ora":47:21)
    %20 = sir.const 32 : !sir.u256 loc("arithmetic_test.ora":47:21)
    %21 = sir.malloc %20 : !sir.u256 : !sir.ptr<1> loc("arithmetic_test.ora":47:21)
    sir.store %21 : !sir.ptr<1>, %19 : !sir.u256 loc("arithmetic_test.ora":47:21)

    sir.return %21 : !sir.ptr<1>, %20 : !sir.u256 loc("arithmetic_test.ora":47:21)
  } loc("arithmetic_test.ora":44:12)
} loc("arithmetic_test.ora":1:10)

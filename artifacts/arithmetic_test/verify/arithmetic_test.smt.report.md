# SMT Encoding Report

## 1. Run Metadata
- Source file: `ora-example/arithmetic_test.ora`
- Generated at (unix): `1771588681`
- Verification mode: `full`
- verify_calls: `true`
- verify_state: `true`
- parallel: `true`
- timeout_ms: `60000`

## 2. Summary
- Total queries: `13`
- SAT: `6`
- UNSAT: `6`
- UNKNOWN: `1`
- Failed obligations: `0`
- Inconsistent assumption bases: `0`
- Proven guards: `0`
- Violatable guards: `0`
- Verification success: `true`
- Verification errors: `0`
- Verification diagnostics: `0`

## 3. Query Kind Counts
- base: `6`
- obligation: `7`
- loop_invariant_step: `0`
- loop_invariant_post: `0`
- guard_satisfy: `0`
- guard_violate: `0`

## 4. Findings
- No verification findings.
## 5. Query Catalog
### Q1 - base
- Function: `multiply`
- Location: `loc("arithmetic_test.ora":20:0`
- Status: `SAT`
- Elapsed ms: `0`
- Constraint count: `2`
- SMT bytes: `268`
- SMT hash: `0xc4582c92fe85fa52`
- SMT-LIB:
```smt2
(declare-fun v_40924005120 () (_ BitVec 256))
(declare-fun v_40924005168 () (_ BitVec 256))
(assert (bvule v_40924005120 ((_ zero_extend 128) #xffffffffffffffffffffffffffffffff)))
(assert (bvule v_40924005168 ((_ zero_extend 128) #xffffffffffffffffffffffffffffffff)))

```

### Q2 - obligation
- Function: `multiply`
- Location: `loc("arithmetic_test.ora":23:0`
- Status: `UNKNOWN`
- Elapsed ms: `60247`
- Constraint count: `3`
- SMT bytes: `597`
- SMT hash: `0x2a79773b5c5697dd`
- Obligation kind: `contract invariant`
- SMT-LIB:
```smt2
(declare-fun v_40924005120 () (_ BitVec 256))
(declare-fun v_40924005168 () (_ BitVec 256))
(assert (bvule v_40924005120 ((_ zero_extend 128) #xffffffffffffffffffffffffffffffff)))
(assert (bvule v_40924005168 ((_ zero_extend 128) #xffffffffffffffffffffffffffffffff)))
(assert (let ((a!1 (not (= (bvudiv (bvmul v_40924005120 v_40924005168) v_40924005168)
                   v_40924005120))))
(let ((a!2 (xor (and a!1
                     (not (= v_40924005168
                             #x0000000000000000000000000000000000000000000000000000000000000000)))
                true)))
  (not a!2))))

```

### Q3 - base
- Function: `divide`
- Location: `loc("arithmetic_test.ora":39:0`
- Status: `SAT`
- Elapsed ms: `0`
- Constraint count: `1`
- SMT bytes: `154`
- SMT hash: `0xf799b5aa52e42694`
- SMT-LIB:
```smt2
(declare-fun v_40924005552 () (_ BitVec 256))
(assert (not (= v_40924005552
        #x0000000000000000000000000000000000000000000000000000000000000000)))

```

### Q4 - obligation
- Function: `divide`
- Location: `loc("arithmetic_test.ora":41:0`
- Status: `UNSAT`
- Elapsed ms: `0`
- Constraint count: `2`
- SMT bytes: `273`
- SMT hash: `0x5805dea9a0c5c65f`
- Obligation kind: `contract invariant`
- SMT-LIB:
```smt2
(declare-fun v_40924005552 () (_ BitVec 256))
(assert (not (= v_40924005552
        #x0000000000000000000000000000000000000000000000000000000000000000)))
(assert (not (not (= v_40924005552
             #x0000000000000000000000000000000000000000000000000000000000000000))))

```

### Q5 - base
- Function: `subtract`
- Location: `loc("arithmetic_test.ora":33:0`
- Status: `SAT`
- Elapsed ms: `0`
- Constraint count: `1`
- SMT bytes: `137`
- SMT hash: `0xdb09c607ee67d84c`
- SMT-LIB:
```smt2
(declare-fun v_40924005456 () (_ BitVec 256))
(declare-fun v_40924005408 () (_ BitVec 256))
(assert (bvuge v_40924005408 v_40924005456))

```

### Q6 - obligation
- Function: `subtract`
- Location: `loc("arithmetic_test.ora":35:0`
- Status: `UNSAT`
- Elapsed ms: `0`
- Constraint count: `2`
- SMT bytes: `199`
- SMT hash: `0x5c57011da66a476`
- Obligation kind: `contract invariant`
- SMT-LIB:
```smt2
(declare-fun v_40924005456 () (_ BitVec 256))
(declare-fun v_40924005408 () (_ BitVec 256))
(assert (bvuge v_40924005408 v_40924005456))
(assert (not (xor (bvult v_40924005408 v_40924005456) true)))

```

### Q7 - base
- Function: `increment`
- Location: `loc("arithmetic_test.ora":10:0`
- Status: `SAT`
- Elapsed ms: `0`
- Constraint count: `1`
- SMT bytes: `143`
- SMT hash: `0x628289a82ecb61ad`
- SMT-LIB:
```smt2
(declare-fun g_counter () (_ BitVec 256))
(assert (bvult g_counter
       #xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff))

```

### Q8 - obligation
- Function: `increment`
- Location: `loc("arithmetic_test.ora":12:0`
- Status: `UNSAT`
- Elapsed ms: `83`
- Constraint count: `2`
- SMT bytes: `323`
- SMT hash: `0x5fca6c8be4689257`
- Obligation kind: `contract invariant`
- SMT-LIB:
```smt2
(declare-fun g_counter () (_ BitVec 256))
(assert (bvult g_counter
       #xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff))
(assert (not (xor (bvult (bvadd g_counter
                        #x0000000000000000000000000000000000000000000000000000000000000001)
                 g_counter)
          true)))

```

### Q9 - base
- Function: `add`
- Location: `loc("arithmetic_test.ora":27:0`
- Status: `SAT`
- Elapsed ms: `0`
- Constraint count: `1`
- SMT bytes: `233`
- SMT hash: `0x76688839f47bfa40`
- SMT-LIB:
```smt2
(declare-fun v_40924005360 () (_ BitVec 256))
(declare-fun v_40924005312 () (_ BitVec 256))
(assert (bvule v_40924005312
       (bvsub #xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
              v_40924005360)))

```

### Q10 - obligation
- Function: `add`
- Location: `loc("arithmetic_test.ora":27:0`
- Status: `UNSAT`
- Elapsed ms: `0`
- Constraint count: `2`
- SMT bytes: `375`
- SMT hash: `0xdeee7c2bb466ba6f`
- Obligation kind: `contract invariant`
- SMT-LIB:
```smt2
(declare-fun v_40924005360 () (_ BitVec 256))
(declare-fun v_40924005312 () (_ BitVec 256))
(assert (bvule v_40924005312
       (bvsub #xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
              v_40924005360)))
(assert (not (xor (bvult #xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
                 v_40924005360)
          true)))

```

### Q11 - obligation
- Function: `add`
- Location: `loc("arithmetic_test.ora":29:0`
- Status: `UNSAT`
- Elapsed ms: `100`
- Constraint count: `2`
- SMT bytes: `317`
- SMT hash: `0x9bbbc05c45a2573f`
- Obligation kind: `contract invariant`
- SMT-LIB:
```smt2
(declare-fun v_40924005360 () (_ BitVec 256))
(declare-fun v_40924005312 () (_ BitVec 256))
(assert (bvule v_40924005312
       (bvsub #xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
              v_40924005360)))
(assert (not (xor (bvult (bvadd v_40924005312 v_40924005360) v_40924005312) true)))

```

### Q12 - base
- Function: `modulo`
- Location: `loc("arithmetic_test.ora":45:0`
- Status: `SAT`
- Elapsed ms: `0`
- Constraint count: `1`
- SMT bytes: `154`
- SMT hash: `0x35a7a80640c61423`
- SMT-LIB:
```smt2
(declare-fun v_40924005648 () (_ BitVec 256))
(assert (not (= v_40924005648
        #x0000000000000000000000000000000000000000000000000000000000000000)))

```

### Q13 - obligation
- Function: `modulo`
- Location: `loc("arithmetic_test.ora":47:0`
- Status: `UNSAT`
- Elapsed ms: `0`
- Constraint count: `2`
- SMT bytes: `273`
- SMT hash: `0x65c4e74c424d4855`
- Obligation kind: `contract invariant`
- SMT-LIB:
```smt2
(declare-fun v_40924005648 () (_ BitVec 256))
(assert (not (= v_40924005648
        #x0000000000000000000000000000000000000000000000000000000000000000)))
(assert (not (not (= v_40924005648
             #x0000000000000000000000000000000000000000000000000000000000000000))))

```


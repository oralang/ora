# Expression Lowering Refactoring Plan

## Current State
- `expressions/mod.zig`: 3397 lines - too large
- All expression lowering logic in a single file

## Proposed Module Structure

### 1. `helpers.zig` (~300 lines)
Common helper functions:
- `createConstant`, `createErrorPlaceholder`
- `createArithmeticOp`, `createComparisonOp`
- `getCommonType`, `convertToType`
- `fileLoc`, `createBoolConstant`, `createTypedConstant`
- `predicateStringToInt`

### 2. `literals.zig` (~400 lines)
Literal expression lowering:
- `lowerLiteral`
- `extractIntegerFromLiteral`, `extractIntegerFromExpr`

### 3. `operators.zig` (~300 lines)
Binary and unary operators:
- `lowerBinary`
- `lowerUnary`
- `insertExactDivisionGuard`

### 4. `access.zig` (~500 lines)
Identifier, field, and index access:
- `lowerIdentifier`
- `lowerFieldAccess`
- `lowerIndex`
- `createStructFieldExtract`, `createPseudoFieldAccess`
- `createLengthAccess`, `createArrayIndexLoad`, `createMapIndexLoad`
- `convertIndexToIndexType`

### 5. `calls.zig` (~200 lines)
Function calls:
- `lowerCall`
- `processNormalCall`, `lowerBuiltinCall`
- `createDirectFunctionCall`, `createMethodCall`
- `lowerBuiltinConstant`

### 6. `assignments.zig` (~400 lines)
Assignments:
- `lowerAssignment`
- `lowerCompoundAssignment`
- `lowerLValue`, `storeLValue`

### 7. `advanced.zig` (~1200 lines)
Advanced expressions:
- `lowerCast`, `lowerComptime`, `lowerOld`
- `lowerTuple`, `lowerSwitchExpression`
- `lowerQuantified`, `lowerTry`
- `lowerErrorReturn`, `lowerErrorCast`
- `lowerShift`, `lowerStructInstantiation`
- `lowerAnonymousStruct`, `lowerRange`
- `lowerLabeledBlock`, `lowerDestructuring`
- `lowerEnumLiteral`, `lowerArrayLiteral`
- `createEmptyArray`, `createInitializedArray`
- `createEmptyStruct`, `createInitializedStruct`
- `createTupleType`, `createExpressionCapture`
- `createDefaultValueForType`
- `extractTypeInfo`, `getTypeString`
- `addVerificationAttributes`, `createVerificationMetadata`

### 8. `mod.zig` (~200 lines)
Main struct and dispatch:
- `ExpressionLowerer` struct definition
- `init`
- `lowerExpression` (dispatch function)
- Imports and re-exports

## Implementation Strategy
1. Create `helpers.zig` first (most dependencies)
2. Create `literals.zig` 
3. Create other modules incrementally
4. Update `mod.zig` to import and delegate to modules
5. Test after each module extraction

## Benefits
- Reduced file size (from 3397 to ~200-500 lines per file)
- Better organization and maintainability
- Easier to navigate and understand
- Clearer separation of concerns


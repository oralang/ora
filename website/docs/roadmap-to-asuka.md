---
sidebar_position: 5
---

# Roadmap to ASUKA Release

ASUKA is Ora's first stable alpha milestone: a coherent front-end pipeline with
clear language semantics and a usable backend path.

## Release goals

- Stable front-end: lexer, parser, type resolution, Ora MLIR emission.
- Defined semantics for regions, refinements, and error unions.
- Sensei-IR (SIR) and EVM backend integration with end-to-end examples.
- Documentation that reflects compiler reality (no stale metrics).

## Current focus

- Completing Ora -> Sensei-IR (SIR) lowering for all supported constructs.
- Improving diagnostic quality and failure transparency.
- Expanding example coverage and test clarity.
- Tightening formal verification integration boundaries.

## Milestones (high-level)

1. **Backend completeness**
   - Full Sensei-IR (SIR) lowering coverage
   - EVM emission path for core contracts

2. **Standard library baseline**
   - Core utilities and safe patterns
   - Reference examples tied to tests

3. **Documentation alignment**
   - Language docs aligned with compiler behavior
   - Research docs aligned with current architecture

4. **Stabilization**
   - Regression test coverage
   - Clear error messages
   - Release notes and migration guidance

## Contributing priorities

- Backend lowering and legalization
- Error handling and diagnostics
- Tests for refinements, effects, and regions
- Documentation updates that remove ambiguity

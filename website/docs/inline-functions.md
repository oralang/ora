# Inline Functions (Planned)

Ora previously experimented with an `inline` keyword. The current compiler
reserves this idea, but it is **not** implemented or enforced in the active
pipeline.

## Status

- `inline` is **not** a supported keyword in the current compiler.
- Inline decisions are not exposed as user-level controls.

## Rationale

Inlining is a backend optimization that affects code size, call overhead, and
verification cost. We plan to reintroduce an inline mechanism once the backend
and cost model are stable enough to make the trade-offs explicit.

## Future design questions

- Should inlining be a compiler hint or a verified guarantee?
- How does inlining interact with verification and source-level traceability?
- What is the cost model for inlining in a gas-constrained environment?


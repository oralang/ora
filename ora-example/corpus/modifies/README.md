# `modifies` Sema Corpus

This directory contains source-level fixtures for user-declared `modifies`
clauses. These files pin the sema side of the feature: supported path syntax,
empty `modifies()`, fail-closed unsupported syntax, and the subset check between
actual current-contract writes and declared write paths.

Files prefixed with `pass_` must typecheck without diagnostics. Files prefixed
with `fail_` must produce the expected diagnostic while still remaining useful
as parser/sema fixtures.

Run these fixtures through the shared SMT/modifies corpus check:

```sh
scripts/check-smt-modifies-corpus.sh
```

The script runs the passing sema fixtures through the full compiler and
verifier pipeline, then checks each failing fixture for its expected diagnostic.

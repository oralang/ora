# The Ora Compiler Field Guide

*A practical, reader-first book for onboarding contributors and new compiler engineers.*

## How to use this guide

This guide is written in two “lanes”:

- **Lane A — Contributor:** no compiler background required. You’ll ship tests, repros, docs, diagnostics, and small safe changes.
- **Lane B — Compiler Engineer:** you’ll touch parsing, semantics, type resolution, MLIR lowering, legality, and verification.

You can read the whole book, or follow the lane markers inside chapters.


> **Guiding rule:** Ora is a pipeline. Every phase produces an artifact. When something breaks, find the **first** artifact that looks wrong, then fix the phase that produced it.


## Table of contents

### Main chapters
- [Welcome](00-welcome.md)
- [Your First Win](01-first-win.md)
- [The Compiler Shape](02-compiler-shape.md)
- [Contribution Tracks](03-contribution-tracks.md)
- [One Story End-to-End](04-walkthrough.md)
- [A Walkthrough That Intentionally Fails](04b-failing-walkthrough.md)
- [The Core Systems](05-core-systems.md)
- [Debugging Playbook](06-debugging-playbook.md)
- [Your First Compiler PR](07-first-compiler-pr.md)
- [Advanced Topics](08-advanced.md)

### Appendices
- [Appendix A — Glossary](appendix-a-glossary.md)
- [Appendix B — Codebase Map](appendix-b-codebase-map.md)
- [Appendix C — Add a Feature Checklist](appendix-c-feature-checklist.md)
- [Appendix D — Where to Add Tests](appendix-d-tests.md)

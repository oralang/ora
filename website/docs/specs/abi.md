---
sidebar_position: 2
---

# Ora ABI Specification v0.1

The complete Application Binary Interface (ABI) specification for Ora contracts.

## Overview

The Ora ABI defines the interface layer for smart contracts, covering type identity, code generation, and UI integration.

Ora ABI is split into two layers:

1. **Manifest (authoritative):** a self-describing interface + type graph + metadata.
2. **Wire profiles (optional):** concrete byte encoding rules used for calls/returns/errors/events.

## Specification Documents

### Main Specification

[Ora ABI v0.1](abi/ora-abi-v0.1) - Complete ABI specification covering:
- Goals and non-goals
- Manifest structure
- Type system
- Function and event definitions
- Wire profiles
- Tooling support

### Compiler Appendix

[Ora ABI v0.1 Compiler Appendix](abi/ora-abi-v0.1-compiler-appendix) - Compiler-specific implementation details and extensions.

## Quick Links

- [ABI Main Specification](abi/ora-abi-v0.1)
- [ABI Compiler Appendix](abi/ora-abi-v0.1-compiler-appendix)


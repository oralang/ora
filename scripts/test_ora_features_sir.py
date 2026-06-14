#!/usr/bin/env python3
"""Compatibility wrapper for the SIR mode of test_ora_features.py."""

from test_ora_features import main


if __name__ == "__main__":
    main(default_emit="sir")

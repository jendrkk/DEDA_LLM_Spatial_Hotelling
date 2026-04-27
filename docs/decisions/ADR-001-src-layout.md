# ADR-001: Use src layout for the Python package

**Status:** Accepted
**Date:** 2024-11

## Context

The initial prototype placed the `hotelling/` package at the repository root.
This is a common Python anti-pattern that causes test runs to silently import
from the working-tree sources rather than the installed package, masking
import-time issues and making editable-install behaviour hard to reason about.

## Decision

Adopt the PEP 517 / Flit / setuptools **src layout**: all importable code
lives under `src/hotelling/`.  The `pyproject.toml` sets
`[tool.setuptools.packages.find] where = ["src"]`.

## Consequences

* `pip install -e "."` now installs from `src/`, so tests always run against
  the installed package rather than the raw file tree.
* The old root-level `hotelling/` package has been removed.
* IDEs must be configured to mark `src/` as a source root.

## References

* https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
* https://setuptools.pypa.io/en/latest/userguide/package_discovery.html

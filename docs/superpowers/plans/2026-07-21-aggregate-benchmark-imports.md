# Aggregate Benchmark Imports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace copied ASV benchmark implementations with private imports while preserving all 91 platform aggregate groups and weights.

**Architecture:** Each aggregate module imports the original benchmark classes under private aliases and retains only aggregate metadata. `_aggregate_common.py` remains the shared runner. A structural test checks discovery counts, metadata alignment, singleton weights, and absence of copied-source markers.

**Tech Stack:** Python 3.14, ASV benchmark modules, pytest, Ruff, Git.

## Global Constraints

- Preserve exactly 51 Kunpeng 920B and 40 Kunpeng 950 aggregate groups.
- Preserve case order, parameter indices, method names, class types, and multi-case `run_repeat` values.
- Require `run_repeat = (1,)` for every single-case group.
- Imported benchmark classes must remain private to ASV discovery.
- Do not change NumPy runtime implementation or aggregate timing semantics.

---

### Task 1: Structural regression test

**Files:**
- Create: `benchmarks/tests/test_platform_aggregate_structure.py`

**Interfaces:**
- Consumes: aggregate modules below `benchmarks/benchmarks/aggregate_920b` and `aggregate_950`.
- Produces: a pytest gate for group counts, metadata alignment, singleton weights, and source-copy removal.

- [ ] Write a test that parses public aggregate classes and fails while generated files still contain `Inlined official source`.
- [ ] Run `python -m pytest benchmarks/tests/test_platform_aggregate_structure.py -q` and confirm the copied-source assertion fails.
- [ ] Keep the failing test unchanged for Task 2.

### Task 2: Replace snapshots with imports

**Files:**
- Modify: `benchmarks/benchmarks/aggregate_920b/*.py`
- Modify: `benchmarks/benchmarks/aggregate_950/*.py`

**Interfaces:**
- Consumes: original classes from sibling `bench_*.py` modules and `_AggregateBenchmark`/`select_case_params` from `_aggregate_common.py`.
- Produces: metadata-only aggregate modules with private class aliases.

- [ ] For every generated module, identify each `_Official_*` class and its original sibling module/class.
- [ ] Replace each copied source prefix with relative imports using the same `_Official_*` aliases.
- [ ] Preserve the aggregate class suffix byte-for-byte apart from import placement.
- [ ] Run the structural test and confirm all 91 groups pass.
- [ ] Run ASV discovery and confirm only the expected public aggregate classes are exposed.

### Task 3: Restore normal linting and verify

**Files:**
- Modify: `ruff.toml`

**Interfaces:**
- Consumes: metadata-only aggregate modules from Task 2.
- Produces: normal repository lint coverage for both aggregate directories.

- [ ] Remove the two aggregate paths from `extend-exclude`.
- [ ] Run Ruff against the aggregate modules and fix only actionable formatting/import findings.
- [ ] Run compileall, structural pytest, ASV discovery, and `git diff --check`.
- [ ] Commit the implementation and push `perf/numpy-platform-aggregate-benchmarks-pr` to refresh PR #131.

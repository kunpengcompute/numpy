# Platform aggregate benchmark import design

## Goal

Keep the Kunpeng 920B and 950 aggregate ASV suites platform-weighted while
removing copied upstream benchmark implementations from generated files.

## Structure

Each platform aggregate module imports the benchmark classes it uses from the
existing module in `benchmarks/benchmarks`. Imported classes use private aliases
so ASV does not discover them as additional public benchmarks. Platform modules
contain only imports, aggregate class metadata, and the shared aggregate runner
import.

`benchmarks/benchmarks/_aggregate_common.py` remains responsible for selecting
one upstream parameter combination and for setup, timed repetition, and teardown.
The original benchmark class remains responsible for its own data preparation
and timed method.

## Preserved behavior

- The 920B suite exposes exactly 51 aggregate groups.
- The 950 suite exposes exactly 40 aggregate groups.
- Existing case ordering, parameter indices, method names, class types, and
  `run_repeat` values remain unchanged.
- Every group containing one case has `run_repeat = (1,)`.
- Imported upstream benchmark classes are not independently discovered from an
  aggregate module.

## Linting

The two aggregate directories are removed from Ruff's `extend-exclude`. The
small metadata modules must pass the repository's normal benchmark lint rules.

## Validation

Automated validation parses and imports all aggregate modules, checks aligned
metadata lengths and singleton weights, and runs ASV benchmark discovery to
confirm the expected public aggregate names without imported-class duplicates.
Compilation, Ruff, and whitespace checks complete the gate.

## Non-goals

This change does not alter NumPy implementation code, benchmark inputs,
platform weights, aggregate timing semantics, or the unrelated incremental
coverage pipeline fix already present in the pull request.

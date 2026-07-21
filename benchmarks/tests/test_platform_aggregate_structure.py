import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

BENCHMARK_ROOT = Path(__file__).parents[1] / "benchmarks"
PLATFORMS = {"aggregate_920b": 51, "aggregate_950": 40}


@pytest.mark.parametrize(("platform", "expected_groups"), PLATFORMS.items())
def test_aggregate_modules_are_metadata_only(platform, expected_groups):
    aggregate_classes = []
    platform_root = BENCHMARK_ROOT / platform

    assert (platform_root / "__init__.py").is_file()

    for path in sorted(platform_root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "Inlined official source" not in source, path

        tree = ast.parse(source, filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            assignments = {
                statement.targets[0].id: statement.value
                for statement in node.body
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            }
            metadata = {
                name: assignments.get(name)
                for name in (
                    "case_params",
                    "run_repeat",
                    "case_methods",
                    "case_types",
                )
            }
            if not all(isinstance(value, ast.Tuple) for value in metadata.values()):
                continue

            aggregate_classes.append((path, node.name))
            lengths = {len(value.elts) for value in metadata.values()}
            assert len(lengths) == 1, (path, node.name, lengths)

            repeats = ast.literal_eval(metadata["run_repeat"])
            assert all(isinstance(value, int) and value >= 1 for value in repeats)
            if len(repeats) == 1:
                assert repeats == (1,), (path, node.name, repeats)

    assert len(aggregate_classes) == expected_groups


def test_aggregate_modules_import_without_public_upstream_aliases(tmp_path):
    script = f"""
import importlib
from pathlib import Path

root = Path({str(BENCHMARK_ROOT)!r})
for platform, expected in (("aggregate_920b", 51), ("aggregate_950", 40)):
    discovered = []
    for path in (root / platform).glob("*.py"):
        module = importlib.import_module(
            f"benchmarks.{{platform}}.{{path.stem}}"
        )
        discovered.extend(
            name
            for name, value in vars(module).items()
            if isinstance(value, type)
            and not name.startswith("_")
            and hasattr(value, "run_repeat")
        )
    assert len(discovered) == expected, (platform, discovered)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BENCHMARK_ROOT.parent)

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
    )

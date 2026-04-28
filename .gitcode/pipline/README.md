# Linux ARM CI 脚本

本目录存放的是 Linux ARM CI 相关 job 的 shell 脚本版本。

容器镜像依赖准备请参考 [container_setup.md](container_setup.md)。

## 对应关系

- `lint.sh` -> 上游 `Linux tests / lint`
- `smoke_test.sh` -> 上游 `Linux tests / smoke_test`
- `benchmark.sh` -> 上游 `Linux tests / benchmark`
- `full.sh` -> 上游 `Linux tests / full`
- `incremental_coverage.sh` -> 新增 `incremental_coverage`

## 使用方式

可以在仓库根目录或任意子目录下执行这些脚本：

```bash
.gitcode/pipline/lint.sh
.gitcode/pipline/smoke_test.sh
.gitcode/pipline/benchmark.sh
.gitcode/pipline/full.sh
.gitcode/pipline/incremental_coverage.sh
```

所有脚本都支持以下环境变量：

- `USE_VENV`: `0` 表示直接使用当前 Python 解释器，`1` 表示创建或使用 venv。默认值：`0`
- `SKIP_SYSTEM_DEPS`: `1` 表示跳过系统包管理器安装，适合 runner 已预装依赖的场景。默认值：`1`
- `PY_DEPS_MODE`: `check` 表示只检查 Python 依赖是否已存在，`install` 表示按需安装。默认值：`check`
- `CI_DUMP_MESON_LOG`: `1` 表示脚本失败时打印 `build/meson-logs/meson-log.txt`。默认值：`0`

各 job 的专属变量：

- `smoke_test.sh`
  - `SMOKE_TEST_ARGS`: 传给 `spin test --` 的参数
- `benchmark.sh`
  - `BENCHMARK_ARGS`: 传给 `spin bench` 的 benchmark 参数
- `full.sh`
  - `FULL_PYTEST_ARGS`: 仅传给 `pytest` 的参数；不要包含 `--cov` 或 `--cov-report`
  - `FULL_PYTHONOPTIMIZE`: 设为 `2` 时使用 `PYTHONOPTIMIZE=2` 运行 full suite；默认不设置 `PYTHONOPTIMIZE`
  - `FULL_GCC_COVERAGE_ICE_WORKAROUND`: `1` 表示 GCC 12 以下的 coverage build 使用 Meson `-Doptimization=0` 以规避编译器 ICE。默认值：`1`
- `incremental_coverage.sh`
  - `COMPARE_BRANCH`: diff 对比基线，默认 `origin/main`
  - `DIFF_COVER_FAIL_UNDER`: 增量覆盖率最低阈值
  - `RUN_FULL_JOB_IF_MISSING`: 设为 `1` 时，若缺少 coverage XML 则先运行 `full.sh`

## 说明

- `common.sh` 在未跳过系统依赖时，会自动检测并使用 `apt-get`、`dnf` 或 `yum`。
- `PY_DEPS_MODE=check` 是预构建 CI 镜像的推荐模式；如果是临时环境，需要脚本自己安装 Python 依赖时，可使用 `PY_DEPS_MODE=install`。

- `smoke_test.sh` 会针对当前解释器运行；如果平台需要 Python 版本矩阵，可以为不同 Python 环境分别调用一次该脚本。
- `full.sh` 会同时生成 Python 覆盖率和 C/C++ 覆盖率产物，便于 `incremental_coverage.sh` 复用同一份覆盖率结果。上游 `Linux tests / full` 使用 `PYTHONOPTIMIZE=2 pytest numpy --durations=10 --timeout=600 --cov-report=html:build/coverage`，但 C/C++ 覆盖率仍是 TODO；本脚本在此基础上通过 `spin test --gcov` 和 `pytest-cov` 同时采集 Python 与 C/C++ 覆盖率。
- `full.sh` 默认不设置 `PYTHONOPTIMIZE`，并会清除调用环境里已有的 `PYTHONOPTIMIZE`。原因是 `PYTHONOPTIMIZE=2` 会移除 docstring 和部分签名元数据，可能导致依赖运行时签名信息的测试（例如 `numpy.ma` 的 signature 测试）失败；默认关闭可让覆盖率任务覆盖普通运行时路径并稳定产出 Python/C 覆盖率。若需要复现上游 full job 的优化模式，可显式执行 `FULL_PYTHONOPTIMIZE=2 .gitcode/pipline/full.sh`，但这可能重新触发这类签名元数据相关失败，且覆盖率结果会反映 `-OO` 下的运行路径。
- `full.sh` 会为派生的扩展构建追加 `LDFLAGS=--coverage`。这样 `numpy/random/tests/test_extending.py::test_cython` 这类在测试中临时调用 Meson 构建的 Cython/C++ 扩展，可以在链接覆盖率插桩后的 NumPy 静态库时同时链接 gcov runtime，避免导入临时 `.so` 时出现未解析的 coverage 符号。
- `full.sh` 在 Python `3.12` 及以上版本会自动追加 `--ignore=numpy/distutils/tests`，因为这些版本里已经没有 `numpy.distutils`。
- `benchmark.sh` 保留了上游 workflow 中 `asv machine` 与 `spin bench --quick` 分开的执行方式。

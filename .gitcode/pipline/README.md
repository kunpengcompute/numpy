# Linux ARM CI 脚本

本目录存放的是 Linux ARM CI 相关 job 的 shell 脚本版本。

容器镜像依赖准备请参考
[container_setup.md](/C:/d/work/py_proj/doc/numpy/tools/ci/linux-arm/container_setup.md)。

## 对应关系

- `lint.sh` -> 上游 `Linux tests / lint`
- `smoke_test.sh` -> 上游 `Linux tests / smoke_test`
- `benchmark.sh` -> 上游 `Linux tests / benchmark`
- `full.sh` -> 上游 `Linux tests / full`
- `incremental_coverage.sh` -> 新增 `incremental_coverage`

## 使用方式

可以在仓库根目录或任意子目录下执行这些脚本：

```bash
tools/ci/linux-arm/lint.sh
tools/ci/linux-arm/smoke_test.sh
tools/ci/linux-arm/benchmark.sh
tools/ci/linux-arm/full.sh
tools/ci/linux-arm/incremental_coverage.sh
```

所有脚本都支持以下环境变量：

- `USE_VENV`: `0` 表示直接使用当前 Python 解释器，`1` 表示创建或使用 venv。默认值：`0`
- `SKIP_SYSTEM_DEPS`: `1` 表示跳过系统包管理器安装，适合 runner 已预装依赖的场景。默认值：`1`
- `PY_DEPS_MODE`: `check` 表示只检查 Python 依赖是否已存在，`install` 表示按需安装。默认值：`check`

各 job 的专属变量：

- `smoke_test.sh`
  - `SMOKE_TEST_ARGS`: 传给 `spin test --` 的参数
- `benchmark.sh`
  - `BENCHMARK_ARGS`: 传给 `spin bench` 的 benchmark 参数
- `full.sh`
  - `FULL_PYTEST_ARGS`: 仅传给 `pytest` 的参数；不要包含 `--cov` 或 `--cov-report`
- `incremental_coverage.sh`
  - `COMPARE_BRANCH`: diff 对比基线，默认 `origin/main`
  - `DIFF_COVER_FAIL_UNDER`: 增量覆盖率最低阈值
  - `RUN_FULL_JOB_IF_MISSING`: 设为 `1` 时，若缺少 coverage XML 则先运行 `full.sh`

## 说明

- `common.sh` 在未跳过系统依赖时，会自动检测并使用 `apt-get`、`dnf` 或 `yum`。
- `PY_DEPS_MODE=check` 是预构建 CI 镜像的推荐模式；如果是临时环境，需要脚本自己安装 Python 依赖时，可使用 `PY_DEPS_MODE=install`。

- `smoke_test.sh` 会针对当前解释器运行；如果平台需要 Python 版本矩阵，可以为不同 Python 环境分别调用一次该脚本。
- `full.sh` 会同时生成 HTML 和 XML 两种覆盖率产物，便于 `incremental_coverage.sh` 复用同一份覆盖率结果。当前脚本使用的是 `coverage run -m pytest`，在 `pytest` 结束后再生成这些产物，而不是依赖 `pytest-cov` 在测试过程中直接输出。
- `full.sh` 在 Python `3.12` 及以上版本会自动追加 `--ignore=numpy/distutils/tests`，因为这些版本里已经没有 `numpy.distutils`。
- `benchmark.sh` 保留了上游 workflow 中 `asv machine` 与 `spin bench --quick` 分开的执行方式。

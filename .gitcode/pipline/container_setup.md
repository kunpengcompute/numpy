# Linux ARM CI 容器准备说明

本文档说明了在全新容器镜像中运行本目录 shell 脚本所需的依赖。

## 推荐模式

预构建镜像推荐使用以下方式：

```bash
export USE_VENV=0
export SKIP_SYSTEM_DEPS=1
export PY_DEPS_MODE=check
```

在这种模式下，镜像中应提前具备全部系统依赖和 Python 依赖。

## 系统层依赖

建议在镜像中安装以下软件包：

- `bash`
- `git`
- `conda`
- `gcc`
- `g++`
- `gfortran`
- `pkg-config`
- `make`
- `util-linux`

说明：

- `util-linux` 提供了 `benchmark.sh` 在可用时会使用的 `script` 命令；没有它脚本仍然可以运行。
- `git` 不仅用于仓库操作，也用于执行 `git submodule update --init --recursive`。

## Conda 环境

创建一个 Python `3.11.15` 的 conda 环境：

```bash
conda create -y -n numpy-ci python=3.11.15
conda activate numpy-ci
```

推荐默认使用 Python `3.11.15`，因为它与当前 `full.sh` 的验证路径最匹配。

请确认环境中可用 `pip`，因为脚本会调用 `python -m pip`。

## Python 依赖

在 conda 环境中安装以下 requirements 文件：

- `requirements/build_requirements.txt`
- `requirements/test_requirements.txt`
- `requirements/linter_requirements.txt`
- `requirements/ci32_requirements.txt`

另外还需要安装这些额外包：

- `asv<0.6.5`
- `virtualenv`
- `packaging`
- `diff-cover`

示例：

```bash
python -m pip install \
  -r requirements/build_requirements.txt \
  -r requirements/test_requirements.txt \
  -r requirements/linter_requirements.txt \
  -r requirements/ci32_requirements.txt \
  "asv<0.6.5" \
  virtualenv \
  packaging \
  diff-cover
```

## 仓库准备

代码拉取后，在运行任何 CI 脚本前先初始化子模块：

```bash
git submodule update --init --recursive
```

这一步是必须的，因为这些脚本依赖 `vendored-meson/meson` 等子模块。

## 适用范围

完成上述准备后，可以直接运行以下脚本：

- `.gitcode/pipline/lint.sh`
- `.gitcode/pipline/smoke_test.sh`
- `.gitcode/pipline/benchmark.sh`
- `.gitcode/pipline/full.sh`
- `.gitcode/pipline/incremental_coverage.sh`

## 可选的安装模式

如果后续某个环境希望由脚本按需安装 Python 依赖，可以切换为：

```bash
export PY_DEPS_MODE=install
```

在这种模式下，镜像仍然需要提供系统层依赖和 conda 环境，但 Python 包的 `pip install` 将由脚本自己完成。

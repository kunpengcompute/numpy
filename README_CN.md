# BoostKit NumPy

BoostKit NumPy 是基于上游社区 NumPy 的鲲鹏平台性能优化项目，属于鲲鹏 BoostKit 应用使能套件生态的一部分，旨在提升 NumPy 在 ARM64 / 鲲鹏环境下的执行性能。

本项目面向 NumPy 的性能相关实现进行优化，并通过 benchmark 用例对优化效果进行评估。在提升性能的同时，本项目保持与上游 NumPy 行为、接口语义和数值结果的一致性。

## 项目状态

本项目处于持续开发和优化阶段。

当前工作重点包括 ARM64 / 鲲鹏平台性能优化、benchmark 结果验证，以及保持与上游 NumPy 行为的兼容性。

## 环境要求

推荐在 Linux ARM64 / aarch64 环境下构建和运行本项目，重点目标平台为鲲鹏服务器。x86_64 Linux 环境可用于 baseline 对比和兼容性验证。

推荐环境如下：

- Python 3.14
- GCC 12.2.1 / 12.3.1
- Meson / meson-python
- ASV
- spin / pytest

具体构建依赖请以 `pyproject.toml`、`meson.build` 和 `doc/source/building` 中的说明为准。

## 特性概览

### ARM64 / 鲲鹏性能优化

本项目面向 ARM64 架构进行 NumPy 性能优化，重点关注 NumPy 在鲲鹏服务器上的运行表现。

优化工作覆盖 NumPy 中的性能相关实现，具体优化内容以版本变更说明和 benchmark 结果为准。优化目标是在保持 NumPy 原有行为一致性的前提下，提升关键场景下的执行性能。

### Benchmark 性能验证

本项目使用 ASV benchmark 对性能变化进行评估。benchmark 结果用于验证优化收益、分析性能波动，并识别可能存在的劣化用例。

### 上游 NumPy 兼容性

本项目基于上游社区 NumPy，保持与上游 NumPy 公共 API、用户可见行为、异常行为和数值语义的一致性。

性能优化不应改变 NumPy 原有接口语义，也不应影响已有测试用例的正确性。

## 功能测试

运行完整功能测试：

```bash
spin test -m full
```

也可以根据需要使用 pytest 运行指定测试：

```bash
python -m pytest numpy/path/to/test_file.py
```

## 性能测试

本项目使用 ASV 进行性能测试。

运行指定 benchmark：

```bash
asv run -b benchmark_name
```

对比 baseline commit 与优化 commit：

```bash
asv continuous BASELINE_COMMIT OPTIMIZED_COMMIT -b benchmark_name
```

## 更多文档

详细的变更说明、构建指南和开发说明请参阅以下文档或目录：

- [doc/release/upcoming_changes/00001.arm64_kunpeng_optimization.rst](doc/release/upcoming_changes/00001.arm64_kunpeng_optimization.rst) — ARM64 / 鲲鹏性能优化概述，包含本版本主要优化方向、性能劣化修复、benchmark 覆盖和验证说明。
- [doc/release/upcoming_changes](doc/release/upcoming_changes) — 即将发布版本的变更说明目录，包含新特性、修复项和其他变更记录模板。
- [doc/source/building](doc/source/building) — 源码构建相关文档，包含 BLAS/LAPACK、编译器选项、交叉编译、FFT 后端、Meson 构建机制和二进制分发等说明。
- [doc/source/dev](doc/source/dev) — 开发者文档，包含开发环境、开发流程、调试方法、文档构建、FFT 后端开发、发布流程和代码评审指南等内容。

## License

本项目遵循上游 NumPy 及相关第三方依赖的许可证要求。

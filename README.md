# BoostKit NumPy

## NumPy介绍

BoostKit NumPy是基于上游社区NumPy的鲲鹏平台性能优化项目，属于鲲鹏BoostKit应用使能套件生态的一部分，旨在提升NumPy在ARM64/鲲鹏环境下的执行性能。

本项目面向NumPy的性能相关实现进行优化，并通过benchmark用例对优化效果进行评估。在提升性能的同时，本项目保持与上游NumPy行为、接口语义和数值结果的一致性。

## 特性说明

### ARM64/鲲鹏性能优化

本项目面向 ARM64架构进行NumPy性能优化，重点关注NumPy在鲲鹏服务器上的运行表现。

优化工作覆盖NumPy中的性能相关实现，具体优化内容以版本变更说明和benchmark 结果为准。优化目标是在保持NumPy原有行为一致性的前提下，提升关键场景下的执行性能。

### Benchmark 性能验证

本项目使用 ASV benchmark 对性能变化进行评估。benchmark 结果用于验证优化收益、分析性能波动，并识别可能存在的劣化用例。

### 上游 NumPy 兼容性

本项目基于上游社区 NumPy，保持与上游 NumPy 公共 API、用户可见行为、异常行为和数值语义的一致性。

性能优化不应改变 NumPy 原有接口语义，也不应影响已有测试用例的正确性。

## 项目状态

本项目处于持续开发和优化阶段。

当前工作重点包括ARM64/鲲鹏平台性能优化、benchmark 结果验证，以及保持与上游 NumPy 行为的兼容性。

具体构建依赖请以`pyproject.toml`、`meson.build`和`docs/source/building`中的说明为准。

## 容器环境部署

容器环境部署通过基于镜像构建容器环境放方式来部署

### 基于镜像构建容器环境

基于镜像构建容器环境，首先需要获取镜像。通过命令直接拉取预构建镜像（推荐）

### 步骤 获取镜像

### 选项：直接拉取预构建镜像

拉取镜像的命令为：

```shell
docker pull swr.cn-north-4.myhuaweicloud.com/cloud_boostkit/openeuler22.03_lts_sp3:numpy-2.4.3
```

### 镜像配置

拉取镜像后需要配置环境变量，从而正确识别 conda 预装环境和完整 Python 工具链。
```shell
 export PATH=/opt/miniforge3/bin:$PATH
```

## 更多文档

|名称|路径|简介|
|--|--|--|
|快速入门|[quick_start.md](./docs/zh/quick_start.md)|提供功能测试与性能测试（ASV）指导，面向开发/测试人员。|
|鲲鹏优化版使用指南|[user_guide.md](./docs/zh/user_guide.md)|提供鲲鹏优化版NumPy发行版wheel包的下载校验、安装与生效验证指导。|
|版本说明书|[release_notes.md](./docs/zh/release_notes.md)|提供NumPy的基础信息和特性更新信息。|

## License

本项目遵循上游NumPy及相关第三方依赖的许可证要求。

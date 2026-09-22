# 快速入门

如果需求在鲲鹏服务器上安装KML、构建带KML FFT后端的BoostKit NumPy，并验证 `kmlfft` 已经实际生效，请参阅[KML FFT 构建与使用指南](../../KML_FFT_GUIDE_CN.md)。

指南同时提供完整HPCKit安装和从HPCKit中只提取KML的极简安装方式，适合第一次接触HPCKit、KML或NumPy源码构建的用户。

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

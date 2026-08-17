# KML FFT 构建与使用指南

本文面向第一次接触 HPCKit、KML 和 NumPy 源码构建的用户，介绍如何在
鲲鹏服务器上安装鲲鹏数学库（Kunpeng Math Library，KML），构建带有
KML FFT 后端的 BoostKit NumPy，并验证 KML FFT 已经真正生效。

完成本文后，你将能够：

- 从 HPCKit 中安装完整开发套件，或者只提取并安装 KML。
- 找到当前鲲鹏 CPU 对应的 KML FFT 动态库目录。
- 使用 `spin` 或 `pip` 构建带 KML FFT 支持的 NumPy。
- 在 `pocketfft` 和 `kmlfft` 之间切换。
- 根据错误现象排查头文件、链接库、运行库和构建缓存问题。

> [!IMPORTANT]
> 本文以 `HPCKit_26.1.RC1_Linux-aarch64.tar.gz` 为经过验证的示例。
> 如果下载的是其他版本，请将命令中的 `26.1.RC1` 替换为实际版本。
> HPCKit 和 KML 中的二进制库面向 Linux aarch64，不能在 x86_64
> 机器上直接编译或运行。
>
> 除特别说明外，本文中的终端命令均应在 Bash 中执行。
> `/path/to/download` 和 `/path/to/numpy` 是示例占位符，执行前请
> 分别替换为 HPCKit 下载目录和 BoostKit NumPy 源码目录的真实路径。

## 1. 选择安装方式

HPCKit 压缩包中包含编译器、MPI、数学库和其他 HPC 组件。BoostKit
NumPy 的 KML FFT 后端只依赖其中的 KML。

| 安装方式 | 适用场景 | KML GCC 版安装目录示例 | 空间和时间 |
| --- | --- | --- | --- |
| 完整安装 HPCKit | 还需要编译器、MPI 或其他 HPCKit 组件 | `/opt/HPCKit/26.1.RC1/kml/gcc` | 安装脚本要求约 10 GiB 可用空间 |
| 只安装 KML（推荐） | 只需要 NumPy 的 KML FFT 加速 | `$HOME/.local/kml/gcc` | 26.1.RC1 GCC 版安装后约 785 MiB |

“只安装 KML”是指不安装 HPCKit 的其他组件。KML 本身还包含 BLAS、
稀疏计算、向量数学库和 FFT 等子库，随包安装脚本不会进一步只复制
FFT 文件。

当前项目不通过 `pkg-config` 探测 KML FFT。构建时需要显式提供
`kfft.h` 所在目录，以及同时包含 `libkfft.so` 和 `libkfftf.so`
的目录。

## 2. 安装前检查

### 2.1 检查机器架构

在目标服务器执行：

```bash
uname -m
```

预期输出：

```text
aarch64
```

如果输出 `x86_64`，请切换到鲲鹏/aarch64 服务器。KML 的
`kfft.h` 包含 ARM NEON 头文件，动态库也是 ARM aarch64 ELF 文件。

### 2.2 检查基础工具

```bash
python3 --version
gcc --version
g++ --version
git --version
tar --version
```

本项目要求 Python 3.11 或更高版本，项目推荐环境可参见
[README_CN.md](README_CN.md)。如果选择极简 KML 安装，还需要系统中
已有可用的 GCC/G++ 工具链和 GCC OpenMP 运行库 `libgomp.so.1`。
源码构建还需要 Python 开发头文件；RPM 系发行版中的软件包通常名为
`python3-devel`，DEB 系发行版中通常名为 `python3-dev`。创建虚拟
环境失败时，还需要安装发行版提供的 Python `venv` 组件。

### 2.3 检查磁盘空间

```bash
df -h .
```

- 完整 HPCKit 安装器会检查安装位置是否至少有约 10 GiB 可用空间。
- 极简 KML 仍需保留 HPCKit 下载包、KML 解压目录和安装目录；安装完成
  前建议准备至少 6 GiB 可用空间。安装成功后可以删除解压目录以回收
  空间，但不要删除正在使用的 KML 安装目录。

## 3. 下载 HPCKit

从[鲲鹏 HPCKit 官方下载页面](https://www.hikunpeng.com/developer/hpc/hpckit-download)
下载 Linux aarch64 安装包。本文使用：

```text
HPCKit_26.1.RC1_Linux-aarch64.tar.gz
```

进入下载目录并确认文件存在：

```bash
cd /path/to/download
ls -lh HPCKit_26.1.RC1_Linux-aarch64.tar.gz
```

如果下载页面同时提供 SHA256 校验值，建议在安装前核对：

```bash
sha256sum HPCKit_26.1.RC1_Linux-aarch64.tar.gz
```

下面的“完整安装”和“极简安装”二选一即可。

## 4. 方式一：完整安装 HPCKit

如果还需要 HPCKit 中的编译器、MPI 或其他组件，可以按照下载页面
对应版本的《HPCKit 安装指南》完成安装。也可以使用包内的一键安装
参数安装默认选中的组件。

## 5. 方式二：从 HPCKit 中极简安装 KML

这是只使用 KML FFT 时的推荐方式。外层 HPCKit 包中需要的文件只有：

```text
HPCKit_26.1.RC1_Linux-aarch64/
└── package/
    └── KunpengHPCKit-kml.26.1.RC1.tar.gz
```

### 5.1 只提取 KML 子包

回到 HPCKit 下载包所在目录：

```bash
cd /path/to/download
```

从 2 GiB 以上的 HPCKit 外层压缩包中只提取 KML 子包：

```bash
tar -xzf HPCKit_26.1.RC1_Linux-aarch64.tar.gz \
    HPCKit_26.1.RC1_Linux-aarch64/package/KunpengHPCKit-kml.26.1.RC1.tar.gz
```

再解压 KML 子包：

```bash
tar -xzf \
    HPCKit_26.1.RC1_Linux-aarch64/package/KunpengHPCKit-kml.26.1.RC1.tar.gz

cd KunpengHPCKit-kml.26.1.RC1
```

可以先查看 KML 安装器支持的参数：

```bash
bash install.sh --help
```

### 5.2 推荐：安装 GCC 版 KML 到用户目录

```bash
bash install.sh \
    --prefix="$HOME/.local/kml" \
    --use-gcc
```

`--prefix` 指定的是 KML 的父目录，安装器还会追加编译器目录。因此
实际 KML 根目录是：

```text
$HOME/.local/kml/gcc
```

加载 KML 环境：

```bash
source "$HOME/.local/kml/gcc/env/setvars.sh"
```

### 5.3 可选：安装到系统目录

```bash
sudo bash install.sh \
    --prefix=/opt/kml \
    --use-gcc

source /opt/kml/gcc/env/setvars.sh
```

此时 KML 根目录为 `/opt/kml/gcc`。

> [!WARNING]
> 不要省略 `--use-gcc` 或 `--use-bisheng`。26.1.RC1 子包的安装脚本
> 在未选择任何编译器版本时仍可能返回成功，但不会安装任何文件。
> 本项目推荐 GCC 路线；只有在 NumPy 也使用匹配的 BiSheng 工具链
> 构建时，才选择 `--use-bisheng`。

## 6. 加载并验证 KML 环境

无论选择哪种安装方式，每次打开新终端后，都需要先加载对应环境：

| 安装方式 | GCC 环境初始化命令 |
| --- | --- |
| 完整 HPCKit，系统安装 | `source /opt/HPCKit/latest/setvars.sh --use-gcc` |
| 完整 HPCKit，用户安装 | `source "$HOME/.local/HPCKit/latest/setvars.sh" --use-gcc` |
| 极简 KML，系统安装 | `source /opt/kml/gcc/env/setvars.sh` |
| 极简 KML，用户安装 | `source "$HOME/.local/kml/gcc/env/setvars.sh"` |

环境脚本会设置 `KMLROOT`、头文件搜索路径和动态库搜索路径，并根据
`/proc/cpuinfo` 选择 `neon`、`sve`、`sve_no_f64mm` 或 `sve512`
等库目录。

先确认 `KMLROOT`：

```bash
printf 'KMLROOT=%s\n' "$KMLROOT"
test -f "$KMLROOT/include/kfft.h" &&
    echo "[OK] 找到 $KMLROOT/include/kfft.h"
```

然后从 KML 配置好的动态库路径中，找出同时包含双精度和单精度 FFT
库的 CPU 专用目录：

```bash
KML_FFT_LIB_DIR=""

IFS=: read -r -a kml_search_dirs <<< "$LD_LIBRARY_PATH"
for kml_search_dir in "${kml_search_dirs[@]}"; do
    if [[ -n "$KMLROOT" &&
          "$kml_search_dir" == "$KMLROOT"/lib/* &&
          -f "$kml_search_dir/libkfft.so" &&
          -f "$kml_search_dir/libkfftf.so" ]]; then
        KML_FFT_LIB_DIR="$kml_search_dir"
        break
    fi
done

printf 'KML_FFT_LIB_DIR=%s\n' "$KML_FFT_LIB_DIR"
```

预期输出类似：

```text
KML_FFT_LIB_DIR=/opt/kml/gcc/lib/sve
```

不同鲲鹏 CPU 的目录可能不同，不要直接照抄上面的 `sve`。继续检查
三个构建必需文件：

```bash
if [[ -n "$KML_FFT_LIB_DIR" &&
      -f "$KMLROOT/include/kfft.h" &&
      -f "$KML_FFT_LIB_DIR/libkfft.so" &&
      -f "$KML_FFT_LIB_DIR/libkfftf.so" ]]; then
    export KML_FFT_LIB_DIR
    echo "[OK] KML FFT 头文件和动态库检查通过"
else
    unset KML_FFT_LIB_DIR
    echo "[ERROR] KML FFT 头文件或动态库缺失，请不要继续构建" >&2
fi
```

只有看到 `[OK]` 才能继续。如果看到 `[ERROR]`，请先查看
[常见问题](#12-常见问题)，不要继续构建。

## 7. 准备 NumPy 构建环境

进入 BoostKit NumPy 源码根目录：

```bash
cd /path/to/numpy
```

初始化 Git 子模块：

```bash
git submodule update --init
```

创建独立 Python 虚拟环境：

```bash
python3 -m venv venv-kml
source venv-kml/bin/activate
```

安装构建依赖和测试依赖：

```bash
python -m pip install -r requirements/build_requirements.txt
python -m pip install -r requirements/test_requirements.txt
```

确认当前终端仍然保留了 KML 环境：

```bash
printf 'KMLROOT=%s\n' "$KMLROOT"
printf 'KML_FFT_LIB_DIR=%s\n' "$KML_FFT_LIB_DIR"
```

如果输出为空，请重新执行第 6 节对应的环境初始化和库目录查找命令。

## 8. 构建带 KML FFT 的 NumPy

### 8.1 推荐：使用 `spin`

在源码根目录执行：

```bash
spin build --clean -- \
    -Dfft-backend=kml \
    -Dfft-include-dir="$KMLROOT/include" \
    -Dfft-lib-dir="$KML_FFT_LIB_DIR"
```

三个参数的含义如下：

- `fft-backend=kml`：选择 KML FFT 构建分支。
- `fft-include-dir`：指向包含 `kfft.h` 的目录。
- `fft-lib-dir`：指向同时包含 `libkfft.so` 和 `libkfftf.so` 的
  CPU 专用目录。

`--clean` 用于避免之前不带 KML 的 Meson 配置被重复使用。首次构建
也可以保留该参数。

### 8.2 可选：使用 `pip` 安装

如果目标是把 NumPy 安装到当前虚拟环境，而不是进行源码开发，可以
选择：

```bash
python -m pip install --no-build-isolation \
    -Csetup-args=-Dfft-backend=kml \
    -Csetup-args=-Dfft-include-dir="$KMLROOT/include" \
    -Csetup-args=-Dfft-lib-dir="$KML_FFT_LIB_DIR" \
    .
```

不要在同一个源码目录中混用 `spin` 构建和 `pip install -e .`
editable 安装。如果使用普通 `pip install .`，安装后建议离开 NumPy
源码目录再运行 Python，以免导入源码树而不是已安装版本。

## 9. 验证 KML 扩展已经构建

使用 `spin` 构建时执行：

```bash
spin python -- -c \
    "from numpy.fft import _kml_fft_umath; print(_kml_fft_umath.__file__)"
```

使用 `pip` 安装时，在 NumPy 源码目录之外执行：

```bash
python -c \
    "from numpy.fft import _kml_fft_umath; print(_kml_fft_umath.__file__)"
```

命令应打印一个 `_kml_fft_umath` 扩展模块路径。如果出现
`ImportError`，说明 KML 扩展没有构建成功，或者运行时动态库路径
不完整。

还可以确认实际导入的是本次构建的 NumPy：

```bash
spin python -- -c "import numpy as np; print(np.__file__)"
```

如果使用 `pip` 路线，请将 `spin python` 替换为 `python`。

## 10. 使用 KML FFT

即使已经构建 KML 扩展，在未设置全局后端或环境变量时，NumPy 的默认
FFT 后端仍然是 `pocketfft`。可以在一个代码块内临时切换到
`kmlfft`：

```python
import numpy as np

x = np.arange(16, dtype=np.float64)

print("默认后端：", np.fft.get_backend())

with np.fft.set_backend("pocketfft"):
    reference = np.fft.fft(x)

with np.fft.set_backend("kmlfft"):
    print("当前后端：", np.fft.get_backend())
    result = np.fft.fft(x)

np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-12)
print("KML FFT 计算结果验证通过")
```

保存为 `verify_kml_fft.py` 后，`spin` 构建可执行：

```bash
spin python verify_kml_fft.py
```

也可以将 KML FFT 设置为当前进程的全局后端：

```python
import numpy as np

np.fft.set_global_backend("kmlfft")
print(np.fft.get_backend())

result = np.fft.fft(np.arange(16, dtype=np.float64))

np.fft.reset_backend()
```

或者在启动 Python 前设置环境变量：

```bash
NUMPY_FFT_BACKEND=kmlfft spin python -- -c \
    'import numpy as np; assert np.fft.get_backend() == "kmlfft"; print(np.fft.get_backend())'
```

`NUMPY_FFT_BACKEND` 在 `numpy.fft` 第一次导入时读取，因此必须在
首次导入 `numpy.fft` 前设置。这里的 `assert` 可以防止 KML 未注册
时仅发出警告后回退到 `pocketfft`，却被误认为验证成功。

## 11. 运行测试

先运行 FFT 后端专项测试：

```bash
spin test -m full -t numpy/fft/tests/test_fft_backend.py -- -rs
```

`-rs` 会显示测试跳过原因。即使 KML 可用，专门模拟“KML 不可用”的
反向测试仍可能按设计跳过；需要重点确认没有因为
`kmlfft backend not available` 而跳过 KML 功能测试。

让通用 FFT 功能测试显式选择 KML 后端：

```bash
NUMPY_FFT_BACKEND=kmlfft spin test -m full \
    -t numpy/fft/tests/test_pocketfft.py -- -rs
```

再根据需要运行 FFT 全部测试或完整 NumPy 测试：

```bash
spin test -m full -t numpy/fft/tests
spin test -m full
```

仅看到测试“没有失败”还不够：KML 不可用时，部分 KML 专项测试会被
跳过。第 9 节的 `_kml_fft_umath` 直接导入，以及第 10 节中
`get_backend() == "kmlfft"` 的检查，才是确认 KML 后端已经注册的
关键步骤。

## 12. 常见问题

### 12.1 `uname -m` 输出 `x86_64`

原因：当前机器不是 Linux aarch64。

处理：切换到鲲鹏/aarch64 服务器后重新安装和构建。不能直接在
x86_64 上链接 HPCKit 中的 aarch64 KML 动态库。

### 12.2 `KMLROOT` 为空

原因：当前终端没有加载 KML 环境。

处理：根据安装方式重新执行：

```bash
source /opt/HPCKit/latest/setvars.sh --use-gcc
```

或者：

```bash
source "$HOME/.local/kml/gcc/env/setvars.sh"
```

### 12.3 极简安装命令成功，但安装目录为空

原因：漏写了 `--use-gcc` 或 `--use-bisheng`。

处理：重新执行安装命令，并明确选择与 NumPy 编译器匹配的版本：

```bash
bash install.sh --prefix="$HOME/.local/kml" --use-gcc
```

### 12.4 Meson 报找不到 `kfft` 或 `kfftf`

原因：`fft-lib-dir` 指向了 `$KMLROOT/lib`，而不是实际 CPU 对应的
子目录，或者当前目录中缺少其中一个动态库。

检查：

```bash
ls -l "$KML_FFT_LIB_DIR/libkfft.so"
ls -l "$KML_FFT_LIB_DIR/libkfftf.so"
```

处理：重新执行第 6 节的自动查找代码，再使用
`-Dfft-lib-dir="$KML_FFT_LIB_DIR"` 构建。

### 12.5 编译器报找不到 `kfft.h`

检查：

```bash
ls -l "$KMLROOT/include/kfft.h"
```

处理：确保 `fft-include-dir` 使用的是
`-Dfft-include-dir="$KMLROOT/include"`，而不是 KML 安装前缀。

### 12.6 `Unknown FFT backend: 'kmlfft'`

原因：`_kml_fft_umath` 没有构建，或者因动态库加载失败而未注册。
普通 `import numpy` 成功，或者默认 `get_backend()` 返回
`pocketfft`，都不能证明 KML 扩展可用；注册 KML 后端时发生的
`ImportError` 会被静默处理。

先直接导入扩展以显示真实错误：

```bash
spin python -c "from numpy.fft import _kml_fft_umath"
```

如果此前已经构建过不带 KML 的 NumPy，清理后重新配置：

```bash
spin build --clean -- \
    -Dfft-backend=kml \
    -Dfft-include-dir="$KMLROOT/include" \
    -Dfft-lib-dir="$KML_FFT_LIB_DIR"
```

### 12.7 导入时报 `libkfft.so.1` 或 `libkfftf.so.1` 找不到

原因：构建时找到了链接库，但启动 Python 的新终端没有加载 KML
运行环境。

处理：先重新 `source` KML 环境，再启动 Python。也可以检查扩展的
动态库依赖：

```bash
KML_EXTENSION="$(find build-install -type f \
    -name '_kml_fft_umath*.so' -print -quit)"

if [[ -n "$KML_EXTENSION" ]]; then
    ldd "$KML_EXTENSION" | grep -E 'kfft|kfftf|gomp|not found'
else
    echo "[ERROR] build-install 中没有找到 KML FFT 扩展" >&2
fi
```

这条命令不导入扩展，因此即使动态库缺失也可以运行。如果使用 `pip`
安装，可以先从当前虚拟环境的安装目录中找到扩展，再执行 `ldd`：

```bash
KML_SITE_PACKAGES="$(python -c \
    'import sysconfig; print(sysconfig.get_path("platlib"))')"
KML_EXTENSION="$(find "$KML_SITE_PACKAGES/numpy/fft" -type f \
    -name '_kml_fft_umath*.so' -print -quit)"

if [[ -n "$KML_EXTENSION" ]]; then
    ldd "$KML_EXTENSION" | grep -E 'kfft|kfftf|gomp|not found'
else
    echo "[ERROR] 当前虚拟环境中没有找到 KML FFT 扩展" >&2
fi
```

### 12.8 导入时报 `libgomp.so.1` 找不到

原因：GCC 版 KML FFT 依赖 GCC OpenMP 运行库。

处理：确认使用的 GCC 工具链已完整安装，并通过系统包管理器安装与
当前 GCC 匹配的 `libgomp` 运行库。

### 12.9 构建成功，但导入的是另一份 NumPy

检查：

```bash
spin python -c "import numpy as np; print(np.__file__)"
```

使用 `spin` 构建时应通过 `spin python`、`spin test` 或
`spin ipython` 运行。使用 `pip install .` 时，应离开 NumPy 源码
目录再运行普通 `python`。

### 12.10 GCC 与 BiSheng 版本混用

KML 子包同时带有 `gcclib` 和 `bishenglib`。使用 GCC/G++ 构建
NumPy 时选择 `--use-gcc` 并加载 `gcc/env/setvars.sh`；使用 BiSheng
工具链时才选择 `--use-bisheng`。头文件、链接库和运行环境应来自
同一编译器目录。

## 13. 当前实现限制

- 只要 `fft-include-dir` 或 `fft-lib-dir` 为空，KML C 扩展就不会
  构建，NumPy 仍只使用内置 `pocketfft`。
- KML FFT 构建需要同时链接 `libkfft.so`（双精度）和
  `libkfftf.so`（单精度）。
- 当前构建系统使用 Meson `find_library` 和显式目录，不支持通过
  KML `pkg-config` 文件自动发现。
- 构建了 KML 后端不代表默认启用；默认仍是 `pocketfft`，需要通过
  上下文、全局设置或环境变量选择 `kmlfft`。
- KML 后端声明支持 `float16`、`float32`、`complex64`、
  `float64` 和 `complex128`。`longdouble` 和 `clongdouble`
  会按 dtype 静默回退到 `pocketfft`；因此
  `get_backend() == "kmlfft"` 表示 KML 已被选择，不保证每一种
  dtype 都实际由 KML 计算。
- 每个新终端都需要重新加载 KML 环境，确保运行时能找到动态库。

更多构建机制和后端架构说明可参阅：

- [FFT 后端构建说明](doc/source/building/fft_backend.rst)
- [FFT 后端架构说明](doc/source/dev/fft-backend.rst)

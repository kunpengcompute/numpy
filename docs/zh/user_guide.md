# NumPy 2.4.3鲲鹏（ARM64）优化版使用指南

本指南说明如何在鲲鹏（ARM64）服务器上安装boostkit/numpy发布的优化版wheel包，并验证优化版本已成功生效。

产品版本信息如下。

| 产品名称 | Kunpeng BoostKit |
| ---- | ---- |
| 产品版本 | 26.2.RC1 |
| 软件名称 | NumPy |
| 软件版本 | 2.4.3 |

发行版本信息如下。

| 项目 | 内容 |
| ---- | ---- |
| 代码仓 | [boostkit/numpy](https://gitcode.com/boostkit/numpy) |
| Release | v2.4.3.0-rc2 |
| wheel文件 | numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl |
| 说明 | 面向ARM64/鲲鹏平台优化关键计算路径，与上游NumPy API、行为、异常语义和数值结果兼容 |

## 安装前环境检查

### 确认CPU架构为ARM64

```bash
uname -m
```

预期输出：

```output
aarch64
```

### 确认Python 3.14解释器可用（必需）

wheel为`cp314`规格，**只能安装在Python 3.14上**，其他Python版本会报"not a supported wheel on this platform"。本指南统一使用`python3.14`命令（不修改系统已有的`python`命令指向），请先确认其可用：

```bash
python3.14 --version
```

预期输出：

```output
Python 3.14.x
```

确认pip可用：

```bash
python3.14 -m pip --version
```

若`python3.14`命令不存在或版本不对，任选一种方式获得Python 3.14：

- 方式一（root）：安装python3.14，提供/usr/bin/python3.14命令。

  ```bash
  dnf install -y python3.14
  ```

- 方式二（推荐，任何环境通用）：conda创建3.14环境并激活，激活后python与python3.14均指向3.14。

  ```bash
  conda create -n py314 python=3.14 -y
  conda activate py314
  ```

> ![](./public_sys-resources/icon-note.gif) **说明：**
> 本指南不覆盖、不改名系统已有的`python`命令（避免影响整机Python环境）。若你习惯使用`python`，请在激活虚拟环境后再使用。

### 确认glibc版本大于或等于2.34（必需）

wheel为`manylinux_2_34`规格，要求系统glibc版本大于或等于2.34：

```bash
ldd --version | head -1
```

预期输出（示例）：

```output
ldd (GNU libc) 2.34
```

### 检查是否已安装旧版NumPy

```bash
python3.14 -m pip show numpy
```

- 若已存在**上游**NumPy（非2.4.3），建议使用下文[安装](#安装)一节的方式一（venv虚拟环境隔离安装），或按方式二先卸载旧版再安装优化版。
- 若同一台机器有多个Python，注意始终用`python3.14 -m pip`而不是裸`pip`，避免装错解释器；并再次确认`python3.14 --version`为3.14.x。

## 下载wheel并校验完整性

创建下载目录并进入：

```bash
mkdir -p ~/whl-install && cd ~/whl-install
```

下载wheel与官方sha256校验文件：

```bash
curl -fLO "https://gitcode.com/boostkit/numpy/releases/download/v2.4.3.0-rc2/numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl"
curl -fLO "https://gitcode.com/boostkit/numpy/releases/download/v2.4.3.0-rc2/numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl.sha256"
```

校验完整性，文件名需与校验文件中记录的一致：

```bash
sha256sum -c numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl.sha256
```

预期输出：

```output
numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl: OK
```

若`sha256sum -c`因文件名不一致报错，可直接对比哈希值：

```bash
sha256sum numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl
cat numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl.sha256
```

两行第一列哈希必须完全相同。

## 安装

### 方式一（推荐）：venv虚拟环境隔离安装

进入下载目录，创建Python 3.14虚拟环境并激活：

```bash
cd ~/whl-install
python3.14 -m venv numpy-env
source numpy-env/bin/activate
```

确认在venv内使用的是Python 3.14，并升级pip：

```bash
python --version
python -m pip install --upgrade pip
```

安装优化版wheel：

```bash
python -m pip install numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl
```

### 方式二：卸载旧版后直接安装

```bash
python3.14 -m pip uninstall -y numpy
python3.14 -m pip install ~/whl-install/numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl
```

> ![](./public_sys-resources/icon-note.gif) **说明：**
> `pip install`本地wheel时pip不会联网改包；若之后执行`pip install -U numpy`之类命令，可能从PyPI拉取上游版本覆盖优化版，请避免，或配置pip只从内网源安装。

## 验证安装成功

### 版本检查（核心）

```bash
python3.14 - <<'EOF'
import sys
print("python :", sys.version.split()[0])
import numpy
print("numpy  :", numpy.__version__)
assert numpy.__version__ == "2.4.3", "不是优化版NumPy！"
from numpy.version import git_revision
print("git_rev:", git_revision)
assert git_revision == "558304423d330034cfdc40b1f55175b4675d437c", "git_revision与发布制品不符！"
print("VERSION_CHECK_OK")
EOF
```

其中`git_revision`断言为制品源码级校验，须与Release发布制品一致，可同时拦截两类问题：装到非发布制品源码构建的包（如旧rc/本地构建），或误装上游同版本号wheel（上游包该字段为其自身提交，断言必然失败）。

预期输出：

```output
python : 3.14.4
numpy  : 2.4.3
git_rev: 558304423d330034cfdc40b1f55175b4675d437c
VERSION_CHECK_OK
```

### 确认安装的wheel平台标签

```bash
python3.14 - <<'EOF'
import importlib.metadata as im
d = im.distribution("numpy")
tag = [l.split(": ")[1] for l in d.read_text("WHEEL").splitlines() if l.startswith("Tag:")][0]
print("wheel tag:", tag)
assert tag == "cp314-cp314-manylinux_2_34_aarch64"
print("WHEEL_TAG_OK")
EOF
```

预期输出：

```output
wheel tag: cp314-cp314-manylinux_2_34_aarch64
WHEEL_TAG_OK
```

### 构建信息确认（aarch64优化构建）

```bash
python3.14 -c "import numpy; numpy.show_config()"
```

关键确认点（已实测样例）：

```output
"Machine Information": { "host": { "cpu": "aarch64", ... }, "build": { "cpu": "aarch64", ... } }
"Compilers": { "c": { "name": "gcc", "version": "12.2.1", ... } }
"Build Dependencies": { "blas": { "name": "openblas", ... } }
```

`build.cpu == "aarch64"`即说明安装的是ARM64/鲲鹏平台的构建产物。
（可选：`pip install pyyaml`可让`show_config()`输出更友好。）

### 功能正确性冒烟和性能参考

功能正确性冒烟验证，以下脚本分别验证矩阵乘、排序、ufunc计算的输出一致性：

```bash
python3.14 - <<'EOF'
import numpy as np
rng = np.random.default_rng(42)
a = rng.standard_normal((200, 300))
assert np.allclose(a @ a.T, a.copy() @ a.T.copy())
x = rng.integers(0, 10**6, 10**7)
s = np.sort(x)
assert s[0] == x.min() and s[-1] == x.max() and np.diff(s).min() >= 0
assert np.allclose(np.exp(a[:5]), np.exp(a[:5].copy()))
print("CORRECTNESS_SMOKE_OK")
EOF
```

预期输出：

```output
CORRECTNESS_SMOKE_OK
```

性能参考，记录基准便于对比优化效果：

```bash
python3.14 - <<'EOF'
import numpy as np, time
rng = np.random.default_rng(42)
x = rng.integers(0, 10**6, 10**7)
t0 = time.perf_counter(); np.sort(x); t1 = time.perf_counter()
print(f"np.sort(1e7 int64): {1000*(t1-t0):.1f} ms")
EOF
```

预期输出（耗时因负载而异，172.0 ms为920b实机参考值）：

```output
np.sort(1e7 int64): 172.0 ms
```

## FAQ

### python3.14 --version不是3.14.x或命令不存在？

未安装Python 3.14。按[安装前环境检查](#安装前环境检查)一节安装python3.14（方式一）或使用conda环境（方式二）。

### 安装时报ERROR: not a supported wheel on this platform？

Python版本不是3.14，或架构不是aarch64。按[安装前环境检查](#安装前环境检查)一节重新检查。

### 安装后numpy.__version__不是2.4.3？

环境里存在多个Python或旧版numpy。用`python3.14 -m pip`明确解释器，或改用venv方式安装。

### 版本检查时git_revision断言失败？

安装的不是本Release发布制品：可能装了旧rc、本地构建或上游同版本号wheel。重新执行[下载wheel并校验完整性](#下载wheel并校验完整性)、[安装](#安装)两节。

### 执行pip install -U numpy后性能回退？

该命令会从PyPI拉取上游版本覆盖优化版，请重新执行[安装](#安装)一节。

### 出现libgomp/glibc相关加载错误？

系统glibc版本小于2.34，需升级操作系统或更换满足manylinux_2_34的环境。

## 卸载

方式二安装时执行：

```bash
python3.14 -m pip uninstall -y numpy
```

方式一安装时直接删除虚拟环境：

```bash
deactivate && rm -rf ~/whl-install/numpy-env
```

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| ---- | ---- | ---- |
| 01 | 2026-09-30 | 第一次正式发布。 |

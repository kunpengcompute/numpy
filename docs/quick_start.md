# NumPy 2.4.3 鲲鹏（ARM64）优化版使用指南

本指南说明如何在鲲鹏（ARM64）服务器上安装 boostkit/numpy 发布的优化版 wheel 包，并验证优化版本已成功生效。

发行版信息：

| 项目 | 内容 |
|---|---|
| 代码仓 | gitcode.com/boostkit/numpy |
| Release | `v2.4.3.0-rc2` |
| wheel 文件 | `numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl` |
| 说明 | 面向 ARM64/鲲鹏平台优化关键计算路径，与上游 NumPy API、行为、异常语义和数值结果兼容 |

> 本指南中所有命令均已在鲲鹏 920b 实机（AlmaLinux 9.4, aarch64）端到端验证通过。

## 1. 安装前环境检查

### 1.1 确认 CPU 架构为 ARM64

```bash
uname -m
```

预期输出：

```
aarch64
```

### 1.2 确认 Python 3.14 解释器可用（必需）

wheel 为 `cp314` 规格，**只能安装在 Python 3.14 上**，其他 Python 版本会报 "not a supported wheel on this platform"。本指南统一使用 `python3.14` 命令（不修改系统已有的 `python` 命令指向），请先确认其可用：

```bash
python3.14 --version
# 预期输出: Python 3.14.x

# 确认 pip 可用
python3.14 -m pip --version
```

若 `python3.14` 命令不存在或版本不对，任选一种方式获得 Python 3.14：

```bash
# 方式一（root）：安装 python3.14，提供 /usr/bin/python3.14 命令
dnf install -y python3.14

# 方式二（推荐，任何环境通用）：conda 创建 3.14 环境并激活
# 激活后 python 与 python3.14 均指向 3.14
conda create -n py314 python=3.14 -y
conda activate py314
```

> 本指南不覆盖、不改名系统已有的 `python` 命令（避免影响整机 Python 环境）。若你习惯使用 `python`，请在激活虚拟环境后再使用。

### 1.3 确认 glibc 版本 ≥ 2.34（必需）

wheel 为 `manylinux_2_34` 规格，要求系统 glibc ≥ 2.34：

```bash
ldd --version | head -1
```

预期输出（示例）：

```
ldd (GNU libc) 2.34
```

### 1.4 检查是否已安装旧版 NumPy

```bash
python3.14 -m pip show numpy
```

- 若已存在**上游** NumPy（非 2.4.3），建议使用下文第 3.1 节的 venv 隔离方式，或按 3.2 节先卸载旧版再安装优化版。
- 若同一台机器有多个 Python，注意始终用 `python3.14 -m pip ...` 而不是裸 `pip`，避免装错解释器；并再次确认 `python3.14 --version` 为 3.14.x。

## 2. 下载 wheel 并校验完整性

```bash
mkdir -p ~/whl-install && cd ~/whl-install

# 下载 wheel 与官方 sha256 校验文件
curl -fLO "https://gitcode.com/boostkit/numpy/releases/download/v2.4.3.0-rc2/numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl"
curl -fLO "https://gitcode.com/boostkit/numpy/releases/download/v2.4.3.0-rc2/numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl.sha256"

# 校验（文件名需与校验文件中记录的一致）
sha256sum -c numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl.sha256
```

预期输出：

```
numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl: OK
```

若 `sha256sum -c` 因文件名不一致报错，可直接对比哈希值：

```bash
sha256sum numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl
cat    numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl.sha256
# 两行第一列哈希必须完全相同
```

## 3. 安装

### 方式 A（推荐）：venv 虚拟环境隔离安装

```bash
cd ~/whl-install

# 创建 Python 3.14 虚拟环境
python3.14 -m venv numpy-env
source numpy-env/bin/activate

python --version          # 确认在 venv 内使用的是 3.14
python -m pip install --upgrade pip
python -m pip install numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl
```

### 方式 B：卸载旧版后直接安装

```bash
python3.14 -m pip uninstall -y numpy
python3.14 -m pip install ~/whl-install/numpy-2.4.3-cp314-cp314-manylinux_2_34_aarch64.whl
```

> 注意：`pip install` 本地 wheel 时 pip 不会联网改包；若之后执行 `pip install -U numpy` 之类命令，可能从 PyPI 拉取上游版本覆盖优化版，请避免，或配置 pip 只从内网源安装。

## 4. 验证安装成功

### 4.1 版本检查（核心）

```bash
python3.14 - <<'EOF'
import sys
print("python :", sys.version.split()[0])
import numpy
print("numpy  :", numpy.__version__)
assert numpy.__version__ == "2.4.3", "不是优化版 NumPy！"
# 制品源码级校验：git_revision 必须与 Release 发布制品一致
from numpy.version import git_revision
print("git_rev:", git_revision)
assert git_revision == "558304423d330034cfdc40b1f55175b4675d437c", "git_revision 与发布制品不符！"
print("VERSION_CHECK_OK")
EOF
```

预期输出：

```
python : 3.14.4
numpy  : 2.4.3
git_rev: 558304423d330034cfdc40b1f55175b4675d437c
VERSION_CHECK_OK
```

`git_revision` 断言可同时拦截两类问题：装到非发布制品源码构建的包（如旧 rc/本地构建），或误装上游同版本号 wheel（上游包该字段为其自身提交，断言必然失败）。

### 4.2 确认安装的 wheel 平台标签

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

```
wheel tag: cp314-cp314-manylinux_2_34_aarch64
WHEEL_TAG_OK
```

### 4.3 构建信息确认（aarch64 优化构建）

```bash
python3.14 -c "import numpy; numpy.show_config()"
```

关键确认点（已实测样例）：

```
"Machine Information": { "host": { "cpu": "aarch64", ... }, "build": { "cpu": "aarch64", ... } }
"Compilers": { "c": { "name": "gcc", "version": "12.2.1", ... } }
"Build Dependencies": { "blas": { "name": "openblas", ... } }
```

`build.cpu == "aarch64"` 即说明安装的是 ARM64/鲲鹏平台的构建产物。
（可选：`pip install pyyaml` 可让 `show_config()` 输出更友好。）

### 4.4 功能正确性冒烟 + 性能参考

```bash
python3.14 - <<'EOF'
import numpy as np, time

# 正确性冒烟
rng = np.random.default_rng(42)
a = rng.standard_normal((200, 300))
assert np.allclose(a @ a.T, a.copy() @ a.T.copy())      # 矩阵乘
x = rng.integers(0, 10**6, 10**7)
s = np.sort(x)
assert s[0] == x.min() and s[-1] == x.max() and np.diff(s).min() >= 0   # 排序
assert np.allclose(np.exp(a[:5]), np.exp(a[:5].copy())) # ufunc
print("CORRECTNESS_SMOKE_OK")

# 性能参考（记录基准，便于对比优化效果）
t0 = time.perf_counter(); np.sort(x); t1 = time.perf_counter()
print(f"np.sort(1e7 int64): {1000*(t1-t0):.1f} ms")
EOF
```

预期输出（耗时因负载而异）：

```
CORRECTNESS_SMOKE_OK
np.sort(1e7 int64): 172.0 ms     # 920b 实机参考值
```

## 5. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `python3.14 --version` 不是 3.14.x 或命令不存在 | 未安装 3.14。按第 1.2 节安装 `python3.14`（方式一）或使用 conda 环境（方式二）。 |
| `ERROR: ... not a supported wheel on this platform` | Python 版本不是 3.14，或架构不是 aarch64。按第 1 节重新检查。 |
| 安装后 `numpy.__version__` 不是 2.4.3 | 环境里存在多个 Python/旧版 numpy。用 `python3.14 -m pip` 明确解释器；或改用 venv 方式。 |
| 4.1 节 `git_revision` 断言失败 | 安装的不是本 Release 发布制品：可能装了旧 rc、本地构建或上游同版本号 wheel。重新执行第 2、3 节下载校验与安装。 |
| `pip install -U numpy` 后性能回退 | 该命令会从 PyPI 拉取上游版本覆盖优化版，请重新执行第 3 节安装。 |
| `libgomp`/glibc 相关加载错误 | 系统 glibc < 2.34，需升级操作系统或更换满足 manylinux_2_34 的环境。 |

## 6. 卸载

```bash
python3.14 -m pip uninstall -y numpy     # 方式 B 安装时
# 或（方式 A）：直接删除虚拟环境
deactivate && rm -rf ~/whl-install/numpy-env
```

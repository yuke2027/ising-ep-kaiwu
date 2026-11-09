# Kaiwu SDK 1.4.1 实操手册

> 目标：从零完成环境安装、本地经典求解、QUBO/Ising 建模、约束检查、精度适配和 SPQC 真机提交。
>
> 文档基线：Kaiwu SDK 1.4.1 中文文档，读取日期 2026-09-21。
>
> 当前工作区检查结果：默认 Python 为 **3.9.6**，且未安装 `kaiwu`。官方文档要求 **Python 3.10**，所以必须先完成第 1 节。

## 0. 先理解整条流水线

```text
现实问题
  ↓ 定义“什么算更好”
Binary(0/1) 或 Spin(-1/+1) 变量
  ↓
目标函数 + 约束惩罚项
  ↓
QUBO 模型 / Ising 模型
  ↓ 必要时做 8-bit 精度适配
经典求解器（本地调试）或 CIMOptimizer（SPQC 真机）
  ↓
解向量 → 业务答案 → 约束验证
```

核心规则：**求解器返回低能量解，不等于自动返回合法业务答案。最终一定要检查约束。**

---

## 1. 安装 Python 3.10 和 Kaiwu SDK

### 1.1 检查 Python

```bash
python3.10 --version
```

预期结果类似：

```text
Python 3.10.x
```

Kaiwu SDK 文档目前只声明支持 Python 3.10，不区分 3.10 的小版本。

### 1.2 创建隔离环境

macOS / Linux：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python --version
```

Windows PowerShell：

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
```

检查输出必须是 `Python 3.10.x`。

### 1.3 更新安装工具

```bash
python -m pip install --upgrade pip
```

### 1.4 安装企业版 wheel

1. 登录专用量子计算云平台。
2. 获取用户 ID、SDK 授权码和适合操作系统/CPU 架构的 SDK 压缩包。
3. 解压，找到 `.whl` 文件。
4. 使用实际文件路径安装：

```bash
python -m pip install "/绝对路径/实际下载的-kaiwu-wheel文件.whl"
```

不要直接照抄文档中旧的 `1.4.0` 文件名；当前站点标题和更新日志为 1.4.1，但安装页仍残留 1.4.0 示例。

### 1.5 导入检查

```bash
python -c "import kaiwu as kw; print('Kaiwu import OK'); print(getattr(kw, '__version__', 'version unavailable'))"
```

如果出现 `ModuleNotFoundError`：

```bash
which python
python -m pip show kaiwu
```

Windows：

```powershell
Get-Command python
python -m pip show kaiwu
```

两个命令必须指向同一个虚拟环境。

### 1.6 初始化 license

首次使用 `CIMOptimizer` 或 `SimulatedAnnealingOptimizer` 时，SDK可能会提示输入 `user_id` 和 `sdk_code`。也可以显式初始化：

```python
import getpass
import kaiwu as kw

user_id = input("user_id: ").strip()
sdk_code = getpass.getpass("sdk_code: ").strip()
kw.license.init(user_id, sdk_code)
```

安全要求：

- 不要把授权码硬编码到脚本。
- 不要提交授权码或 license 文件到 Git。
- 生成 license 需要可访问国内网络。
- 授权过期后，到专用量子云平台重新申请。

---

## 2. 第一个可运行例子：本地求解 Max-Cut

这是最短的入门路径，因为 Max-Cut 可以直接写成 Ising 矩阵。

### 2.1 问题意思

把图上的顶点分成两组，让跨组的边尽量多。

- 自旋 `+1`：分到 A 组。
- 自旋 `-1`：分到 B 组。
- 求解器最小化 Ising 能量。

### 2.2 保存为 `maxcut_local.py`

```python
import numpy as np
import kaiwu as kw

# 邻接矩阵：1 表示有边，0 表示无边。
adjacency = np.array(
    [
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0],
    ],
    dtype=float,
)

# Kaiwu 求最小能量；Max-Cut 要最大化割边，所以取负号。
ising_matrix = -adjacency

optimizer = kw.classical.SimulatedAnnealingOptimizer(
    initial_temperature=100,
    alpha=0.99,
    cutoff_temperature=0.001,
    iterations_per_t=10,
    size_limit=100,
    rand_seed=0,
)

solutions = optimizer.solve(ising_matrix, sort_solutions=True)
best = solutions[0]

# 对称邻接矩阵会把每条边计算两次，因此公式除以 4。
max_cut = (np.sum(adjacency) - adjacency.dot(best).dot(best)) / 4

print("best spin:", best)
print("group A:", np.flatnonzero(best == 1).tolist())
print("group B:", np.flatnonzero(best == -1).tolist())
print("max cut:", max_cut)
```

运行：

```bash
python maxcut_local.py
```

验收：

- 输出向量只包含 `-1` 和 `1`。
- A、B 两组覆盖全部顶点且没有重复。
- 相同随机种子应便于复现；模拟退火仍属于启发式方法，不保证每次都找到全局最优。

---

## 3. QUBO 建模的标准写法

### 3.1 最小模板

```python
import kaiwu as kw

x = kw.core.ndarray(4, "x", kw.core.Binary)

model = kw.core.QuboModel()
model.set_objective(3 * x[0] + x[1] - 2 * x[2] + x[0] * x[3])
model.add_constraint(x.sum() == 2, name="choose_two", penalty=10)

optimizer = kw.classical.SimulatedAnnealingOptimizer(rand_seed=0)
solution_dict, qubo_value = optimizer.solve_qubo(model)

unsatisfied_count, constraint_values = model.verify_constraint(solution_dict)

print("solution:", solution_dict)
print("QUBO value:", qubo_value)
print("unsatisfied constraints:", unsatisfied_count)
print("constraint values:", constraint_values)
```

### 3.2 四个必要动作

1. 用 `Binary`、`Integer` 或 `ndarray` 创建变量。
2. 用 `set_objective()` 设置目标。
3. 用 `add_constraint()` 添加约束。
4. 求解后调用 `verify_constraint()`。

### 3.3 数组变量

```python
x = kw.core.ndarray((5, 5), "x", kw.core.Binary)
```

含义：创建 25 个二值变量，适合表示“对象 × 位置”“任务 × 机器”等二维分配。

常用操作：

```python
x.sum()
x.sum(axis=0)
x.sum(axis=1)
kw.core.quicksum(x.ravel().tolist())
kw.core.zeros((5, 5))
```

### 3.4 从解字典恢复数组

```python
x_value = kw.core.get_val(x, solution_dict)
print(x_value)
```

---

## 4. 完整 QUBO 例子：旅行商问题 TSP

### 4.1 变量定义

`x[u, j] = 1` 表示城市 `u` 在路线的第 `j` 个位置被访问。

必须满足：

- 每个城市只出现一次。
- 每个位置只放一个城市。
- 不存在的边不能成为相邻路线。
- 总距离尽量小。

### 4.2 保存为 `tsp_local.py`

```python
import numpy as np
import kaiwu as kw


def is_edge_used(var_x, u, v, n):
    """返回边 u→v 是否在循环路线中被使用的 QUBO 表达式。"""
    return kw.core.quicksum(
        [var_x[u, j] * var_x[v, j + 1] for j in range(-1, n - 1)]
    )


# 0 表示无边；非零值为距离。
distance = np.array(
    [
        [0, 0, 4, 4, 2],
        [0, 0, 4, 2, 0],
        [4, 4, 0, 3, 3],
        [4, 2, 3, 0, 4],
        [2, 0, 3, 4, 0],
    ],
    dtype=float,
)

n = distance.shape[0]
x = kw.core.ndarray((n, n), "x", kw.core.Binary)

edges = [(u, v) for u in range(n) for v in range(n) if distance[u, v] != 0]
no_edges = [(u, v) for u in range(n) for v in range(n) if distance[u, v] == 0]

model = kw.core.QuboModel()
model.set_objective(
    kw.core.quicksum(
        [distance[u, v] * is_edge_used(x, u, v, n) for u, v in edges]
    )
)

# 每个位置只能有一个城市。
model.add_constraint(x.sum(axis=0) == 1, "position_once", penalty=20)

# 每个城市只能出现一次。
model.add_constraint(x.sum(axis=1) == 1, "city_once", penalty=20)

# 无边的两个城市不能相邻。
model.add_constraint(
    kw.core.quicksum([is_edge_used(x, u, v, n) for u, v in no_edges]),
    "connected_path",
    penalty=20,
)

optimizer = kw.classical.SimulatedAnnealingOptimizer(
    initial_temperature=100,
    alpha=0.999,
    cutoff_temperature=0.001,
    iterations_per_t=10,
    size_limit=100,
    rand_seed=0,
)

solution_dict, qubo_value = optimizer.solve_qubo(model)
unsatisfied_count, constraint_values = model.verify_constraint(solution_dict)

print("QUBO value:", qubo_value)
print("unsatisfied constraints:", unsatisfied_count)
print("constraint values:", constraint_values)

if unsatisfied_count == 0:
    x_value = kw.core.get_val(x, solution_dict)
    route = np.array(np.nonzero(x_value.T))[1]
    path_cost = kw.core.get_val(model.objective, solution_dict)
    print("valid route:", route.tolist())
    print("path cost:", path_cost)
else:
    print("当前解不合法：先调整惩罚系数或求解参数。")
```

运行：

```bash
python tsp_local.py
```

官方 5 城市示例给出的合法路径成本为 `15.0`，示例路线之一为 `[3, 1, 2, 4, 0]`。模拟退火具有随机性，路线方向或循环起点可能不同，但成本可以相同。

---

## 5. 惩罚系数怎么设置

惩罚系数相当于“违反规则的罚款”。

- 太小：求解器可能宁愿违反约束来获得更低目标值。
- 太大：目标函数被淹没，还可能扩大矩阵动态范围，造成真机精度损失。

### 5.1 自动估计单个约束的最低惩罚

```python
import kaiwu as kw

x = kw.core.ndarray(4, "x", kw.core.Binary)
constraint_expression = (x[0] + x[1] + 2 * x[3] - 2) ** 2 + 2 * x[2]
objective = 3 * (x[0] - x[1]) ** 2 - 2 * x[2]

penalty = kw.core.get_min_penalty(objective, constraint_expression)
qubo_expression = objective + penalty * constraint_expression

print("estimated penalty:", penalty)
```

### 5.2 让模型初始化惩罚系数

```python
model.initialize_penalties()
```

自动值仍是估计值。最终验收标准是：

```python
unsatisfied_count, details = model.verify_constraint(solution_dict)
assert unsatisfied_count == 0, details
```

---

## 6. QUBO 与 Ising 转换

### 6.1 QUBO 矩阵转 Ising 矩阵

```python
qubo_matrix = model.get_matrix()
ising_matrix, bias = kw.core.qubo_matrix_to_ising_matrix(qubo_matrix)

print("Ising shape:", ising_matrix.shape)
print("bias:", bias)
```

### 6.2 Ising 转回 QUBO

```python
restored_qubo, restored_bias = kw.core.ising_matrix_to_qubo_matrix(
    ising_matrix,
    remove_linear_bit=True,
)
```

### 6.3 为什么 Ising 多一个变量

QUBO 的 `0/1` 变量通过下面的关系转为自旋：

```text
s = 2x - 1
x = (s + 1) / 2
```

部分 SPQC 不能直接处理 Ising 线性项，所以 SDK 增加一个辅助自旋，把线性项变成二次耦合项。这通常会让 Ising 矩阵边长增加 1，属于正常行为。

---

## 7. 8-bit 精度适配

SPQC 当前文档限制 Ising 系数为 8 位有符号整数范围：

```text
[-128, 127]
```

重点不是原始数字是否超过 127，而是能否整体同比例缩放到该范围，同时保留足够的相对差异。

### 7.1 先检查位宽

```python
info = kw.preprocess.calculate_ising_matrix_bit_width(ising_matrix, bit_width=8)
print(info)
```

QUBO 矩阵：

```python
info = kw.preprocess.calculate_qubo_matrix_bit_width(qubo_matrix, bit_width=8)
print(info)
```

典型返回值：

```python
{"precision": 8, "multiplier": 1.0}
```

如果返回无穷大，表示简单缩放仍不能满足精度要求。

### 7.2 方法一：直接取整

```python
adjusted = kw.preprocess.adjust_ising_matrix_precision(ising_matrix, bit_width=8)
```

适用：系数大小比较接近。

风险：如果同时存在 `0.198` 和 `50`，小系数可能都被压成同一个整数，改变最优解。

### 7.3 方法二：动态范围压缩

```python
mutated = kw.preprocess.perform_precision_adaption_mutate(
    ising_matrix,
    iterations=100,
    heuristic="greedy",
    decision="heuristic",
)
```

特点：尝试修改系数并保持最优解，但能压缩多少取决于矩阵本身。

### 7.4 方法三：变量或项拆分

```python
split_matrix, split_indices = kw.preprocess.perform_precision_adaption_split(
    ising_matrix,
    param_bit=8,
    min_increment=None,
    penalty=None,
    round_to_increment=True,
    method="var",  # 也可以是 "item"
)
```

求解拆分后的矩阵：

```python
optimizer = kw.classical.SimulatedAnnealingOptimizer(rand_seed=0)
split_solutions = optimizer.solve(split_matrix, sort_solutions=True)
split_best = split_solutions[0]

original_solution = kw.preprocess.restore_split_solution(
    split_best,
    split_indices,
)
```

代价：精度更容易满足，但辅助变量会增加。

### 7.5 方法四：PrecisionReducer 自动处理

本地求解器：

```python
base_optimizer = kw.classical.SimulatedAnnealingOptimizer(rand_seed=0)
optimizer = kw.preprocess.PrecisionReducer(
    base_optimizer,
    precision=8,
    target_bits=None,
    only_feasible_solution=False,
)

solution_dict, qubo_value = optimizer.solve_qubo(model)
```

真机求解器也可以被同样包装，见第 9 节。

### 7.6 选择建议

| 情况 | 建议 |
|---|---|
| 系数大小接近 | 直接取整 |
| 动态范围较大，但矩阵有压缩空间 | 动态范围压缩 |
| 必须达到特定位宽，允许增加变量 | 变量/项拆分 |
| 希望自动串联截断与拆分 | `PrecisionReducer` |

---

## 8. 高阶模型 HOBO

QUBO 只能包含最高二阶项，例如 `x1*x2`。如果出现 `x1*x2*x3`，就是 HOBO。

```python
import kaiwu as kw

a = kw.core.Binary("a")
b = kw.core.Binary("b")
c = kw.core.Binary("c")

hobo = kw.hobo.HoboModel(
    objective=a * b * c - a - b - c,
    hobo_default_penalty=10,
)

qubo_model = hobo.reduce()

optimizer = kw.classical.SimulatedAnnealingOptimizer(rand_seed=0)
solution_dict, qubo_value = optimizer.solve_qubo(qubo_model)

unsatisfied_hobo, hobo_details = hobo.verify_hobo_constraint(solution_dict)

print("solution:", solution_dict)
print("QUBO value:", qubo_value)
print("unsatisfied HOBO constraints:", unsatisfied_hobo)
print("HOBO details:", hobo_details)
```

原理：SDK 引入辅助变量，例如令 `y = a*b`，再用惩罚项强制 `y` 与 `a*b` 一致，把三阶项降成二阶项。

---

## 9. 提交 SPQC 真机任务

> 这一步会向云平台创建真实计算任务。先在本地经典求解器上确认模型、约束和精度。

### 9.1 Optimization 模式

适合更重视结果质量、可以等待平台动态追加计算资源的任务。

```python
from pathlib import Path

import kaiwu as kw
from kaiwu.cim import TaskMode

checkpoint_dir = Path("./kaiwu-checkpoints").resolve()
checkpoint_dir.mkdir(parents=True, exist_ok=True)
kw.common.CheckpointManager.save_dir = str(checkpoint_dir)

base_optimizer = kw.cim.CIMOptimizer(
    task_name="my_qubo_task_v1",
    wait=True,
    interval=1,
    task_mode=TaskMode.OPTIMIZATION,
)

optimizer = kw.preprocess.PrecisionReducer(base_optimizer, precision=8)
solution_dict, qubo_value = optimizer.solve_qubo(model)

unsatisfied_count, details = model.verify_constraint(solution_dict)
print("QUBO value:", qubo_value)
print("unsatisfied constraints:", unsatisfied_count)
print("details:", details)
```

注意：

- `task_name` 是当前 1.4.1 API 的必填参数。
- 任务唯一标识与 `task_name` 和 Ising 矩阵共同相关。
- 查询已有任务时不要随意改变 `task_name`。
- 如果同一个矩阵需要创建新任务，可以更换 `task_name`。
- `interval` 是轮询间隔，单位为分钟，最小值为 1。
- `project_no` 可选，用于指定云平台项目。

### 9.2 Sampling 模式

适合需要大量候选解、接受更高随机性的场景。

```python
optimizer = kw.cim.CIMOptimizer(
    task_name="my_sampling_task_v1",
    wait=True,
    interval=1,
    task_mode=TaskMode.SAMPLING,
    sample_number=100,
)

solutions = optimizer.solve(ising_matrix, sort_solutions=True)
```

`sample_number` 范围为 10–2000，默认 10。

### 9.3 只上传矩阵到云平台

```python
import pandas as pd

qubo_matrix = model.get_matrix()
pd.DataFrame(qubo_matrix).to_csv(
    "qubo_matrix.csv",
    index=False,
    header=False,
)
```

云平台操作顺序：

1. 进入控制台，选择真机并创建任务。
2. 填写任务名称并上传 CSV 矩阵。
3. 确认任务和真机配置。
4. 等待校验和排队。
5. 任务完成后查看解向量、目标值演化和执行时间。

---

## 10. 带约束模型的混合求解

`PenaltyMethodOptimizer` 会组合底层优化器和循环控制器，迭代处理惩罚系数。

```python
import kaiwu as kw

controller = kw.common.SolverLoopController(
    max_repeat_step=10,
    no_improve_limit=5,
)

base_optimizer = kw.classical.SimulatedAnnealingOptimizer(
    rand_seed=0,
    size_limit=100,
)

solver = kw.hybrid.PenaltyMethodOptimizer(base_optimizer, controller)
solution_dict, objective_value = solver.solve_qubo(model)

unsatisfied_count, details = model.verify_constraint(solution_dict)
print(objective_value, unsatisfied_count, details)
```

如需断点恢复：

```python
kw.common.CheckpointManager.save_dir = "/绝对路径/kaiwu-checkpoints"
```

---

## 11. 日志、能量和解池

### 11.1 打开调试日志

```python
import kaiwu as kw

kw.common.set_log_level("DEBUG")
kw.common.set_log_path("/绝对路径/kaiwu-debug.log")
```

### 11.2 重新计算 Ising 能量

```python
energies = kw.common.hamiltonian(ising_matrix, solutions)
print(energies)
```

### 11.3 检查矩阵对称性

```python
kw.common.check_symmetric(ising_matrix)
```

### 11.4 获取模拟退火过程

```python
optimizer = kw.classical.SimulatedAnnealingOptimizer(
    flag_evolution_history=True,
    rand_seed=0,
)
optimizer.solve(ising_matrix)
history = optimizer.get_ha_history()
print(history)
```

---

## 12. 常见故障排查

### 12.1 `ModuleNotFoundError: No module named 'kaiwu'`

依次执行：

```bash
python --version
which python
python -m pip show kaiwu
python -c "import kaiwu; print(kaiwu.__file__)"
```

常见原因：wheel 安装到了另一个 Python 环境。

### 12.2 Python 版本错误

必须看到：

```text
Python 3.10.x
```

当前工作区默认 Python 3.9.6，不能直接用于官方 Kaiwu SDK 1.4.1。

### 12.3 macOS 架构错误

Intel 与 Apple Silicon 的 wheel 可能不同。按官方说明，若一种架构包报错，可核对并尝试平台提供的另一版本；仍失败则联系 SDK 提供方。

检查机器架构：

```bash
uname -m
```

典型输出：

- `arm64`：Apple Silicon。
- `x86_64`：Intel，或运行在 Rosetta 环境。

### 12.4 license 生成失败

- 确认国内网络可用。
- 尽量在本机生成，不要先在受限服务器环境中尝试。
- 检查用户 ID 和授权码是否过期。
- 授权过期后到云平台重新申请。

### 12.5 返回了解，但约束不满足

```python
unsatisfied_count, details = model.verify_constraint(solution_dict)
print(unsatisfied_count)
print(details)
```

按顺序处理：

1. 检查约束表达式方向和索引。
2. 适度增加对应约束的惩罚系数。
3. 增加退火深度或重复运行。
4. 尝试禁忌搜索在候选解邻域继续优化。
5. 检查过大的惩罚系数是否造成精度问题。

### 12.6 真机任务重复创建

同一个矩阵如果使用不同 `task_name`，可能创建不同任务。查询或复用缓存时保持任务名不变。

### 12.7 精度适配后答案变差

可能原因：直接取整把不同小系数映射成了相同整数。

处理顺序：

1. 检查动态范围。
2. 比较适配前后矩阵的能量排序。
3. 改用动态范围压缩。
4. 改用变量/项拆分。
5. 恢复原变量后重新计算原始目标值，不要只比较适配后矩阵能量。

---

## 13. 推荐项目结构

```text
kaiwu-project/
├── .venv/                  # 不提交 Git
├── data/
│   └── input.csv
├── checkpoints/            # 不提交敏感或大型运行状态
├── outputs/
│   ├── qubo_matrix.csv
│   └── result.json
├── src/
│   ├── model.py            # 变量、目标、约束
│   ├── solve_local.py      # 本地经典求解
│   ├── solve_spqc.py       # 真机提交
│   └── validate.py         # 业务答案和约束验证
├── tests/
│   └── test_small_case.py  # 小规模可穷举答案
├── .gitignore
└── README.md
```

建议 `.gitignore`：

```gitignore
.venv/
__pycache__/
*.pyc
checkpoints/
kaiwu-checkpoints/
*.log
.env
```

不要提交 license、授权码或用户 ID。

---

## 14. 上真机前的验收清单

### 模型

- [ ] 每个决策变量有明确业务含义。
- [ ] 目标函数方向正确：SDK 默认寻找更低能量。
- [ ] 所有硬约束都有名称。
- [ ] 用小规模数据手算或穷举过正确答案。

### 本地求解

- [ ] `verify_constraint()` 返回 0 个未满足约束。
- [ ] 还原后的业务答案没有重复、遗漏或越界。
- [ ] 用原始业务目标重新计算过结果。
- [ ] 多个随机种子下结果基本稳定，或已保留多个候选解。

### 精度

- [ ] 已把 QUBO 转成最终提交的 Ising 表示。
- [ ] 已检查 8-bit 位宽。
- [ ] 极端系数没有被直接取整后吞掉。
- [ ] 拆分后的解已恢复到原变量空间。

### 真机

- [ ] license 有效。
- [ ] checkpoint 路径可写。
- [ ] `task_name` 唯一且可追踪。
- [ ] 已选择 `OPTIMIZATION` 或 `SAMPLING`。
- [ ] 提交前记录了矩阵哈希、参数、SDK 版本和时间。

---

## 15. API 速查表

| 需求 | API |
|---|---|
| 二值变量 | `kw.core.Binary(name)` |
| 自旋变量 | `kw.core.Spin(name)` |
| 整数变量 | `kw.core.Integer(name, min_value, max_value)` |
| 变量数组 | `kw.core.ndarray(shape, name, var_func)` |
| 快速求和 | `kw.core.quicksum(items)` |
| QUBO 模型 | `kw.core.QuboModel()` |
| 设置目标 | `model.set_objective(expr)` |
| 添加约束 | `model.add_constraint(expr, name, penalty=...)` |
| 约束检查 | `model.verify_constraint(solution_dict)` |
| 恢复表达式值 | `kw.core.get_val(expr, solution_dict)` |
| QUBO→Ising | `kw.core.qubo_matrix_to_ising_matrix(matrix)` |
| Ising→QUBO | `kw.core.ising_matrix_to_qubo_matrix(matrix)` |
| 自动惩罚估计 | `kw.core.get_min_penalty(objective, constraint)` |
| 模拟退火 | `kw.classical.SimulatedAnnealingOptimizer` |
| 禁忌搜索 | `kw.classical.TabuSearchOptimizer` |
| 暴力求解 | `kw.classical.BruteForceOptimizer` |
| 真机求解 | `kw.cim.CIMOptimizer` |
| 位宽检查 | `kw.preprocess.calculate_ising_matrix_bit_width` |
| 直接精度调整 | `kw.preprocess.adjust_ising_matrix_precision` |
| 动态范围压缩 | `kw.preprocess.perform_precision_adaption_mutate` |
| 变量/项拆分 | `kw.preprocess.perform_precision_adaption_split` |
| 恢复拆分解 | `kw.preprocess.restore_split_solution` |
| 自动精度适配 | `kw.preprocess.PrecisionReducer` |
| HOBO 降阶 | `kw.hobo.HoboModel.reduce()` |
| 混合约束求解 | `kw.hybrid.PenaltyMethodOptimizer` |
| 日志级别 | `kw.common.set_log_level()` |
| 日志文件 | `kw.common.set_log_path()` |
| checkpoint | `kw.common.CheckpointManager.save_dir` |
| 计算能量 | `kw.common.hamiltonian()` |

---

## 16. 官方资料

- [Kaiwu SDK 1.4.1 文档首页](https://kaiwu-sdk-docs.qboson.com/zh/latest/)
- [安装说明](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/getting_started/sdk_installation_instructions.html)
- [TSP QUBO 教程](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/getting_started/tsp.html)
- [Max-Cut Ising 教程](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/getting_started/max_cut.html)
- [真机：云平台或 SDK](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/getting_started/tutorial4_cloud_platform_usage_cases.html)
- [参数精度](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/advanced/precision.html)
- [自动确定惩罚系数](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/advanced/auto_penalty.html)
- [云平台计算模式](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/advanced/mode.html)
- [模块 API 手册](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/modules/index.html)
- [常见问题](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/faq/frequently_asked_question.html)
- [更新日志](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/about/changelog.html)

---

## 17. 当前验证状态

- 已检查 Markdown 结构、命令顺序和代码块完整性。
- API 名称、参数和流程依据官方 1.4.1 文档整理。
- 当前机器只有 Python 3.9.6，且没有安装专有 `kaiwu` wheel，因此本手册中的 Kaiwu 代码尚未在当前工作区实际运行。
- 完成 Python 3.10 与企业版 wheel 安装后，应优先运行第 2 节 Max-Cut 和第 3 节最小 QUBO 模板作为冒烟测试。

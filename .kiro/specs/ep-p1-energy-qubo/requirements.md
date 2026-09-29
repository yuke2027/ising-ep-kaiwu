# Requirements Document

## Introduction

本规格是 OSPP 2026 项目 26b1f0072"基于 Kaiwu-PyTorch-Plugin 实现均衡传播算法"五期计划中的 P1（补做申请书里程碑 M1"能量和 argmin 等价"），排期第 3–5 周（2026-09-29 至 2026-10-19）。本规格从总需求 `.kiro/specs/kpp-equilibrium-propagation/requirements.md` 抽取子集，逐条保持原意与数值阈值，仅做本期范围裁剪（裁剪处在条目后注明）。总需求文件在本期不作修改；两者表述不一致时以总需求为准。

**前置条件（P0 `ep-p0-upstream-setup` 退出门禁）**

- Plugin_Repository 已克隆到工作区 `upstream/kaiwu-pytorch-plugin`（即 Upstream_Clone）。
- Python 3.10 + `kaiwu==1.4.1` 环境可用。
- 上游既有测试在该环境下的回归结果已记录。

**代码位置**：Upstream_Clone 的 `src/kaiwu/torch_plugin/` 下新建 `equilibrium_propagation.py`（EPConfig、EP_Module）、`_ep_dynamics.py`（Continuous_Backend）、`_ep_qubo.py`（QUBO_Builder）与最小 `_ep_solver.py`（Solver_Adapter）；测试写入 Upstream_Clone 的 `tests/test_ep_*.py`。

**起点现状**

- 总需求 22.1 列出的五个源模块（`equilibrium_propagation.py`、`_ep_dynamics.py`、`_ep_qubo.py`、`_ep_solver.py`、`_ep_training.py`）在 Upstream_Clone 中均不存在。本期新建前四个，`_ep_training.py` 由 P2 新建。
- Migration_Source 只作为迁移参考，在 Upstream_Clone 中按本规格约定重写，不被 Upstream_Clone 中的代码导入。可迁移的思路与写法：
  - QUBO 常数处理：`src/ising_ep/mapping.py` 的 `build_qubo`、`qubo_energy`；
  - 辅助自旋解码：`mapping.py` 的 `decode_aux`/`decode_aux_alt`，以及 `tests/integration/test_kaiwu_mapping.py` 中基于 `kaiwu.core.get_sol_dict` 的解码检查；
  - 按原始能量复算与并列裁决：`mapping.py` 的 `energy_equivalence_report`、`minimizer_indices` 与 `examples/mnist_kaiwu/` 的求解流程；
  - fake SDK 与延迟导入写法：`tests/unit/test_solve.py` 的 `FakeCore`/`fake_kaiwu` 替身，以及 `examples/mnist_kaiwu/solve.py` 的 `_import_kaiwu` 注入点。
- 不可复用部分：±1 自旋表示（本期为 [0,1] Layer_State 与 {0,1} Logical_Bit）、单隐层结构（本期支持任意合法 `layer_sizes`）、无连续后端、无 K 位编码。

**退出门禁**（全部满足方可进入 P2）

1. 本期全部 Level_1_Test 通过，单进程 CPU 总耗时不超过 120 秒（需求 9.1，对应总需求 17.1）。
2. 能量测试与 QUBO 等价测试覆盖 `[2, 3, 1]` 与至少一个含 2 个隐藏层的结构（需求 9.2，对应总需求 17.2）。
3. 本期新增公共 API 均有 docstring（需求 10.1，对应总需求 21.7）。
4. 文件布局、导出、SPDX 头与 lint 符合需求 10.2–10.5（对应总需求 22.1–22.4），本期导出 `EPConfig` 与 `EquilibriumPropagation`。

**本期范围**

| 本规格需求 | 对应总需求 |
|---|---|
| 1 EP 统一配置 | 1.1–1.12 |
| 2 模型结构、参数与 PyTorch 兼容 | 2.1–2.8 |
| 3 能量、代价与总能量 | 3.1–3.6 |
| 4 连续参考后端 | 4.1–4.7、15.3 |
| 5 K 位状态编码与 QUBO 构造 | 5.1–5.9、11.11、15.4 |
| 6 SDK 矩阵约定、Ising 转换与解码 | 6.1–6.8、7.7、7.12 |
| 7 穷举后端 | 8.1–8.2 |
| 8 错误类型层级 | 15.1 |
| 9 本期测试套件 | 17.1、17.2、17.3、17.9、17.10、17.12、17.13（均为本期子集） |
| 10 公共 API 文档与上游集成规范 | 21.7、22.1–22.4 |

**不在本期**（由 P2–P4 承担）：总需求 7.1–7.6、7.8–7.11；8.3–8.10；9；10；11.1–11.10；12–14；15.2、15.5–15.9；16；17.4–17.8、17.11，以及 17.13 中聚合权重归一与 Best_So_Far 非增部分；18–20；21.1–21.6；22.5–22.10；23；24。

**本期裁剪说明**

- EPConfig 本期完整实现字段、校验与往返；`aggregation`、`topk`、`aggregation_temperature`、`initial_state_policy` 与 `gradient_estimator` 的运行时行为在 P2 实现。
- Solver_Adapter 为最小实现：仅承担 Kaiwu 问题构造、Ising 转换、变量映射与解码、Kaiwu 版本读取及 Exhaustive_Backend 调用。SA_Backend、CIM_Backend、Aggregation、重试、`to_dict()`、版本告警（7.8）与 Kaiwu 缺失时的求解错误（7.10）在 P2；本期对 Kaiwu 转换 API 不可调用的处理仅限需求 9.5 的测试跳过。
- 系数范围本期仅实现 `coefficient_policy="error"`（15.4）；`scale` 策略（15.5）在 P2。`coefficient_limit`、`dynamic_range_limit` 与 `coefficient_policy` 的默认值在设计阶段确定并写入 docstring。
- Equilibrium_Result 本期由 Continuous_Backend 直接返回；`model.equilibrate()` 统一入口（7.2）在 P2。

## Glossary

术语沿用总需求定义；下列为本期使用的术语及本期补充说明。

- **Kaiwu_SDK**：玻色量子 Kaiwu SDK，固定为 `kaiwu==1.4.1`。真实版本以 `importlib.metadata.version("kaiwu")` 为准（`kw.__version__` 返回依赖包 kaiwu-community 的版本 1.0.7）。1.4.1 不含 `kaiwu.conversion` 模块，QUBO/Ising 转换位于 `kaiwu.core`；支持的 Python 版本为 3.10。
- **Plugin_Repository**：上游仓库 `https://github.com/qboson/kaiwu-pytorch-plugin`（Apache-2.0）。源码位于 `src/kaiwu/torch_plugin/`，测试位于 `tests/test_*.py`，`__init__.py` 导出公共 API。
- **Upstream_Clone**：P0 克隆到工作区 `upstream/kaiwu-pytorch-plugin` 的 Plugin_Repository 工作副本；本期全部代码与测试写入其中。
- **EP_Changeset**：本期在 Upstream_Clone 中新增或修改的文件集合，后续期作为 Upstream_PR 的组成部分提交。
- **Upstream_PR**：提交到 Plugin_Repository 的 Pull Request 及其提交内容（本期不创建）。
- **Migration_Source**：前期可行性验证（Preliminary_Validation）中的 `src/ising_ep/model.py`、`src/ising_ep/mapping.py`、`examples/mnist_kaiwu/`、`tests/unit/` 与 `tests/integration/`，为 MNIST/100 单隐层 ±1 自旋 Ising EP 实现，只读。
- **EP_Module**：`kaiwu.torch_plugin.EquilibriumPropagation`（`equilibrium_propagation.py`），`torch.nn.Module` 子类，持有可学习参数并提供能量计算。`equilibrate()`、`train_batch()` 与 `predict()` 在后续期实现。
- **EPConfig**：`equilibrium_propagation.py` 中的配置类，统一保存网络结构、β、相位预算、状态步长、量化位宽、聚合方式、初始状态策略与随机种子，提供 `to_dict()` 与 `from_dict()`。
- **MAX_STATE_BITS**：`state_bits` 允许的最大值，取值不小于 3。
- **Layer_State**：第 l 层状态 `s^l`，`layer_sizes = [n_0, …, n_L]`。`s^0 = x` 为钳制输入，不作为求解变量；`s^1..s^L` 为待求平衡状态，取值于 [0,1]。
- **E_θ**：能量 `E_θ(s;x) = ½Σ_{l=1..L}‖s^l‖² − Σ_{l=0..L−1}(s^l)ᵀW^l s^{l+1} − Σ_{l=1..L}(b^l)ᵀs^l`，`W^l` 形状为 `(n_l, n_{l+1})`，`b^l` 形状为 `(n_l,)`。
- **C**：代价 `C(s^L, y) = ½‖s^L − y‖²`，`y` 为 one-hot 或 [0,1] 目标向量。
- **F_β**：总能量 `F_β = E_θ + β·C`。
- **Free_Phase**：β = 0 的平衡求解。
- **Nudged_Phase**：β > 0 的平衡求解。
- **Phase_Pair**：同一样本在同一参数快照下的一次 Free_Phase 与一次 Nudged_Phase。
- **Equilibrium_Result**：平衡求解的结果，包含 Layer_State、按原始 F_β 复算的能量、Stop_Reason、Budget 与求解器元数据。本期由 Continuous_Backend 直接返回，字段见需求 4.6。
- **Continuous_Backend**：`_ep_dynamics.py` 中的 `ContinuousDynamics`，在 [0,1] 盒约束内对 F_β 做投影梯度下降的确定性参考后端，输出 `energy_trace`。
- **Discrete_Backend**：在 State_Grid 上直接最小化量化后 F_β 的后端统称，包括 Exhaustive_Backend、SA_Backend 与 CIM_Backend。
- **Exhaustive_Backend**：枚举全部 Logical_Bit 赋值的确定性后端，仅用于 Micro_Network。
- **SA_Backend**：基于 `kaiwu.classical.SimulatedAnnealingOptimizer` 的模拟退火后端（P2 实现）。
- **CIM_Backend**：基于 `kaiwu.cim.CIMOptimizer` 的相干伊辛机后端（P2 及以后实现）。
- **Solver_Adapter**：`_ep_solver.py` 中的 `SolverAdapter`，统一各后端的调用与解码。本期为最小实现，范围见"本期裁剪说明"。
- **K_Bit_Encoding**：`s_i(q) = Σ_{k=0..K−1} 2^k q_{i,k} / (2^K − 1)`，`q_{i,k} ∈ {0,1}`，`K = state_bits`。
- **State_Grid**：K_Bit_Encoding 可表示的状态集合 `{0, 1/(2^K−1), …, 1}`。
- **Logical_Bit**：QUBO 二值变量 `q_{i,k}`；网络共有 `K·Σ_{l=1..L} n_l` 个。
- **QUBO_Builder**：`_ep_qubo.py` 中的 `EPQuboBuilder`，负责输入消元、K 位展开、Coefficient_Table、SDK 矩阵转换与解码辅助。
- **Coefficient_Table**：内部 QUBO 表示；每个无序变量对 `(i, j)`（i < j）的二次系数只记一次，线性系数与 Constant_Offset 单独保存。
- **SDK_Matrix_Convention**：Kaiwu 1.4.1 `kaiwu.core.qubo_matrix_to_qubo_model` 的上三角约定：目标为 `qᵀQq`，对角元为线性系数，`Q[i, j]`（i < j）为二次系数，下三角元为 0。
- **Constant_Offset**：满足 `F_β(s(q)) = qᵀQq + Constant_Offset` 的常数（β > 0 时包含 `β·½‖y‖²`）。
- **Ising_Objective**：Kaiwu 1.4.1 Ising 模型目标 `H(σ) = −σᵀJσ + ising_bias`，J 与 `ising_bias` 分别取自 `get_matrix()` 与 `get_bias()`。
- **Auxiliary_Spin**：`kaiwu.core.qubo_model_to_ising_model` 为表达线性项引入的额外自旋；物理解码对 Auxiliary_Spin 的两种取值（±1）给出同一结果。
- **Budget**：一次平衡求解的资源描述；Continuous_Backend 为 `free_steps`/`nudged_steps` 步数上限与实际步数。
- **Stop_Reason**：停止原因；Continuous_Backend 取 `converged`、`max_steps` 或 `energy_increase`。
- **ε_mono**：能量单调性容差 `1e-12·max(1, |F_β(s_t)|)`。
- **EP_Error**：本模块显式异常的公共基类；子类区分配置、动力学、系数范围、配对、求解、授权与数值错误，配置错误同时继承 `ValueError`。
- **Micro_Network**：用于全枚举校验的微型网络（如 `[2, 2, 2]`、`[2, 3, 1]`；含 2 个隐藏层的示例如 `[2, 2, 2, 1]`），Logical_Bit 数不超过 `exhaustive_max_bits`。
- **Level_1_Test**：无需 Kaiwu 授权与网络即可运行的确定性测试（Continuous_Backend、Exhaustive_Backend、纯 Torch 映射与 fake SDK）。本期三方等价测试的 Ising 部分另需已安装 Kaiwu_SDK（无需授权），不可调用时按需求 9.5 跳过。
- **Level_2_Test**：需要 Kaiwu 授权（真实 SA 调用）或 CIM 授权/网络/配额的测试，以 pytest 标记标识，条件不满足时显式跳过。
- **Test_Suite**：Upstream_Clone 中 `tests/test_ep_*.py` 的 pytest 测试集合。
- **Documentation**：插件 README、示例 README 与公共 API docstring；本期仅涉及公共 API docstring，README 在后续期。

## Requirements

### Requirement 1: EP 统一配置

**User Story:** 作为插件使用者，我希望用一个 `EPConfig` 统一声明网络结构、β、相位预算、状态步长、量化位宽、聚合方式、初始状态策略与随机种子，以便实验配置可校验、可保存、可复现。

#### Acceptance Criteria

1. THE EPConfig SHALL 至少提供字段 `layer_sizes`、`beta`、`beta_min`、`free_steps`、`nudged_steps`、`step_size`、`max_backtracks`、`residual_tol`、`state_tol`、`state_bits`、`aggregation`、`topk`、`aggregation_temperature`、`initial_state_policy`、`gradient_estimator` 与 `seed`。（对应总需求 1.1）
2. WHEN 调用方以 `EPConfig(layer_sizes=[64, 32, 10], beta=0.2, free_steps=50, nudged_steps=25, state_bits=1, aggregation="best", seed=2026)` 构造配置，THE EPConfig SHALL 创建成功，并为未指定字段填入 Documentation 中列出的默认值。（对应总需求 1.2）
3. THE EPConfig SHALL 以 `aggregation="best"`、`initial_state_policy="shared"` 与 `gradient_estimator="one_sided"` 作为默认值。（对应总需求 1.3）
4. IF `layer_sizes` 少于 2 个元素或含非正整数，THEN THE EPConfig SHALL 抛出配置错误，错误信息包含字段名与收到的值。（对应总需求 1.4）
5. IF `beta` 非有限或小于 `beta_min`，THEN THE EPConfig SHALL 抛出配置错误，错误信息包含 `beta` 与 `beta_min` 的数值。（对应总需求 1.5）
6. IF `free_steps` 或 `nudged_steps` 不是正整数、`max_backtracks` 不是非负整数，或 `beta_min`、`step_size`、`residual_tol`、`state_tol` 不是正有限数，THEN THE EPConfig SHALL 抛出配置错误。（对应总需求 1.6）
7. IF `state_bits` 不是 1 到 `MAX_STATE_BITS`（不小于 3）之间的整数，THEN THE EPConfig SHALL 抛出配置错误。（对应总需求 1.7）
8. IF `aggregation` 不属于 {`"best"`, `"topk_mean"`, `"energy_weighted"`}、`initial_state_policy` 不属于 {`"shared"`, `"warm_start"`}，或 `gradient_estimator` 不属于 {`"one_sided"`, `"symmetric"`}，THEN THE EPConfig SHALL 抛出配置错误并列出允许取值。（对应总需求 1.8）
9. IF `aggregation="topk_mean"` 而 `topk` 不是正整数，或 `aggregation="energy_weighted"` 而 `aggregation_temperature` 不是正有限数，THEN THE EPConfig SHALL 抛出配置错误。（对应总需求 1.9）
10. WHEN 调用 `cfg.to_dict()`，THE EPConfig SHALL 返回仅含 JSON 可序列化值（`str`、`int`、`float`、`bool`、`list`、`dict`、`None`）的字典。（对应总需求 1.10）
11. THE EPConfig SHALL 对任意合法实例 `cfg` 满足 `EPConfig.from_dict(cfg.to_dict()) == cfg`，并在经 `json.dumps`/`json.loads` 往返后仍满足该等式（往返性质）。（对应总需求 1.11）
12. IF `EPConfig.from_dict` 收到缺少必需键或含未知键的字典，THEN THE EPConfig SHALL 抛出配置错误并列出缺失键与未知键。（对应总需求 1.12）

### Requirement 2: 模型结构、参数与 PyTorch 兼容

**User Story:** 作为 PyTorch 使用者，我希望 `EquilibriumPropagation` 是标准 `torch.nn.Module`，支持多层结构并兼容 `Dataset`、`DataLoader`、`torch.optim` 与 `state_dict`，以便像普通模型一样训练与管理。

#### Acceptance Criteria

1. THE EP_Module SHALL 继承 `torch.nn.Module`，并为 `layer_sizes=[n_0, …, n_L]` 注册 L 个耦合参数 `W^l`（形状 `(n_l, n_{l+1})`，l = 0..L−1）与 L 个偏置参数 `b^l`（形状 `(n_l,)`，l = 1..L），参数仅包含相邻层耦合与偏置。（对应总需求 2.1）
2. THE EP_Module SHALL 接受长度不小于 2 的任意合法 `layer_sizes`，即支持 0 个、1 个或多个隐藏层。（对应总需求 2.2）
3. WHEN 以相同 EPConfig（含相同 `seed`）两次构造 EP_Module，THE EP_Module SHALL 生成逐元素相同的全部参数。（对应总需求 2.3）
4. THE EP_Module SHALL 默认以 `torch.float64` 创建参数，并使用 Documentation 中记录的初始化分布与尺度规则。（对应总需求 2.4）
5. THE EP_Module SHALL 使用确定且记录在 Documentation 中的 `state_dict` 键名，使键集合与各张量形状仅由 `layer_sizes` 决定。（对应总需求 2.5）
6. WHEN 调用方传入来自 `torch.utils.data.DataLoader` 的批数据（`x` 为形状 `(B, n_0)` 的 `float32` 或 `float64` 张量，`y` 为形状 `(B,)` 的类别索引或形状 `(B, n_L)` 的 [0,1] 目标），THE EP_Module SHALL 将 `x` 转换为参数 dtype，并将类别索引转换为 one-hot 目标。（对应总需求 2.6）
7. IF `x` 或 `y` 的形状与 `layer_sizes` 不符、含 NaN 或 Inf，或类别索引超出 `[0, n_L)`，THEN THE EP_Module SHALL 抛出 `ValueError`，错误信息包含参数名与期望形状或取值范围。（对应总需求 2.7）
8. THE EP_Module SHALL 通过 `model.parameters()` 暴露全部可学习参数，使 `torch.optim.SGD` 与 `torch.optim.Adam` 可直接用于参数更新。（对应总需求 2.8）

### Requirement 3: 能量、代价与总能量

**User Story:** 作为算法开发者，我希望模块严格按申请书公式计算 E_θ、C 与 F_β 并支持对参数自动微分，以便两相位求解、QUBO 构造与梯度共享同一能量定义。

#### Acceptance Criteria

1. THE EP_Module SHALL 按 E_θ 的定义（`s^0 = x`）计算能量，对批输入返回形状 `(B,)` 的张量，对形状 `(n_0,)` 的单样本输入返回标量。（对应总需求 3.1）
2. THE EP_Module SHALL 按 C 的定义计算代价，并按 `F_β = E_θ + β·C` 计算总能量。（对应总需求 3.2）
3. WHEN 以分离（detached）的 Layer_State 计算 E_θ 并调用 `backward()`，THE EP_Module SHALL 得到满足 `∂E_θ/∂W^l = −s^l(s^{l+1})ᵀ` 与 `∂E_θ/∂b^l = −s^l` 的参数梯度（与解析式的最大绝对误差 ≤ 1e-12）。（对应总需求 3.3）
4. THE EP_Module SHALL 对任意合法输入与状态满足 `F_0 = E_θ`（逐元素相等）。（对应总需求 3.4）
5. IF 传入的 Layer_State 数量不等于 L、某层宽度不等于 `n_l`，或状态值超出 [0,1] 的幅度大于 1e-12，THEN THE EP_Module SHALL 抛出 `ValueError`。（对应总需求 3.5）
6. IF 传入的 `beta` 为负数或非有限值，THEN THE EP_Module SHALL 抛出 `ValueError`。（对应总需求 3.6）

### Requirement 4: 连续参考后端

**User Story:** 作为算法开发者，我希望有确定性的连续状态后端，在 [0,1] 盒约束内使 F_β 非增并按残差与状态变化判停，以便作为梯度方向与"下降并稳定"行为的参考。

#### Acceptance Criteria

1. THE Continuous_Backend SHALL 在每一步把 `s^1..s^L` 投影到 [0,1]，并保持 `s^0 = x` 不变。（对应总需求 4.1）
2. THE Continuous_Backend SHALL 使每个被接受的步骤满足 `F_β(s_{t+1}) ≤ F_β(s_t) + ε_mono`。（对应总需求 4.2）
3. IF 按当前步长计算的一步使 F_β 上升超过 ε_mono，THEN THE Continuous_Backend SHALL 将该步步长减半后重算，至多 `max_backtracks` 次。（对应总需求 4.3）
4. WHEN 投影梯度残差 `‖s − Π_[0,1](s − ∇_s F_β(s))‖_∞ ≤ residual_tol` 且相邻步状态变化 `‖s_{t+1} − s_t‖_∞ ≤ state_tol`，THE Continuous_Backend SHALL 停止并记录 `stop_reason="converged"`。（对应总需求 4.4）
5. WHEN Free_Phase 达到 `free_steps` 步或 Nudged_Phase 达到 `nudged_steps` 步仍未满足收敛条件，THE Continuous_Backend SHALL 停止并记录 `stop_reason="max_steps"`。（对应总需求 4.5）
6. THE Continuous_Backend SHALL 在 Equilibrium_Result 中返回逐步 `energy_trace`、最终残差、最终状态变化、实际步数、回溯次数与 Stop_Reason。（对应总需求 4.6）
7. WHEN 在同一环境中以相同参数、输入、初始状态与配置重复运行，THE Continuous_Backend SHALL 返回逐元素相同的状态与 `energy_trace`。（对应总需求 4.7）
8. IF Continuous_Backend 回溯 `max_backtracks` 次后 F_β 仍上升超过 ε_mono，THEN THE Continuous_Backend SHALL 记录 `stop_reason="energy_increase"` 并抛出动力学错误，错误信息包含相位、步数与能量增量。（对应总需求 15.3）

### Requirement 5: K 位状态编码与 QUBO 构造

**User Story:** 作为算法开发者，我希望把每个非输入神经元按 K 位定点编码代入 F_β 得到 QUBO，并显式保存常数偏移，以便离散后端最小化的正是量化后的 F_β。

#### Acceptance Criteria

1. THE QUBO_Builder SHALL 按 K_Bit_Encoding 对 `s^1..s^L` 的每个神经元编码，使可表示状态恰为 State_Grid。（对应总需求 5.1）
2. THE QUBO_Builder SHALL 对任意 K 与任意 `q ∈ {0,1}^K` 满足 `encode(decode(q)) = q`，并对 State_Grid 中任意值 v 满足 `decode(encode(v)) = v`（往返性质）。（对应总需求 5.2）
3. THE QUBO_Builder SHALL 生成恰好 `K·Σ_{l=1..L} n_l` 个 Logical_Bit；对 `layer_sizes=[64, 32, 10]` 且 K = 1/2/3，Logical_Bit 数分别为 42/84/126。（对应总需求 5.3）
4. THE QUBO_Builder SHALL 将输入 x 作为常量，把 `−xᵀW^0 s^1` 并入 `s^1` 对应 Logical_Bit 的线性系数，使问题变量中不含输入层。（对应总需求 5.4）
5. THE QUBO_Builder SHALL 在 Coefficient_Table 中对每个无序变量对只记录一个二次系数，并单独记录线性系数与 Constant_Offset。（对应总需求 5.5）
6. THE QUBO_Builder SHALL 对任意合法参数、输入、目标、β ≥ 0 与任意 Logical_Bit 赋值 q，满足 `F_β(s(q)) = Σ_i a_i q_i + Σ_{i<j} c_ij q_i q_j + Constant_Offset`（绝对误差 ≤ 1e-9）。（对应总需求 5.6）
7. WHEN β = 0，THE QUBO_Builder SHALL 生成与目标 y 无关的 Coefficient_Table 与 Constant_Offset。（对应总需求 5.7）
8. THE QUBO_Builder SHALL 为每个问题报告 Logical_Bit 数、非零二次项数、最大绝对系数、最小非零绝对系数，以及最大绝对系数与最小非零绝对系数之比（动态范围）。（对应总需求 5.8）
9. WHEN 以相同 `layer_sizes` 与 `state_bits` 重复构造问题，THE QUBO_Builder SHALL 复用同一位索引布局与稀疏结构模板并只更新系数，模板构造计数保持为 1。（对应总需求 5.9）
10. WHEN 以 β = 0 构造同一样本的两个相位问题，THE QUBO_Builder SHALL 生成逐元素相同的 Coefficient_Table 与 Constant_Offset。（对应总需求 11.11）
11. IF 任一 QUBO 系数绝对值超过 `coefficient_limit` 或动态范围超过 `dynamic_range_limit`，且 `coefficient_policy="error"`，THEN THE QUBO_Builder SHALL 抛出系数范围错误并报告超限的统计量。（对应总需求 15.4；`scale` 策略 15.5 在 P2）

### Requirement 6: SDK 矩阵约定、Ising 转换与解码

**User Story:** 作为算法开发者，我希望 Coefficient_Table 按 Kaiwu 1.4.1 的矩阵约定转换并经官方 API 得到 Ising 模型，再把解还原为层状态，以便 F_β、QUBO 与 Ising 三者仅差常数且解码后的 argmin 一致。

#### Acceptance Criteria

1. WHEN QUBO_Builder 为 Discrete_Backend 导出矩阵，THE QUBO_Builder SHALL 按 SDK_Matrix_Convention 生成上三角矩阵 Q：对角元为线性系数，`Q[i, j]`（i < j）为二次系数，下三角元为 0。（对应总需求 6.1）
2. THE QUBO_Builder SHALL 对任意 Logical_Bit 赋值 q 满足 `qᵀQq + Constant_Offset = F_β(s(q))`（绝对误差 ≤ 1e-9），以排除非对角项加倍。（对应总需求 6.2）
3. WHEN 构造 Kaiwu 问题，THE Solver_Adapter SHALL 调用 `kaiwu.core.qubo_matrix_to_qubo_model(Q)` 与 `kaiwu.core.qubo_model_to_ising_model(...)`，并从 Ising 模型读取 J、变量表与 `ising_bias`。（对应总需求 6.3）
4. THE Solver_Adapter SHALL 依据 Kaiwu 变量表建立 Logical_Bit 与 Ising 变量的一一映射，并识别至多 1 个 Auxiliary_Spin。（对应总需求 6.4）
5. THE Solver_Adapter SHALL 对任意自旋解 σ 满足 `H(σ) + Constant_Offset = F_β(s(q(σ)))`（H 为 Ising_Objective，绝对误差 ≤ 1e-8）。（对应总需求 6.5）
6. THE Solver_Adapter SHALL 通过 `kaiwu.core.get_sol_dict` 恢复 Logical_Bit 取值，并经 K_Bit_Encoding 还原为 [0,1] Layer_State。（对应总需求 6.6）
7. THE Solver_Adapter SHALL 对任意自旋解 σ 及其全局翻转 −σ（Auxiliary_Spin 的两种规范）解码出相同的 Logical_Bit 与 Layer_State。（对应总需求 6.7）
8. IF Kaiwu 变量表无法与 Logical_Bit 一一对应或含多于 1 个 Auxiliary_Spin，THEN THE Solver_Adapter SHALL 抛出求解错误，错误信息包含期望与实际变量数。（对应总需求 6.8）
9. THE Solver_Adapter SHALL 通过 `importlib.metadata.version("kaiwu")` 读取 Kaiwu 版本。（对应总需求 7.7）
10. THE Solver_Adapter SHALL 将 Kaiwu_SDK 导入延迟到首次调用依赖 Kaiwu_SDK 的功能时执行，使 Continuous_Backend 与 Exhaustive_Backend（含其依赖的 EPConfig、EP_Module 与 QUBO_Builder）在未安装 Kaiwu_SDK 时可用。（对应总需求 7.12；本期依赖 Kaiwu_SDK 的功能为本需求第 3–8 条，SA_Backend 与 CIM_Backend 于 P2 纳入同一规则）

### Requirement 7: 穷举后端

**User Story:** 作为算法开发者，我希望穷举后端为微型网络给出确定性精确解，以便单元测试确定且可作为离散后端的参考。

#### Acceptance Criteria

1. WHEN Logical_Bit 数不超过 `exhaustive_max_bits`（默认 16），THE Exhaustive_Backend SHALL 枚举全部 `2^n` 个赋值，并返回复算 F_β 与最低值之差在 1e-9 以内的全部状态。（对应总需求 8.1）
2. IF Logical_Bit 数超过 `exhaustive_max_bits`，THEN THE Exhaustive_Backend SHALL 在枚举开始前抛出求解错误，错误信息包含位数与上限。（对应总需求 8.2）

### Requirement 8: 错误类型层级

**User Story:** 作为研究者，我希望模块的显式错误有统一基类与可区分的子类，以便异常不会被静默吞掉，调用方可按错误类别捕获与处理。

#### Acceptance Criteria

1. THE EP_Module SHALL 使全部显式错误继承自 EP_Error，并提供可区分的配置、动力学、系数范围、配对、求解、授权与数值错误子类。（对应总需求 15.1）

### Requirement 9: 本期测试套件

**User Story:** 作为维护者，我希望本期的能量、连续后端与 QUBO/Ising 映射有确定性的 pytest 覆盖且不依赖授权与网络，以便上游 CI 稳定运行，并作为 P1 退出门禁的依据。

#### Acceptance Criteria

1. THE Test_Suite SHALL 以 `tests/test_ep_*.py` 组织，并使本期全部 Level_1_Test 在无 Kaiwu 授权、无网络的环境中通过，单进程 CPU 总耗时不超过 120 秒。（对应总需求 17.1）
2. THE Test_Suite SHALL 在能量测试与 QUBO 等价测试中覆盖 `[2, 3, 1]` 与至少一个含 2 个隐藏层的结构。（对应总需求 17.2；训练测试的覆盖在 P2）
3. THE Test_Suite SHALL 包含"两相位能量收敛"测试：固定 Micro_Network 与初始状态，在 β = 0 与 β > 0 下断言 Continuous_Backend 的 `energy_trace` 在 ε_mono 内非增，且以 `stop_reason="converged"` 结束。（对应总需求 17.3；随机后端部分 17.4 在 P2）
4. THE Test_Suite SHALL 包含"三方能量等价"全枚举测试：对 K = 1 与 K = 2、β = 0 与 β > 0 的 Micro_Network 枚举全部 Logical_Bit 赋值，断言 F_β、QUBO 与 Ising_Objective 两两仅差常数（残差 ≤ 1e-8），并断言输入项消元、Auxiliary_Spin 两种规范的解码与三方 argmin 集合一致。（对应总需求 17.9）
5. IF Kaiwu 转换 API 在当前环境不可调用，THEN THE Test_Suite SHALL 仅显式跳过三方等价测试中的 Ising 部分，并继续执行 F_β↔QUBO 部分。（对应总需求 17.10）
6. THE Test_Suite SHALL 注册用于标识 Level_2_Test 的 pytest 标记。（对应总需求 17.12）
7. IF 带 Level_2_Test 标记的测试运行时 Kaiwu 未安装、授权不可用或 CIM 不可用，THEN THE Test_Suite SHALL 以写明具体原因的 skip 信息跳过该测试。（对应总需求 17.12）
8. THE Test_Suite SHALL 为 EPConfig 往返与 K_Bit_Encoding 往返提供以固定种子随机生成至少 100 组输入的性质测试。（对应总需求 17.13；聚合权重归一与 Best_So_Far 非增部分在 P2）

### Requirement 10: 公共 API 文档与上游集成规范

**User Story:** 作为上游维护者，我希望本期新增代码遵循仓库既有结构、许可与代码风格并带完整 docstring，以便后续 PR 可以评审与合并。

#### Acceptance Criteria

1. THE EP_Module SHALL 为本期新增的全部公共类与函数提供 docstring，写明参数、返回值、张量形状、dtype 与可能抛出的错误。（对应总需求 21.7）
2. THE EP_Changeset SHALL 将本期源码放在 Upstream_Clone `src/kaiwu/torch_plugin/` 下的 `equilibrium_propagation.py`、`_ep_dynamics.py`、`_ep_qubo.py` 与 `_ep_solver.py`，测试放在 `tests/test_ep_*.py`。（对应总需求 22.1；`_ep_training.py` 与 `example/<示例名>/` 在后续期）
3. THE EP_Changeset SHALL 在 `src/kaiwu/torch_plugin/__init__.py` 中导出 `EquilibriumPropagation` 与 `EPConfig`，并保留全部既有导出。（对应总需求 22.2；`EPTrainer` 与 `SolverAdapter` 的导出在后续期）
4. THE EP_Changeset SHALL 在每个新增源文件头部包含 `SPDX-License-Identifier: Apache-2.0`。（对应总需求 22.3）
5. THE EP_Changeset SHALL 使新增 Python 代码符合 PEP 8，并通过仓库既有的 lint 与格式检查。（对应总需求 22.4）

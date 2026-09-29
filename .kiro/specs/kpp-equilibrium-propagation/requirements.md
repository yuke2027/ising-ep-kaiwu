# Requirements Document

## Introduction

本规格依据 OSPP 2026（开源之夏）项目申请书（项目编号 26b1f0072，题目"基于 Kaiwu-PyTorch-Plugin 实现均衡传播算法"，开发周期 12 周），在上游仓库 `qboson/kaiwu-pytorch-plugin`（Apache-2.0）中实现可复用的均衡传播（Equilibrium Propagation, EP）模块。模块以 `torch.nn.Module` 形式连接 PyTorch 的模型/数据接口与 Kaiwu SDK 的模拟退火（SA）和相干伊辛机（CIM）后端，覆盖自由相位、微扰相位、K 位状态编码、QUBO/Ising 构造、局部梯度更新、预测与模型保存，并在 sklearn Digits 上与同规模 BP 网络做可复现对比。

**范围与约束**

- SDK 基线：使用公共 PyPI 上的 `kaiwu==1.4.1`，替代申请书中"以 1.3.1 为验收基线、不直接升级"的策略。该变更属于相对申请书的计划变更，须在结项报告中说明并保留导师确认记录（已获书面确认，存放位置待补，见需求 24.4）；不实现 1.3.1/1.4.x 双版本适配层或兼容矩阵。
- 模型范围：输入层、零到多个隐藏层与输出层；首版仅含相邻层二次耦合，非输入层状态有界于 [0,1]。
- 后端定位：SA 为可复现主基线并完成全部核心验收；CIM 在授权与配额允许时做扩展验证，受限时明确说明；确定性单元测试由连续后端与穷举后端承担。
- 结项约束：成果须以 PR 形式提交到上游仓库并被合并（结项时间以合并时间为准，合并阶段新开的 PR 不计入成果）；开发周期约自 2026 年 9 月中旬至 12 月上中旬，以 OSPP 系统显示为准（本文档撰写于 2026-09-28）。
- 前期工作：本工作区的 MNIST/100 单隐层 ±1 自旋 Ising EP（直接调用 Kaiwu 1.4.1 SA，三种子平均测试准确率 0.6233）提供可复用的 QUBO 常数处理、辅助自旋解码、按原始能量复算与确定性并列裁决、配对种子、`init_solution` 热启动与 fake SDK 测试写法。该工作只作为"前期可行性验证"，不是主交付；历史 `runs/` 产物只读。

**非目标**

- 高阶（三次及以上）能量项、非相邻层耦合、大规模深层网络。
- 默认把整个 batch 拼成块对角超大矩阵求解。
- 兼容 Kaiwu 1.3.1；承诺跨依赖版本逐位复现。
- 预填、估计或虚构任何实验数值；把 CIM 结果作为强制验收项。

**里程碑**（供任务阶段排期）：M1（第 1–3 周）能量/矩阵约定、连续后端、K 位 QUBO 与全枚举校验；M2（第 4–6 周）SA/CIM 适配、两相位梯度、多层、异常处理、保存加载；M3（第 7–9 周）参数化 pytest、Digits 管线、BP 对照、β 主实验；M4（第 10–12 周）预算/位宽/聚合消融、CIM 验证（可用时）、文档、回归与 PR。

## Glossary

- **Kaiwu_SDK**：玻色量子 Kaiwu SDK，本项目固定为 `kaiwu==1.4.1`。真实版本以 `importlib.metadata.version("kaiwu")` 为准（`kw.__version__` 返回依赖包 kaiwu-community 的版本 1.0.7）。1.4.1 不含 `kaiwu.conversion` 模块；`PrecisionReducer` 位于 `kaiwu.preprocess`；授权通过 `kw.license.init(user_id=..., sdk_code=...)` 完成。1.4.1 在公共 PyPI 仅提供 cp310 wheel（win_amd64、manylinux1_x86_64、macOS x86_64/arm64），因此支持的 Python 版本为 3.10；其依赖固定为 `numpy==2.2.6`、`matplotlib==3.10.3` 与 `kaiwu-community==1.0.7`。
- **Plugin_Repository**：上游仓库 `https://github.com/qboson/kaiwu-pytorch-plugin`（Apache-2.0）。源码位于 `src/kaiwu/torch_plugin/`，示例位于 `example/<name>/`，测试位于 `tests/test_*.py`，`__init__.py` 导出公共 API，当前 `__version__ = "0.3.0"`。
- **EP_Module**：`kaiwu.torch_plugin.EquilibriumPropagation`（`equilibrium_propagation.py`），`torch.nn.Module` 子类，持有可学习参数并提供能量计算、`equilibrate()`、`train_batch()` 与 `predict()`。
- **EPConfig**：`equilibrium_propagation.py` 中的配置类，统一保存网络结构、β、相位预算、状态步长、量化位宽、聚合方式、初始状态策略与随机种子，提供 `to_dict()` 与 `from_dict()`。
- **Layer_State**：第 l 层状态 `s^l`，`layer_sizes = [n_0, …, n_L]`。`s^0 = x` 为钳制输入，不作为求解变量；`s^1..s^L` 为待求平衡状态，取值于 [0,1]。
- **E_θ**：能量 `E_θ(s;x) = ½Σ_{l=1..L}‖s^l‖² − Σ_{l=0..L−1}(s^l)ᵀW^l s^{l+1} − Σ_{l=1..L}(b^l)ᵀs^l`，`W^l` 形状为 `(n_l, n_{l+1})`，`b^l` 形状为 `(n_l,)`。
- **C**：代价 `C(s^L, y) = ½‖s^L − y‖²`，`y` 为 one-hot 或 [0,1] 目标向量。
- **F_β**：总能量 `F_β = E_θ + β·C`。
- **Free_Phase**：β = 0 的平衡求解。
- **Nudged_Phase**：β > 0 的平衡求解。
- **Phase_Pair**：同一样本在同一 Parameter_Snapshot 下的一次 Free_Phase 与一次 Nudged_Phase。
- **Parameter_Snapshot**：一个批次开始求解前的参数副本；批内全部 Phase_Pair 基于同一快照。
- **EP_Gradient**：`ĝ_θ = [∂_θE_θ(s_nudged) − ∂_θE_θ(s_free)] / β`。因 `∂E_θ/∂W^l = −s^l(s^{l+1})ᵀ`、`∂E_θ/∂b^l = −s^l`，有 `g_W^l = [corr_free − corr_nudged] / β`、`g_b^l = [s_free^l − s_nudged^l] / β`，可直接写入最小化型优化器（无需再反号）。
- **Equilibrium_Result**：`equilibrate()` 的返回，包含 Layer_State、按原始 F_β 复算的能量、Best_So_Far、Stop_Reason、Budget 与求解器元数据。
- **Continuous_Backend**：`_ep_dynamics.py` 中的 `ContinuousDynamics`，在 [0,1] 盒约束内对 F_β 做投影梯度下降的确定性参考后端，输出 `energy_trace`。
- **Discrete_Backend**：在 State_Grid 上直接最小化量化后 F_β 的后端统称，包括 Exhaustive_Backend、SA_Backend 与 CIM_Backend。
- **Exhaustive_Backend**：枚举全部 Logical_Bit 赋值的确定性后端，仅用于 Micro_Network。
- **SA_Backend**：基于 `kaiwu.classical.SimulatedAnnealingOptimizer.single_process_solve` 的模拟退火后端，为可复现主基线。
- **CIM_Backend**：基于 `kaiwu.cim.CIMOptimizer` 的相干伊辛机后端，依赖授权、网络与配额，用于扩展验证。
- **Solver_Adapter**：`_ep_solver.py` 中的 `SolverAdapter`，统一 Exhaustive_Backend、SA_Backend、CIM_Backend 与 Continuous_Backend 的调用、候选解码、聚合与元数据。
- **K_Bit_Encoding**：`s_i(q) = Σ_{k=0..K−1} 2^k q_{i,k} / (2^K − 1)`，`q_{i,k} ∈ {0,1}`，`K = state_bits`。
- **State_Grid**：K_Bit_Encoding 可表示的状态集合 `{0, 1/(2^K−1), …, 1}`。
- **Logical_Bit**：QUBO 二值变量 `q_{i,k}`；网络共有 `K·Σ_{l=1..L} n_l` 个（64—32—10 网络为 42K 个）。
- **QUBO_Builder**：`_ep_qubo.py` 中的 `EPQuboBuilder`，负责输入消元、K 位展开、Coefficient_Table、SDK 矩阵转换与解码辅助。
- **Coefficient_Table**：内部 QUBO 表示；每个无序变量对 `(i, j)`（i < j）的二次系数只记一次，线性系数与 Constant_Offset 单独保存。
- **SDK_Matrix_Convention**：Kaiwu 1.4.1 `kaiwu.core.qubo_matrix_to_qubo_model` 的上三角约定：目标为 `qᵀQq`，对角元为线性系数，`Q[i, j]`（i < j）为二次系数，下三角元为 0。
- **Constant_Offset**：满足 `F_β(s(q)) = qᵀQq + Constant_Offset` 的常数（β > 0 时包含 `β·½‖y‖²`）。
- **Ising_Objective**：Kaiwu 1.4.1 Ising 模型目标 `H(σ) = −σᵀJσ + ising_bias`，J 与 `ising_bias` 分别取自 `get_matrix()` 与 `get_bias()`；前期集成测试已验证 `H(σ)` 与 `qᵀQq` 相等。
- **Auxiliary_Spin**：`kaiwu.core.qubo_model_to_ising_model` 为表达线性项引入的额外自旋；物理解码对 Auxiliary_Spin 的两种取值（±1）给出同一结果。
- **Candidate**：一次后端调用返回的一个解，解码为 Layer_State 后参与聚合。
- **Aggregation**：把去重后的 Candidate 映射为非负且和为 1 的权重：`best`（best-of-run，默认）、`topk_mean`、`energy_weighted`。
- **Best_So_Far**：按候选返回顺序与重复顺序计算的复算 F_β 运行最小值；非增只是记录定义，不证明收敛。
- **Budget**：一次平衡求解的资源描述。Continuous_Backend 为 `free_steps`/`nudged_steps` 步数上限与实际步数；SA_Backend 为退火参数、温度层数、每层迭代数、候选数与重复次数；CIM_Backend 为 `task_mode`、样本数与重复次数。
- **Paired_Seed**：同一 Phase_Pair 两个相位共用的 SA 随机种子，由 `(seed, epoch, 样本索引)` 确定性派生。
- **Initial_State_Policy**：Nudged_Phase 初始状态策略：`shared`（与 Free_Phase 同一初始状态，默认，对应申请书"两相位仅 β·C 不同"）或 `warm_start`（以 Free_Phase 所选状态为初始状态，对应前期实现的热启动）。
- **Repeat_Dispersion**：多次重复求解所选复算 F_β 的均值、标准差、最小值与最大值。
- **Stop_Reason**：停止原因。Continuous_Backend 取 `converged`、`max_steps` 或 `energy_increase`；SA_Backend 与 CIM_Backend 取 `budget_exhausted` 或 `no_improvement`。
- **ε_mono**：能量单调性容差 `1e-12·max(1, |F_β(s_t)|)`。
- **EP_Error**：本模块显式异常的公共基类；子类区分配置、动力学、系数范围、配对、求解、授权与数值错误，配置错误同时继承 `ValueError`。
- **EP_Trainer**：`_ep_training.py` 中的 `EPTrainer`，负责两相位梯度、批处理、日志、评估与检查点；`model.train_batch()` 委托 EP_Trainer 实现。
- **Micro_Network**：用于全枚举与梯度校验的微型网络（如 `[2, 2, 2]`、`[2, 3, 1]`），Logical_Bit 数不超过 `exhaustive_max_bits`。
- **Level_1_Test**：无需 Kaiwu 授权与网络即可运行的确定性测试（Continuous_Backend、Exhaustive_Backend、纯 Torch 映射与 fake SDK）。
- **Level_2_Test**：需要 Kaiwu 授权（真实 SA 调用）或 CIM 授权/网络/配额的测试，以 pytest 标记标识，条件不满足时显式跳过。
- **Test_Suite**：`tests/test_ep_*.py` 中的 pytest 测试集合。
- **Digits_Benchmark**：`sklearn.datasets.load_digits` 数据集：1,797 个 8×8 样本、64 维、10 类。
- **Digits_Example**：`example/<示例名>/` 下的 Digits 分类示例，一条命令运行 EP 与 BP 对照。
- **BP_Baseline**：与 EP 同为 64—32—10 结构、以反向传播训练的 MLP 对照模型。
- **Smoke_Config**：Digits_Example 的快速配置：不超过 100 个训练样本、不超过 100 个测试样本、1 个 epoch、1 个种子。
- **Experiment_Runner**：Digits_Example 中按实验矩阵运行、汇总并写出产物的入口。
- **Run_Manifest**：每次运行写出的可复现性记录文件（JSON）。
- **Documentation**：插件 README、示例 README 与公共 API docstring。
- **License_Initializer**：示例入口与 Level_2_Test 夹具使用的授权初始化辅助函数。
- **Upstream_PR**：提交到 Plugin_Repository 的 Pull Request 及其提交内容。
- **Final_Report**：提交到 OSPP 系统的结项报告。
- **Preliminary_Validation**：本工作区 MNIST/100 ±1 自旋 Ising EP 工作及其只读历史产物。

## Requirements

### Requirement 1: EP 统一配置

**User Story:** 作为插件使用者，我希望用一个 `EPConfig` 统一声明网络结构、β、相位预算、状态步长、量化位宽、聚合方式、初始状态策略与随机种子，以便实验配置可校验、可保存、可复现。

#### Acceptance Criteria

1. THE EPConfig SHALL 至少提供字段 `layer_sizes`、`beta`、`beta_min`、`free_steps`、`nudged_steps`、`step_size`、`max_backtracks`、`residual_tol`、`state_tol`、`state_bits`、`aggregation`、`topk`、`aggregation_temperature`、`initial_state_policy`、`gradient_estimator` 与 `seed`。
2. WHEN 调用方以 `EPConfig(layer_sizes=[64, 32, 10], beta=0.2, free_steps=50, nudged_steps=25, state_bits=1, aggregation="best", seed=2026)` 构造配置，THE EPConfig SHALL 创建成功，并为未指定字段填入 Documentation 中列出的默认值。
3. THE EPConfig SHALL 以 `aggregation="best"`、`initial_state_policy="shared"` 与 `gradient_estimator="one_sided"` 作为默认值。
4. IF `layer_sizes` 少于 2 个元素或含非正整数，THEN THE EPConfig SHALL 抛出配置错误，错误信息包含字段名与收到的值。
5. IF `beta` 非有限或小于 `beta_min`，THEN THE EPConfig SHALL 抛出配置错误，错误信息包含 `beta` 与 `beta_min` 的数值。
6. IF `free_steps` 或 `nudged_steps` 不是正整数、`max_backtracks` 不是非负整数，或 `beta_min`、`step_size`、`residual_tol`、`state_tol` 不是正有限数，THEN THE EPConfig SHALL 抛出配置错误。
7. IF `state_bits` 不是 1 到 `MAX_STATE_BITS`（不小于 3）之间的整数，THEN THE EPConfig SHALL 抛出配置错误。
8. IF `aggregation` 不属于 {`"best"`, `"topk_mean"`, `"energy_weighted"`}、`initial_state_policy` 不属于 {`"shared"`, `"warm_start"`}，或 `gradient_estimator` 不属于 {`"one_sided"`, `"symmetric"`}，THEN THE EPConfig SHALL 抛出配置错误并列出允许取值。
9. IF `aggregation="topk_mean"` 而 `topk` 不是正整数，或 `aggregation="energy_weighted"` 而 `aggregation_temperature` 不是正有限数，THEN THE EPConfig SHALL 抛出配置错误。
10. WHEN 调用 `cfg.to_dict()`，THE EPConfig SHALL 返回仅含 JSON 可序列化值（`str`、`int`、`float`、`bool`、`list`、`dict`、`None`）的字典。
11. THE EPConfig SHALL 对任意合法实例 `cfg` 满足 `EPConfig.from_dict(cfg.to_dict()) == cfg`，并在经 `json.dumps`/`json.loads` 往返后仍满足该等式（往返性质）。
12. IF `EPConfig.from_dict` 收到缺少必需键或含未知键的字典，THEN THE EPConfig SHALL 抛出配置错误并列出缺失键与未知键。

### Requirement 2: 模型结构、参数与 PyTorch 兼容

**User Story:** 作为 PyTorch 使用者，我希望 `EquilibriumPropagation` 是标准 `torch.nn.Module`，支持多层结构并兼容 `Dataset`、`DataLoader`、`torch.optim` 与 `state_dict`，以便像普通模型一样训练与管理。

#### Acceptance Criteria

1. THE EP_Module SHALL 继承 `torch.nn.Module`，并为 `layer_sizes=[n_0, …, n_L]` 注册 L 个耦合参数 `W^l`（形状 `(n_l, n_{l+1})`，l = 0..L−1）与 L 个偏置参数 `b^l`（形状 `(n_l,)`，l = 1..L），参数仅包含相邻层耦合与偏置。
2. THE EP_Module SHALL 接受长度不小于 2 的任意合法 `layer_sizes`，即支持 0 个、1 个或多个隐藏层。
3. WHEN 以相同 EPConfig（含相同 `seed`）两次构造 EP_Module，THE EP_Module SHALL 生成逐元素相同的全部参数。
4. THE EP_Module SHALL 默认以 `torch.float64` 创建参数，并使用 Documentation 中记录的初始化分布与尺度规则。
5. THE EP_Module SHALL 使用确定且记录在 Documentation 中的 `state_dict` 键名，使键集合与各张量形状仅由 `layer_sizes` 决定。
6. WHEN 调用方传入来自 `torch.utils.data.DataLoader` 的批数据（`x` 为形状 `(B, n_0)` 的 `float32` 或 `float64` 张量，`y` 为形状 `(B,)` 的类别索引或形状 `(B, n_L)` 的 [0,1] 目标），THE EP_Module SHALL 将 `x` 转换为参数 dtype，并将类别索引转换为 one-hot 目标。
7. IF `x` 或 `y` 的形状与 `layer_sizes` 不符、含 NaN 或 Inf，或类别索引超出 `[0, n_L)`，THEN THE EP_Module SHALL 抛出 `ValueError`，错误信息包含参数名与期望形状或取值范围。
8. THE EP_Module SHALL 通过 `model.parameters()` 暴露全部可学习参数，使 `torch.optim.SGD` 与 `torch.optim.Adam` 可直接用于参数更新。

### Requirement 3: 能量、代价与总能量

**User Story:** 作为算法开发者，我希望模块严格按申请书公式计算 E_θ、C 与 F_β 并支持对参数自动微分，以便两相位求解、QUBO 构造与梯度共享同一能量定义。

#### Acceptance Criteria

1. THE EP_Module SHALL 按 E_θ 的定义（`s^0 = x`）计算能量，对批输入返回形状 `(B,)` 的张量，对形状 `(n_0,)` 的单样本输入返回标量。
2. THE EP_Module SHALL 按 C 的定义计算代价，并按 `F_β = E_θ + β·C` 计算总能量。
3. WHEN 以分离（detached）的 Layer_State 计算 E_θ 并调用 `backward()`，THE EP_Module SHALL 得到满足 `∂E_θ/∂W^l = −s^l(s^{l+1})ᵀ` 与 `∂E_θ/∂b^l = −s^l` 的参数梯度（与解析式的最大绝对误差 ≤ 1e-12）。
4. THE EP_Module SHALL 对任意合法输入与状态满足 `F_0 = E_θ`（逐元素相等）。
5. IF 传入的 Layer_State 数量不等于 L、某层宽度不等于 `n_l`，或状态值超出 [0,1] 的幅度大于 1e-12，THEN THE EP_Module SHALL 抛出 `ValueError`。
6. IF 传入的 `beta` 为负数或非有限值，THEN THE EP_Module SHALL 抛出 `ValueError`。

### Requirement 4: 连续参考后端

**User Story:** 作为算法开发者，我希望有确定性的连续状态后端，在 [0,1] 盒约束内使 F_β 非增并按残差与状态变化判停，以便作为梯度方向与"下降并稳定"行为的参考。

#### Acceptance Criteria

1. THE Continuous_Backend SHALL 在每一步把 `s^1..s^L` 投影到 [0,1]，并保持 `s^0 = x` 不变。
2. THE Continuous_Backend SHALL 使每个被接受的步骤满足 `F_β(s_{t+1}) ≤ F_β(s_t) + ε_mono`。
3. IF 按当前步长计算的一步使 F_β 上升超过 ε_mono，THEN THE Continuous_Backend SHALL 将该步步长减半后重算，至多 `max_backtracks` 次。
4. WHEN 投影梯度残差 `‖s − Π_[0,1](s − ∇_s F_β(s))‖_∞ ≤ residual_tol` 且相邻步状态变化 `‖s_{t+1} − s_t‖_∞ ≤ state_tol`，THE Continuous_Backend SHALL 停止并记录 `stop_reason="converged"`。
5. WHEN Free_Phase 达到 `free_steps` 步或 Nudged_Phase 达到 `nudged_steps` 步仍未满足收敛条件，THE Continuous_Backend SHALL 停止并记录 `stop_reason="max_steps"`。
6. THE Continuous_Backend SHALL 在 Equilibrium_Result 中返回逐步 `energy_trace`、最终残差、最终状态变化、实际步数、回溯次数与 Stop_Reason。
7. WHEN 在同一环境中以相同参数、输入、初始状态与配置重复运行，THE Continuous_Backend SHALL 返回逐元素相同的状态与 `energy_trace`。

### Requirement 5: K 位状态编码与 QUBO 构造

**User Story:** 作为算法开发者，我希望把每个非输入神经元按 K 位定点编码代入 F_β 得到 QUBO，并显式保存常数偏移，以便离散后端最小化的正是量化后的 F_β。

#### Acceptance Criteria

1. THE QUBO_Builder SHALL 按 K_Bit_Encoding 对 `s^1..s^L` 的每个神经元编码，使可表示状态恰为 State_Grid。
2. THE QUBO_Builder SHALL 对任意 K 与任意 `q ∈ {0,1}^K` 满足 `encode(decode(q)) = q`，并对 State_Grid 中任意值 v 满足 `decode(encode(v)) = v`（往返性质）。
3. THE QUBO_Builder SHALL 生成恰好 `K·Σ_{l=1..L} n_l` 个 Logical_Bit；对 `layer_sizes=[64, 32, 10]` 且 K = 1/2/3，Logical_Bit 数分别为 42/84/126。
4. THE QUBO_Builder SHALL 将输入 x 作为常量，把 `−xᵀW^0 s^1` 并入 `s^1` 对应 Logical_Bit 的线性系数，使问题变量中不含输入层。
5. THE QUBO_Builder SHALL 在 Coefficient_Table 中对每个无序变量对只记录一个二次系数，并单独记录线性系数与 Constant_Offset。
6. THE QUBO_Builder SHALL 对任意合法参数、输入、目标、β ≥ 0 与任意 Logical_Bit 赋值 q，满足 `F_β(s(q)) = Σ_i a_i q_i + Σ_{i<j} c_ij q_i q_j + Constant_Offset`（绝对误差 ≤ 1e-9）。
7. WHEN β = 0，THE QUBO_Builder SHALL 生成与目标 y 无关的 Coefficient_Table 与 Constant_Offset。
8. THE QUBO_Builder SHALL 为每个问题报告 Logical_Bit 数、非零二次项数、最大绝对系数、最小非零绝对系数，以及最大绝对系数与最小非零绝对系数之比（动态范围）。
9. WHEN 以相同 `layer_sizes` 与 `state_bits` 重复构造问题，THE QUBO_Builder SHALL 复用同一位索引布局与稀疏结构模板并只更新系数，模板构造计数保持为 1。

### Requirement 6: SDK 矩阵约定、Ising 转换与解码

**User Story:** 作为算法开发者，我希望 Coefficient_Table 按 Kaiwu 1.4.1 的矩阵约定转换并经官方 API 得到 Ising 模型，再把解还原为层状态，以便 F_β、QUBO 与 Ising 三者仅差常数且解码后的 argmin 一致。

#### Acceptance Criteria

1. WHEN QUBO_Builder 为 Discrete_Backend 导出矩阵，THE QUBO_Builder SHALL 按 SDK_Matrix_Convention 生成上三角矩阵 Q：对角元为线性系数，`Q[i, j]`（i < j）为二次系数，下三角元为 0。
2. THE QUBO_Builder SHALL 对任意 Logical_Bit 赋值 q 满足 `qᵀQq + Constant_Offset = F_β(s(q))`（绝对误差 ≤ 1e-9），以排除非对角项加倍。
3. WHEN 构造 Kaiwu 问题，THE Solver_Adapter SHALL 调用 `kaiwu.core.qubo_matrix_to_qubo_model(Q)` 与 `kaiwu.core.qubo_model_to_ising_model(...)`，并从 Ising 模型读取 J、变量表与 `ising_bias`。
4. THE Solver_Adapter SHALL 依据 Kaiwu 变量表建立 Logical_Bit 与 Ising 变量的一一映射，并识别至多 1 个 Auxiliary_Spin。
5. THE Solver_Adapter SHALL 对任意自旋解 σ 满足 `H(σ) + Constant_Offset = F_β(s(q(σ)))`（H 为 Ising_Objective，绝对误差 ≤ 1e-8）。
6. THE Solver_Adapter SHALL 通过 `kaiwu.core.get_sol_dict` 恢复 Logical_Bit 取值，并经 K_Bit_Encoding 还原为 [0,1] Layer_State。
7. THE Solver_Adapter SHALL 对任意自旋解 σ 及其全局翻转 −σ（Auxiliary_Spin 的两种规范）解码出相同的 Logical_Bit 与 Layer_State。
8. IF Kaiwu 变量表无法与 Logical_Bit 一一对应或含多于 1 个 Auxiliary_Spin，THEN THE Solver_Adapter SHALL 抛出求解错误，错误信息包含期望与实际变量数。

### Requirement 7: 求解器适配层与 Kaiwu 1.4.1 边界

**User Story:** 作为插件使用者，我希望通过统一的 `SolverAdapter` 选择穷举、SA 或 CIM 并得到结构一致的结果与元数据，以便切换后端时训练与评估代码保持不变。

#### Acceptance Criteria

1. THE Solver_Adapter SHALL 提供 Exhaustive_Backend、SA_Backend 与 CIM_Backend，并使三者与 Continuous_Backend 返回同一结构的 Equilibrium_Result。
2. WHEN 调用 `model.equilibrate(x, solver=solver, beta=β, y=y)`，THE EP_Module SHALL 通过所给后端为每个样本返回一个 Equilibrium_Result。
3. IF 调用 `equilibrate` 时 β > 0 而未提供 `y`，THEN THE EP_Module SHALL 抛出 `ValueError`。
4. THE Solver_Adapter SHALL 在 Equilibrium_Result 中返回 Layer_State、复算 F_β、Best_So_Far 序列、Stop_Reason、Budget、`n_candidates` 与 `n_unique`。
5. THE Solver_Adapter SHALL 在 Equilibrium_Result 中附带 JSON 可序列化的元数据：后端名、Kaiwu 版本、种子、Logical_Bit 数、Ising 变量数、Auxiliary_Spin 名称、Initial_State_Policy 与各阶段计时。
6. THE Solver_Adapter SHALL 通过 `to_dict()` 输出不含凭据的 JSON 可序列化求解器配置，至少包含后端名、各后端参数及 `repeats`、`patience`、`max_retries`、`exhaustive_max_bits`、`coefficient_policy`、`coefficient_limit`、`dynamic_range_limit` 与 `precision_policy`。
7. THE Solver_Adapter SHALL 通过 `importlib.metadata.version("kaiwu")` 读取 Kaiwu 版本。
8. IF 已安装的 Kaiwu 版本不是 `1.4.1`，THEN THE Solver_Adapter SHALL 发出警告，警告信息包含已验证版本 `1.4.1` 与实际版本。
9. THE Solver_Adapter SHALL 仅调用 Kaiwu 1.4.1 中已核实的 API：`kaiwu.core.qubo_matrix_to_qubo_model`、`kaiwu.core.qubo_model_to_ising_model`、`kaiwu.core.get_sol_dict`、`kaiwu.classical.SimulatedAnnealingOptimizer`、`kaiwu.cim.CIMOptimizer` 与 `kaiwu.preprocess.PrecisionReducer`。
10. IF `kaiwu` 未安装或上述任一 API 缺失，THEN THE Solver_Adapter SHALL 在首次使用对应后端时抛出求解错误，错误信息指明缺失的模块或属性名。
11. THE Solver_Adapter SHALL 仅使用调用方配置的后端执行求解，并在元数据中记录该后端名。
12. THE Solver_Adapter SHALL 将 Kaiwu 导入延迟到首次使用 SA_Backend 或 CIM_Backend 时执行，使 Continuous_Backend 与 Exhaustive_Backend 在未安装 Kaiwu 时可用。

### Requirement 8: 穷举后端与 SA 后端

**User Story:** 作为算法开发者，我希望穷举后端为微型网络给出确定性精确解，SA 后端以配对种子和完整预算记录提供可复现主基线，以便单元测试确定、实验可复现。

#### Acceptance Criteria

1. WHEN Logical_Bit 数不超过 `exhaustive_max_bits`（默认 16），THE Exhaustive_Backend SHALL 枚举全部 `2^n` 个赋值，并返回复算 F_β 与最低值之差在 1e-9 以内的全部状态。
2. IF Logical_Bit 数超过 `exhaustive_max_bits`，THEN THE Exhaustive_Backend SHALL 在枚举开始前抛出求解错误，错误信息包含位数与上限。
3. WHEN 选择 SA_Backend，THE Solver_Adapter SHALL 以 `initial_temperature`、`alpha`、`cutoff_temperature`、`iterations_per_t`、`size_limit`、`rand_seed` 与 `process_num` 构造 `kaiwu.classical.SimulatedAnnealingOptimizer`，并调用 `single_process_solve(ising_matrix=..., init_solution=..., rand_seed=...)`。
4. WHEN 在同一环境中以相同 Ising 矩阵、初始解、退火参数与 `rand_seed` 重复调用，THE SA_Backend SHALL 返回逐元素相同的候选集合。
5. WHERE 配置了重复次数 `repeats = R`（R > 1），THE SA_Backend SHALL 执行 R 次独立求解，第 r 次使用由 Paired_Seed 与 r 确定性派生的种子，并按重复顺序与候选顺序计算 Best_So_Far。
6. THE SA_Backend SHALL 在 Budget 中记录全部退火参数、温度层数、每层迭代数、请求与实际返回的候选数，以及计划与实际执行的重复次数。
7. THE SA_Backend SHALL 报告 Repeat_Dispersion，并在 R = 1 时将标准差记为 0。
8. THE SA_Backend SHALL 仅以 `budget_exhausted` 或 `no_improvement` 作为 Stop_Reason。
9. WHERE 配置了 `patience = P`，WHEN Best_So_Far 连续 P 次重复无改进，THE SA_Backend SHALL 提前停止，并记录 `stop_reason="no_improvement"` 与无改进窗口长度。
10. THE Solver_Adapter SHALL 在结果与报告中把 Best_So_Far 标注为运行最小值记录，并与 Continuous_Backend 的收敛判定分开呈现。

### Requirement 9: CIM 后端与精度约束

**User Story:** 作为研究者，我希望在授权与配额允许时用 CIM 做扩展验证，并在精度、授权或配额受限时得到显式错误或记录，以便 CIM 结果可信且不影响 SA 主基线的核心验收。

#### Acceptance Criteria

1. WHERE 配置了 CIM_Backend，THE Solver_Adapter SHALL 通过 `kaiwu.cim.CIMOptimizer` 提交 Ising 矩阵，并在元数据中记录 `task_mode`、请求样本数、计划重复次数与实际完成的重复次数。
2. WHEN 使用 CIM_Backend，THE Solver_Adapter SHALL 在提交前判定 Ising 矩阵能否以 8 位精度表示，并记录缩放因子、量化后最大绝对系数误差与所选精度处理方式。
3. IF Ising 矩阵无法以 8 位精度表示且 `precision_policy="error"`，THEN THE Solver_Adapter SHALL 抛出系数范围错误并报告动态范围。
4. WHERE `precision_policy="reduce"`，THE Solver_Adapter SHALL 使用 `kaiwu.preprocess.PrecisionReducer` 或统一正缩放完成精度处理，并按未量化的原始 F_β 复算 CIM 返回样本的能量。
5. IF 精度处理后同一 Phase_Pair 两个相位的 Ising 矩阵逐元素相同而 β > 0，THEN THE Solver_Adapter SHALL 抛出系数范围错误，说明 β·C 引入的差异在量化中丢失。
6. THE CIM_Backend SHALL 将 `energy_trace` 置为空、在元数据中记录 `trace_unavailable_reason`，并仅基于 CIM 返回样本计算 Best_So_Far。
7. IF `initial_state_policy="warm_start"` 与 CIM_Backend 组合使用而 Kaiwu 1.4.1 的 `CIMOptimizer` 不接受初始解，THEN THE Solver_Adapter SHALL 在提交前抛出配置错误。
8. IF CIM 授权、网络或配额不可用，THEN THE Solver_Adapter SHALL 抛出求解错误，错误信息区分授权、网络与配额原因且不含凭据值。

### Requirement 10: 候选聚合

**User Story:** 作为研究者，我希望默认以 best-of-run 选择平衡状态，并可切换 top-k 平均与能量加权作为消融，以便比较聚合方式对梯度方差与分类性能的影响。

#### Acceptance Criteria

1. THE Solver_Adapter SHALL 对每个 Candidate 解码后按原始 F_β（而非 SDK 目标值）复算能量，并以复算能量作为全部 Aggregation 的排序依据。
2. THE Solver_Adapter SHALL 在聚合前按解码后的 Layer_State 对 Candidate 去重，并记录 `n_candidates` 与 `n_unique`。
3. WHERE `aggregation="best"`，THE Solver_Adapter SHALL 对复算 F_β 最低的唯一候选赋权重 1。
4. WHEN 多个唯一候选的复算 F_β 在 1e-9 容差内并列最低，THE Solver_Adapter SHALL 选择展平后 Layer_State 字典序最小的候选。
5. WHERE `aggregation="topk_mean"`，THE Solver_Adapter SHALL 对复算 F_β 最低的 `k_eff = min(topk, n_unique)` 个唯一候选赋等权重 `1/k_eff`，并记录 `k_eff`。
6. WHERE `aggregation="energy_weighted"`，THE Solver_Adapter SHALL 对唯一候选赋权重 `w_i ∝ exp(−(F_i − F_min) / aggregation_temperature)`。
7. THE Solver_Adapter SHALL 对任意候选集合与任意 Aggregation 输出非负且和为 1（误差 ≤ 1e-12）的权重。
8. THE Solver_Adapter SHALL 以聚合权重下的加权平均 Layer_State 作为 Equilibrium_Result 的报告状态。
9. WHILE 使用 Continuous_Backend，THE Solver_Adapter SHALL 返回权重为 1 的单一状态，并在元数据中记录 `n_candidates=1`。

### Requirement 11: 两相位配对与初始状态策略

**User Story:** 作为研究者，我希望自由相位与微扰相位除 β·C 外保持同后端、同预算、同精度、同种子，并把初始状态策略作为可配置且被记录的选项，以便梯度差异只来自微扰项或被明确记录的额外差异。

#### Acceptance Criteria

1. WHEN EP_Trainer 执行一个 Phase_Pair，THE EP_Trainer SHALL 在两个相位使用同一 Parameter_Snapshot、同一后端与同一 `state_bits`。
2. WHILE 使用 Discrete_Backend，THE EP_Trainer SHALL 在同一 Phase_Pair 的两个相位使用相同的 Budget 与精度处理设置。
3. WHILE 使用 SA_Backend，THE EP_Trainer SHALL 在同一 Phase_Pair 的两个相位使用同一 Paired_Seed，多次重复时逐次配对。
4. WHILE 使用 CIM_Backend，THE EP_Trainer SHALL 在同一 Phase_Pair 的两个相位使用相同的 `task_mode`、样本数与重复次数。
5. THE EP_Trainer SHALL 由 `(seed, epoch, 样本索引)` 确定性派生 Paired_Seed，并写入两个相位的元数据。
6. IF 同一 Phase_Pair 两个相位记录的后端、Budget（Discrete_Backend）、精度设置或 Paired_Seed 不一致，THEN THE EP_Trainer SHALL 在参数更新前抛出配对错误。
7. WHEN 调用方未提供初始状态，THE EP_Module SHALL 由 `(seed, 样本索引)` 按 Documentation 记录的规则确定性生成 Free_Phase 初始状态。
8. WHERE `initial_state_policy="shared"`，THE EP_Trainer SHALL 使 Nudged_Phase 与 Free_Phase 从同一初始状态出发（SA_Backend 下两相位传入相同的 `init_solution` 或均不传入）。
9. WHERE `initial_state_policy="warm_start"`，THE EP_Trainer SHALL 以 Free_Phase 所选状态作为 Nudged_Phase 初始状态（SA_Backend 下以含 Auxiliary_Spin 的完整自旋解作为 `init_solution`）。
10. THE EP_Trainer SHALL 在每个 Phase_Pair 的元数据与每次运行的配置产物中记录 Initial_State_Policy。
11. WHEN 以 β = 0 构造同一样本的两个相位问题，THE QUBO_Builder SHALL 生成逐元素相同的 Coefficient_Table 与 Constant_Offset。

### Requirement 12: EP 梯度与训练接口

**User Story:** 作为 PyTorch 使用者，我希望 `train_batch` 用两相位平衡状态计算局部 EP 梯度并交给 `torch.optim` 执行一次最小化更新，以便用标准优化器与 DataLoader 训练 EP 模型。

#### Acceptance Criteria

1. WHEN 调用 `model.train_batch(x, y, solver=solver, optimizer=optimizer)`，THE EP_Trainer SHALL 在求解前保存 Parameter_Snapshot，并基于该快照对批内每个样本完成 Phase_Pair。
2. THE EP_Trainer SHALL 按 `ĝ_θ = [G_nudged − G_free] / β` 计算每个样本的 EP_Gradient（`G_phase` 为该相位候选上 `∂_θE_θ` 按 Aggregation 权重的加权平均；`best` 时即所选单一状态的 `∂_θE_θ`），并对批内样本取均值。
3. WHEN 批均值梯度计算完成，THE EP_Trainer SHALL 清零既有梯度、将批均值 EP_Gradient 原样写入各参数的 `.grad`，并调用一次 `optimizer.step()`。
4. IF 求解期间任一参数与 Parameter_Snapshot 不再逐元素相同，THEN THE EP_Trainer SHALL 抛出配对错误且不调用 `optimizer.step()`。
5. WHEN `train_batch` 返回，THE EP_Trainer SHALL 提供指标字典，包含批大小、Free_Phase 平均代价、批内预测准确率、零更新样本比例、两相位状态差范数、各阶段计时与 `solver_calls`。
6. WHERE 配置了 `gradient_estimator="symmetric"`，THE EP_Trainer SHALL 以 +β 与 −β 两个微扰相位计算 `[G_{+β} − G_{−β}] / (2β)`，并在元数据中记录估计器类型。
7. WHEN 调用 `EPTrainer.fit(train_loader, epochs=...)`，THE EP_Trainer SHALL 逐批调用 `train_batch`，在每个 epoch 结束时执行评估并写出一条日志记录，并按配置保存检查点。

### Requirement 13: 预测与评估

**User Story:** 作为使用者，我希望 `predict` 用自由相位平衡状态给出类别并有统一的评估接口，以便在 Digits 上报告准确率与误差。

#### Acceptance Criteria

1. WHEN 调用 `model.predict(x, solver=solver)`，THE EP_Module SHALL 以 β = 0 执行 Free_Phase，并返回形状 `(B,)` 的 `torch.long` 类别索引，类别为输出层报告状态 `s^L` 的最大分量索引。
2. WHEN 输出层多个分量在 1e-12 容差内并列最大，THE EP_Module SHALL 返回并列分量中最小的索引，并在详细结果中标记该样本为并列。
3. WHERE 调用方传入 `return_details=True`，THE EP_Module SHALL 同时返回每个样本的 Equilibrium_Result。
4. THE EP_Module SHALL 在 `predict` 调用前后保持全部参数逐元素相同。
5. WHEN 在同一环境中以相同参数、输入、后端配置与种子重复调用 `predict`，THE EP_Module SHALL 返回相同的类别索引。
6. WHEN 调用 `EPTrainer.evaluate(loader, solver=...)`，THE EP_Trainer SHALL 返回准确率、分类误差（1 − 准确率）、样本数、并列样本数与评估耗时。

### Requirement 14: 保存与加载

**User Story:** 作为使用者，我希望通过 `state_dict` 与 `EPConfig.to_dict()` 保存并恢复模型与训练状态，以便训练结果可复用、可复验。

#### Acceptance Criteria

1. THE EP_Module SHALL 支持 `torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, path)` 形式的检查点，且 `model.state_dict()` 仅含参数张量。
2. WHEN 以 `EquilibriumPropagation(EPConfig.from_dict(ckpt["config"]))` 重建模型并调用 `load_state_dict(ckpt["model"])`，THE EP_Module SHALL 恢复逐元素相同的全部参数。
3. WHEN 恢复后的模型在同一环境中使用与保存前相同的输入、后端配置与种子，THE EP_Module SHALL 给出与保存前绝对误差 ≤ 1e-12 的能量值与相同的预测类别。
4. THE EP_Trainer SHALL 在训练检查点中保存 EPConfig、预处理参数、类别映射、求解器配置（不含凭据）、epoch 计数与优化器 `state_dict`。
5. WHEN 从训练检查点恢复 EP_Trainer，THE EP_Trainer SHALL 恢复参数、优化器状态与 epoch 计数后继续训练。
6. IF `load_state_dict` 收到的键集合或张量形状与当前 `layer_sizes` 不一致，THEN THE EP_Module SHALL 抛出错误，错误信息包含不一致的键名与形状。

### Requirement 15: 错误处理与显式失败

**User Story:** 作为研究者，我希望 β 过小、能量上升、系数超范围、重复解与求解失败都给出显式错误或有记录的回退，以便异常不会被静默吞掉并污染实验结果。

#### Acceptance Criteria

1. THE EP_Module SHALL 使全部显式错误继承自 EP_Error，并提供可区分的配置、动力学、系数范围、配对、求解、授权与数值错误子类。
2. IF 训练或 Nudged_Phase 使用的 β 小于 `beta_min`，THEN THE EP_Trainer SHALL 在任何求解调用之前抛出配置错误。
3. IF Continuous_Backend 回溯 `max_backtracks` 次后 F_β 仍上升超过 ε_mono，THEN THE Continuous_Backend SHALL 记录 `stop_reason="energy_increase"` 并抛出动力学错误，错误信息包含相位、步数与能量增量。
4. IF 任一 QUBO 系数绝对值超过 `coefficient_limit` 或动态范围超过 `dynamic_range_limit`，且 `coefficient_policy="error"`，THEN THE QUBO_Builder SHALL 抛出系数范围错误并报告超限的统计量。
5. WHERE `coefficient_policy="scale"`，THE Solver_Adapter SHALL 以统一正缩放后的矩阵求解并记录缩放因子，同时按未缩放的原始 F_β 复算候选能量。
6. THE EP_Trainer SHALL 在每个 epoch 日志中记录重复候选比例（`1 − n_unique / n_candidates` 的均值）与 `k_eff < topk` 的问题数。
7. IF SDK 调用抛出异常、返回空结果、返回宽度与变量数不符，或返回值不属于 {−1, +1} 或 {0, 1}，THEN THE Solver_Adapter SHALL 抛出求解错误，错误信息包含后端、相位、样本索引与原因类别。
8. WHERE 配置了 `max_retries = N`，WHEN SA 或 CIM 求解失败，THE Solver_Adapter SHALL 以相同后端与相同配置重试至多 N 次，并在元数据中记录重试次数与每次失败的原因类别。
9. IF 任一参数的 EP_Gradient 含 NaN 或 Inf，THEN THE EP_Trainer SHALL 抛出数值错误且不调用 `optimizer.step()`，使参数保持更新前的值。

### Requirement 16: 性能计时与计数

**User Story:** 作为研究者，我希望分别记录矩阵构造、求解、解码/梯度与总 epoch 时间以及变量数和求解调用次数，以便比较后端、位宽与预算的资源开销。

#### Acceptance Criteria

1. THE EP_Trainer SHALL 默认为批内每个样本的每个相位独立构造并求解一个问题。
2. THE EP_Trainer SHALL 对每个 epoch 分别记录矩阵构造时间、求解时间、解码与梯度时间和总 epoch 时间（单位秒，基于 `time.perf_counter`）。
3. THE EP_Trainer SHALL 对每个 epoch 记录 `solver_calls`，以及每个问题的 Logical_Bit 数、Ising 变量数与 Auxiliary_Spin 数。
4. THE EP_Trainer SHALL 使 `solver_calls` 等于实际执行的后端求解次数；对无重试的 SA_Backend 满足 `solver_calls = 样本数 × 相位数 × 实际重复次数`。
5. THE Experiment_Runner SHALL 将上述计时与计数写入逐 epoch 的指标 CSV 与 JSON。

### Requirement 17: 测试套件（五类测试与全枚举等价）

**User Story:** 作为维护者，我希望有覆盖申请书五类测试与全枚举等价的 pytest 套件，确定性断言不依赖随机后端与授权，以便上游 CI 稳定运行且 Kaiwu 相关验证可按条件执行。

#### Acceptance Criteria

1. THE Test_Suite SHALL 以 `tests/test_ep_*.py` 组织，并使全部 Level_1_Test 在无 Kaiwu 授权、无网络的环境中通过，单进程 CPU 总耗时不超过 120 秒。
2. THE Test_Suite SHALL 在能量、QUBO 等价与训练测试中覆盖 `[2, 3, 1]` 与至少一个含 2 个隐藏层的结构。
3. THE Test_Suite SHALL 包含"两相位能量收敛"测试：固定 Micro_Network 与初始状态，在 β = 0 与 β > 0 下断言 Continuous_Backend 的 `energy_trace` 在 ε_mono 内非增，且以 `stop_reason="converged"` 结束。
4. THE Test_Suite SHALL 在"两相位能量收敛"测试中对随机后端仅断言 Budget 已记录与 Best_So_Far 非增（SA 允许上坡，CIM 不保证单调）。
5. THE Test_Suite SHALL 包含"β = 0 一致性"测试：在相同矩阵、初始状态、配置与 Paired_Seed 下断言两相位 Coefficient_Table 逐元素相同、Exhaustive_Backend 两相位状态逐元素相同、Continuous_Backend 在 `shared` 策略且步数上限相同时状态逐元素相同、在 `warm_start` 策略下最大绝对差不超过 `state_tol`。
6. THE Test_Suite SHALL 包含"梯度方向"测试：在平衡点位于 (0,1) 内部的固定 2—3—1 连续网络上以 β = 1e-3 计算 EP_Gradient，与 `C(s*_free(θ), y)` 的中心有限差分或自动微分参考梯度比较，断言余弦相似度 ≥ 0.99 且相对误差 ≤ 0.05，并输出相对误差。
7. THE Test_Suite SHALL 包含"参数更新"测试：执行一次 `train_batch` 后断言至少一个参数发生非零变化、全部参数与梯度有限（无 NaN/Inf），且 `optimizer.step()` 恰被调用一次。
8. THE Test_Suite SHALL 包含"保存加载"测试：固定输入、后端与种子，断言保存前后 `state_dict` 逐元素相同、能量绝对误差 ≤ 1e-12、预测类别相同。
9. THE Test_Suite SHALL 包含"三方能量等价"全枚举测试：对 K = 1 与 K = 2、β = 0 与 β > 0 的 Micro_Network 枚举全部 Logical_Bit 赋值，断言 F_β、QUBO 与 Ising_Objective 两两仅差常数（残差 ≤ 1e-8），并断言输入项消元、Auxiliary_Spin 两种规范的解码与三方 argmin 集合一致。
10. IF Kaiwu 转换 API 在当前环境不可调用，THEN THE Test_Suite SHALL 仅显式跳过三方等价测试中的 Ising 部分，并继续执行 F_β↔QUBO 部分。
11. THE Test_Suite SHALL 在 Level_1_Test 中使用受控的 Kaiwu 替身（fake SDK）验证 SA_Backend 与 CIM_Backend 的参数传递、Paired_Seed、`init_solution`、候选规范化、解码与错误映射。
12. THE Test_Suite SHALL 用已注册的 pytest 标记标识 Level_2_Test，并在 Kaiwu 未安装、授权不可用或 CIM 不可用时以写明具体原因的 skip 信息跳过。
13. THE Test_Suite SHALL 为 EPConfig 往返、K_Bit_Encoding 往返、聚合权重归一与 Best_So_Far 非增提供以固定种子随机生成至少 100 组输入的性质测试。

### Requirement 18: Digits 分类示例与 BP 对照

**User Story:** 作为评审者，我希望用一条命令在 sklearn Digits 上运行 EP 与同规模 BP 的对照，并有快速 smoke 配置，以便复现申请书中的对比实验。

#### Acceptance Criteria

1. THE Digits_Example SHALL 使用 `sklearn.datasets.load_digits` 加载 Digits_Benchmark，将像素值除以 16 归一化到 [0,1]，按固定种子分层划分训练集与测试集，并保存划分索引文件。
2. WHEN 指定的划分索引文件已存在，THE Digits_Example SHALL 从该文件读取训练与测试索引。
3. THE Digits_Example SHALL 使 EP 与 BP_Baseline 均采用 64—32—10 结构，并共享预处理、划分索引、初始化规则、种子与评估时点（每个 epoch 结束）。
4. THE BP_Baseline SHALL 在实验开始前固定激活函数、损失函数、优化器与学习率，将其写入配置产物，并在同一实验矩阵的全部种子中保持相同。
5. WHEN 用户执行 Documentation 中的单条命令，THE Digits_Example SHALL 依次运行离散 EP（SA_Backend）、连续 EP 与 BP_Baseline，并写出对照结果。
6. WHEN 用户以 Smoke_Config 执行同一命令，THE Digits_Example SHALL 在 Smoke_Config 的样本、epoch 与种子限制内完成全流程，并写出与完整实验同结构的产物。
7. IF Kaiwu 授权不可用，THEN THE Digits_Example SHALL 完成连续 EP 与 BP_Baseline，并把 SA 与 CIM 条目记录为"未运行"及具体原因。
8. THE Digits_Example SHALL 同时报告等 epoch 对照与等训练时间（或等计算预算）对照，并记录每个 epoch 的累计训练时间。
9. WHERE CIM_Backend 可用且已配置，THE Digits_Example SHALL 将离散 EP（CIM_Backend）加入主对照并标注实际重复次数。

### Requirement 19: 实验矩阵、消融与结果产物

**User Story:** 作为研究者，我希望实验矩阵覆盖主对照、β、平衡预算、量化位宽与解聚合，每项至少 3 个固定种子并保存完整产物，以便结论可追溯且不含预填或虚构数值。

#### Acceptance Criteria

1. THE Experiment_Runner SHALL 支持主对照（离散 EP + SA、连续 EP、BP-MLP），并报告训练准确率、测试准确率、分类误差与每个 epoch 耗时。
2. THE Experiment_Runner SHALL 支持 β ∈ {0.05, 0.1, 0.2, 0.5, 1.0} 扫描，并报告梯度方向（相对 Continuous_Backend 小 β 参考梯度的余弦相似度）、两相位状态差、稳定性（零更新比例与非有限梯度次数）与测试准确率。
3. WHEN 预实验缩减 β 取值，THE Experiment_Runner SHALL 在配置产物与报告中记录实际使用的取值及缩减原因。
4. THE Experiment_Runner SHALL 支持低、中、高三档平衡预算，报告准确率—时间关系、相对同一问题已知最低能量的能量差与收敛比例，并在报告中写明能量差与收敛比例的计算定义。
5. THE Experiment_Runner SHALL 支持 K = 1 与 K = 2 的量化位宽对比，并报告量化误差、Logical_Bit 数、耗时与准确率。
6. WHERE 实验配置启用 K = 3，THE Experiment_Runner SHALL 以与 K = 1、K = 2 相同的指标报告 K = 3 结果。
7. IF K = 3 因资源限制未启用，THEN THE Experiment_Runner SHALL 在报告中将 K = 3 标注为"未运行"并写明原因。
8. THE Experiment_Runner SHALL 支持 `best`、`topk_mean` 与 `energy_weighted` 聚合对比，并报告梯度方差与分类性能。
9. THE Experiment_Runner SHALL 对每个实验条目运行至少 3 个写入配置的固定种子，并报告均值、标准差、最小值与最大值。
10. IF 某条目因配额或授权限制实际完成的种子数或重复次数少于计划值，THEN THE Experiment_Runner SHALL 在结果表中标注实际次数与限制原因。
11. THE Experiment_Runner SHALL 仅从本次运行生成的指标文件计算并渲染报告数值，并对未运行条目显示"未运行"及原因。
12. THE Experiment_Runner SHALL 为每次运行保存配置、数据划分索引、逐 epoch 指标 CSV 与 JSON、模型检查点、日志、图表，以及由指标文件生成图表的绘图脚本。
13. IF 指定的输出目录已存在且非空，THEN THE Experiment_Runner SHALL 拒绝运行并提示使用新目录。
14. IF 正式实验（非 Smoke_Config）启动时已安装的 Kaiwu 版本不是 `1.4.1`，THEN THE Experiment_Runner SHALL 拒绝运行并记录期望版本与实际版本。

### Requirement 20: 可复现性记录

**User Story:** 作为评审者，我希望每次运行记录随机源、环境与版本信息，以便在同一环境中重复结果并理解跨平台差异。

#### Acceptance Criteria

1. WHEN 运行开始，THE Experiment_Runner SHALL 以配置种子设置 Python `random`、NumPy 与 PyTorch 随机源。
2. THE Experiment_Runner SHALL 由根种子确定性派生数据划分、模型初始化、初始状态与 SA `rand_seed`，并在 Run_Manifest 中记录根种子与派生规则。
3. THE Run_Manifest SHALL 记录 Git commit 哈希及工作区是否含未提交修改、Python/PyTorch/NumPy/scikit-learn 版本、`importlib.metadata.version("kaiwu")` 返回的 Kaiwu 版本、操作系统、CPU 与设备信息，以及 `torch.get_num_threads()` 线程数。
4. WHEN 在同一环境、相同线程数下以相同配置与种子重复运行 Continuous_Backend、Exhaustive_Backend 或 SA_Backend 实验，THE Experiment_Runner SHALL 产生逐元素相同的参数与指标（计时字段除外）。
5. THE Documentation SHALL 区分"同环境重复（逐元素一致）"与"跨平台数值近似一致（给出容差）"，并声明不承诺跨依赖版本逐位复现。

### Requirement 21: README 与 API 文档

**User Story:** 作为新用户，我希望文档说明算法、API、环境、随机种子、完整命令、后端限制与保存加载，以便不读源码即可安装、运行与复现。

#### Acceptance Criteria

1. THE Documentation SHALL 说明 E_θ、C、F_β、两相位流程、EP_Gradient 及其写入最小化优化器的符号约定，以及 K_Bit_Encoding、QUBO/Ising 映射、Constant_Offset 与 SDK_Matrix_Convention。
2. THE Documentation SHALL 给出 `EPConfig`、`EquilibriumPropagation`、`equilibrate`、`train_batch`、`predict`、`EPTrainer` 与 `SolverAdapter` 的用法示例，且示例可使用 Continuous_Backend 或 Exhaustive_Backend 在 Level_1_Test 环境中运行。
3. THE Documentation SHALL 说明 `kaiwu==1.4.1` 的安装命令与支持的 Python 版本（3.10，与 Kaiwu_SDK 1.4.1 仅提供 cp310 wheel 一致）、以 `importlib.metadata.version("kaiwu")` 检查版本的方法（并注明 `kw.__version__` 返回 kaiwu-community 版本），以及通过 `KAIWU_USER_ID`、`KAIWU_SDK_CODE` 配置授权的方式。
4. THE Documentation SHALL 给出 Digits 完整实验、Smoke_Config、Level_1_Test 与 Level_2_Test 的完整命令。
5. THE Documentation SHALL 说明后端限制：SA 允许上坡、CIM 不保证单调且不提供内部轨迹、Best_So_Far 仅为记录定义、CIM 8 位精度与配额限制、`warm_start` 与 CIM 的兼容性，以及 `exhaustive_max_bits` 上限。
6. THE Documentation SHALL 说明随机种子派生规则、Initial_State_Policy、检查点格式与可复现性边界。
7. THE EP_Module SHALL 为全部公共类与函数提供 docstring，写明参数、返回值、张量形状、dtype 与可能抛出的错误。

### Requirement 22: 上游集成、依赖与 PR 规范

**User Story:** 作为上游维护者，我希望 EP 模块遵循仓库既有结构、许可与代码风格，并附带依赖升级回归结果，以便 PR 可以评审与合并。

#### Acceptance Criteria

1. THE Upstream_PR SHALL 将源码放在 `src/kaiwu/torch_plugin/` 下的 `equilibrium_propagation.py`、`_ep_dynamics.py`、`_ep_qubo.py`、`_ep_solver.py` 与 `_ep_training.py`，示例放在 `example/<示例名>/`，测试放在 `tests/test_ep_*.py`。
2. THE Upstream_PR SHALL 在 `src/kaiwu/torch_plugin/__init__.py` 中导出 `EquilibriumPropagation`、`EPConfig`、`EPTrainer` 与 `SolverAdapter`，并保留全部既有导出。
3. THE Upstream_PR SHALL 在每个新增源文件头部包含 `SPDX-License-Identifier: Apache-2.0`。
4. THE Upstream_PR SHALL 使新增 Python 代码符合 PEP 8，并通过仓库既有的 lint 与格式检查。
5. THE Upstream_PR SHALL 将 `requirements/requirements.txt` 中的 Kaiwu 依赖改为 `kaiwu==1.4.1`，并在示例级依赖文件中以精确版本声明仅示例使用的依赖（如 `scikit-learn`、`matplotlib`），其中与 Kaiwu_SDK 依赖重叠的包须与 Kaiwu_SDK 的固定版本一致（`numpy==2.2.6`、`matplotlib==3.10.3`）。
6. WHEN 依赖升级为 `kaiwu==1.4.1`，THE Upstream_PR SHALL 附带回归记录，逐项列出上游既有测试（如 `test_bm.py`、`test_rbm.py`、`test_qvae.py`）与示例在 1.4.1 下的通过、失败或跳过结果。
7. IF 回归发现既有模块、示例或 README 在 1.4.1 下失效（如 `from kaiwu.cim import CIMOptimizer, PrecisionReducer`），THEN THE Upstream_PR SHALL 在描述中列出失效项，并按导师或维护者确认的方式处理（本 PR 修复、单独小 PR，或提交 issue 交由维护者处理）。
8. THE Upstream_PR SHALL 在描述中包含变更摘要、已运行的测试及结果、未完成或受阻的功能（如 CIM 未验证项），且标题不超过 70 个字符。
9. THE Upstream_PR SHALL 仅提交源码、测试、示例、绘图脚本、文档与经维护者同意的汇总结果文件；完整实验产物、数据集文件、`.env`、授权文件与凭据保留在提交之外。
10. THE Upstream_PR SHALL 仅在维护者要求时修改 `__version__`（当前为 `0.3.0`）。

### Requirement 23: 凭据与授权安全

**User Story:** 作为项目参与者，我希望 Kaiwu 凭据只从环境变量读取且不出现在任何产物中，以便公开仓库、日志与报告不泄露授权信息。

#### Acceptance Criteria

1. THE License_Initializer SHALL 仅从环境变量 `KAIWU_USER_ID` 与 `KAIWU_SDK_CODE`（可由本地未提交的 `.env` 加载）读取凭据，并调用 `kw.license.init(user_id=..., sdk_code=...)`。
2. THE License_Initializer SHALL 仅在示例入口或 Level_2_Test 夹具显式调用时执行授权初始化。
3. IF 凭据缺失或授权初始化失败，THEN THE License_Initializer SHALL 抛出授权错误，错误信息仅包含缺失的变量名或 SDK 异常类型名。
4. THE Experiment_Runner SHALL 在日志、指标、Run_Manifest、检查点、图表与报告中仅记录凭据变量是否已设置的布尔值。
5. THE Solver_Adapter SHALL 在求解错误与元数据中仅保留 SDK 异常的类型名与原因类别。
6. THE Upstream_PR SHALL 在 `.gitignore` 中包含 `.env` 与授权文件模式，并提供仅含空值的 `.env.example`。
7. THE Test_Suite SHALL 包含一项 Level_2_Test：在凭据环境变量已设置时运行 Smoke_Config，并断言生成的全部文本产物均不包含凭据值。

### Requirement 24: OSPP 结项材料与流程约束

**User Story:** 作为 OSPP 参与学生，我希望交付物与结项报告满足官方结项要求，并对照申请书说明完成情况与偏差，以便通过导师与组委会考核。

#### Acceptance Criteria

1. THE Upstream_PR SHALL 提交到 `https://github.com/qboson/kaiwu-pytorch-plugin`，并在 OSPP 系统显示的开发周期内、合并阶段开始前创建。
2. THE Upstream_PR SHALL 由公开 GitHub 账号提交，且提交作者邮箱与该账号的公开邮箱均与 OSPP 报名所用 git email 一致。
3. THE Final_Report SHALL 包含四部分：已完成工作（逐项对应申请书目标与交付物）、遇到的问题与解决方案、测试用例与结果、后续工作（应用、发版、维护）。
4. THE Final_Report SHALL 在"与申请书的偏差"小节列出每项计划变更及原因，至少包括 Kaiwu 基线由 1.3.1 改为 1.4.1（附导师确认记录的存放位置）、Initial_State_Policy 的引入，以及任何缩减项（如 K = 3 或 CIM 验证）。
5. THE Final_Report SHALL 将 Preliminary_Validation 标注为"前期可行性验证"，与 Digits 主交付分开陈述，并注明 Preliminary_Validation 的数值来自只读历史产物。
6. THE Final_Report SHALL 对照 M1–M4 里程碑列出实际完成时间与对应 PR 或提交链接。
7. THE Final_Report SHALL 仅引用由 Experiment_Runner 产物得到的实验数值，并为每个数值给出产物路径或链接。
8. THE Final_Report SHALL 在 OSPP 系统显示的开发周期结束之前，与 Upstream_PR 链接一起提交到 OSPP 系统。
9. WHEN 维护者发布包含 EP 模块的版本，THE Final_Report SHALL 记录该版本号。
10. IF 结项时尚未发版，THEN THE Final_Report SHALL 记录未发版状态与后续发版计划。

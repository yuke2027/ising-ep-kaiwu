# Implementation Plan: ep-p1-energy-qubo

## Overview

在 Upstream_Clone（`upstream/kaiwu-pytorch-plugin`）内按依赖方向自下而上实现：`_ep_dynamics.py` → `_ep_qubo.py` → `_ep_solver.py` → `equilibrium_propagation.py` → `__init__.py`。源码写入 `src/kaiwu/torch_plugin/`，测试写入 `tests/test_ep_*.py` 与 `tests/_ep_gen.py`，全部测试在 `upstream/.venv` 中运行。实现语言为 Python（PyTorch，float64）。

采用降级方案：源码按通用 K 实现，测试先让 K = 1 全链路（编码、QUBO、Ising、解码、穷举）通过，再扩到 K = 2；三方等价门禁只要求 K = 1 与 K = 2，K = 3 仅做编码往返与一次 F_β↔QUBO 抽查。

## Tasks

- [ ] 1. 前置核对
  - [ ] 1.1 核对 P0 退出门禁
    - 确认 `records/ep-p0/status-report.md` 标记"P1 可开工"
    - 确认 Upstream_Clone 位于 `upstream/kaiwu-pytorch-plugin`，Upstream_Venv 位于 `upstream/.venv`，且其中 Python 3.10 与 `importlib.metadata.version("kaiwu") == "1.4.1"`
    - 任一条件不满足时停止后续任务并询问用户
    - _Requirements: 10.2, 6.9_
  - [ ] 1.2 逐项核实 design.md"待核实假设"5 项并回写结果
    - 假设 1：检查 `kaiwu/__init__.py` 是否急切导入 SDK 子模块，确定子进程屏蔽方案或 `sys.meta_path` 拦截备选
    - 假设 2：查找上游 pytest 配置（`pyproject.toml`/`pytest.ini`/`setup.cfg`/`tests/conftest.py`），确定 `ep_level2` 标记注册位置，以及 `tests/` 是否为包（决定 `_ep_gen.py` 导入方式）
    - 假设 3：按 `pyproject.toml`、`setup.cfg`、`.pre-commit-config.yaml`、CI 工作流顺序确定 lint/格式工具与行宽
    - 假设 4：确认 `src/kaiwu/torch_plugin/__init__.py` 的导出风格与 `__all__`
    - 假设 5：用真实 `kaiwu.core.qubo_matrix_to_qubo_model`/`qubo_model_to_ising_model` 对小矩阵探测变量名格式与 Auxiliary_Spin 形态
    - 产出：更新 `.kiro/specs/ep-p1-energy-qubo/design.md`"待核实假设"一节（逐项写明"确认/改用备选"）；与默认方案不符且备选也不可行时停止并询问用户
    - _Requirements: 6.4, 6.10, 9.6, 10.3, 10.5_

- [ ] 2. 底层模块 `_ep_dynamics.py`
  - [ ] 2.1 建立错误层级、常量与数据结构
    - 产出：`src/kaiwu/torch_plugin/_ep_dynamics.py`（首行含 `SPDX-License-Identifier: Apache-2.0`）
    - 定义 `EPError` 及七个子类（`EPConfigError` 同时继承 `ValueError`）、`MAX_STATE_BITS = 4`
    - 定义 frozen dataclass `EquilibriumResult`、`Budget`、`SolverMetadata`（P2 字段留 `None` 默认值）
    - _Requirements: 8.1, 4.6, 10.4_
  - [ ] 2.2 实现能量纯函数
    - 产出：`_ep_dynamics.py` 中 `energy_fn`、`cost_fn`、`total_energy_fn`、`state_gradient_fn`
    - `beta == 0` 时 `total_energy_fn` 直接返回 `energy_fn`；状态梯度解析实现，不走 autograd
    - _Requirements: 3.1, 3.2, 3.4_
  - [ ] 2.3 实现 ContinuousDynamics
    - 产出：`_ep_dynamics.py` 中 `ContinuousDynamics`（`__init__`、`from_config`、`run`）
    - 投影到 [0,1]、逐样本步长回溯、ε_mono、`converged`/`max_steps` 判停、`energy_trace (T+1, B)`；回溯耗尽抛 `EPDynamicsError` 并挂部分结果
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_
  - [ ] 2.4 建立测试生成器模块
    - 产出：`tests/_ep_gen.py`（`gen_layer_sizes`、`gen_params`、`gen_state`、`gen_config`、`gen_invalid_config`、`assert_constant_offset`，固定种子 `20260929 + case_offset`，失败信息含种子与序号）
    - _Requirements: 9.1, 9.8_
  - [ ] 2.5 编写两相位能量收敛测试
    - 产出：`tests/test_ep_dynamics.py`
    - 固定 Micro_Network 与初始状态，在 β = 0 与 β > 0 下断言 `energy_trace` 在 ε_mono 内非增且 `stop_reason="converged"`；另含 `max_steps` 示例与 `max_backtracks=0` + 过大步长触发 `energy_increase` 的示例
    - _Requirements: 9.3, 4.4, 4.5, 4.8_
  - [ ]* 2.6 编写错误层级性质测试
    - **Property 21: 错误层级**
    - **Validates: Requirements 8.1**
    - 产出：`tests/test_ep_config.py`
  - [ ]* 2.7 编写连续后端投影与单调性质测试
    - **Property 7: 连续后端投影与单调**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.8, 9.3**
    - 产出：`tests/test_ep_dynamics.py`
  - [ ]* 2.8 编写连续后端确定性性质测试
    - **Property 8: 连续后端确定性**
    - **Validates: Requirements 4.7**
    - 产出：`tests/test_ep_dynamics.py`

- [ ] 3. 检查点：确认本阶段测试全部通过，有问题询问用户

- [ ] 4. QUBO 构造 `_ep_qubo.py`
  - [ ] 4.1 实现 BitLayout
    - 产出：`src/kaiwu/torch_plugin/_ep_qubo.py`（SPDX 头）中 `BitLayout`
    - 索引 `g(l, i, k) = K·(offset_l + i) + k`；`bit_weights`、`quad_index`（神经元内对 + 相邻层块，i < j 字典序）、`index`、`encode`（非 State_Grid 值抛 `EPConfigError`）、`decode`
    - _Requirements: 5.1, 5.2, 5.3_
  - [ ] 4.2 编写 K 位编码往返性质测试
    - **Property 9: K 位编码往返**
    - **Validates: Requirements 5.1, 5.2, 9.8**
    - 产出：`tests/test_ep_qubo.py`；K ∈ [1, MAX_STATE_BITS]，固定种子至少 100 组输入
  - [ ]* 4.3 编写 Logical_Bit 计数性质测试
    - **Property 10: Logical_Bit 计数**
    - **Validates: Requirements 5.3**
    - 产出：`tests/test_ep_qubo.py`；含 `[64, 32, 10]` 在 K = 1/2/3 时 42/84/126 的示例
  - [ ] 4.4 实现 EPQuboBuilder 与数据结构
    - 产出：`_ep_qubo.py` 中 `CoefficientTable`、`CoefficientStats`、`EPQuboBuilder`（`from_config`、`layout` 实例级缓存与 `template_builds`、`build`、`to_upper_triangular`、`evaluate`）
    - 按设计的系数推导表实现输入消元、Kronecker 层间块、`Constant_Offset = ½β‖y‖²`（β = 0 不触碰 y）、合并统计、`coefficient_policy="error"` 超限抛 `EPCoefficientRangeError`
    - _Requirements: 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 6.1, 6.2_
  - [ ] 4.5 编写 F_β↔QUBO 全枚举等价测试（K = 1）
    - 产出：`tests/test_ep_qubo.py`
    - 结构 `[3, 2]`、`[2, 2, 2]`、`[2, 3, 1]`、`[2, 2, 2, 1]`，β ∈ {0, 0.5}；枚举全部 q，断言 `F_β(decode(q)) − qᵀQq` 为常数且等于 `Constant_Offset`（≤ 1e-9），断言无输入层变量、Q 下三角为 0、F_β 与 QUBO 的 argmin 集合一致
    - 该部分不依赖 Kaiwu，Ising 不可用时照常执行
    - _Requirements: 9.4, 9.5, 9.2, 5.4, 5.6, 6.2_
  - [ ]* 4.6 编写 F_β 与 Coefficient_Table 等价性质测试
    - **Property 11: F_β 与 Coefficient_Table 等价**
    - **Validates: Requirements 5.4, 5.5, 5.6**
    - 产出：`tests/test_ep_qubo.py`
  - [ ]* 4.7 编写上三角 qᵀQq 等价性质测试
    - **Property 12: 上三角 qᵀQq 等价**
    - **Validates: Requirements 6.1, 6.2**
    - 产出：`tests/test_ep_qubo.py`
  - [ ]* 4.8 编写 β = 0 与 y 无关性质测试
    - **Property 13: β = 0 与 y 无关及两相位相同**
    - **Validates: Requirements 5.7, 5.10**
    - 产出：`tests/test_ep_qubo.py`
  - [ ]* 4.9 编写模板复用性质测试
    - **Property 14: 模板复用**
    - **Validates: Requirements 5.9**
    - 产出：`tests/test_ep_qubo.py`
  - [ ]* 4.10 编写系数统计与超限报错性质测试
    - **Property 15: 系数统计与超限报错**
    - **Validates: Requirements 5.8, 5.11**
    - 产出：`tests/test_ep_qubo.py`

- [ ] 5. 最小求解适配 `_ep_solver.py`
  - [ ] 5.1 实现 ExhaustiveBackend
    - 产出：`src/kaiwu/torch_plugin/_ep_solver.py`（SPDX 头，模块顶层不导入 kaiwu）中 `ExhaustiveBackend`
    - 超过 `max_bits` 时在分配内存前抛 `EPSolverError`（含位数与上限）；分块位移枚举，按原始 F_β 复算，`tie_tol=1e-9` 并列按索引升序
    - _Requirements: 7.1, 7.2_
  - [ ] 5.2 实现 SolverAdapter 最小版本
    - 产出：`_ep_solver.py` 中 `KaiwuIsingProblem`、`_kaiwu_attr`（白名单 6 个 API）、默认提供者（`importlib.import_module` + 缓存）、`SolverAdapter`（`from_config`、`build_kaiwu_problem`、`ising_energy`、`decode_spins`、`solve_exhaustive`、`kaiwu_version`）
    - 变量映射按 1.2 核实的变量名规则；辅助自旋至多 1 个，不符抛 `EPSolverError`（含期望与实际变量数）；解码前按 `σ·σ_aux` 规范化再调 `get_sol_dict`，返回值须在 {0,1}
    - _Requirements: 6.3, 6.4, 6.6, 6.7, 6.8, 6.9, 6.10_
  - [ ] 5.3 建立 fake Kaiwu 提供者工厂
    - 产出：`tests/_ep_gen.py` 中 fake `core` 模块（`FakeQuboModel`、`FakeIsingModel`、`get_sol_dict`）与记录属性访问的包装器，可配置变量表以构造数量不符、重复名、多个辅助自旋等异常场景
    - _Requirements: 6.4, 6.8, 6.10_
  - [ ] 5.4 注册 Level_2 标记与跳过工具
    - 产出：按 1.2 核实的位置注册 `ep_level2`（`pyproject.toml` 的 `markers` 或 `tests/conftest.py` 的 `pytest_configure`）；`tests/_ep_gen.py` 中 `require_level2(need_license=..., need_cim=...)`；`tests/test_ep_solver.py` 中一个带标记的占位用例验证 skip 信息写明缺失项
    - _Requirements: 9.6, 9.7_
  - [ ] 5.5 编写三方等价 Ising 部分测试（K = 1）
    - **Property 16: Ising 等价**
    - **Validates: Requirements 6.3, 6.5, 9.4, 9.5**
    - 产出：`tests/test_ep_kaiwu_conversion.py`；开头 `pytest.importorskip("kaiwu.core")` 并以最小矩阵探测转换 API，不可调用时仅跳过本文件并写明异常类型；4 个结构 × β ∈ {0, 0.5} × 3 个种子，全枚举 σ 断言 `H(σ) + Constant_Offset = F_β`（≤ 1e-8）；含 `kaiwu_version()` 示例
  - [ ] 5.6 编写三方 argmin 一致测试（K = 1）
    - **Property 18: 三方 argmin 一致**
    - **Validates: Requirements 7.1, 9.4**
    - 产出：`tests/test_ep_kaiwu_conversion.py`；断言 σ 与 −σ 经 `decode_spins` 得到相同位与 Layer_State，F_β、qᵀQq、Ising 三方最小值集合在 Layer_State 上相同且等于 `ExhaustiveBackend` 的 `candidates`
  - [ ]* 5.7 编写变量映射与 ±σ 解码性质测试（fake 部分）
    - **Property 17: 变量映射与 ±σ 解码一致**
    - **Validates: Requirements 6.4, 6.6, 6.7, 6.8**
    - 产出：`tests/test_ep_solver.py`
  - [ ]* 5.8 编写穷举完整性与上限性质测试
    - **Property 19: 穷举完整性与上限**
    - **Validates: Requirements 7.1, 7.2**
    - 产出：`tests/test_ep_solver.py`

- [ ] 6. 检查点：确认 K = 1 全链路（编码、QUBO、Ising、解码、穷举）测试全部通过，有问题询问用户

- [ ] 7. 扩展到 K = 2
  - [ ] 7.1 F_β↔QUBO 全枚举测试扩到 K = 2
    - 产出：`tests/test_ep_qubo.py`；4.5 的测试参数化加入 K = 2（最多 10 位），另加 `[2, 3, 1]` 在 K = 3 的一次抽查
    - _Requirements: 9.4, 9.2_
  - [ ] 7.2 三方等价 Ising 测试扩到 K = 2
    - 产出：`tests/test_ep_kaiwu_conversion.py`；5.5、5.6 参数化加入 K = 2（合计 48 次转换）
    - _Requirements: 9.4, 9.5, 9.2_

- [ ] 8. 顶层模块 `equilibrium_propagation.py`
  - [ ] 8.1 实现 EPConfig
    - 产出：`src/kaiwu/torch_plugin/equilibrium_propagation.py`（SPDX 头）中 frozen dataclass `EPConfig`
    - 按设计默认值表定义全部字段；`__post_init__` 校验（信息含字段名、数值、允许取值）；`layer_sizes` 规范化为 tuple；`to_dict`/`from_dict`（一次列出缺失键与未知键）
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12_
  - [ ] 8.2 编写 EPConfig 往返性质测试
    - **Property 1: EPConfig 往返**
    - **Validates: Requirements 1.10, 1.11, 9.8**
    - 产出：`tests/test_ep_config.py`；固定种子至少 100 组，含 `json.dumps`/`json.loads` 往返
  - [ ]* 8.3 编写 EPConfig 非法输入性质测试
    - **Property 2: EPConfig 非法输入被拒**
    - **Validates: Requirements 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.12**
    - 产出：`tests/test_ep_config.py`
  - [ ] 8.4 实现 EquilibriumPropagation 并再导出错误类型
    - 产出：`equilibrium_propagation.py` 中 `EquilibriumPropagation`（`register_parameter` 注册 `W_0..W_{L−1}`、`b_1..b_L`；私有 `torch.Generator` 按 seed 初始化 `U(±1/√n_l)`、偏置 0；`weights`/`biases` 属性；`prepare_batch`；`energy`/`cost`/`total_energy` 委托纯函数）；再导出八个错误类
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 8.1_
  - [ ]* 8.5 编写参数结构与确定性性质测试
    - **Property 3: 参数结构与确定性**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.5**
    - 产出：`tests/test_ep_model.py`
  - [ ]* 8.6 编写输入规范化与校验性质测试
    - **Property 4: 输入规范化与校验**
    - **Validates: Requirements 2.6, 2.7, 3.5, 3.6**
    - 产出：`tests/test_ep_model.py`
  - [ ]* 8.7 编写能量公式性质测试
    - **Property 5: 能量公式与 F_0 = E_θ**
    - **Validates: Requirements 3.1, 3.2, 3.4**
    - 产出：`tests/test_ep_model.py`；覆盖 `[2, 3, 1]` 与 `[2, 2, 2, 1]`
  - [ ]* 8.8 编写解析参数梯度性质测试
    - **Property 6: 解析参数梯度**
    - **Validates: Requirements 3.3**
    - 产出：`tests/test_ep_model.py`
  - [ ]* 8.9 编写 PyTorch 兼容单元测试
    - 产出：`tests/test_ep_model.py`；float64 默认 dtype、`DataLoader` 批输入、`torch.optim.SGD`/`Adam` 单步更新
    - _Requirements: 2.4, 2.6, 2.8_
  - [ ]* 8.10 编写 EPConfig 默认值单元测试
    - 产出：`tests/test_ep_config.py`；需求 1.2 的构造示例与 1.3 的默认值
    - _Requirements: 1.2, 1.3_

- [ ] 9. 导出与导入隔离
  - [ ] 9.1 更新 `__init__.py` 导出
    - 产出：`src/kaiwu/torch_plugin/__init__.py`；按 1.2 核实的风格追加 `EPConfig`、`EquilibriumPropagation`（并入 `__all__`），保留全部既有导出与顺序
    - _Requirements: 10.3_
  - [ ] 9.2 编写 SPDX 头静态检查
    - 产出：`tests/test_ep_import_isolation.py`；断言四个新源文件前 3 行含 `SPDX-License-Identifier: Apache-2.0`
    - _Requirements: 10.4, 10.2_
  - [ ]* 9.3 编写延迟导入可用性性质测试
    - **Property 20: 延迟导入可用性**
    - **Validates: Requirements 6.10**
    - 产出：`tests/test_ep_import_isolation.py`；子进程屏蔽 `kaiwu.core/classical/cim/preprocess`（或 1.2 确认的备选）后跑最小流程，`ast` 扫描顶层导入，fake 提供者访问记录全部在白名单内

- [ ] 10. 检查点：确认本阶段测试全部通过，有问题询问用户

- [ ] 11. P1 退出门禁
  - [ ] 11.1 补全公共 API docstring
    - 产出：四个新源文件；全部公共类与函数写明参数、返回值、张量形状、dtype 与可能抛出的错误，EPConfig 与 EPQuboBuilder 的 docstring 写明默认值（含 `coefficient_limit`、`dynamic_range_limit`、`coefficient_policy`）、初始化规则与 `state_dict` 键名
    - _Requirements: 10.1, 2.4, 2.5_
  - [ ] 11.2 运行 lint 与格式检查并修复
    - 产出：按 1.2 确定的上游工具对四个源文件、`tests/test_ep_*.py`、`tests/_ep_gen.py` 检查并修复至通过
    - _Requirements: 10.5_
  - [ ] 11.3 运行 Level_1 全量测试并核对门禁
    - 在 `upstream/.venv` 中单进程运行 `pytest tests/test_ep_*.py --durations=10`：全部通过且总耗时 ≤ 120 秒（超出时先缩减种子数，不缩减结构覆盖）
    - 核对能量测试与 QUBO 等价测试覆盖 `[2, 3, 1]` 与 `[2, 2, 2, 1]`；核对 `from kaiwu.torch_plugin import EPConfig, EquilibriumPropagation` 可用且上游既有测试相对 P0 记录无新增失败
    - 任一项未达标时停止并询问用户
    - _Requirements: 9.1, 9.2, 10.2, 10.3_

## Notes

- 带 `*` 的子任务为可选，可跳过以加快 MVP；需求 9.3、9.4、9.5、9.6/9.7、9.8 对应的测试（2.5、4.2、4.5、5.4、5.5、5.6、7.1、7.2、8.2）属于退出门禁，不可跳过
- 所有代码与测试位于 `upstream/kaiwu-pytorch-plugin`，测试在 `upstream/.venv` 中运行；Migration_Source 只作参考，不被导入
- 三方等价门禁只要求 K = 1 与 K = 2；K = 3 仅编码往返与一次抽查
- P1 结束时的能量/QUBO PR（推送、建 PR）不在本任务列表中，由用户确认后另行处理

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["2.1"] },
    { "id": 3, "tasks": ["2.2", "2.4", "2.6"] },
    { "id": 4, "tasks": ["2.3"] },
    { "id": 5, "tasks": ["2.5", "4.1"] },
    { "id": 6, "tasks": ["2.7", "4.2"] },
    { "id": 7, "tasks": ["2.8", "4.3", "4.4"] },
    { "id": 8, "tasks": ["4.5", "5.1"] },
    { "id": 9, "tasks": ["4.6", "5.2", "5.3"] },
    { "id": 10, "tasks": ["4.7", "5.4", "5.5"] },
    { "id": 11, "tasks": ["4.8", "5.6", "5.7"] },
    { "id": 12, "tasks": ["4.9", "5.8"] },
    { "id": 13, "tasks": ["4.10", "7.2"] },
    { "id": 14, "tasks": ["7.1", "8.1"] },
    { "id": 15, "tasks": ["8.2", "8.4"] },
    { "id": 16, "tasks": ["8.3", "8.5", "9.1"] },
    { "id": 17, "tasks": ["8.6", "8.10", "9.2"] },
    { "id": 18, "tasks": ["8.7", "9.3"] },
    { "id": 19, "tasks": ["8.8"] },
    { "id": 20, "tasks": ["8.9"] },
    { "id": 21, "tasks": ["11.1"] },
    { "id": 22, "tasks": ["11.2"] },
    { "id": 23, "tasks": ["11.3"] }
  ]
}
```

# Requirements Document

## Introduction

本规格是 OSPP 2026 项目 26b1f0072（"基于 Kaiwu-PyTorch-Plugin 实现均衡传播算法"）五个分期规格（P0–P4）中的 P0：上游仓库接入与 Kaiwu 1.4.1 依赖升级准备。P0 排期为开发周期第 3 周（2026-09-29 至 2026-10-05），其中 2026-10-01 至 2026-10-05 与国庆假期重叠，导师、维护者与 Kaiwu 平台的回复可能延迟；因此依赖外部方的事项在退出门禁中按"记录状态即满足"处理。P0 不编写 EP 功能代码，能量与 QUBO 由后续规格 P1（`ep-p1-energy-qubo`）承担。

### 起点现状（2026-09-28 核对）

- 上游 `qboson/kaiwu-pytorch-plugin` 尚未 fork，也未克隆到本地；工作区根目录没有 `.git`，`upstream/` 目录不存在。
- 上游 main 的依赖声明为 `numpy==2.2.6`、`torch==2.7.0`、`kaiwu==1.3.1`。
- `kaiwu==1.4.1` 在公共 PyPI 仅提供 cp310 wheel，固定依赖 `numpy==2.2.6`、`matplotlib==3.10.3`、`kaiwu-community==1.0.7`；工作区 `vendor/kaiwu-1.4.1-cp310-none-win_amd64.whl` 与之一致。
- 工作区根目录已有 Python 3.10.11 虚拟环境 `.venv`，属于 Preliminary_Validation，不作为上游回归环境。
- 上游 CI 配置以及既有 lint/格式工具尚未核实。
- 导师已书面确认 SDK 基线由 1.3.1 改为 1.4.1，确认记录的存放位置待定。
- CIM 授权与配额状态未知。

### 退出门禁（P1 开工前提）

硬门禁 G1–G9 全部满足后 P1 方可开工；状态门禁 S1–S3 只要求在 P0_Status_Report 中记录当前状态，不要求外部方已完成。推送与创建 PR 不属于门禁。

| 编号 | 门禁内容 | 依据 |
|---|---|---|
| G1 | Fork_Clone 位于 `upstream/kaiwu-pytorch-plugin`，Base_Commit 已记录，Work_Branch 已基于 Base_Commit 创建 | 需求 2 |
| G2 | Upstream_Venv 为 Python 3.10，版本核对全部通过，Environment_Snapshot 已写出 | 需求 3 |
| G3 | Work_Branch 含把 `requirements/requirements.txt` 改为 `kaiwu==1.4.1` 的本地提交 | 需求 4 |
| G4 | Regression_Record 中每个 Regression_Item 均有结果，每个非通过项均有原因类别 | 需求 5 |
| G5 | Breakage_List 已写出，每个条目有建议 Handling_Mode 与 Confirmation_Status（允许"待确认"） | 需求 6 |
| G6 | Work_Branch 的 `.gitignore` 与 `.env.example` 已提交，忽略规则核对与 Credential_Scan 通过 | 需求 7 |
| G7 | Work_Branch 相对 Base_Commit 新增的全部提交，作者邮箱与提交者邮箱均等于 OSPP_Git_Email | 需求 8 |
| G8 | Mentor_Confirmation_Record 已归档，Stable_Reference 已记录 | 需求 11 |
| G9 | CI_Inventory 已写出 | 需求 10 |
| S1 | CIM_Authorization_Status 已记录 | 需求 12 |
| S2 | PR_Strategy 及 Confirmation_Status 已记录 | 需求 9 |
| S3 | GitHub 公开账号与公开邮箱的用户确认结果已记录 | 需求 8 |

### 与总需求的关系

总需求为 `.kiro/specs/kpp-equilibrium-propagation/requirements.md`（Master_Spec），P0 只读引用、不修改。每条需求下以"对应总需求"标注所落实的 Master_Spec 验收标准编号；标注为"准备工作"的需求不直接对应验收标准，只为所列条目提供前置条件。P0 覆盖的 Master_Spec 条目如下：

- 22.5：`requirements/requirements.txt` 改为 `kaiwu==1.4.1`（示例级依赖文件由后续分期处理）。
- 22.6：上游既有测试与示例在 1.4.1 下的逐项回归记录。
- 22.7：失效项清单与处理方式。
- 23.6：`.gitignore` 含 `.env` 与授权文件模式，并提供空值 `.env.example`。
- 24.2：提交作者邮箱与 OSPP 报名 git email 一致。
- 24.4（仅归档部分）：导师确认记录的存放位置确定并可引用；Final_Report 偏差小节的写作不在 P0 范围。

Master_Spec 中 22.6、22.7、23.6、24.2 以 Upstream_PR 为主体；P0 产出这些条目所需的分支内容与记录，Upstream_PR 的创建与合并在后续分期完成。

### 非目标

- 编写 EP 模块代码、测试或示例。
- 提交 CIM 任务或运行 CIM 实验。
- 实现 1.3.1/1.4.x 双版本适配或兼容矩阵（1.3.1 对照运行仅用于区分既有缺陷，见需求 5.12）。
- 撰写 Final_Report。

## Glossary

- **Master_Spec**：总需求文档 `.kiro/specs/kpp-equilibrium-propagation/requirements.md`。本文沿用其中 Kaiwu_SDK、Plugin_Repository、Upstream_PR、Final_Report、Preliminary_Validation、Logical_Bit、Auxiliary_Spin、SA_Backend、CIM_Backend 的定义。
- **Plugin_Repository**：上游仓库 `https://github.com/qboson/kaiwu-pytorch-plugin`（Apache-2.0）。
- **Fork_Repository**：用户以本人公开 GitHub 账号 fork 的 Plugin_Repository 副本。
- **Fork_Clone**：Fork_Repository 在工作区内的本地克隆，路径为 `upstream/kaiwu-pytorch-plugin`。
- **Base_Commit**：克隆当日 Plugin_Repository `main` 分支的提交 SHA，作为 Work_Branch 的起点与回归基准。
- **Work_Branch**：Fork_Clone 中基于 Base_Commit 创建的本地分支，默认名 `deps/kaiwu-1.4.1`，承载依赖升级、忽略规则与失效项修复。
- **Dep_PR**：增量 PR 策略下承载依赖升级及其回归记录的小型 Upstream_PR。
- **Upstream_Venv**：专用于 Fork_Clone 回归的 Python 3.10 虚拟环境，默认路径 `upstream/.venv`，位于 Fork_Clone 之外，与工作区根目录 `.venv` 分离。
- **Kaiwu_SDK**：`kaiwu==1.4.1`，版本以 `importlib.metadata.version("kaiwu")` 为准。
- **Vendor_Wheel**：工作区文件 `vendor/kaiwu-1.4.1-cp310-none-win_amd64.whl`。
- **Setup_Automation**：P0 中由 Kiro 代理或脚本执行的全部自动化步骤，包括命令执行、文件编辑与记录生成。
- **Manual_Operation**：由用户本人执行的操作：在 GitHub 上 fork、设置 git 提交身份、执行任何 Git_Config_Write、提交 CIM 授权申请、归档 Mentor_Confirmation_Record、确认 GitHub 公开邮箱。
- **User_Confirmation**：用户在会话中针对某一具体操作（含目标与内容）给出的明确同意。
- **Git_Config_Write**：执行 `git config` 的写入类子命令、执行 `git remote add/set-url/remove`，或直接编辑任一作用域的 git 配置文件。`git clone` 初始化 `origin`、创建本地分支，以及经 User_Confirmation 的 `git push -u` 写入的跟踪信息属于这些命令的固有效果，不计入 Git_Config_Write。
- **OSPP_Git_Email**：用户在 OSPP 报名时登记的 git email，由用户提供。
- **Credential_Variables**：环境变量 `KAIWU_USER_ID` 与 `KAIWU_SDK_CODE`。
- **Authorization_File_Pattern**：匹配 Kaiwu 授权文件的 `.gitignore` 模式，至少包含前期工作区 `.gitignore` 使用的 `*.lic`，以及 P0 中核实到的其他授权文件名模式。
- **Credential_Scan**：推送前与写出 P0 记录前执行的检查：(a) 已跟踪或已暂存文件中不含 `.env` 与匹配 Authorization_File_Pattern 的文件；(b) Credential_Variables 已设置时，待推送差异与 P0_Record_Dir 全部文本文件中不含凭据值。
- **Sanitized_Message**：去除凭据值，并把绝对路径替换为相对 Fork_Clone 根目录的路径或 `<HOME>` 占位符后的异常信息首行。
- **Dependency_Declaration**：Fork_Clone 中声明 Kaiwu 版本的任一位置，如 `requirements/requirements.txt`、`pyproject.toml`、`setup.py`、`setup.cfg`、示例级 requirements 文件、CI 配置与 README 安装命令。
- **Environment_Snapshot**：P0_Record_Dir 中记录 Upstream_Venv 的 Python 版本、操作系统与架构、pip 版本、`pip freeze` 输出、关键包版本核对结果、Kaiwu_SDK 安装来源与生成时间的 JSON 文件。
- **Regression_Item**：回归对象的最小单位：一个 pytest 测试用例（测试文件收集失败时为该测试文件）或一个示例入口。
- **Regression_Record**：P0_Record_Dir 中逐项记录 Regression_Item 在 Kaiwu_SDK 1.4.1 下结果的文件，含 Markdown 与 JSON 两种格式。
- **Static_Import_Check**：对 `src/`、`tests/`、`example/` 与 README 代码块中每条 `kaiwu` 导入语句，在 Upstream_Venv 中核对模块与名称是否可解析，不执行授权与求解。
- **Breakage_List**：P0_Record_Dir 中登记 1.4.1 下失效项及处理方式的文件，含 Markdown 与 JSON 两种格式。
- **Handling_Mode**：失效项处理方式，取值与 Master_Spec 22.7 一致：`本 PR 修复`（"本 PR"指承载依赖升级的 PR，增量策略下即 Dep_PR）、`单独小 PR`、`提交 issue 交由维护者处理`。
- **Confirmation_Status**：待导师或维护者确认事项的状态：`待确认`、`导师已确认`、`维护者已确认`、`已调整`。
- **CI_Inventory**：P0_Record_Dir 中记录 Base_Commit 的 CI 工作流与 lint/格式工具配置的文件。
- **PR_Strategy**：Upstream_PR 的拆分策略。默认值为增量 PR：Dep_PR 单独提交；P1 结束时提交能量/QUBO PR。
- **Mentor_Confirmation_Record**：导师对"SDK 基线由 `kaiwu==1.3.1` 改为 `kaiwu==1.4.1`"的书面确认。
- **Stable_Reference**：Mentor_Confirmation_Record 的可引用位置：P0_Record_Dir 内的文件路径，或 OSPP 系统、GitHub issue/评论等可供评审访问的 URL。
- **CIM_Authorization_Status**：CIM 授权与配额申请状态：`未申请`、`已申请待回复`、`已获批`、`被拒绝`。
- **P0_Record_Dir**：P0 记录目录，默认路径为工作区根目录下的 `records/ep-p0/`，位于 Fork_Clone 之外。
- **P0_Status_Report**：P0_Record_Dir 中汇总门禁状态、确认事项、交接信息与总需求修订建议的 Markdown 文件。
- **Hard_Gate**：退出门禁中的 G1–G9，满足前 P1 不开工。
- **Status_Gate**：退出门禁中的 S1–S3，记录状态即满足。

## Requirements

### Requirement 1: 操作边界与用户确认

**对应总需求：** 无直接对应验收标准（流程约束，为 24.2 与 23.1 提供执行前提）

**User Story:** 作为 OSPP 参与学生，我希望涉及账号、身份、远端仓库与外部申请的操作由我本人执行或经我明确确认，以便自动化步骤不会以我的名义做出难以撤销的操作。

#### Acceptance Criteria

1. THE Setup_Automation SHALL 将 fork Plugin_Repository、设置 git 提交身份、Git_Config_Write、提交 CIM 授权申请、归档 Mentor_Confirmation_Record 与确认 GitHub 公开邮箱交由用户作为 Manual_Operation 执行。
2. WHEN 某一步骤依赖 Manual_Operation，THE Setup_Automation SHALL 向用户说明该操作的目的、建议命令或页面入口与完成后的核对方法，并在用户报告完成后执行依赖该操作的后续步骤。
3. THE Setup_Automation SHALL 仅以只读命令（如 `git config --get`）访问 git 配置。
4. WHEN 准备执行 `git push`，THE Setup_Automation SHALL 先取得针对该次推送的 User_Confirmation，确认内容包含目标远端、目标分支与待推送提交的 SHA 列表。
5. IF 某次推送未取得 User_Confirmation，THEN THE Setup_Automation SHALL 停止该次推送，并在 P0_Status_Report 中将该推送记录为"待确认"。
6. WHEN 准备创建 Upstream_PR 或删除 Fork_Clone 中的文件，THE Setup_Automation SHALL 先取得针对该操作的 User_Confirmation。

### Requirement 2: fork 与克隆上游仓库

**对应总需求：** 准备工作（为 22.1、24.1 提供仓库前提）

**User Story:** 作为 OSPP 参与学生，我希望上游仓库被 fork 并克隆到工作区内的固定位置，且分支起点可追溯，以便后续分期都在同一基准上开发。

#### Acceptance Criteria

1. WHEN 用户报告已完成 fork 并提供 Fork_Repository 的 URL，THE Setup_Automation SHALL 将 Fork_Repository 克隆到 `upstream/kaiwu-pytorch-plugin`。
2. IF `upstream/kaiwu-pytorch-plugin` 已存在且非空，THEN THE Setup_Automation SHALL 停止克隆、保留现有内容，并向用户报告该路径的现状。
3. THE Setup_Automation SHALL 使用不含凭据或访问令牌的 URL 执行克隆与拉取。
4. WHEN 克隆完成，THE Setup_Automation SHALL 以 `git fetch https://github.com/qboson/kaiwu-pytorch-plugin.git main`（不新增远端）获取 Plugin_Repository `main` 的最新提交，并将该提交 SHA 记录为 Base_Commit。
5. IF Fork_Clone 中 `main` 的 HEAD 与 Base_Commit 不同，THEN THE Setup_Automation SHALL 在 P0_Status_Report 中记录两个 SHA。
6. WHEN Base_Commit 已获取，THE Setup_Automation SHALL 基于 Base_Commit 在 Fork_Clone 中创建本地 Work_Branch。
7. THE Setup_Automation SHALL 在 P0_Status_Report 中记录 Fork_Repository URL、Fork_Clone 路径、Base_Commit、Work_Branch 名称与克隆日期。
8. WHEN 克隆完成，THE Setup_Automation SHALL 核对 Base_Commit 中 `src/kaiwu/torch_plugin/`、`src/kaiwu/torch_plugin/__init__.py` 的 `__version__`、`example/`、`tests/test_*.py`、`requirements/requirements.txt` 与 Apache-2.0 许可文件是否与 Master_Spec 的描述一致，并记录核对结果。
9. IF 克隆或拉取因网络或访问权限失败，THEN THE Setup_Automation SHALL 记录失败的命令与错误类别，并暂停依赖 Fork_Clone 的步骤，直至用户报告问题已处理。

### Requirement 3: Python 3.10 虚拟环境与 Kaiwu 1.4.1

**对应总需求：** 准备工作（为 22.5、22.6 的回归运行与 7.7–7.9 的 API 核实提供环境）

**User Story:** 作为算法开发者，我希望有一个版本已核对的 Python 3.10 + Kaiwu 1.4.1 专用环境，以便回归结果与后续开发可复现。

#### Acceptance Criteria

1. THE Setup_Automation SHALL 以 Python 3.10 解释器在 `upstream/.venv` 创建 Upstream_Venv。
2. IF 本机未找到 Python 3.10 解释器，THEN THE Setup_Automation SHALL 停止创建 Upstream_Venv，并报告原因为 Kaiwu_SDK 1.4.1 仅提供 cp310 wheel。
3. WHEN 安装 Kaiwu_SDK，THE Setup_Automation SHALL 在 Upstream_Venv 中从公共 PyPI 安装 `kaiwu==1.4.1`。
4. IF 从公共 PyPI 安装 `kaiwu==1.4.1` 失败，THEN THE Setup_Automation SHALL 改用 Vendor_Wheel 安装，并在 Environment_Snapshot 中记录安装来源与 Vendor_Wheel 的 SHA-256。
5. WHEN Kaiwu_SDK 安装完成，THE Setup_Automation SHALL 在 Upstream_Venv 中按 Work_Branch 升级后的依赖声明与上游声明的测试依赖安装其余依赖。
6. IF 依赖解析出现版本冲突（如上游声明的 `matplotlib` 版本不同于 `3.10.3`），THEN THE Setup_Automation SHALL 保留 Kaiwu_SDK 固定的版本，并把该冲突登记为 Breakage_List 条目。
7. WHEN 依赖安装完成，THE Setup_Automation SHALL 以 `importlib.metadata.version` 核对 `kaiwu==1.4.1`、`kaiwu-community==1.0.7`、`numpy==2.2.6`、`matplotlib==3.10.3` 与 `torch==2.7.0`，并将逐项结果写入 Environment_Snapshot。
8. IF 任一核对项的实际版本不同于期望版本，THEN THE Setup_Automation SHALL 在 Environment_Snapshot 中记录期望版本与实际版本，并在 P0_Status_Report 中将 G2 标记为"未满足"。
9. THE Setup_Automation SHALL 在 Environment_Snapshot 中记录 Python 版本、操作系统与架构、pip 版本、`pip freeze` 输出、Kaiwu_SDK 安装来源与生成时间。
10. WHEN 版本核对全部通过，THE Setup_Automation SHALL 在不执行授权初始化与求解的条件下，逐项核对 Master_Spec 7.9 所列 6 个 API（`kaiwu.core.qubo_matrix_to_qubo_model`、`kaiwu.core.qubo_model_to_ising_model`、`kaiwu.core.get_sol_dict`、`kaiwu.classical.SimulatedAnnealingOptimizer`、`kaiwu.cim.CIMOptimizer`、`kaiwu.preprocess.PrecisionReducer`）可导入，并记录每项结果。

### Requirement 4: 依赖声明升级为 kaiwu==1.4.1

**对应总需求：** 22.5（`requirements/requirements.txt` 部分）

**User Story:** 作为上游维护者，我希望依赖升级以一个范围清晰的提交完成，并列出仓库中所有 Kaiwu 版本声明，以便评审升级影响。

#### Acceptance Criteria

1. THE Setup_Automation SHALL 在 Work_Branch 上将 `requirements/requirements.txt` 中的 Kaiwu 依赖行改为 `kaiwu==1.4.1`，并保持该文件其余各行的内容与行尾符不变。
2. IF `requirements/requirements.txt` 不存在或不含 Kaiwu 依赖行，THEN THE Setup_Automation SHALL 暂停修改，记录 Fork_Clone 中实际的 Dependency_Declaration 位置，并请用户确定修改对象。
3. THE Setup_Automation SHALL 列出 Fork_Clone 中全部 Dependency_Declaration，并为每处记录相对路径、行号与声明值。
4. IF `requirements/requirements.txt` 以外的 Dependency_Declaration 声明 `kaiwu==1.3.1` 或与 1.4.1 不兼容的版本范围，THEN THE Setup_Automation SHALL 将每处登记为 Breakage_List 条目。
5. WHEN 需求 4.1 的修改完成且需求 8.1 的身份核对结果为一致，THE Setup_Automation SHALL 以显式文件路径暂存，并在 Work_Branch 上创建仅含依赖升级的本地提交。
6. THE Setup_Automation SHALL 参照 Base_Commit 历史中的上游提交信息风格撰写依赖升级提交的提交信息，并在提交信息中写明 `kaiwu 1.3.1 → 1.4.1`。

### Requirement 5: 既有测试与示例回归记录

**对应总需求：** 22.6

**User Story:** 作为上游维护者，我希望看到上游既有测试与示例在 Kaiwu 1.4.1 下的逐项结果，以便判断依赖升级是否可以合并。

#### Acceptance Criteria

1. THE Setup_Automation SHALL 将 Base_Commit 中 `tests/` 下的全部 pytest 测试用例（如 `test_bm.py`、`test_rbm.py`、`test_qvae.py` 中的用例）与 `example/` 下的全部示例入口登记为 Regression_Item，并为每项记录相对 Fork_Clone 根目录的路径与类型（测试或示例）。
2. WHEN 在 Upstream_Venv 中对 Work_Branch 的依赖升级提交运行 pytest，THE Setup_Automation SHALL 在 Regression_Record 中为每个测试用例记录 pytest 节点 ID 与结果（通过、失败、错误、跳过、预期失败或超时）。
3. THE Setup_Automation SHALL 以该示例 README 给出的运行命令执行每个示例入口（无 README 时以目录内含 `if __name__ == "__main__"` 的脚本为入口），并在 Regression_Record 中记录实际命令与结果。
4. THE Setup_Automation SHALL 对单个示例入口施加默认 600 秒、对一次 pytest 运行施加默认 1800 秒的运行上限，并在 Regression_Record 中记录实际使用的上限值。
5. IF 某 Regression_Item 达到运行上限，THEN THE Setup_Automation SHALL 将该项记为"超时"，并记录已运行时长以及截至超时的输出中是否出现异常。
6. IF 某示例无法以非交互方式执行（如需要图形界面、人工输入或 CIM 授权），THEN THE Setup_Automation SHALL 将该示例记为"跳过"并写明原因。
7. THE Setup_Automation SHALL 对 `src/`、`tests/`、`example/` 与 README 代码块执行 Static_Import_Check，并在 Regression_Record 中为每条 `kaiwu` 导入记录相对路径、行号与是否可解析。
8. THE Setup_Automation SHALL 为每个结果不是"通过"的 Regression_Item 记录原因类别（API 变更、依赖冲突、授权、网络或配额、数据获取、既有缺陷、未知）与 Sanitized_Message。
9. WHEN 用户报告 Credential_Variables 已设置，THE Setup_Automation SHALL 重跑原因类别为"授权"的 Regression_Item，并在 Regression_Record 中同时保留首次结果与重跑结果。
10. THE Regression_Record SHALL 包含 Base_Commit、被测提交 SHA、Environment_Snapshot 的相对路径、运行命令、运行上限、运行日期、Credential_Variables 是否已设置（布尔值）以及各结果类别的计数。
11. THE Regression_Record SHALL 满足各结果类别计数之和等于已登记的 Regression_Item 总数（完整性性质）。
12. WHERE Kaiwu 1.3.1 可在另一独立虚拟环境中安装，THE Setup_Automation SHALL 在 Base_Commit 上运行相同的 Regression_Item 作为对照，并将对照结果与 1.4.1 结果并列记录；原因类别"既有缺陷"仅用于在该对照中同样失败的项。
13. THE Setup_Automation SHALL 以 Markdown（供 Dep_PR 描述引用）与 JSON 两种格式将 Regression_Record 写入 P0_Record_Dir，并使两种格式的逐项结果一致。
14. WHEN 回归运行结束，THE Setup_Automation SHALL 列出 Fork_Clone 中由运行产生的未跟踪文件，并使这些文件保持未暂存状态。

### Requirement 6: 失效项清单与处理方式

**对应总需求：** 22.7

**User Story:** 作为上游维护者，我希望每个在 1.4.1 下失效的模块、示例或 README 片段都被登记并给出经确认的处理方式，以便依赖升级不引入未说明的破坏。

#### Acceptance Criteria

1. THE Setup_Automation SHALL 为 Regression_Record 中原因类别为 API 变更、依赖冲突、既有缺陷或未知的每个非通过项，以及 Static_Import_Check 中每条不可解析的导入，在 Breakage_List 中新建一个条目或关联到一个已有条目。
2. THE Breakage_List SHALL 为每个条目记录编号、位置（相对路径与行号）、现象（异常类型与 Sanitized_Message）、原因类别、关联的 Regression_Item 或 Static_Import_Check 结果、1.4.1 下的替代写法（已知时）、建议 Handling_Mode 与 Confirmation_Status。
3. WHEN Fork_Clone 中出现 `from kaiwu.cim import CIMOptimizer, PrecisionReducer`，THE Setup_Automation SHALL 在 Breakage_List 中登记该导入，替代写法为 `from kaiwu.cim import CIMOptimizer` 与 `from kaiwu.preprocess import PrecisionReducer`。
4. WHEN Fork_Clone 中出现对 `kaiwu.conversion` 的导入，THE Setup_Automation SHALL 在 Breakage_List 中登记该导入，并记录 Kaiwu_SDK 1.4.1 中对应功能的位置或注明"未找到对应功能"。
5. THE Setup_Automation SHALL 为原因类别为 API 变更或依赖冲突的条目建议 Handling_Mode `本 PR 修复`，为原因类别为既有缺陷的条目建议 `提交 issue 交由维护者处理`，并将原因类别为未知的条目的建议记为"待导师确认"。
6. THE Breakage_List SHALL 为每个条目记录 Confirmation_Status 与确认记录的引用位置，未获确认的条目记为 `待确认`。
7. WHILE 某条目的 Confirmation_Status 为 `待确认`，THE Setup_Automation SHALL 仅以独立于依赖升级提交的本地提交在 Work_Branch 上准备该条目的修复，使修复可单独移除或迁移到其他分支。
8. WHEN 某条目的修复提交完成，THE Setup_Automation SHALL 重跑关联的 Regression_Item，并在 Regression_Record 中同时保留修复前与修复后的结果。
9. WHEN Regression_Record 与 Static_Import_Check 中没有需要登记的项，THE Breakage_List SHALL 记录"无失效项"与条目数 0。
10. THE Breakage_List SHALL 满足双向可追溯：每个条目至少关联一个 Regression_Item 或 Static_Import_Check 结果，且需求 6.1 所列每一项恰好关联一个条目。
11. THE Setup_Automation SHALL 以 Markdown 与 JSON 两种格式将 Breakage_List 写入 P0_Record_Dir，并使两种格式的条目一致。

### Requirement 7: 凭据保护与忽略规则

**对应总需求：** 23.6（7.1–7.5）；7.6–7.9 将 23.1、23.4 的凭据原则应用于 P0 的 Setup_Automation

**User Story:** 作为项目参与者，我希望 fork 分支从一开始就忽略 `.env` 与授权文件，且自动化步骤只从环境变量取凭据、不在任何产物中留下凭据，以便公开仓库与记录不泄露授权信息。

#### Acceptance Criteria

1. THE Setup_Automation SHALL 在 Work_Branch 根目录 `.gitignore` 中加入 `.env` 与全部 Authorization_File_Pattern，并保留 `.gitignore` 的既有规则。
2. THE Setup_Automation SHALL 在 Work_Branch 根目录提供 `.env.example`，其中非注释行仅为 `KAIWU_USER_ID=` 与 `KAIWU_SDK_CODE=` 两个空值条目。
3. WHEN `.gitignore` 与 `.env.example` 修改完成，THE Setup_Automation SHALL 以 `git check-ignore` 核对 `.env` 与每个 Authorization_File_Pattern 的示例文件名均被忽略、`.env.example` 未被忽略，并将核对结果写入 P0_Status_Report。
4. THE Setup_Automation SHALL 在 P0_Status_Report 中为每个 Authorization_File_Pattern 记录来源（Kaiwu_SDK 文档、实际观察到的授权文件或前期工作区 `.gitignore`），无法核实授权文件名时记为"未核实"并保留 `*.lic`。
5. THE Setup_Automation SHALL 以独立于依赖升级提交的本地提交提交 `.gitignore` 与 `.env.example` 的变更。
6. THE Setup_Automation SHALL 仅从进程环境变量读取 Credential_Variables；`.env` 由用户自行加载到环境变量，Setup_Automation 不读取 `.env` 文件内容。
7. THE Setup_Automation SHALL 在命令行参数、终端输出、P0 记录、Environment_Snapshot 与提交内容中仅以"已设置/未设置"表示 Credential_Variables 的状态。
8. WHEN 准备推送或写出 P0 记录，THE Setup_Automation SHALL 执行 Credential_Scan，并在 P0_Status_Report 中记录扫描时间、扫描范围与 Credential_Variables 是否已设置。
9. IF Credential_Scan 发现 `.env`、授权文件或凭据值，THEN THE Setup_Automation SHALL 阻止推送，并仅报告命中的文件路径与行号。

### Requirement 8: 提交作者身份

**对应总需求：** 24.2

**User Story:** 作为 OSPP 参与学生，我希望所有提交的作者邮箱与 OSPP 报名的 git email 一致，以便贡献能被 OSPP 正确认定。

#### Acceptance Criteria

1. WHEN 用户提供 OSPP_Git_Email，THE Setup_Automation SHALL 以只读命令读取 Fork_Clone 中生效的 `user.name` 与 `user.email`，并将 `user.email` 与 OSPP_Git_Email 比较。
2. IF 生效的 `user.email` 未设置或不等于 OSPP_Git_Email，THEN THE Setup_Automation SHALL 在创建任何提交之前停止，并向用户给出设置提交身份的建议命令，由用户作为 Manual_Operation 执行。
3. WHEN 准备推送 Work_Branch，THE Setup_Automation SHALL 核对 Work_Branch 相对 Base_Commit 新增的每个提交的作者邮箱与提交者邮箱均等于 OSPP_Git_Email。
4. IF 任一新增提交的作者邮箱或提交者邮箱不等于 OSPP_Git_Email，THEN THE Setup_Automation SHALL 阻止推送，列出不一致提交的 SHA，并在取得 User_Confirmation 后执行修正。
5. THE Setup_Automation SHALL 请用户确认 GitHub 账号为公开账号且公开邮箱等于 OSPP_Git_Email（Manual_Operation），并在 P0_Status_Report 中记录确认结果与日期。
6. THE P0_Status_Report SHALL 以"一致/不一致"记录每次身份核对的结果与所核对的提交 SHA。

### Requirement 9: 推送与 PR 粒度策略

**对应总需求：** 准备工作（为 24.1 与 22.8 确定提交方式）

**User Story:** 作为 OSPP 参与学生，我希望在 P0 就确定 PR 的拆分方式与推送规则，以便依赖升级可以尽早评审，后续功能 PR 不被阻塞。

#### Acceptance Criteria

1. THE Setup_Automation SHALL 仅将 `main` 以外的分支以非强制方式推送到 Fork_Repository。
2. WHEN 经 User_Confirmation 首次推送某分支，THE Setup_Automation SHALL 使用 `git push -u origin <分支名>`。
3. THE P0_Status_Report SHALL 记录默认 PR_Strategy：Dep_PR 承载依赖升级、Regression_Record 摘要、Handling_Mode 为 `本 PR 修复` 的修复以及 `.gitignore`/`.env.example` 变更；P1 结束时另提交能量/QUBO PR。
4. THE P0_Status_Report SHALL 记录 PR_Strategy 的 Confirmation_Status 与确认记录的引用位置，未获导师确认时记为 `待确认`。
5. WHEN 导师调整 PR_Strategy，THE Setup_Automation SHALL 在 P0_Status_Report 中记录调整后的策略、调整日期与受影响的后续分期。
6. WHERE PR_Strategy 为增量 PR，THE Setup_Automation SHALL 在 P0_Record_Dir 中起草 Dep_PR 描述，包含不超过 70 个字符的标题、变更摘要、已运行测试及结果（引用 Regression_Record）、失效项与处理方式（引用 Breakage_List）以及未完成或受阻事项。
7. WHEN 用户确认创建 Dep_PR，THE Setup_Automation SHALL 以 Work_Branch 为源分支、Plugin_Repository `main` 为目标分支创建 Dep_PR，并在 P0_Status_Report 中记录 PR 链接。

### Requirement 10: 上游 CI 与 lint/格式工具核实

**对应总需求：** 准备工作（为 22.4 确定"仓库既有的 lint 与格式检查"）

**User Story:** 作为算法开发者，我希望在写代码之前就知道上游 CI 跑什么、用哪些 lint 与格式工具，以及依赖升级是否与 CI 冲突，以便新代码一次通过检查。

#### Acceptance Criteria

1. WHEN Fork_Clone 就绪，THE Setup_Automation SHALL 检查 Base_Commit 中的 CI 与代码风格配置（`.github/workflows/`、`pyproject.toml`、`setup.cfg`、`tox.ini`、`.flake8`、`ruff.toml`、`.pre-commit-config.yaml` 等），并将结果写入 CI_Inventory。
2. THE CI_Inventory SHALL 为每个 CI 工作流记录文件路径、触发事件、Python 版本矩阵、依赖安装方式、执行的命令，以及所需密钥的名称（仅名称）。
3. THE CI_Inventory SHALL 为每个 lint 与格式工具记录名称、配置文件路径、版本约束与本地等价命令。
4. WHERE Base_Commit 声明了 lint 或格式工具，THE Setup_Automation SHALL 在 Upstream_Venv 中对 Base_Commit 运行这些检查，并在 CI_Inventory 中记录所用工具版本与基线结果（通过或问题数）。
5. IF CI 的 Python 版本矩阵不含 3.10，或 CI 的依赖安装方式无法安装 `kaiwu==1.4.1`，THEN THE Setup_Automation SHALL 将该项登记为 Breakage_List 条目，原因类别为"依赖冲突"。
6. IF Base_Commit 中未发现 CI 配置或 lint/格式工具配置，THEN THE CI_Inventory SHALL 记录"未发现"，并将"Master_Spec 22.4 所指的检查工具"列为待导师或维护者确认的事项。
7. WHERE Plugin_Repository 的 GitHub Actions 运行记录可公开访问，THE CI_Inventory SHALL 记录 `main` 分支最近一次 CI 运行的状态、日期与链接。

### Requirement 11: 导师确认记录归档

**对应总需求：** 24.4（归档部分：导师确认记录的存放位置确定并可引用）

**User Story:** 作为 OSPP 参与学生，我希望导师对 1.3.1→1.4.1 的书面确认有一个稳定、可引用的存放位置，以便结项报告的偏差说明有据可查。

#### Acceptance Criteria

1. THE Setup_Automation SHALL 请用户将 Mentor_Confirmation_Record 归档到用户选定的 Stable_Reference（Manual_Operation），可选位置包括 P0_Record_Dir 下的文件、OSPP 系统中的沟通记录与 GitHub issue/评论链接。
2. WHEN 用户报告归档完成，THE Setup_Automation SHALL 在 P0_Status_Report 中记录 Stable_Reference、确认日期、确认人角色（导师）与确认范围（SDK 基线由 `kaiwu==1.3.1` 改为 `kaiwu==1.4.1`，不做双版本适配）。
3. WHERE Stable_Reference 为本地文件路径，THE Setup_Automation SHALL 仅核对该文件存在，并在 P0_Status_Report 中记录核对结果。
4. WHERE Stable_Reference 为 URL，THE Setup_Automation SHALL 请用户确认 OSPP 评审方可访问该 URL，并在 P0_Status_Report 中记录确认结果。
5. THE Setup_Automation SHALL 使 Mentor_Confirmation_Record 的正文保持在 Fork_Clone 之外，并在 Dep_PR 描述中仅写明"SDK 基线变更已获导师书面确认"。

### Requirement 12: CIM 授权申请状态

**对应总需求：** 准备工作（为 9.8、18.9、19.10 与 24.4 中 CIM 缩减项说明提供状态依据）

**User Story:** 作为研究者，我希望尽早提交 CIM 授权申请并记录状态，以便后续分期能按授权情况安排 CIM 扩展验证，并在受限时有据说明。

#### Acceptance Criteria

1. THE Setup_Automation SHALL 向用户提供 CIM 授权与配额申请所需的信息清单，至少包含项目用途与预计问题规模（64—32—10 网络在 K = 1/2/3 时为 42/84/126 个 Logical_Bit，另加至多 1 个 Auxiliary_Spin），申请由用户作为 Manual_Operation 提交。
2. THE P0_Status_Report SHALL 记录 CIM_Authorization_Status、申请日期、申请渠道，以及已知的配额与限制。
3. WHILE CIM_Authorization_Status 不是 `已获批`，THE P0_Status_Report SHALL 注明 SA_Backend 承担全部核心验收、CIM 验证待授权后进行。
4. IF 用户在 P0 期间未提交 CIM 申请，THEN THE P0_Status_Report SHALL 记录未提交原因与计划提交日期。
5. THE Setup_Automation SHALL 将 CIM 任务的提交安排到后续分期，P0 只记录 CIM 状态。

### Requirement 13: 退出门禁与 P0 状态报告

**对应总需求：** 无直接对应验收标准（P0 → P1 交接）

**User Story:** 作为 OSPP 参与学生，我希望在一份报告里看到 P0 各门禁是否满足、谁在阻塞以及 P1 需要的交接信息，以便判断 P1 能否开工。

#### Acceptance Criteria

1. THE Setup_Automation SHALL 在 P0_Record_Dir 中维护 P0_Status_Report，为 G1–G9 与 S1–S3 逐项记录状态（满足、未满足或阻塞）、证据文件的相对路径或引用、负责方（用户、导师、维护者、Kaiwu 平台或 Setup_Automation）与更新日期。
2. WHEN 全部 Hard_Gate 为"满足"且全部 Status_Gate 已记录状态，THE Setup_Automation SHALL 在 P0_Status_Report 中标记"P1 可开工"并记录日期。
3. IF 截至 2026-10-05 仍有 Hard_Gate 未满足，THEN THE Setup_Automation SHALL 在 P0_Status_Report 中标记"P1 暂缓开工"，并为每个未满足的 Hard_Gate 记录阻塞原因、负责方与预计完成日期。
4. WHILE 处于 2026-10-01 至 2026-10-05 的国庆假期，THE Setup_Automation SHALL 在 P0_Status_Report 中为每个等待导师、维护者或 Kaiwu 平台回复的事项记录请求发出日期与渠道。
5. THE Setup_Automation SHALL 保持 Master_Spec 内容不变，并将 P0 中发现的与 Master_Spec 不一致的事实（如仓库结构、依赖声明或 API 位置）记录为 P0_Status_Report 中的"总需求修订建议"，由用户决定是否修订。
6. THE P0_Status_Report SHALL 为 P1 列出交接信息：Fork_Clone 路径、Work_Branch 名称与最新提交 SHA、Upstream_Venv 路径与激活命令、需求 3.10 的 API 核对结果，以及 CI_Inventory 中的 lint 与格式命令。
7. THE Setup_Automation SHALL 在 P0_Status_Report 与其他 P0 记录中使用相对工作区根目录或相对 Fork_Clone 根目录的路径。

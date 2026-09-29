# Implementation Plan：ep-p0-upstream-setup

## Overview

按 design.md "执行流程"的依赖顺序组织：先实现 `tools/ep_p0/` 中跑通流程所需的最小辅助脚本（仅标准库，入口 `python -m tools.ep_p0 <cmd>`），再按 A0–A14 执行，Manual_Operation（M-a～M-d、M1–M4）单独成任务。JSON 为唯一数据源，Markdown 由 `render` 生成。实现语言为 Python（≥3.10）。9 条 Correctness Properties 的 Hypothesis 性质测试均为可选子任务。

Manual_Operation 任务统一约定：由用户执行，代理给出说明（目的、建议命令或页面入口、核对方法）并在用户报告完成后核对；代理不得自行执行。

## Tasks

- [ ] 1. 辅助脚本基础（tools/ep_p0）
  - [x] 1.1 包骨架与数据模式
    - 产出：`tools/ep_p0/__init__.py`、`tools/ep_p0/__main__.py`、`tools/ep_p0/schema.py`
    - `__main__.py` 用固定的"子命令→模块"映射表延迟导入各模块的 `main(argv)`，退出码 0/1/2；后续任务只新增模块，不改此文件
    - `schema.py`：2.1 全部枚举与中文标签、`compute_counts`、`validate_regression`（非通过⇔reason 非空、`sum(counts)==total==len(items)`、有效结果取最后一次 attempt）
    - _Requirements: 5.2, 5.8, 5.10, 5.11, 13.1_
  - [x] 1.2 路径脱敏、只读 git 包装与仓库辅助
    - 产出：`tools/ep_p0/sanitize.py`、`tools/ep_p0/gitro.py`、`tools/ep_p0/repo.py`
    - `sanitize()` 按设计顺序处理（首行、`<REDACTED>`、`<HOME>`、相对 Fork_Clone 根、`<ABS>`、`/` 分隔、300 字符），`\`/`/` 与大小写不敏感
    - `gitro.py` 只放行白名单子命令，其余抛错；`repo.py`：克隆 URL 校验、克隆/fetch stderr 分类（network/access/other）、结构核对（2.8 所列路径与 `__version__`）
    - 子进程环境剔除 Credential_Variables 的工具函数放在 `sanitize.py`
    - _Requirements: 1.3, 2.3, 2.8, 2.9, 5.8, 7.6, 7.7, 13.7_
  - [ ]* 1.3 单元测试：gitro 拒绝写入类子命令、克隆 URL 校验（含 `@`/userinfo 拒绝）、stderr 分类
    - 产出：`tools/ep_p0/tests/test_basics.py`（git 测试在 `tmp_path` 中 `git init`，设 `GIT_CONFIG_NOSYSTEM=1`、`GIT_CONFIG_GLOBAL` 指向临时空文件）
    - _Requirements: 1.3, 2.3, 2.9_
  - [ ]* 1.4 性质测试：回归计数完整性
    - 产出：`tools/ep_p0/tests/test_prop_counts.py`
    - **Property 1: 回归计数完整性**
    - **Validates: Requirements 5.2, 5.9, 5.11, 6.8**

- [x] 2. A0：记录目录与 status.json 初始化
  - [x] 2.1 实现 status.json 读写
    - 产出：`tools/ep_p0/status.py`（`init`/`load`/`save`，按设计分节建空结构；`gates` 初值为 G1–G9 `unmet`、S1–S3 未记录；写盘前经 `sanitize`）
    - _Requirements: 13.1, 13.7_
  - [x] 2.2 执行 A0
    - 建 `records/ep-p0/`，运行 `python -m tools.ep_p0 init` 生成 `records/ep-p0/status.json`
    - 用工作区根 `.venv` 运行 `.\.venv\Scripts\python.exe -m pytest tools/ep_p0/tests -q`（缺少时安装 `pytest==8.3.3`、`hypothesis==6.112.0`）
    - _Requirements: 13.1_

- [ ] 3. Manual_Operation：并行发起的外部事项（M-a～M-d）
  - [ ] 3.1 M-a CIM 授权申请（由用户执行，代理给出说明并在用户报告完成后核对）
    - 代理提供信息清单：项目用途、64—32—10 网络 K=1/2/3 时 42/84/126 Logical_Bit、另加至多 1 个 Auxiliary_Spin
    - 用户报告后写入 `status.json.manual.cim`（状态、申请日期、渠道、已知配额）与 `waits[]`；未提交时记录原因与计划日期；未获批时注明 SA_Backend 承担核心验收
    - _Requirements: 1.1, 1.2, 12.1, 12.2, 12.3, 12.4, 12.5, 13.4_
  - [ ] 3.2 M-b 归档 Mentor_Confirmation_Record（由用户执行，代理给出说明并在用户报告完成后核对）
    - 可选位置：`records/ep-p0/mentor-confirmation/` 下文件、OSPP 系统记录、GitHub issue/评论 URL
    - 本地文件只核对存在；URL 请用户确认评审方可访问；写入 `manual.mentor_record`（Stable_Reference、日期、角色、范围、verified）；正文不进入 Fork_Clone
    - _Requirements: 1.1, 1.2, 11.1, 11.2, 11.3, 11.4, 11.5_
  - [ ] 3.3 M-c 确认 GitHub 公开账号与公开邮箱（由用户执行，代理给出说明并在用户报告完成后核对）
    - 请用户提供 OSPP_Git_Email，确认公开邮箱与之相同；写入 `manual.public_email`（结果、日期）
    - _Requirements: 1.1, 1.2, 8.5_
  - [ ] 3.4 M-d 向导师发出 PR_Strategy 确认请求（由用户执行，代理给出说明并在用户报告完成后核对）
    - 代理提供默认增量策略说明文本；写入 `manual.pr_strategy`（默认策略、`pending`、引用位置）与 `waits[]`；导师调整时记录新策略、日期与受影响分期
    - _Requirements: 9.3, 9.4, 9.5, 13.4_

- [ ] 4. M1 + A1：fork 与克隆
  - [ ] 4.1 M1 在 GitHub fork Plugin_Repository（由用户执行，代理给出说明并在用户报告完成后核对）
    - 代理给出 fork 页面入口；用户提供 Fork_Repository URL，代理用 `repo.py` 校验 URL
    - _Requirements: 1.1, 1.2, 2.1_
  - [ ] 4.2 执行 A1
    - 非空目录检查（非空则停止并报告现状，不删除）→ `git clone` 到 `upstream/kaiwu-pytorch-plugin` → 按 URL fetch 上游 main（不新增远端）→ 记录 Base_Commit 与 fork main HEAD（不同则都记）→ `git switch -c deps/kaiwu-1.4.1 <base>` → 结构核对
    - 写入 `status.json.repo`；失败时记录去凭据命令与错误类别，G1 记 `blocked`（owner user）并暂停后续步骤
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

- [ ] 5. A2：依赖声明清单与 CI 只读盘点
  - [ ] 5.1 实现 `decls` 与 CI 配置解析
    - 产出：`tools/ep_p0/decls.py`（requirements*.txt、pyproject.toml、setup.py/.cfg、workflows、README 安装命令中 kaiwu 声明的路径、行号、值；标记 1.3.1 或不兼容范围）、`tools/ep_p0/ci.py`（workflow 触发事件、Python 矩阵、安装方式、命令、密钥名称；lint/格式工具名称、配置路径、版本约束、本地命令）
    - _Requirements: 4.3, 4.4, 10.1, 10.2, 10.3_
  - [ ] 5.2 执行 A2
    - 写入 `status.json.dependency_declarations[]`（D-*）与 `status.json.ci`（C-* 候选：矩阵不含 3.10 或无法安装 1.4.1；未发现配置时 `not_found` 并加入待确认事项）
    - _Requirements: 4.3, 4.4, 10.1, 10.2, 10.3, 10.5, 10.6_

- [ ] 6. A3：requirements.txt 字节级修改
  - [ ] 6.1 实现 `bump-req`
    - 产出：`tools/ep_p0/bumpreq.py`（按字节替换唯一 kaiwu 行并保留行尾；缺文件或无 kaiwu 行时退出码 2、不写文件并列出实际声明位置）
    - _Requirements: 4.1, 4.2_
  - [ ]* 6.2 性质测试：requirements.txt 单行替换
    - 产出：`tools/ep_p0/tests/test_prop_bumpreq.py`
    - **Property 7: requirements.txt 单行替换**
    - **Validates: Requirements 4.1, 4.2**
  - [ ] 6.3 执行 A3
    - 对 `upstream/kaiwu-pytorch-plugin/requirements/requirements.txt` 运行 `bump-req --to 1.4.1`，暂不提交；退出码 2 时暂停并请用户确定修改对象
    - _Requirements: 4.1, 4.2_

- [ ] 7. A4 + A5：Upstream_Venv 与 Environment_Snapshot
  - [ ] 7.1 实现 `snapshot`
    - 产出：`tools/ep_p0/snapshot.py`（子进程调用 venv 解释器查询版本、`pip --version`、`pip freeze`、平台信息；vendor 来源计算 SHA-256；`--api-check` 仅导入 6 个 API；`dependency_conflicts[]` 记 E-*；`credentials_set` 布尔值）
    - _Requirements: 3.4, 3.6, 3.7, 3.8, 3.9, 3.10, 7.7_
  - [ ] 7.2 执行 A4 与 A5
    - `py -3.10 -m venv upstream\.venv`（缺 3.10 时停止，G2 记 `blocked`，原因"Kaiwu_SDK 1.4.1 仅提供 cp310 wheel"）→ PyPI 安装 `kaiwu==1.4.1`，失败回退 Vendor_Wheel → 按 `upstream/.p0-runs/constraints-kaiwu141.txt` 安装其余依赖、测试依赖与声明的 lint 工具 → `pip check`
    - 运行 `snapshot --api-check` 写出 `records/ep-p0/environment.json`；版本不符时 G2 记 `unmet`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_

- [ ] 8. M2 + A6：提交身份与依赖升级提交 C1
  - [ ] 8.1 M2 设置 git 提交身份（由用户执行，代理给出说明并在用户报告完成后核对）
    - 代理给出建议命令（如 `git -C upstream\kaiwu-pytorch-plugin config user.email "<OSPP_Git_Email>"`），不自行执行任何 Git_Config_Write
    - _Requirements: 1.1, 1.2, 8.2_
  - [ ] 8.2 执行 A6
    - 只读 `git config --get user.name/user.email` 与 OSPP_Git_Email 比较，不一致则停止并回到 8.1
    - 参照 Base_Commit 提交信息风格写 `upstream/.p0-runs/msg-c1.txt`（含 `kaiwu 1.3.1 → 1.4.1`），`git add -- requirements/requirements.txt` 后 `git commit -F`；核对结果写入 `status.json`
    - _Requirements: 4.5, 4.6, 8.1, 8.2_

- [ ] 9. 检查点：确认 tools/ep_p0/tests 全部通过，有问题询问用户
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. A7 + A8：回归运行与 Static_Import_Check
  - [ ] 10.1 实现原因分类
    - 产出：`tools/ep_p0/classify.py`（按设计 7 条顺序规则，关键词表可扩展；通过结果返回空）
    - _Requirements: 5.8, 5.12_
  - [ ]* 10.2 性质测试：非通过项原因完整且"既有缺陷"只来自对照
    - 产出：`tools/ep_p0/tests/test_prop_classify.py`
    - **Property 2: 非通过项原因完整且"既有缺陷"只来自对照**
    - **Validates: Requirements 5.8, 5.12**
  - [ ] 10.3 实现回归子命令
    - 产出：`tools/ep_p0/regress.py`（`inventory`、`run-tests`、`run-examples`、`junit2reg`：逐文件 pytest `-v -p no:cacheprovider --junitxml`、超时杀进程树、junit 缺失时从 `-v` 日志恢复、attempts 只追加、示例 `MPLBACKEND=Agg`、非交互不可执行记 skipped 并写原因）
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.8, 5.10_
  - [ ]* 10.4 单元测试：collect-only 与收集失败解析、junit collection error 映射、超时日志恢复
    - 产出：`tools/ep_p0/tests/test_regress.py`
    - _Requirements: 5.1, 5.2, 5.5_
  - [ ] 10.5 实现 Static_Import_Check
    - 产出：`tools/ep_p0/imports.py`（AST 扫描 `.py`，README python 代码块解析失败退回正则；`sdk` 在 venv 子进程解析（120 s 上限），`internal` 对照源码树解析；`kaiwu.conversion` 按设计查找替代位置）
    - _Requirements: 5.7, 6.3, 6.4_
  - [ ]* 10.6 性质测试：静态导入扫描完整
    - 产出：`tools/ep_p0/tests/test_prop_imports.py`（临时目录伪 `kaiwu` 包，当前解释器作 `--venv` 替身）
    - **Property 8: 静态导入扫描完整**
    - **Validates: Requirements 5.7**
  - [ ] 10.7 执行 A7
    - `inventory` → `run-tests --limit 1800 --phase initial` → `run-examples --limit 600` → `imports`，写出 `records/ep-p0/regression.json`
    - 结束后 `git status --porcelain --untracked-files=all` 列出运行产生的未跟踪文件，保持未暂存
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.10, 5.11, 5.14_
  - [ ] 10.8 执行 A8：1.3.1 对照（可安装时）
    - `git archive` 导出 Base_Commit 到 `upstream/.p0-runs/base-src/`，建 `upstream/.venv-kaiwu131/`，运行相同 Regression_Item；结果写 `upstream/.p0-runs/control-kaiwu131.json` 与 `records/ep-p0/environment-kaiwu131.json`；不可安装时记录原因
    - _Requirements: 5.12_

- [ ] 11. A9：Breakage_List
  - [ ] 11.1 实现 `breakage`
    - 产出：`tools/ep_p0/breakage.py`（登记范围、去重键、按 key 合并保留编号与 confirmation、建议 Handling_Mode 映射、PrecisionReducer 已知项、`links` 5 类来源、回写 `breakage_id`、无条目时 `none: true`）
    - _Requirements: 3.6, 4.4, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.9, 6.10, 10.5_
  - [ ]* 11.2 性质测试：失效项双向可追溯与重建幂等
    - 产出：`tools/ep_p0/tests/test_prop_breakage.py`（含 PrecisionReducer 与 kaiwu.conversion 两个已知项示例）
    - **Property 4: 失效项双向可追溯与重建幂等**
    - **Validates: Requirements 3.6, 4.4, 6.1, 6.5, 6.6, 6.9, 6.10**
  - [ ] 11.3 执行 A9
    - 把 control-kaiwu131.json 合入 `regression.json` 的 `control_kaiwu131` 与 `control` 节，重新分类后运行 `breakage`，写出 `records/ep-p0/breakage.json`（C-* 已在 13.1 写入 status.json 时一并合并）
    - _Requirements: 5.12, 6.1, 6.2, 6.5, 6.6, 6.9, 6.10_

- [ ] 12. A10：`.gitignore` 与 `.env.example`（提交 C2）
  - [ ] 12.1 实现忽略规则辅助
    - 产出：`tools/ep_p0/ignore.py`（追加带注释规则块、跳过已有规则、沿用原行尾；生成仅含两个空值条目的 `.env.example`；解析 `git check-ignore` 结果）
    - _Requirements: 7.1, 7.2, 7.3_
  - [ ]* 12.2 单元测试：`.gitignore` 追加保留既有规则、`.env.example` 内容、check-ignore 结果解析
    - 产出：`tools/ep_p0/tests/test_ignore.py`
    - _Requirements: 7.1, 7.2, 7.3_
  - [ ] 12.3 执行 A10
    - 修改 Fork_Clone 根 `.gitignore`（`.env`、`*.lic` 与核实到的模式）并新建 `.env.example`；`git check-ignore -v -- .env sample.lic` 与 `git check-ignore -- .env.example`（期望退出码 1）
    - 写入 `status.json.ignore_checks[]` 与 `auth_patterns[]`（来源，无法核实记 `unverified` 并保留 `*.lic`）；身份核对一致后以显式路径提交 C2
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1_

- [ ] 13. A11：lint 基线与 CI 运行记录
  - [ ] 13.1 执行 A11
    - 在 Upstream_Venv 对 Base_Commit 运行已声明的 lint/格式检查，记录工具版本与基线结果（通过或问题数）
    - 查询 Plugin_Repository main 最近一次公开 Actions 运行（状态、日期、链接），写入 `status.json.ci.latest_run`；C-* 条目确认后写入 `status.json.ci`
    - _Requirements: 10.4, 10.5, 10.6, 10.7_

- [ ] 14. 检查点：确认 tools/ep_p0/tests 全部通过且 regression.json 通过 `validate_regression`，有问题询问用户
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. A12：`fix_in_pr` 条目的独立修复提交
  - [ ] 15.1 执行 A12
    - 每个 `fix_in_pr` 条目一个独立本地提交（提交信息引用 `B-xxx`，C3…），记入 `fix_commits`；重跑关联项以 `post_fix` 追加 attempts，再运行 `breakage` 合并
    - _Requirements: 6.7, 6.8_

- [ ] 16. M3 + A13：授权项重跑（可选）
  - [ ] 16.1 M3 在进程环境中设置 Credential_Variables（由用户执行，代理给出说明并在用户报告完成后核对）
    - 代理说明用户自行把 `.env` 加载到环境变量；代理不读取 `.env`，只核对"已设置/未设置"
    - _Requirements: 1.2, 5.9, 7.6, 7.7_
  - [ ] 16.2 执行 A13
    - 对 reason 为 `auth` 的项以 `--phase auth_rerun` 重跑，保留首次结果；`credentials_set` 只写布尔值
    - _Requirements: 5.9, 7.7_

- [ ] 17. A14：扫描、身份核对、渲染与门禁判定工具
  - [ ] 17.1 实现 `scan-creds` 与 `identity`
    - 产出：`tools/ep_p0/scan.py`（按路径匹配 `.env`/授权模式、不读 `.env` 内容；凭据已设置时逐字查找 diff 新增行、提交信息与 records 文本文件；只输出 `{path, line, kind}`，命中退出码 1）、`tools/ep_p0/identity.py`（`git log --format=%H%x09%ae%x09%ce <range>`，"一致/不一致"与不一致 SHA 列表）
    - _Requirements: 7.8, 7.9, 8.3, 8.4, 8.6_
  - [ ]* 17.2 性质测试：Credential_Scan 与 Sanitized_Message 无泄漏
    - 产出：`tools/ep_p0/tests/test_prop_scan.py`（`monkeypatch.setenv` 注入随机凭据）
    - **Property 5: Credential_Scan 与 Sanitized_Message 无泄漏**
    - **Validates: Requirements 5.8, 7.6, 7.7, 7.8, 7.9, 13.7**
  - [ ]* 17.3 性质测试：新增提交邮箱一致
    - 产出：`tools/ep_p0/tests/test_prop_identity.py`
    - **Property 6: 新增提交邮箱一致**
    - **Validates: Requirements 8.3, 8.4, 8.6**
  - [ ] 17.4 实现 `render`
    - 产出：`tools/ep_p0/render.py`（regression.md、breakage.md、ci-inventory.md、status-report.md、dep-pr-draft.md；内存渲染后先扫描再写盘；`parse_*` 反解析；Dep_PR 标题 ≤70 字符，导师确认只写"SDK 基线变更已获导师书面确认"；未确认推送显示"待确认"）
    - _Requirements: 1.5, 5.13, 6.11, 9.3, 9.6, 10.2, 10.3, 11.5, 13.1, 13.5, 13.6, 13.7_
  - [ ]* 17.5 性质测试：MD/JSON 一致（round-trip）
    - 产出：`tools/ep_p0/tests/test_prop_render.py`
    - **Property 3: MD/JSON 一致（round-trip）**
    - **Validates: Requirements 5.13, 6.11, 9.6**
  - [ ] 17.6 实现 `gates`
    - 产出：`tools/ep_p0/gates.py`（按设计表判定 G1–G9、S1–S3；`go`/`defer`/`pending`；`defer` 时要求每个非 met 的 G 带 blocker；判定后调用 render 重渲染 status-report.md）
    - _Requirements: 13.1, 13.2, 13.3_
  - [ ]* 17.7 性质测试：门禁判定
    - 产出：`tools/ep_p0/tests/test_prop_gates.py`
    - **Property 9: 门禁判定**
    - **Validates: Requirements 13.2, 13.3**
  - [ ] 17.8 执行 A14
    - 运行 `identity --range <base>..HEAD --email <OSPP_Git_Email>` 与 `scan-creds`（范围含待推送差异与 `records/ep-p0/`），写入 `identity_checks[]`、`credential_scans[]`（时间、范围、credentials_set）
    - 运行 `render` 生成全部 MD 与 dep-pr-draft.md；把与 Master_Spec 不一致的事实写入 `master_spec_suggestions[]`，并在待确认事项中注明 6.10 证据来源扩展为 5 类
    - _Requirements: 5.13, 6.11, 7.8, 8.3, 8.6, 9.6, 13.5, 13.7_

- [ ] 18. 检查点：确认 tools/ep_p0/tests 全部通过、最近一次 Credential_Scan 与身份核对通过，有问题询问用户
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 19. M4：推送与创建 Dep_PR（需用户逐次确认，非门禁）
  - [ ] 19.1 推送 Work_Branch（需用户逐次确认，非门禁）
    - 每次推送前重做身份核对与 Credential_Scan，向用户列出远端、分支与 `git log --format=%H <base>..HEAD`；确认后 `git push -u origin deps/kaiwu-1.4.1`（禁止 main、`--force`、`--no-verify`）
    - 未确认记为 `pending_confirmation`，拒绝记为 `declined`，写入 `status.json.pushes[]`；身份不一致时经确认后按设计对未推送提交执行 `--reset-author` 修正
    - _Requirements: 1.4, 1.5, 8.3, 8.4, 9.1, 9.2_
  - [ ] 19.2 创建 Dep_PR（需用户逐次确认，非门禁）
    - 用户确认后以 Work_Branch 为源、Plugin_Repository main 为目标，使用 dep-pr-draft.md 创建 Dep_PR，PR 链接写入 status.json
    - _Requirements: 1.6, 9.6, 9.7_

- [ ] 20. 门禁汇总与 P1 交接
  - [ ] 20.1 运行 `gates --today` 并写出结论
    - 运行 `python -m tools.ep_p0 gates --records records/ep-p0 --today <当日>`（最终判定用 `--today 2026-10-05`）
    - status-report.md 写出"P1 可开工"或"P1 暂缓开工"及日期；未满足的 G 逐项写阻塞原因、负责方、预计完成日期；`waits[]` 列出请求日期与渠道；S1–S3 当前状态
    - P1 交接信息：Fork_Clone 路径、Work_Branch 名称与最新 SHA、Upstream_Venv 路径与激活命令（`upstream\.venv\Scripts\Activate.ps1`）、6 个 API 核对结果、CI_Inventory 中的 lint/格式命令
    - 渲染后再运行一次 `scan-creds` 覆盖 `records/ep-p0/`
    - _Requirements: 12.3, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7_

## Notes

- 带 `*` 的子任务为可选测试，可跳过以保留 P1 时间；Property 1–9 各对应一个可选子任务。
- Manual_Operation 任务（3.1–3.4、4.1、8.1、16.1）代理不得自行执行；推送与建 PR（19.x）每次单独取得 User_Confirmation，不影响门禁判定。
- `__main__.py` 的子命令映射在 1.1 一次写定，后续模块任务互不修改同一文件，便于并行。
- 排期：09-29 完成 1–8；09-30 完成 10–13；10-01～10-05 完成 15–20，外部回复只写入 `waits[]`。

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "4.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "5.1", "6.1", "7.1", "10.1", "10.5", "11.1", "12.1", "17.1", "17.6"] },
    { "id": 2, "tasks": ["1.3", "1.4", "2.2", "6.2", "10.2", "10.3", "10.6", "11.2", "12.2", "17.3", "17.4", "17.7"] },
    { "id": 3, "tasks": ["3.1", "10.4", "17.2", "17.5"] },
    { "id": 4, "tasks": ["3.2"] },
    { "id": 5, "tasks": ["3.3"] },
    { "id": 6, "tasks": ["3.4"] },
    { "id": 7, "tasks": ["4.2"] },
    { "id": 8, "tasks": ["5.2", "8.1", "10.8"] },
    { "id": 9, "tasks": ["6.3"] },
    { "id": 10, "tasks": ["7.2"] },
    { "id": 11, "tasks": ["8.2"] },
    { "id": 12, "tasks": ["10.7", "13.1"] },
    { "id": 13, "tasks": ["11.3", "12.3"] },
    { "id": 14, "tasks": ["15.1", "16.1"] },
    { "id": 15, "tasks": ["16.2"] },
    { "id": 16, "tasks": ["17.8"] },
    { "id": 17, "tasks": ["19.1"] },
    { "id": 18, "tasks": ["19.2"] },
    { "id": 19, "tasks": ["20.1"] }
  ]
}
```

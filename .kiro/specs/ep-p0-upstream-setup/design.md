# Design Document：ep-p0-upstream-setup

## Overview

P0 由两部分组成：Kiro 代理按固定顺序执行的 Setup_Automation 步骤，以及工作区 `tools/ep_p0/` 下的辅助脚本。设计决策：

- JSON 是唯一数据源（`regression.json`、`breakage.json`、`status.json`），Markdown 全部由 `render` 生成，不手工编辑，保证 MD/JSON 一致。
- 脚本只用标准库，任一 Python ≥3.10 可运行；需要在 Upstream_Venv 内求值的探针（包版本、导入解析）以子进程调用 `upstream/.venv/Scripts/python.exe`，子进程环境中剔除 Credential_Variables。
- 脚本不写 git 配置，只通过白名单只读包装 `gitro.py` 调用 git；写操作（add/commit/push）由代理按'凭据、git 安全与 Sanitized_Message'规则执行。
- 写入记录的路径一律相对工作区根或 Fork_Clone 根；异常文本一律经 `sanitize()`。
- 需求一致性说明：6.10 要求每个条目关联 Regression_Item 或 Static_Import_Check 结果，但 3.6、4.4、10.5 产生的条目来源是安装冲突、依赖声明与 CI 配置。设计把证据来源扩展为 5 类（见 Data Models 2.3），可追溯性质按扩展后的来源检验，并在 status-report 的待确认事项中注明。

## Architecture
### 目录与产物布局

```
<workspace>/
├─ upstream/
│  ├─ kaiwu-pytorch-plugin/      Fork_Clone（origin = Fork_Repository）
│  ├─ .venv/                     Upstream_Venv（Python 3.10，kaiwu 1.4.1）
│  ├─ .venv-kaiwu131/            可选：5.12 对照环境
│  └─ .p0-runs/                  原始产物：junit xml、-v 日志、base 源码快照、constraints、提交信息文件；不作为记录发布
├─ records/ep-p0/                P0_Record_Dir
│  ├─ environment.json           Environment_Snapshot（对照环境另写 environment-kaiwu131.json）
│  ├─ regression.json / .md      Regression_Record（含 Static_Import_Check 结果）
│  ├─ breakage.json / .md        Breakage_List
│  ├─ status.json                门禁、确认事项、依赖声明清单、CI 数据、身份核对、扫描与推送记录的结构化源
│  ├─ ci-inventory.md            由 status.json.ci 渲染
│  ├─ status-report.md           P0_Status_Report，由 status.json 渲染
│  ├─ dep-pr-draft.md            Dep_PR 描述草稿
│  └─ mentor-confirmation/       可选：Stable_Reference 为本地文件时由用户放入
└─ tools/ep_p0/                  辅助脚本与 tests/（Fork_Clone 之外，不进上游提交）
```

1.3.1 对照在 `git archive` 导出的 Base_Commit 源码快照（`upstream/.p0-runs/base-src/`）上运行，不切换 Fork_Clone 的工作区。

## Data Models

### 2.1 枚举（`tools/ep_p0/schema.py` 统一定义 JSON 值与中文标签）

| 枚举 | JSON 值 → 标签 |
|---|---|
| result | `passed` 通过、`failed` 失败、`error` 错误、`skipped` 跳过、`xfailed` 预期失败、`timeout` 超时 |
| reason | `api_change` API 变更、`dep_conflict` 依赖冲突、`auth` 授权、`network_quota` 网络或配额、`data_fetch` 数据获取、`preexisting` 既有缺陷、`unknown` 未知 |
| handling_mode | `fix_in_pr` 本 PR 修复、`separate_pr` 单独小 PR、`issue` 提交 issue 交由维护者处理、`pending_mentor` 待导师确认 |
| confirmation | `pending` 待确认、`mentor_confirmed` 导师已确认、`maintainer_confirmed` 维护者已确认、`adjusted` 已调整 |
| gate_status | `met` 满足、`unmet` 未满足、`blocked` 阻塞 |
| owner | `user`、`mentor`、`maintainer`、`kaiwu_platform`、`automation` |
| cim_status | `not_applied` 未申请、`applied_pending` 已申请待回复、`approved` 已获批、`rejected` 被拒绝 |

### 2.2 regression.json

```jsonc
{
  "schema": "ep-p0/regression@1",
  "base_commit": "<40hex>", "tested_commit": "<40hex>",
  "environment_snapshot": "records/ep-p0/environment.json",
  "run_date": "2026-09-30", "credentials_set": false,
  "limits": {"pytest_run_s": 1800, "example_s": 600},
  "commands": ["upstream/.venv/Scripts/python.exe -m pytest tests/test_rbm.py -v ..."],
  "control": {"available": false, "note": "<对照不可用原因>"},
  "items": [{
    "id": "R-001", "kind": "test",            // test | example
    "path": "tests/test_rbm.py",
    "node_id": "tests/test_rbm.py::test_x",   // 文件级（收集失败）与示例为 null
    "command": "...",
    "attempts": [{                            // 只追加；有效结果 = 最后一次
      "phase": "initial",                     // initial | auth_rerun | post_fix
      "fix_commit": null, "result": "failed", "reason": "api_change",
      "exception": "ImportError", "message": "<Sanitized_Message>",
      "duration_s": 3.2, "exception_in_output": null,  // 仅超时时为布尔值
      "note": null, "credentials_set": false, "date": "2026-09-30"
    }],
    "control_kaiwu131": {"result": "passed", "exception": null},  // 无对照为 null
    "breakage_id": "B-001"                    // 无需登记为 null
  }],
  "static_imports": [{
    "id": "I-001", "path": "src/kaiwu/torch_plugin/x.py", "line": 7, "source": "py",  // py | readme
    "statement": "from kaiwu.cim import CIMOptimizer, PrecisionReducer",
    "module": "kaiwu.cim", "names": ["CIMOptimizer", "PrecisionReducer"],
    "scope": "sdk",                           // sdk | internal（kaiwu.torch_plugin.*）
    "resolved": false, "unresolved": ["PrecisionReducer"],
    "message": "<Sanitized_Message>", "breakage_id": "B-001"
  }],
  "counts": {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "xfailed": 0, "timeout": 0},
  "total": 0
}
```

校验规则（`schema.validate_regression`）：有效结果非 `passed` 当且仅当 `reason` 非空；`counts` 由有效结果计算，`sum(counts) == total == len(items)`；`internal` 导入对照 Fork_Clone 源码树解析（文件存在且名称为模块顶层定义），`sdk` 导入在 Upstream_Venv 解析。

### 2.3 breakage.json

```jsonc
{
  "schema": "ep-p0/breakage@1", "base_commit": "<40hex>", "count": 1, "none": false,
  "entries": [{
    "id": "B-001", "key": "api_change|kaiwu.cim:PrecisionReducer",   // 去重键，重建时保持编号
    "location": {"path": "src/kaiwu/torch_plugin/x.py", "line": 7},
    "symptom": {"exception": "ImportError", "message": "<Sanitized_Message>"},
    "reason": "api_change",
    "links": [{"type": "static_import", "id": "I-001"}, {"type": "regression", "id": "R-003"}],
    "replacement": "from kaiwu.cim import CIMOptimizer\nfrom kaiwu.preprocess import PrecisionReducer",
    "handling_mode": {"suggested": "fix_in_pr", "final": null},
    "confirmation": {"status": "pending", "reference": null, "date": null},
    "fix_commits": []
  }]
}
```

`links[].type` 取 `regression`（R-*）、`static_import`（I-*）、`env_conflict`（E-*，environment.json 的 `dependency_conflicts`）、`dependency_declaration`（D-*，status.json）、`ci`（C-*，status.json.ci）。无条目时 `count: 0, none: true`，MD 渲染为"无失效项"。

### 2.4 environment.json 与 status.json 要点

environment.json：`generated_at`（带时区 ISO 8601）、`venv`、`python`、`os`、`machine`、`pip`、`freeze[]`、`kaiwu_source {type: pypi|vendor_wheel, wheel, sha256}`、`version_checks[] {package, expected, actual, ok}`、`api_checks[] {api, importable, message}`、`dependency_conflicts[] {id, package, declared, pinned, source}`、`credentials_set`（布尔）。

status.json 分节：`repo`（fork_url、clone_path、base_commit、fork_main_head、work_branch、clone_date、structure_checks[]）、`dependency_declarations[]`、`ci {workflows[], tools[], latest_run, not_found}`、`ignore_checks[]`、`auth_patterns[] {pattern, source: sdk_docs|observed|prior_gitignore|unverified}`、`identity_checks[] {date, range, commits[{sha, author_ok, committer_ok}], result}`、`credential_scans[] {time, scope[], credentials_set, hits[], passed}`、`pushes[] {remote, branch, shas[], status: pending_confirmation|pushed|declined}`、`manual {mentor_record, public_email, cim, pr_strategy}`、`waits[] {item, party, requested_date, channel}`、`gates[]`、`p1_decision {value: go|defer|pending, date}`、`master_spec_suggestions[]`、`handover`。

门禁条目：`{"id": "G4", "status": "met", "evidence": ["records/ep-p0/regression.json"], "owner": "automation", "updated": "2026-09-30", "blocker": null}`；`blocker` 为 `{reason, owner, eta}`。`gates.evaluate()` 判定规则：

| 门禁 | 判定为 met 的条件 |
|---|---|
| G1 | clone_path 存在；base_commit 为 40 位十六进制；`git merge-base --is-ancestor base work_branch` 成立 |
| G2 | python 以 3.10 开头且 version_checks 全部 ok |
| G3 | base..work_branch 中存在仅改 `requirements/requirements.txt` 且该文件含 `kaiwu==1.4.1` 的提交 |
| G4 | `validate_regression` 通过 |
| G5 | breakage.json 存在，每个条目有 `handling_mode.suggested` 与 `confirmation.status` |
| G6 | `.gitignore`、`.env.example` 已在 work_branch 提交；ignore_checks 全部符合预期；最近一次 credential_scan passed |
| G7 | 最近一次 identity_check 覆盖当前 HEAD 且 result 为一致 |
| G8 | mentor_record.stable_reference 非空且 verified（文件存在，或用户确认 URL 可访问） |
| G9 | ci-inventory.md 存在 |
| S1–S3 | 对应 manual 字段已记录状态（记录即 met） |

等待外部方或 Manual_Operation 的 G 记为 `blocked`，其余不满足记为 `unmet`。全部 G 为 met 且 S 已记录时 `p1_decision = go`；`--today` ≥ 2026-10-05 且仍有 G 不为 met 时 `p1_decision = defer`，并要求每个未满足的 G 带 blocker。

## 执行流程

M 为 Manual_Operation，A 为自动化步骤。每个 M 由代理先说明目的、建议命令或页面入口与核对方法，用户报告完成后再执行后续 A。

| 步骤 | 内容 | 依赖 | 需求 |
|---|---|---|---|
| M-a | 提交 CIM 授权申请（代理提供信息清单：用途、42/84/126 Logical_Bit + ≤1 Auxiliary_Spin） [可并行等待] | 无 | 12 |
| M-b | 归档 Mentor_Confirmation_Record 到 Stable_Reference | 无 | 11 |
| M-c | 确认 GitHub 公开账号与公开邮箱 = OSPP_Git_Email | 无 | 8.5 |
| M-d | 向导师发出 PR_Strategy 确认请求 [可并行等待] | 无 | 9.4 |
| A0 | 建 `records/ep-p0/`、初始化 status.json；运行 `tools/ep_p0/tests` | 无 | 13.1 |
| M1 | 在 GitHub fork，提供 Fork_Repository URL | 无 | 2.1 |
| A1 | 校验 URL → 非空目录检查 → clone → 按 URL fetch 上游 main → Base_Commit → 创建 Work_Branch → 结构核对 | M1 | 2 |
| A2 | 依赖声明清单；CI 与 lint 配置只读盘点 | A1 | 4.3、10.1–10.3 |
| A3 | 字节级修改 requirements.txt（暂不提交） | A2 | 4.1–4.2 |
| A4 | 建 Upstream_Venv，安装 kaiwu（PyPI→vendor 回退）、其余依赖、测试依赖、声明的 lint 工具 | A3 | 3.1–3.6 |
| A5 | environment.json 与 6 个 API 可导入核对 | A4 | 3.7–3.10 |
| M2 | 用户设置 git 提交身份 | A1 | 8.2 |
| A6 | 只读核对 user.email；提交 C1（依赖升级） | A5、M2 | 4.5–4.6、8.1 |
| A7 | 登记 Regression_Item → 逐文件 pytest → 示例 → Static_Import_Check → 分类 | A6 | 5.1–5.8 |
| A8 | 1.3.1 对照（可安装时）[与 A7 并行] | A1 | 5.12 |
| A9 | 生成 Breakage_List | A7、A8 | 6.1–6.6 |
| A10 | `.gitignore`/`.env.example` 修改、check-ignore、提交 C2 | A6 | 7.1–7.5 |
| A11 | lint 基线、查询公开 CI 运行记录、ci-inventory.md | A4 | 10.4–10.7 |
| A12 | 为 `fix_in_pr` 条目准备独立修复提交 C3…，重跑关联项（post_fix） | A9 | 6.7–6.8 |
| M3 | （可选）用户在进程环境中设置 Credential_Variables；A13 重跑 auth 项 | A7 | 5.9 |
| A14 | 身份范围核对、Credential_Scan、渲染全部 MD 与 dep-pr-draft、门禁汇总 | A10–A12 | 7.8、8.3、9.6、13 |
| M4 | 每次推送、创建 PR 前的 User_Confirmation（非门禁） | A14 | 1.4–1.6、9 |

排期：09-29 完成 M-a～M-d 的发出、A0～A6；09-30 完成 A7～A11；10-01～10-05 期间 CIM 回复、导师对 PR_Strategy 与 Breakage 条目的确认、维护者回复均为可并行等待事项，只写入 `waits[]`，不阻塞 G1–G9（G8 只需用户归档已有书面确认）；A12～A14 在假期内完成，10-05 以 `gates --today 2026-10-05` 做最终判定。

## Components and Interfaces

入口：在工作区根执行 `python -m tools.ep_p0 <cmd>`。退出码：0 成功，1 检查未通过（扫描命中、身份不一致、校验失败），2 用法或环境错误。

| 子命令 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `snapshot` | `--venv`、`--source pypi\|vendor`、`--wheel` | environment.json | 子进程查询 `importlib.metadata.version`、`pip --version`、`pip freeze`；vendor 时计算 SHA-256；`--api-check` 核对 6 个 API |
| `decls` | `--clone` | status.json.dependency_declarations | 扫描 requirements*.txt、pyproject.toml、setup.py/.cfg、workflows、README 安装命令中的 kaiwu 版本声明 |
| `bump-req` | `--clone`、`--to 1.4.1` | 修改后的文件 | 按字节替换唯一的 kaiwu 行，保留该行行尾；缺文件或无 kaiwu 行时退出码 2 且不写 |
| `inventory` | `--clone`、`--venv` | `.p0-runs/inventory.json` | `pytest --collect-only -q` 得节点 ID，收集失败的文件登记为文件级；示例取 README 代码块中的 `python …` 命令，无 README 时取含 `__main__` 的脚本 |
| `run-tests` | `--limit 1800`、`--phase` | regression.json | 每个测试文件单独一次 pytest（`-v -p no:cacheprovider --junitxml`），超时杀进程树 |
| `run-examples` | `--limit 600`、`--phase` | regression.json | `MPLBACKEND=Agg`；notebook 用 `jupyter nbconvert --execute`，输出到 `.p0-runs/` |
| `junit2reg` | `--junit`、`--log`、`--phase` | regression.json | junit 为主；junit 缺失时从 `-v` 日志恢复已完成节点，未完成节点记 timeout |
| `imports` | `--clone`、`--venv` | regression.json.static_imports | AST 扫描 `.py`，README 取 python 代码块（解析失败退回正则），子进程解析 |
| `breakage` | regression/environment/status | breakage.json | 按'失效项判定与 Handling_Mode'规则构建，按 `key` 合并已有条目并保留确认字段与编号 |
| `scan-creds` | `--clone`、`--records`、`--range` | status.json.credential_scans | 见'凭据、git 安全与 Sanitized_Message' |
| `identity` | `--clone`、`--range`、`--email` | status.json.identity_checks | `git log --format=%H%x09%ae%x09%ce <range>` |
| `render` | `--records` | 全部 .md | 先在内存渲染并做凭据扫描，通过后再写盘；`render.parse_*` 供测试反解析 |
| `gates` | `--records`、`--today` | status.json.gates、p1_decision | 调用 `gates.evaluate()` 后重渲染 status-report.md |

Windows PowerShell 关键命令（工作区根执行）：

```powershell
$clone = "upstream\kaiwu-pytorch-plugin"; $py = "upstream\.venv\Scripts\python.exe"
if ((Test-Path $clone) -and (Get-ChildItem $clone -Force | Select-Object -First 1)) { throw "目标非空，停止克隆" }
git clone https://github.com/<user>/kaiwu-pytorch-plugin.git $clone
git -C $clone fetch https://github.com/qboson/kaiwu-pytorch-plugin.git main   # 不新增远端
$base = git -C $clone rev-parse FETCH_HEAD; git -C $clone rev-parse main
git -C $clone switch -c deps/kaiwu-1.4.1 $base
git -C $clone config --get user.email                                         # 只读
py -0p; py -3.10 -m venv upstream\.venv
& $py -m pip install --timeout 120 --retries 5 kaiwu==1.4.1
if ($LASTEXITCODE -ne 0) {
  & $py -m pip install vendor\kaiwu-1.4.1-cp310-none-win_amd64.whl
  (Get-FileHash vendor\kaiwu-1.4.1-cp310-none-win_amd64.whl -Algorithm SHA256).Hash
}
& $py -m pip install -r "$clone\requirements\requirements.txt" -c upstream\.p0-runs\constraints-kaiwu141.txt
& $py -m pip check
$env:MPLBACKEND = "Agg"; $env:PYTHONDONTWRITEBYTECODE = "1"
$p = Start-Process -FilePath (Resolve-Path $py) -WorkingDirectory $clone -PassThru -NoNewWindow `
  -ArgumentList "-m","pytest","tests/test_rbm.py","-v","-p","no:cacheprovider","--junitxml=..\.p0-runs\rbm.xml" `
  -RedirectStandardOutput upstream\.p0-runs\rbm.log
if (-not $p.WaitForExit(1800 * 1000)) { taskkill /PID $p.Id /T /F }
git -C $clone add -- requirements/requirements.txt
git -C $clone commit -F upstream\.p0-runs\msg-c1.txt                           # UTF-8 文件，含 "kaiwu 1.3.1 → 1.4.1"
git -C $clone status --porcelain --untracked-files=all                         # 5.14，仅列出不暂存
```

`constraints-kaiwu141.txt` 固定 `numpy==2.2.6`、`matplotlib==3.10.3`、`kaiwu-community==1.0.7`；若与上游声明冲突，pip 报错后按约束安装其余依赖，并在 environment.json 登记 E-* 冲突。未找到 `py -3.10` 时，可读取工作区根 `.venv\pyvenv.cfg` 的 `home` 定位 3.10 解释器。

## 凭据、git 安全与 Sanitized_Message

- 凭据只经 `os.environ` 读取，值只保存在进程内存；命令行参数、终端输出与记录中只出现 `credentials_set: true/false`。任何脚本都不打开 `.env`：扫描遇到名为 `.env` 的文件只按路径报告命中，不读内容。
- Credential_Scan：(a) `git ls-files -z` 与 `git diff --cached --name-only -z` 的路径按 basename 匹配 `.env` 与 Authorization_File_Pattern（fnmatch，大小写不敏感）；(b) 凭据已设置且非空时，逐字查找 `git diff <range>` 新增行、`git log --format=%B <range>`、`records/ep-p0/` 下全部文本文件（含 NUL 字节视为二进制，跳过）。只输出 `{path, line, kind}`，kind 为 `env_file`、`auth_file`、`credential_value`；命中即退出码 1，推送被阻止。
- `.gitignore` 追加带注释的规则块（`.env`、`*.lic` 与核实到的模式），跳过已存在的规则，沿用文件原有行尾；核对命令 `git check-ignore -v -- .env sample.lic`（期望忽略）与 `git check-ignore -- .env.example`（期望退出码 1）。
- git 规则：`gitro.py` 只放行 `config --get`、`config --show-origin --get`、`rev-parse`、`log`、`ls-files`、`diff`、`status`、`check-ignore`、`merge-base`；不执行 `git remote add/set-url`；暂存只用显式路径；不用 `--no-verify`、`--force`；推送只允许非 main 分支，首次推送 `git push -u origin <branch>`，推送前依次完成身份核对、Credential_Scan、User_Confirmation（远端、分支、`git log --format=%H <base>..HEAD` 列表），未确认的推送在 `pushes[]` 记为 `pending_confirmation`。8.4 的修正只针对未推送提交，经确认后执行 `git rebase <base> --exec "git commit --amend --no-edit --reset-author"`。
- 克隆 URL 校验：须匹配 `^https://github\.com/[\w.-]+/[\w.-]+?(\.git)?$`，含 `@` 或 userinfo 的 URL 拒绝。
- `sanitize(text, clone_root, workspace_root, home, secrets)`：取第一条非空行 → 凭据值替换为 `<REDACTED>` → HOME 前缀替换为 `<HOME>` → 工作区内绝对路径改为相对 Fork_Clone 根（如 `../.venv/Lib/site-packages/kaiwu/...`）→ 其余 Windows 绝对路径替换为 `<ABS>` → 统一 `/` 分隔并截断至 300 字符。路径匹配同时处理 `\` 与 `/` 两种写法，且大小写不敏感。

## 失效项判定与 Handling_Mode

原因分类按顺序取第一条命中规则（关键词表放在 `classify.py`，可扩展）：

| 顺序 | 条件 | reason |
|---|---|---|
| 1 | 消息或输出含授权相关词（license、授权、KAIWU_USER_ID、sdk code、未授权等），或示例需要 CIM 授权 | `auth` |
| 2 | 回溯含数据集下载帧（download、torchvision.datasets、urlopen 伴随数据路径） | `data_fetch` |
| 3 | ConnectionError、URLError、socket 超时、HTTP 429/5xx、quota/配额 | `network_quota` |
| 4 | 对照可用，对照结果非 passed 且异常类型相同 | `preexisting` |
| 5 | 涉及 kaiwu 的 ImportError/ModuleNotFoundError/AttributeError/签名 TypeError；静态导入不可解析 | `api_change` |
| 6 | 版本冲突、numpy ABI 错误（`_ARRAY_API not found`、dtype size changed）、E-*/D-*/C-* 证据 | `dep_conflict` |
| 7 | 其余（含无法归类的跳过、预期失败与超时） | `unknown` |

- 登记范围：reason 为 api_change、dep_conflict、preexisting、unknown 的非通过项，全部不可解析的 `sdk` 导入，以及 E-*、不兼容的 D-*、10.5 的 C-*。auth、network_quota、data_fetch 只记入 Regression_Record。
- 合并：去重键为 `reason|模块:名称`（导入类）或 `reason|异常类型|path:line`（其余）。消息中出现 `cannot import name 'X' from 'M'` 的回归项挂到对应导入条目，因此多个测试共享一个条目。每个待登记项恰好写入一个 `breakage_id`。
- 建议规则：api_change、dep_conflict → `fix_in_pr`；preexisting → `issue`；unknown → `pending_mentor`。`separate_pr` 只在导师调整后出现（`confirmation.status = adjusted`）。
- 已知项：`from kaiwu.cim import CIMOptimizer, PrecisionReducer` 登记为 api_change，替代写法为 `from kaiwu.cim import CIMOptimizer` 与 `from kaiwu.preprocess import PrecisionReducer`。对 `kaiwu.conversion` 的导入登记为 api_change，解析器在 `kaiwu.core`、`kaiwu.preprocess` 及 `pkgutil.walk_packages(kaiwu.__path__)` 的子模块中按同名属性查找替代位置（只导入、不调用），找到即写 `from <mod> import <name>`，否则写"未找到对应功能"。
- 修复提交每个条目一个，提交信息引用 `B-xxx`，记入 `fix_commits`。重跑结果以 `post_fix` 追加到 attempts，不覆盖首次结果。

## Error Handling

| 情形 | 处理 |
|---|---|
| 克隆或 fetch 失败 | 按 stderr 分类：`network`（Could not resolve host、timed out、SSL）、`access`（Repository not found、403、Authentication failed）、`other`；记录去凭据的命令与类别，G1 记 `blocked`（owner user），暂停 A2 之后的步骤 |
| 目标目录非空 | 停止，报告 `git -C … status -sb` 或目录项数量，不删除任何内容 |
| 缺 Python 3.10 | 停止创建 venv，原因写"Kaiwu_SDK 1.4.1 仅提供 cp310 wheel"，G2 记 `blocked`（owner user） |
| PyPI 与 vendor 安装都失败 | G2 记 `blocked`，保留 pip 错误的 Sanitized_Message |
| 依赖冲突 | 保留 kaiwu 固定版本，登记 E-*，进入 Breakage_List |
| pytest 单文件超过 1800 s 或示例超过 600 s | `taskkill /T /F` 结束进程树；已完成节点保留结果，运行中与未开始的节点记 timeout，写 `duration_s` 与 `exception_in_output` |
| 导入解析子进程崩溃或超过 120 s | 未完成的导入记 `resolved: false`、reason `unknown` |
| 扫描命中或身份不一致 | 退出码 1，阻止推送；身份不一致时在提交前停止 |
| 国庆期间外部方未回复 | 写入 `waits[]`，相关 S 或确认字段保持 `pending`，不阻塞 G 判定 |

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: 回归计数完整性

对任意 Regression_Item 集合（任意结果组合、任意 attempts 追加序列），`compute_counts` 得到的各类别计数之和等于条目总数，每个类别计数等于有效结果属于该类别的条目数；追加 auth_rerun 或 post_fix 后，此前的 attempts 保持不变。

**Validates: Requirements 5.2, 5.9, 5.11, 6.8**

### Property 2: 非通过项原因完整且"既有缺陷"只来自对照

对任意（异常类型、消息、1.4.1 结果、对照结果）组合，`classify` 对非通过结果返回枚举内的 reason，对通过结果返回空；返回 `preexisting` 时对照结果一定可用且非通过。

**Validates: Requirements 5.8, 5.12**

### Property 3: MD/JSON 一致（round-trip）

对任意通过校验的 regression.json 与 breakage.json，`parse_md(render_md(x))` 得到的逐项字段（ID、路径、节点 ID、结果、原因、条目位置、Handling_Mode、Confirmation_Status）与 x 相同；Dep_PR 草稿标题长度 ≤70 字符并包含全部必需小节。

**Validates: Requirements 5.13, 6.11, 9.6**

### Property 4: 失效项双向可追溯与重建幂等

对任意回归记录、静态导入结果与 E/D/C 证据，`build_breakage` 的输出中每个条目至少有一个 link；每个待登记项恰好被一个条目引用，且该项的 `breakage_id` 指向该条目；`handling_mode.suggested` 等于 reason 的建议规则映射；以既有 breakage.json 再次构建时，已有条目的编号与 confirmation 字段不变。

**Validates: Requirements 3.6, 4.4, 6.1, 6.5, 6.6, 6.9, 6.10**

### Property 5: Credential_Scan 与 Sanitized_Message 无泄漏

对任意文件集合与任意非空凭据值，扫描命中当且仅当某文件含该值或路径为 `.env`/授权文件模式；扫描报告、`sanitize` 输出与全部渲染记录中不含凭据值、HOME 绝对前缀或工作区绝对路径，且 `sanitize` 输出为单行。

**Validates: Requirements 5.8, 7.6, 7.7, 7.8, 7.9, 13.7**

### Property 6: 新增提交邮箱一致

对任意提交列表（SHA、作者邮箱、提交者邮箱）与 OSPP_Git_Email，核对结果为"一致"当且仅当每个提交的两个邮箱都等于 OSPP_Git_Email，且报告的不一致 SHA 集合恰好等于不满足该条件的提交集合。

**Validates: Requirements 8.3, 8.4, 8.6**

### Property 7: requirements.txt 单行替换

对任意由随机行与 LF/CRLF 行尾组成、恰含一条 kaiwu 依赖行的文件，`bump_req` 输出中只有该行变为 `kaiwu==1.4.1` 并保留原行尾，其余字节全部不变；不含 kaiwu 行时不写文件并报错。

**Validates: Requirements 4.1, 4.2**

### Property 8: 静态导入扫描完整

对任意由 kaiwu 与非 kaiwu 导入语句、其他语句随机组成的源码，AST 扫描返回的 `(行号, 模块, 名称)` 集合恰好等于注入的 kaiwu 导入集合。

**Validates: Requirements 5.7**

### Property 9: 门禁判定

对任意 G1–G9、S1–S3 状态组合与日期，`evaluate` 在全部 G 为 met 且 S 已记录时给出 `go`；日期 ≥ 2026-10-05 且存在非 met 的 G 时给出 `defer`，并且每个非 met 的 G 都带 blocker（reason、owner、eta）。

**Validates: Requirements 13.2, 13.3**

## Testing Strategy

- 位置 `tools/ep_p0/tests/`，用工作区根 `.venv` 运行：`.\.venv\Scripts\python.exe -m pytest tools/ep_p0/tests -q`。缺少时安装固定版本 `pytest==8.3.3`、`hypothesis==6.112.0`；已有则沿用现有版本。
- 属性测试（Hypothesis，`@settings(max_examples=100)`）逐条实现 Property 1–9，每个测试带注释 `# Feature: ep-p0-upstream-setup, Property N: <标题>`。生成器覆盖空集合、无失效项、CRLF、Windows 路径大小写与分隔符混用、中文消息、短凭据值等边界。
- 单元测试（示例）：collect-only 输出与收集失败的解析；junit 中 collection error 的映射；超时日志恢复；stderr 错误分类；PrecisionReducer 与 kaiwu.conversion 两个已知项；`gitro` 拒绝写入类子命令；克隆 URL 校验；`.gitignore` 追加保留既有规则；`.env.example` 只含两个空值条目；未确认推送记为 `pending_confirmation`。
- 隔离：不访问网络、不导入真实 kaiwu、不需要授权。需要 git 的测试在 `tmp_path` 里 `git init`，身份通过 `GIT_AUTHOR_*`/`GIT_COMMITTER_*` 环境变量提供，并设置 `GIT_CONFIG_NOSYSTEM=1`、`GIT_CONFIG_GLOBAL` 指向临时空文件，避免任何 Git_Config_Write。导入解析用临时目录中的伪 `kaiwu` 包，以当前解释器作为 `--venv` 替身。凭据用 `monkeypatch.setenv` 注入随机值。
- 真实克隆、安装、回归与推送属于集成步骤，由'执行流程'流程执行，不进入单元测试。

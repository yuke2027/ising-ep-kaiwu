"""status.json 的初始化与读写（``python -m tools.ep_p0 init``）。

对应 design.md "Data Models 2.4"。status.json 是门禁、确认事项、依赖声明清单、
CI 数据、身份核对、扫描与推送记录的结构化源；Markdown 由 ``render`` 生成。

门禁初值的表示：

- 硬门禁 G1–G9：``status = "unmet"``，``updated`` 为初始化日期；
- 状态门禁 S1–S3：``status = null`` 表示"未记录"（``NOT_RECORDED_LABEL``），
  ``updated = null``；用户报告对应 manual 字段后记为 ``"met"``（记录即满足）。

每个门禁条目形如
``{"id", "kind": "hard"|"status", "status", "evidence": [], "owner", "updated", "blocker"}``，
``blocker`` 为 ``null`` 或 ``{reason, owner, eta}``。

``save()`` 写盘规则：

1. 凭据扫描：任一字符串（含键名）包含进程环境中的凭据值即拒写，
   抛 ``CredentialLeakError``（命令行退出码 1）；报告只给出 JSON 位置，不给出值；
2. 路径：整串为工作区内绝对路径的值改写为相对工作区根的 ``/`` 分隔路径；
   其余含 Windows 绝对路径的字符串拒写，抛 ``AbsolutePathError``（退出码 1）；
3. UTF-8、``sort_keys=True``、缩进 2、末尾换行；先写同目录临时文件再 ``os.replace``。

本模块只用标准库，不打开 ``.env``。
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import EXIT_CHECK_FAILED, EXIT_OK, EXIT_USAGE
from .sanitize import credential_values, to_record_path

STATUS_SCHEMA = "ep-p0/status@1"
STATUS_FILE = "status.json"
DEFAULT_RECORDS = "records/ep-p0"
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

NOT_RECORDED_LABEL = "未记录"

# 门禁：id → 初始负责方
HARD_GATES: dict[str, str] = {
    "G1": "automation",   # Fork_Clone、Base_Commit、Work_Branch
    "G2": "automation",   # Upstream_Venv 与版本核对
    "G3": "automation",   # 依赖升级提交
    "G4": "automation",   # Regression_Record 校验
    "G5": "automation",   # Breakage_List
    "G6": "automation",   # .gitignore / .env.example 与 Credential_Scan
    "G7": "automation",   # 提交身份核对
    "G8": "user",         # Mentor_Confirmation_Record 归档
    "G9": "automation",   # CI_Inventory
}
STATUS_GATES: dict[str, str] = {
    "S1": "user",         # CIM_Authorization_Status
    "S2": "mentor",       # PR_Strategy 与 Confirmation_Status
    "S3": "user",         # GitHub 公开账号与公开邮箱
}

# 需求 9.3 的默认增量 PR_Strategy
DEFAULT_PR_STRATEGY_TEXT = (
    "Dep_PR 承载依赖升级、Regression_Record 摘要、Handling_Mode 为“本 PR 修复”的修复"
    "以及 .gitignore/.env.example 变更；P1 结束时另提交能量/QUBO PR。"
)
DEFAULT_PR_STRATEGY_PRS = [
    {
        "name": "Dep_PR",
        "when": "P0",
        "scope": [
            "依赖升级（kaiwu 1.3.1 → 1.4.1）",
            "Regression_Record 摘要",
            "Handling_Mode 为 fix_in_pr 的修复",
            ".gitignore/.env.example 变更",
        ],
    },
    {"name": "能量/QUBO PR", "when": "P1 结束时", "scope": ["能量/QUBO 实现"]},
]

# 整串或片段中的 Windows 绝对路径（盘符或 UNC）
_ABS_FULL = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/])\S*$")
_ABS_ANY = re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/])")


class CredentialLeakError(ValueError):
    """待写内容包含凭据值；``locations`` 为 JSON 位置列表（不含值）。"""

    def __init__(self, locations: list[str]):
        self.locations = locations
        super().__init__("status.json 含凭据值，拒绝写入：" + ", ".join(locations))


class AbsolutePathError(ValueError):
    """待写内容含无法改写为相对路径的绝对路径（13.7）。"""

    def __init__(self, locations: list[str]):
        self.locations = locations
        super().__init__("status.json 含绝对路径，拒绝写入：" + ", ".join(locations))


# ---------------------------------------------------------------------------
# 空结构
# ---------------------------------------------------------------------------

def _gate(gate_id: str, kind: str, owner: str, today: str) -> dict[str, Any]:
    hard = kind == "hard"
    return {
        "id": gate_id,
        "kind": kind,
        "status": "unmet" if hard else None,
        "evidence": [],
        "owner": owner,
        "updated": today if hard else None,
        "blocker": None,
    }


def empty_status(today: str | None = None) -> dict[str, Any]:
    """按 design.md 2.4 分节返回初始 status.json 内容。"""
    day = today or _dt.date.today().isoformat()
    gates = [_gate(g, "hard", o, day) for g, o in HARD_GATES.items()]
    gates += [_gate(s, "status", o, day) for s, o in STATUS_GATES.items()]
    return {
        "schema": STATUS_SCHEMA,
        "created": day,
        "repo": {
            "fork_url": None,
            "clone_path": None,
            "base_commit": None,
            "fork_main_head": None,
            "work_branch": None,
            "clone_date": None,
            "structure_checks": [],
        },
        "dependency_declarations": [],
        "ci": {"workflows": [], "tools": [], "latest_run": None, "not_found": None},
        "ignore_checks": [],
        "auth_patterns": [],
        "identity_checks": [],
        "credential_scans": [],
        "pushes": [],
        "manual": {
            "mentor_record": {
                "stable_reference": None,
                "reference_type": None,     # file | ospp | url
                "date": None,
                "role": None,
                "scope": None,
                "verified": False,
                "verification": None,
            },
            "public_email": {"result": None, "date": None},
            "cim": {
                "status": "not_applied",
                "applied_date": None,
                "channel": None,
                "quota": None,
                "not_applied_reason": None,
                "planned_date": None,
                "updated": None,
            },
            "pr_strategy": {
                "strategy": "incremental",
                "description": DEFAULT_PR_STRATEGY_TEXT,
                "prs": copy.deepcopy(DEFAULT_PR_STRATEGY_PRS),
                "confirmation": "pending",
                "reference": None,
                "date": None,
                "adjustments": [],          # [{strategy, date, affected_phases[]}]
            },
        },
        "waits": [],
        "gates": gates,
        "p1_decision": {"value": "pending", "date": None},
        "master_spec_suggestions": [],
        "handover": {
            "clone_path": None,
            "work_branch": None,
            "head_sha": None,
            "venv_path": None,
            "activate_command": None,
            "api_checks": [],
            "lint_commands": [],
        },
    }


def gate_status_label(status: str | None) -> str:
    """门禁状态的中文标签；``None`` 为"未记录"。"""
    from .schema import label

    return NOT_RECORDED_LABEL if status is None else label("gate_status", status)


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------

def status_path(records: str | os.PathLike[str] = DEFAULT_RECORDS) -> Path:
    return Path(records) / STATUS_FILE


def load(records: str | os.PathLike[str] = DEFAULT_RECORDS) -> dict[str, Any]:
    """读取 status.json；文件缺失抛 FileNotFoundError，schema 不符抛 ValueError。"""
    path = status_path(records)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or data.get("schema") != STATUS_SCHEMA:
        raise ValueError(f"{STATUS_FILE} 的 schema 应为 {STATUS_SCHEMA!r}")
    return data


def _walk_strings(obj: Any, where: str = "$"):
    """产出 (JSON 位置, 字符串)；键名与值都参与。"""
    if isinstance(obj, str):
        yield where, obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            if isinstance(key, str):
                yield f"{where}.{key}<key>", key
            yield from _walk_strings(value, f"{where}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from _walk_strings(value, f"{where}[{index}]")


def find_credentials(obj: Any, secrets: list[str] | None = None) -> list[str]:
    """返回含凭据值的 JSON 位置列表（不含值本身）。"""
    values = [s for s in (credential_values() if secrets is None else secrets) if s]
    if not values:
        return []
    return [where for where, text in _walk_strings(obj) if any(v in text for v in values)]


def relativize_paths(
    obj: Any, workspace_root: str | os.PathLike[str] = WORKSPACE_ROOT
) -> tuple[Any, list[str]]:
    """返回 (改写后的副本, 无法改写的 JSON 位置列表)。

    整串为工作区内绝对路径的值改为相对工作区根；其余含绝对路径的字符串记为错误。
    """
    errors: list[str] = []

    def fix(value: Any, where: str) -> Any:
        if isinstance(value, str):
            if _ABS_FULL.match(value):
                try:
                    return to_record_path(value, workspace_root)
                except ValueError:
                    errors.append(where)
                    return value
            if _ABS_ANY.search(value):
                errors.append(where)
            return value
        if isinstance(value, Mapping):
            return {k: fix(v, f"{where}.{k}") for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [fix(v, f"{where}[{i}]") for i, v in enumerate(value)]
        return value

    return fix(obj, "$"), errors


def dumps(data: Mapping[str, Any]) -> str:
    """确定性序列化：键排序、缩进 2、保留中文、末尾换行。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def save(
    data: Mapping[str, Any],
    records: str | os.PathLike[str] = DEFAULT_RECORDS,
    *,
    workspace_root: str | os.PathLike[str] = WORKSPACE_ROOT,
    secrets: list[str] | None = None,
) -> Path:
    """扫描凭据、改写路径后原子写入 status.json，返回文件路径。

    ``secrets`` 为 None 时取进程环境中的凭据值。
    命中凭据抛 ``CredentialLeakError``，含绝对路径抛 ``AbsolutePathError``；两种情况都不写盘。
    """
    hits = find_credentials(data, secrets)
    if hits:
        raise CredentialLeakError(hits)
    fixed, bad = relativize_paths(data, workspace_root)
    if bad:
        raise AbsolutePathError(bad)
    text = dumps(fixed)

    path = status_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".status.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return path


def init(
    records: str | os.PathLike[str] = DEFAULT_RECORDS,
    today: str | None = None,
    **save_kwargs: Any,
) -> tuple[Path, bool]:
    """建 P0_Record_Dir 与初始 status.json；已存在时不覆盖。返回 (路径, 是否新建)。"""
    path = status_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path, False
    return save(empty_status(today), records, **save_kwargs), True


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.ep_p0 init",
        description="建 P0_Record_Dir 并初始化 status.json（已存在时不覆盖）。",
    )
    parser.add_argument(
        "--records", default=DEFAULT_RECORDS, help=f"P0_Record_Dir（默认 {DEFAULT_RECORDS}）"
    )
    args = parser.parse_args(argv)

    try:
        path, created = init(args.records)
    except (CredentialLeakError, AbsolutePathError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_CHECK_FAILED
    except OSError as exc:
        print(f"无法写入 {STATUS_FILE}：{exc.strerror or exc}", file=sys.stderr)
        return EXIT_USAGE

    shown = Path(args.records).as_posix().rstrip("/") + "/" + STATUS_FILE
    if created:
        print(f"已创建 {shown}")
    else:
        print(f"{shown} 已存在，未覆盖")
    return EXIT_OK


__all__ = [
    "STATUS_SCHEMA",
    "DEFAULT_RECORDS",
    "HARD_GATES",
    "STATUS_GATES",
    "NOT_RECORDED_LABEL",
    "DEFAULT_PR_STRATEGY_TEXT",
    "CredentialLeakError",
    "AbsolutePathError",
    "empty_status",
    "gate_status_label",
    "status_path",
    "load",
    "save",
    "init",
    "find_credentials",
    "relativize_paths",
    "dumps",
    "main",
]

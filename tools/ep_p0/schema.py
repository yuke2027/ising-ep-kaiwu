"""P0 记录的数据模式：枚举（JSON 值 → 中文标签）、回归计数与 regression.json 校验。

对应 design.md "Data Models 2.1 / 2.2"。只用标准库。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

# ---------------------------------------------------------------------------
# 2.1 枚举：dict 保持声明顺序，键为 JSON 值，值为中文标签
# ---------------------------------------------------------------------------

RESULT: dict[str, str] = {
    "passed": "通过",
    "failed": "失败",
    "error": "错误",
    "skipped": "跳过",
    "xfailed": "预期失败",
    "timeout": "超时",
}

REASON: dict[str, str] = {
    "api_change": "API 变更",
    "dep_conflict": "依赖冲突",
    "auth": "授权",
    "network_quota": "网络或配额",
    "data_fetch": "数据获取",
    "preexisting": "既有缺陷",
    "unknown": "未知",
}

HANDLING_MODE: dict[str, str] = {
    "fix_in_pr": "本 PR 修复",
    "separate_pr": "单独小 PR",
    "issue": "提交 issue 交由维护者处理",
    "pending_mentor": "待导师确认",
}

CONFIRMATION: dict[str, str] = {
    "pending": "待确认",
    "mentor_confirmed": "导师已确认",
    "maintainer_confirmed": "维护者已确认",
    "adjusted": "已调整",
}

GATE_STATUS: dict[str, str] = {
    "met": "满足",
    "unmet": "未满足",
    "blocked": "阻塞",
}

OWNER: dict[str, str] = {
    "user": "用户",
    "mentor": "导师",
    "maintainer": "维护者",
    "kaiwu_platform": "Kaiwu 平台",
    "automation": "Setup_Automation",
}

CIM_STATUS: dict[str, str] = {
    "not_applied": "未申请",
    "applied_pending": "已申请待回复",
    "approved": "已获批",
    "rejected": "被拒绝",
}

ENUMS: dict[str, dict[str, str]] = {
    "result": RESULT,
    "reason": REASON,
    "handling_mode": HANDLING_MODE,
    "confirmation": CONFIRMATION,
    "gate_status": GATE_STATUS,
    "owner": OWNER,
    "cim_status": CIM_STATUS,
}

# 2.2 / 2.3 中出现的其他取值集合
ITEM_KINDS = ("test", "example")
ATTEMPT_PHASES = ("initial", "auth_rerun", "post_fix")
IMPORT_SOURCES = ("py", "readme")
IMPORT_SCOPES = ("sdk", "internal")
LINK_TYPES = ("regression", "static_import", "env_conflict", "dependency_declaration", "ci")

REGRESSION_SCHEMA = "ep-p0/regression@1"
BREAKAGE_SCHEMA = "ep-p0/breakage@1"

_SHA40 = re.compile(r"[0-9a-f]{40}")


def label(enum: str, value: str | None) -> str:
    """返回枚举值的中文标签；未知枚举名或取值抛 KeyError。"""
    if value is None:
        return ""
    return ENUMS[enum][value]


def value_of(enum: str, text: str) -> str:
    """按 JSON 值或中文标签反查 JSON 值（供 render.parse_* 使用）。"""
    table = ENUMS[enum]
    if text in table:
        return text
    for key, lab in table.items():
        if lab == text:
            return key
    raise KeyError(f"{enum}: {text!r}")


# ---------------------------------------------------------------------------
# 有效结果与计数
# ---------------------------------------------------------------------------

def effective_attempt(item: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """有效结果 = attempts 的最后一项；无 attempts 返回 None。"""
    attempts = item.get("attempts") or []
    return attempts[-1] if attempts else None


def effective_result(item: Mapping[str, Any]) -> str | None:
    attempt = effective_attempt(item)
    return None if attempt is None else attempt.get("result")


def compute_counts(items: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """按有效结果计数，键为全部 result 取值（缺省为 0）。

    无 attempts 或结果不在枚举内的条目不计入任何类别，从而
    ``sum(counts) != len(items)``，由 ``validate_regression`` 报出。
    """
    counts = {key: 0 for key in RESULT}
    for item in items:
        result = effective_result(item)
        if result in counts:
            counts[result] += 1
    return counts


# ---------------------------------------------------------------------------
# regression.json 校验
# ---------------------------------------------------------------------------

def _is_sha40(value: Any) -> bool:
    return isinstance(value, str) and _SHA40.fullmatch(value) is not None


def _validate_attempt(prefix: str, attempt: Any, errors: list[str]) -> None:
    if not isinstance(attempt, Mapping):
        errors.append(f"{prefix}: attempt 不是对象")
        return
    result = attempt.get("result")
    reason = attempt.get("reason")
    if result not in RESULT:
        errors.append(f"{prefix}: result 非法 {result!r}")
    if reason is not None and reason not in REASON:
        errors.append(f"{prefix}: reason 非法 {reason!r}")
    phase = attempt.get("phase")
    if phase not in ATTEMPT_PHASES:
        errors.append(f"{prefix}: phase 非法 {phase!r}")
    if result in RESULT:
        exc_flag = attempt.get("exception_in_output")
        if result == "timeout" and not isinstance(exc_flag, bool):
            errors.append(f"{prefix}: 超时结果须以布尔值记录 exception_in_output")
        if result != "timeout" and exc_flag is not None:
            errors.append(f"{prefix}: exception_in_output 仅超时结果可为布尔值")


def validate_regression(data: Mapping[str, Any]) -> list[str]:
    """校验 regression.json，返回错误列表（空列表表示通过）。

    规则（design.md 2.2）：
    - 有效结果（最后一次 attempt）非 ``passed`` 当且仅当 ``reason`` 非空；
    - ``counts`` 等于由有效结果计算的计数，且 ``sum(counts) == total == len(items)``；
    - 另做结构检查：schema、40 位 SHA、条目 ID 唯一、枚举取值合法。
    """
    errors: list[str] = []
    if not isinstance(data, Mapping):
        return ["regression: 顶层不是对象"]

    if data.get("schema") != REGRESSION_SCHEMA:
        errors.append(f"schema 应为 {REGRESSION_SCHEMA!r}")
    for key in ("base_commit", "tested_commit"):
        if not _is_sha40(data.get(key)):
            errors.append(f"{key} 不是 40 位小写十六进制 SHA")
    if not isinstance(data.get("credentials_set"), bool):
        errors.append("credentials_set 须为布尔值")

    items = data.get("items")
    if not isinstance(items, list):
        errors.append("items 须为列表")
        items = []

    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            errors.append(f"items[{index}]: 不是对象")
            continue
        item_id = item.get("id") or f"items[{index}]"
        if item_id in seen:
            errors.append(f"{item_id}: ID 重复")
        seen.add(item_id)
        if item.get("kind") not in ITEM_KINDS:
            errors.append(f"{item_id}: kind 非法 {item.get('kind')!r}")
        if not item.get("path"):
            errors.append(f"{item_id}: 缺少 path")

        attempts = item.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            errors.append(f"{item_id}: 缺少结果（attempts 为空）")
            continue
        for n, attempt in enumerate(attempts):
            _validate_attempt(f"{item_id}.attempts[{n}]", attempt, errors)

        last = attempts[-1]
        if isinstance(last, Mapping):
            result, reason = last.get("result"), last.get("reason")
            if result == "passed" and reason:
                errors.append(f"{item_id}: 通过结果不应有 reason")
            if result in RESULT and result != "passed" and not reason:
                errors.append(f"{item_id}: 非通过结果缺少 reason")

    for index, imp in enumerate(data.get("static_imports") or []):
        imp_id = imp.get("id") if isinstance(imp, Mapping) else None
        prefix = imp_id or f"static_imports[{index}]"
        if not isinstance(imp, Mapping):
            errors.append(f"{prefix}: 不是对象")
            continue
        if imp.get("source") not in IMPORT_SOURCES:
            errors.append(f"{prefix}: source 非法 {imp.get('source')!r}")
        if imp.get("scope") not in IMPORT_SCOPES:
            errors.append(f"{prefix}: scope 非法 {imp.get('scope')!r}")
        if not isinstance(imp.get("resolved"), bool):
            errors.append(f"{prefix}: resolved 须为布尔值")

    expected = compute_counts(i for i in items if isinstance(i, Mapping))
    counts = data.get("counts")
    if not isinstance(counts, Mapping):
        errors.append("counts 须为对象")
    else:
        if set(counts) != set(RESULT):
            errors.append(f"counts 键应为 {sorted(RESULT)}")
        if {k: counts.get(k) for k in RESULT} != expected:
            errors.append(f"counts 与有效结果不一致：记录 {dict(counts)}，应为 {expected}")

    total = data.get("total")
    if sum(expected.values()) != len(items):
        errors.append(f"有效结果计数之和 {sum(expected.values())} ≠ 条目数 {len(items)}")
    if total != len(items):
        errors.append(f"total {total!r} ≠ 条目数 {len(items)}")
    return errors


def is_valid_regression(data: Mapping[str, Any]) -> bool:
    return not validate_regression(data)

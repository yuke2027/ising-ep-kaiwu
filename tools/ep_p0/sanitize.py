"""Sanitized_Message 与凭据环境工具。

``sanitize()`` 处理顺序（design.md "凭据、git 安全与 Sanitized_Message"）：

1. 取第一条非空行；
2. 凭据值替换为 ``<REDACTED>``；
3. HOME 前缀替换为 ``<HOME>``；
4. 工作区内绝对路径改为相对 Fork_Clone 根的路径（如 ``../.venv/Lib/...``）；
5. 其余 Windows 绝对路径替换为 ``<ABS>``；
6. 统一 ``/`` 分隔，截断至 300 字符。

路径匹配同时接受 ``\\`` 与 ``/``（含转义产生的 ``\\\\``），且大小写不敏感。
按上述顺序，若工作区位于 HOME 之下，工作区路径会先被替换为 ``<HOME>/...``。
凭据只经 ``os.environ`` 读取；本模块不打开 ``.env``。
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping

CREDENTIAL_VARIABLES = ("KAIWU_USER_ID", "KAIWU_SDK_CODE")
MAX_LEN = 300
REDACTED = "<REDACTED>"
HOME_TOKEN = "<HOME>"
ABS_TOKEN = "<ABS>"

_SEP = r"[\\/]+"
# 路径尾部的一个组成部分：不含空白、引号、分隔符与 Windows 非法字符
_PART = r"[^\\/\s'\"`<>|*?:,;()\[\]{}]+"
_TAIL = rf"(?P<tail>(?:{_SEP}{_PART})*)(?P<trail>{_SEP})?"
# 其余 Windows 绝对路径：盘符路径或 UNC 路径
_WIN_ABS = re.compile(
    rf"(?<![\w.\-])(?:[A-Za-z]:{_SEP}|\\\\{_PART}\\+)(?:{_PART})?(?:{_SEP}{_PART})*(?:{_SEP})?"
)


# ---------------------------------------------------------------------------
# 凭据环境
# ---------------------------------------------------------------------------

def credential_values(environ: Mapping[str, str] | None = None) -> list[str]:
    """返回进程环境中非空的凭据值（只在内存中使用，不得输出）。"""
    env = os.environ if environ is None else environ
    values = []
    for name in CREDENTIAL_VARIABLES:
        value = env.get(name)
        if value:
            values.append(value)
    return values


def credentials_set(environ: Mapping[str, str] | None = None) -> bool:
    """两个 Credential_Variables 均已设置且非空时为 True（记录中只写此布尔值）。"""
    env = os.environ if environ is None else environ
    return all(env.get(name) for name in CREDENTIAL_VARIABLES)


def scrubbed_env(
    base: Mapping[str, str] | None = None, **overrides: str
) -> dict[str, str]:
    """返回剔除 Credential_Variables 的环境副本，供子进程使用。

    键名比较大小写不敏感（Windows 环境变量不区分大小写）。
    ``overrides`` 追加或覆盖变量，但不能重新放入凭据变量。
    """
    source = os.environ if base is None else base
    blocked = {name.upper() for name in CREDENTIAL_VARIABLES}
    env = {k: v for k, v in source.items() if k.upper() not in blocked}
    for key, value in overrides.items():
        if key.upper() in blocked:
            raise ValueError(f"不允许向子进程环境写入 {key}")
        env[key] = value
    return env


# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------

def _parts(path: str | os.PathLike[str]) -> list[str]:
    """把路径拆成组成部分；同时接受 \\ 与 /，去掉空段与 '.'。"""
    text = os.fspath(path)
    return [p for p in re.split(r"[\\/]+", text) if p and p != "."]


def _root_pattern(root: str | os.PathLike[str]) -> re.Pattern[str] | None:
    parts = _parts(root)
    if not parts:
        return None
    text = os.fspath(root)
    lead = ""
    if re.match(r"[\\/]{2}", text):          # UNC：\\server\share
        lead = r"[\\/]{2}"
    elif re.match(r"[\\/]", text):           # POSIX 绝对路径
        lead = _SEP
    body = _SEP.join(re.escape(p) for p in parts)
    # 根之后须为路径尾部或非路径字符，避免 E:\foo 误配 E:\foobar
    return re.compile(
        rf"(?<![\w.\-]){lead}{body}(?![^\\/\s'\"`<>|*?:,;()\[\]{{}}]){_TAIL}",
        re.IGNORECASE,
    )


def _lower(parts: list[str]) -> list[str]:
    return [p.casefold() for p in parts]


def _relative(target: list[str], base: list[str]) -> str:
    """target、base 均为相对同一根的组成部分，返回 base → target 的相对路径。"""
    t, b = _lower(target), _lower(base)
    k = 0
    while k < len(t) and k < len(b) and t[k] == b[k]:
        k += 1
    rel = [".."] * (len(base) - k) + target[k:]
    return "/".join(rel) if rel else "."


def _clone_rel(clone_root, workspace_root) -> list[str]:
    """Fork_Clone 根相对工作区根的组成部分；不在工作区内时返回空（相对工作区根）。"""
    if clone_root is None:
        return []
    ws, cl = _parts(workspace_root), _parts(clone_root)
    if len(cl) >= len(ws) and _lower(cl[: len(ws)]) == _lower(ws):
        return cl[len(ws):]
    return []


# ---------------------------------------------------------------------------
# sanitize
# ---------------------------------------------------------------------------

def first_line(text: object) -> str:
    if text is None:
        return ""
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()
    return ""


def sanitize(
    text: object,
    clone_root: str | os.PathLike[str] | None = None,
    workspace_root: str | os.PathLike[str] | None = None,
    home: str | os.PathLike[str] | None = None,
    secrets: Iterable[str] | None = None,
) -> str:
    """生成单行 Sanitized_Message。

    - ``secrets`` 为 None 时取进程环境中的凭据值；传空列表表示不替换。
    - ``home`` 为 None 时取 ``os.path.expanduser("~")``。
    - ``workspace_root`` 为 None 时跳过第 4 步；``clone_root`` 为 None 或不在
      工作区内时，工作区路径改为相对工作区根。
    """
    secret_list = sorted(
        {s for s in (credential_values() if secrets is None else secrets) if s},
        key=len,
        reverse=True,
    )
    home_root = os.path.expanduser("~") if home is None else home

    # 1. 第一条非空行
    out = first_line(text)

    # 2. 凭据值
    for secret in secret_list:
        out = out.replace(secret, REDACTED)

    # 3. HOME 前缀
    home_re = _root_pattern(home_root) if home_root else None
    if home_re is not None:
        def _home(m: re.Match[str]) -> str:
            tail = _parts(m.group("tail") or "")
            return "/".join([HOME_TOKEN, *tail])
        out = home_re.sub(_home, out)

    # 4. 工作区内绝对路径 → 相对 Fork_Clone 根
    if workspace_root is not None:
        ws_re = _root_pattern(workspace_root)
        base = _clone_rel(clone_root, workspace_root)
        if ws_re is not None:
            def _ws(m: re.Match[str]) -> str:
                return _relative(_parts(m.group("tail") or ""), base)
            out = ws_re.sub(_ws, out)

    # 5. 其余 Windows 绝对路径
    out = _WIN_ABS.sub(ABS_TOKEN, out)

    # 6. 统一分隔符
    out = out.replace("\\", "/")

    # 占位符或相对路径可能意外拼出短凭据值：反复删除直到不再出现
    changed = True
    while changed and secret_list:
        changed = False
        for secret in secret_list:
            if secret in out:
                out = out.replace(secret, "")
                changed = True

    return out[:MAX_LEN]


def to_record_path(
    path: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str],
    clone_root: str | os.PathLike[str] | None = None,
) -> str:
    """把绝对路径转成记录用的相对路径（13.7）。

    位于 Fork_Clone 内时相对 Fork_Clone 根，否则相对工作区根；两者都不在时抛 ValueError。
    """
    target = _parts(path)
    for root in (clone_root, workspace_root):
        if root is None:
            continue
        base = _parts(root)
        if len(target) >= len(base) and _lower(target[: len(base)]) == _lower(base):
            rest = target[len(base):]
            return "/".join(rest) if rest else "."
    raise ValueError("路径不在工作区内")


__all__ = [
    "CREDENTIAL_VARIABLES",
    "credential_values",
    "credentials_set",
    "scrubbed_env",
    "first_line",
    "sanitize",
    "to_record_path",
]

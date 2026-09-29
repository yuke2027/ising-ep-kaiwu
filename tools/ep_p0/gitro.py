"""只读 git 包装：只放行白名单子命令，其余一律抛 ``GitReadOnlyError``。

白名单（design.md）：``config --get``、``config --show-origin --get``、``rev-parse``、
``log``、``ls-files``、``diff``、``status``、``check-ignore``、``merge-base``。

- ``args[0]`` 必须是子命令本身，不接受 ``-c``/``--git-dir`` 等全局选项；
  仓库位置经 ``repo`` 参数以 ``git -C <repo>`` 传入。
- ``config`` 只接受 ``["config", "--get", <key>]`` 与
  ``["config", "--show-origin", "--get", <key>]`` 两种形式。
- 拒绝会写文件或执行外部程序的选项（``--output``、``--ext-diff``、``--textconv`` 等）。
- 子进程环境剔除 Credential_Variables，并设 ``GIT_TERMINAL_PROMPT=0``、
  ``GIT_OPTIONAL_LOCKS=0``（``status`` 不回写索引）。
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence

from .sanitize import scrubbed_env

READONLY_SUBCOMMANDS = frozenset(
    {"config", "rev-parse", "log", "ls-files", "diff", "status", "check-ignore", "merge-base"}
)
# 在白名单子命令中仍会写文件或调用外部程序的选项（按前缀匹配）
_FORBIDDEN_OPTIONS = ("--output", "--ext-diff", "--textconv")
# 与禁用选项前缀重合、但本身是独立只读选项的精确名称
_SAFE_EXACT = frozenset({"--text"})
_CONFIG_KEY = re.compile(r"[A-Za-z0-9][\w.-]*")
DEFAULT_TIMEOUT = 120


class GitReadOnlyError(ValueError):
    """请求的 git 调用不在只读白名单内。"""


def check_args(args: Sequence[str]) -> None:
    """校验参数列表；不在白名单内时抛 ``GitReadOnlyError``。"""
    if isinstance(args, (str, bytes)) or not args:
        raise GitReadOnlyError("参数须为非空的字符串列表")
    if not all(isinstance(a, str) for a in args):
        raise GitReadOnlyError("参数须全部为字符串")
    if any("\x00" in a or "\n" in a for a in args):
        raise GitReadOnlyError("参数含控制字符")

    sub, rest = args[0], list(args[1:])
    if sub not in READONLY_SUBCOMMANDS:
        raise GitReadOnlyError(f"git {sub} 不在只读白名单内")

    if sub == "config":
        if rest[:1] == ["--get"] and len(rest) == 2:
            key = rest[1]
        elif rest[:2] == ["--show-origin", "--get"] and len(rest) == 3:
            key = rest[2]
        else:
            raise GitReadOnlyError("git config 只允许 --get <key> 或 --show-origin --get <key>")
        if not _CONFIG_KEY.fullmatch(key):
            raise GitReadOnlyError("git config 键名非法")
        return

    for arg in rest:
        if arg == "--":
            break
        name = arg.split("=", 1)[0].lower()
        if not name.startswith("--") or name in _SAFE_EXACT:
            continue
        # git 接受无歧义的长选项缩写（如 --out=），因此按"是禁用选项的前缀"判定
        if len(name) > 2 and any(opt.startswith(name) for opt in _FORBIDDEN_OPTIONS):
            raise GitReadOnlyError(f"选项 {name} 不允许")


def is_allowed(args: Sequence[str]) -> bool:
    try:
        check_args(args)
    except GitReadOnlyError:
        return False
    return True


def run(
    args: Sequence[str],
    repo: str | os.PathLike[str] | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    check: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """执行只读 git 命令，返回 ``CompletedProcess``（stdout/stderr 已捕获）。

    ``text=True`` 时按 UTF-8 解码（非法字节替换）；``check=True`` 时非零退出抛
    ``subprocess.CalledProcessError``。
    """
    check_args(args)
    cmd = ["git"]
    if repo is not None:
        cmd += ["-C", os.fspath(repo)]
    cmd += list(args)
    env = scrubbed_env(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    kwargs = {"encoding": "utf-8", "errors": "replace"} if text else {}
    return subprocess.run(
        cmd,
        capture_output=True,
        env=env,
        timeout=timeout,
        check=check,
        stdin=subprocess.DEVNULL,
        **kwargs,
    )


def output(
    args: Sequence[str],
    repo: str | os.PathLike[str] | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """执行只读 git 命令并返回去掉首尾空白的 stdout；非零退出抛 ``CalledProcessError``。"""
    return run(args, repo, timeout=timeout, check=True).stdout.strip()


def config_get(key: str, repo: str | os.PathLike[str] | None = None) -> str | None:
    """只读读取生效的配置值；未设置返回 None。"""
    proc = run(["config", "--get", key], repo)
    return proc.stdout.strip() if proc.returncode == 0 else None


__all__ = [
    "GitReadOnlyError",
    "READONLY_SUBCOMMANDS",
    "check_args",
    "is_allowed",
    "run",
    "output",
    "config_get",
]

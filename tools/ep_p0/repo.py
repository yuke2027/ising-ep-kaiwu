"""仓库辅助：克隆 URL 校验、克隆/fetch 错误分类、克隆目标检查与结构核对（需求 2）。

本模块不执行 clone/fetch（由代理按 design.md 流程执行）；只读 git 调用经 ``gitro``。
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import gitro
from .sanitize import sanitize

UPSTREAM_URL = "https://github.com/qboson/kaiwu-pytorch-plugin.git"
CLONE_REL = "upstream/kaiwu-pytorch-plugin"
WORK_BRANCH = "deps/kaiwu-1.4.1"
EXPECTED_VERSION = "0.3.0"   # Master_Spec 对 __init__.py 中 __version__ 的描述

_CLONE_URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+?(\.git)?$", re.ASCII)

# ---------------------------------------------------------------------------
# 克隆 URL 校验（2.3）
# ---------------------------------------------------------------------------


class CloneUrlError(ValueError):
    pass


def validate_clone_url(url: str) -> str:
    """校验 Fork_Repository URL，合规时原样返回；不合规时抛 ``CloneUrlError``。

    规则：整串匹配 ``^https://github\\.com/[\\w.-]+/[\\w.-]+?(\\.git)?$``（ASCII，
    不接受首尾空白或换行），且不含 ``@`` 或 userinfo（令牌、用户名、密码）。
    """
    if not isinstance(url, str):
        raise CloneUrlError("URL 须为字符串")
    candidate = url
    if "@" in candidate:
        raise CloneUrlError("URL 不得包含 @ 或凭据")
    parts = urlsplit(candidate)
    if parts.username or parts.password:
        raise CloneUrlError("URL 不得包含 userinfo")
    if not _CLONE_URL.fullmatch(candidate):
        raise CloneUrlError("URL 须为 https://github.com/<owner>/<repo>[.git]")
    owner, repo = parts.path.strip("/").split("/")
    if owner in (".", "..") or repo.removesuffix(".git") in ("", ".", ".."):
        raise CloneUrlError("URL 中的账号或仓库名非法")
    return candidate


def is_valid_clone_url(url: str) -> bool:
    try:
        validate_clone_url(url)
    except CloneUrlError:
        return False
    return True


# ---------------------------------------------------------------------------
# 克隆 / fetch 失败分类（2.9）
# ---------------------------------------------------------------------------

_ACCESS_PATTERNS = (
    "repository not found",
    "403",
    "authentication failed",
    "permission denied",
    "could not read username",
    "could not read password",
    "access denied",
    "401",
)
_NETWORK_PATTERNS = (
    "could not resolve host",
    "timed out",
    "ssl",
    "failed to connect",
    "connection refused",
    "connection reset",
    "connection was reset",
    "recv failure",
    "rpc failed",
    "early eof",
    "proxy",
    "network is unreachable",
)


def classify_git_error(stderr: str | None) -> str:
    """把 clone/fetch 的 stderr 分为 ``network``、``access`` 或 ``other``。

    同时命中时优先 ``access``（如 ``unable to access ...: returned error: 403``）。
    """
    text = (stderr or "").lower()
    if any(p in text for p in _ACCESS_PATTERNS):
        return "access"
    if any(p in text for p in _NETWORK_PATTERNS):
        return "network"
    return "other"


def describe_failure(
    command: Sequence[str],
    stderr: str | None,
    *,
    workspace_root: str | os.PathLike[str] | None = None,
    clone_root: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """生成失败记录：去凭据的命令、错误类别与 Sanitized_Message。"""
    kw = {"clone_root": clone_root, "workspace_root": workspace_root}
    lines = (stderr or "").splitlines()
    # git 的首行常是 "Cloning into ..."；优先取 fatal:/error: 行作为消息
    key_line = next(
        (ln for ln in lines if ln.lstrip().lower().startswith(("fatal:", "error:"))),
        stderr,
    )
    return {
        "command": sanitize(" ".join(command), **kw),
        "category": classify_git_error(stderr),
        "message": sanitize(key_line, **kw),
    }


# ---------------------------------------------------------------------------
# 克隆目标检查（2.2）
# ---------------------------------------------------------------------------

def target_state(path: str | os.PathLike[str]) -> dict[str, Any]:
    """返回克隆目标现状：``exists``、``is_dir``、``empty``、``entries``（目录项数）。

    目标可克隆当且仅当不存在，或为空目录（``ok_to_clone``）。只读，不删除任何内容。
    """
    p = Path(path)
    state: dict[str, Any] = {"exists": p.exists(), "is_dir": p.is_dir(), "entries": 0}
    if state["is_dir"]:
        state["entries"] = sum(1 for _ in p.iterdir())
    state["empty"] = state["is_dir"] and state["entries"] == 0
    state["ok_to_clone"] = (not state["exists"]) or state["empty"]
    return state


# ---------------------------------------------------------------------------
# 结构核对（2.8）
# ---------------------------------------------------------------------------

def read_version(init_py: str | os.PathLike[str]) -> str | None:
    """用 AST 读取模块顶层 ``__version__ = "..."``；不存在返回 None。"""
    try:
        tree = ast.parse(Path(init_py).read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "__version__"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                return value.value
    return None


_LICENSE_NAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENCE", "COPYING")


def _is_apache2(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return False
    return "Apache License" in head and "Version 2.0" in head


def _check(item: str, path: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"item": item, "path": path, "ok": bool(ok), "detail": detail}


def check_structure(
    clone_root: str | os.PathLike[str], base_commit: str | None = None
) -> list[dict[str, Any]]:
    """核对 Fork_Clone 结构是否与 Master_Spec 描述一致，返回逐项结果。

    检查工作区文件；给出 ``base_commit`` 时追加一项"工作区与 Base_Commit 一致"
    （HEAD 等于 base 且已跟踪文件无差异），以说明检查对象即 Base_Commit。
    路径一律相对 Fork_Clone 根。
    """
    root = Path(clone_root)
    results: list[dict[str, Any]] = []

    pkg = root / "src" / "kaiwu" / "torch_plugin"
    results.append(_check("源码目录", "src/kaiwu/torch_plugin/", pkg.is_dir()))

    init_py = pkg / "__init__.py"
    version = read_version(init_py) if init_py.is_file() else None
    if not init_py.is_file():
        detail = "文件不存在"
    elif version is None:
        detail = "未找到 __version__"
    else:
        detail = f"__version__ = {version!r}（期望 {EXPECTED_VERSION!r}）"
    results.append(
        _check("__version__", "src/kaiwu/torch_plugin/__init__.py",
               version == EXPECTED_VERSION, detail)
    )

    example = root / "example"
    names = sorted(p.name for p in example.iterdir() if p.is_dir()) if example.is_dir() else []
    results.append(
        _check("示例目录", "example/", example.is_dir(),
               f"子目录：{', '.join(names)}" if names else "无子目录")
    )

    tests_dir = root / "tests"
    tests = sorted(p.name for p in tests_dir.glob("test_*.py")) if tests_dir.is_dir() else []
    results.append(
        _check("测试文件", "tests/test_*.py", bool(tests), ", ".join(tests) or "未找到")
    )

    req = root / "requirements" / "requirements.txt"
    results.append(_check("依赖文件", "requirements/requirements.txt", req.is_file()))

    lic = next((root / n for n in _LICENSE_NAMES if (root / n).is_file()), None)
    results.append(
        _check("Apache-2.0 许可", lic.name if lic else "LICENSE",
               lic is not None and _is_apache2(lic),
               "" if lic else "未找到许可文件")
    )

    if base_commit is not None:
        head = gitro.run(["rev-parse", "HEAD"], root)
        same_head = head.returncode == 0 and head.stdout.strip() == base_commit
        diff = gitro.run(["diff", "--quiet", base_commit, "--"], root)
        clean = diff.returncode == 0
        results.append(
            _check("工作区与 Base_Commit 一致", ".", same_head and clean,
                   f"HEAD {'=' if same_head else '≠'} base；"
                   f"已跟踪文件{'无差异' if clean else '有差异或比较失败'}")
        )
    return results


__all__ = [
    "UPSTREAM_URL",
    "CLONE_REL",
    "WORK_BRANCH",
    "CloneUrlError",
    "validate_clone_url",
    "is_valid_clone_url",
    "classify_git_error",
    "describe_failure",
    "target_state",
    "read_version",
    "check_structure",
]

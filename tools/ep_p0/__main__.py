"""``python -m tools.ep_p0 <cmd> [args...]`` 入口。

子命令到模块的映射在此一次写定，后续任务只新增模块、不修改本文件。
各模块提供 ``main(argv: list[str]) -> int | None``：

- 单子命令模块：``argv`` 为子命令之后的参数；
- 多子命令模块（``regress``）：``argv[0]`` 为子命令名，其后为参数。

返回值 ``None`` 视为 0。退出码：0 成功，1 检查未通过，2 用法或环境错误。
"""

from __future__ import annotations

import importlib
import sys
from typing import NamedTuple

from . import EXIT_OK, EXIT_USAGE


class Route(NamedTuple):
    module: str          # tools.ep_p0 下的模块名
    shared: bool         # True：多子命令共用模块，argv[0] 传子命令名
    summary: str         # --help 中的一句话说明


COMMANDS: dict[str, Route] = {
    "init": Route("status", False, "建 records/ep-p0/ 并初始化 status.json"),
    "snapshot": Route("snapshot", False, "写 Environment_Snapshot（environment.json）"),
    "decls": Route("decls", False, "扫描 Kaiwu 依赖声明 → status.json.dependency_declarations"),
    "bump-req": Route("bumpreq", False, "按字节替换 requirements.txt 中唯一的 kaiwu 行"),
    "inventory": Route("regress", True, "登记 Regression_Item → .p0-runs/inventory.json"),
    "run-tests": Route("regress", True, "逐文件运行 pytest → regression.json"),
    "run-examples": Route("regress", True, "运行示例入口 → regression.json"),
    "junit2reg": Route("regress", True, "junit/-v 日志 → regression.json"),
    "imports": Route("imports", False, "Static_Import_Check → regression.json.static_imports"),
    "breakage": Route("breakage", False, "构建 Breakage_List（breakage.json）"),
    "scan-creds": Route("scan", False, "Credential_Scan → status.json.credential_scans"),
    "identity": Route("identity", False, "提交身份核对 → status.json.identity_checks"),
    "render": Route("render", False, "由 JSON 渲染全部 Markdown 记录"),
    "gates": Route("gates", False, "门禁判定 → status.json.gates、p1_decision"),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        "用法：python -m tools.ep_p0 <cmd> [args...]",
        "",
        "子命令：",
    ]
    for name, route in COMMANDS.items():
        lines.append(f"  {name.ljust(width)}  {route.summary}")
    lines += [
        "",
        "退出码：0 成功，1 检查未通过，2 用法或环境错误。",
        "子命令帮助：python -m tools.ep_p0 <cmd> --help",
    ]
    return "\n".join(lines)


def _configure_streams() -> None:
    # Windows 控制台编码可能无法表示部分字符；替换而不是崩溃。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _configure_streams()
    args = list(sys.argv[1:] if argv is None else argv)

    if not args:
        print(_usage(), file=sys.stderr)
        return EXIT_USAGE
    if args[0] in ("-h", "--help", "help"):
        print(_usage())
        return EXIT_OK

    cmd, rest = args[0], args[1:]
    route = COMMANDS.get(cmd)
    if route is None:
        print(f"未知子命令：{cmd}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return EXIT_USAGE

    qualname = f"{__package__}.{route.module}"
    try:
        module = importlib.import_module(qualname)
    except ModuleNotFoundError as exc:
        # 只把"模块本身不存在"判为未实现；模块内部缺依赖照常抛出。
        if exc.name == qualname:
            print(f"子命令 {cmd} 未实现（缺少模块 {route.module}.py）", file=sys.stderr)
            return EXIT_USAGE
        raise

    entry = getattr(module, "main", None)
    if not callable(entry):
        print(f"子命令 {cmd} 未实现（模块 {route.module} 缺少 main(argv)）", file=sys.stderr)
        return EXIT_USAGE

    code = entry([cmd, *rest] if route.shared else rest)
    return EXIT_OK if code is None else int(code)


if __name__ == "__main__":
    sys.exit(main())

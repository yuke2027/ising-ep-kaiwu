"""ep-p0-upstream-setup 辅助脚本（仅标准库，Python >= 3.10）。

入口：在工作区根执行 ``python -m tools.ep_p0 <cmd>``。
退出码：0 成功，1 检查未通过，2 用法或环境错误。
"""

__all__ = ["EXIT_OK", "EXIT_CHECK_FAILED", "EXIT_USAGE"]

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_USAGE = 2

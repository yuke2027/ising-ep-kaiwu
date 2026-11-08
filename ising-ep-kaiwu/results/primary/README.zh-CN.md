# 归档的主运行

[English](README.md) · [中文](README.zh-CN.md)

这是三次种子 MNIST/100 运行的只读证据，数字被[项目 README](../../README.zh-CN.md)和 [examples/mnist_kaiwu/README.zh-CN.md](../../examples/mnist_kaiwu/README.zh-CN.md) 引用。

不要覆盖这个目录。把 `--output` 指到新路径。

## 来源

2026-09-30 由当前代码产生（`config.json` 记录的时间是 `2026-09-30T14:20:14Z`）。候选由 Kaiwu 的 `get_sorted_solutions` 排序。

## 内容

```
config.json                         模型、训练与退火设置
summary.json                        跨种子统计
seeds/<seed>/initial-model.npz      训练前的参数
seeds/<seed>/status.json            准确率、平局率、逐类结果、学习证据
seeds/<seed>/checkpoint.npz         最后一个 epoch 之后的参数
seeds/<seed>/epochs/<e>/model.npz   每个 epoch 的参数快照
seeds/<seed>/epochs/<e>/dynamics.json   该 epoch 的两相翻转率
seeds/<seed>/test/metrics.json      混淆矩阵与逐类细节
seeds/<seed>/test/predictions.json  逐样本预测
```

## 主要数字

| 种子 | 准确率 | 平局率 |
|------|--------|--------|
| 0 | 0.15 | 0.62 |
| 1 | 0.54 | 0.06 |
| 2 | 0.46 | 0.18 |
| **均值** | **0.3833** | 0.2867 |

784-64-40，5 个 epoch，批大小 10，`lr = 0.01`，`β = 1.0`。每个种子 10,100 次求解器调用：5,000 次自由相 + 5,000 次微扰相 + 100 次评估。

## 读取

```python
import json
from pathlib import Path

root = Path("results/primary")
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
print(summary["accuracy"])          # mean / std / min / max

with __import__("numpy").load(root / "seeds/0/checkpoint.npz") as ckpt:
    print({k: ckpt[k].shape for k in ckpt.files})   # W, V, b_h, b_o
```

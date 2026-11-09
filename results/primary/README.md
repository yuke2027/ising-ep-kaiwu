# Archived primary run

[English](README.md) · [中文](README.zh-CN.md)

Read-only evidence for the three-seed MNIST/100 run quoted in the
[project README](../../README.md) and in
[examples/mnist_kaiwu/README.md](../../examples/mnist_kaiwu/README.md).

Do not overwrite this directory. Point `--output` at a new path instead.

## Provenance

Produced on 2026-09-30 by the current code (`config.json` records
`2026-09-30T14:20:14Z`). Candidates are ranked by Kaiwu's
`get_sorted_solutions`.

## Contents

```
config.json                         Model, training and annealing settings
summary.json                        Cross-seed statistics
seeds/<seed>/initial-model.npz      Parameters before training
seeds/<seed>/status.json            Accuracy, tie rate, per class, learning evidence
seeds/<seed>/checkpoint.npz         Parameters after the final epoch
seeds/<seed>/epochs/<e>/model.npz   Per-epoch parameter snapshot
seeds/<seed>/epochs/<e>/dynamics.json   Two-phase flip rates for that epoch
seeds/<seed>/test/metrics.json      Confusion matrix and per-class detail
seeds/<seed>/test/predictions.json  Per-sample predictions
```

## Headline numbers

| Seed | Accuracy | Tie rate |
|------|----------|----------|
| 0 | 0.15 | 0.62 |
| 1 | 0.54 | 0.06 |
| 2 | 0.46 | 0.18 |
| **mean** | **0.3833** | 0.2867 |

784-64-40, 5 epochs, batch 10, `lr = 0.01`, `β = 1.0`. 10,100 solver calls per
seed: 5,000 free + 5,000 nudged + 100 evaluation.

## Reading it

```python
import json
from pathlib import Path

root = Path("results/primary")
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
print(summary["accuracy"])          # mean / std / min / max

with __import__("numpy").load(root / "seeds/0/checkpoint.npz") as ckpt:
    print({k: ckpt[k].shape for k in ckpt.files})   # W, V, b_h, b_o
```

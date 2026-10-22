# PyTorch + Kaiwu Ising Equilibrium Propagation

This repository refactors the MNIST/100 Ising equilibrium-propagation experiment
to a PyTorch core and direct Kaiwu 1.3.1 API calls, matching
`qboson/kaiwu-pytorch-plugin`. Historical `runs/` artifacts are read-only
evidence and are not overwritten by new runs.

## Environment

Kaiwu SDK 1.3.1 is used with Python 3.10, the same pair as the upstream plugin.
Use the project-local virtual environment; do not use the global interpreter.

Windows PowerShell:

```powershell
py -3.10 --version
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.10 --version
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The version check must print `Python 3.10.x`.

After the public dependencies, install Kaiwu 1.3.1. `pip install kaiwu==1.3.1`
works when PyPI is reachable. Otherwise install the official `cp310` wheel for
this OS and CPU architecture:

```powershell
.\.venv\Scripts\python.exe -m pip install .\vendor\kaiwu-1.3.1-cp310-none-win_amd64.whl
```

```bash
python -m pip install ./vendor/kaiwu-1.3.1-cp310-none-win_amd64.whl
```

Confirm the same interpreter imported Kaiwu 1.3.1:

```powershell
.\.venv\Scripts\python.exe -c "import importlib.metadata as m; print(m.version('kaiwu'))"
```

```bash
python -c "import importlib.metadata as m; print(m.version('kaiwu'))"
```

Create `.env` from `.env.example` and fill in `KAIWU_USER_ID` and
`KAIWU_SDK_CODE`. Do not commit `.env`, license files, or paste credential
values into artifacts, logs, tests, or reports.

## Layout

- `src/ising_ep/model.py`: stacked linear layers (`hidden_sizes`), the EP energy,
  and the QUBO mapping (`IsingEPModel.qubo`, `IsingEPModel.split_state`).
- `src/ising_ep/kaiwu_solver.py`: the only module that imports kaiwu.
- `examples/mnist_kaiwu/`: MNIST/100 data, two-phase training, free-phase
  evaluation, and the experiment CLI.
- `tests/unit`: tests that do not require Kaiwu or credentials.
- `tests/integration`: optional licensed-SDK checks. They skip when the SDK is absent.
- `archive/`: read-only papers, the 1001 delivery, and the pre-cleanup snapshot
  under `archive/ising-ep-kaiwu-delivery/workspace-before-cleanup/`.
  New experiments still write to `runs/`.

`energy` is the parameter-dependent Ising energy used by the training loss
`(E_nudged - E_free) / beta`. The solver minimises `energy + beta * cost`.
The cost does not depend on the parameters, so both expressions share a
parameter gradient. One integer `--n-hidden` is a single hidden layer;
`--hidden-sizes 128,64` stacks hidden layers. Per-connection scales
`--init-gains` and rates `--lr-weights` / `--lr-biases` default to 1 and the
global `--lr`. The default network is 784-64-40.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit
.\.venv\Scripts\python.exe -m pytest tests -m integration
.\.venv\Scripts\python.exe -m pylint src examples tests
.\.venv\Scripts\python.exe -m pydocstyle --convention=google src examples
```

Integration tests explicitly skip when `kaiwu` or its license is unavailable.

## Official run

The direct API path uses `kaiwu.qubo.qubo_matrix_to_qubo_model`,
`kaiwu.conversion.qubo_model_to_ising_model`, and
`kaiwu.classical.SimulatedAnnealingOptimizer.single_process_solve`. Ranking and
the auxiliary-spin gauge fix are done in this package because Kaiwu 1.3.1 has
no `get_sorted_solutions`. There is no local annealer fallback.

Pytest finds `ising_ep` through `pythonpath` in `pyproject.toml`. Running the
experiment module needs the same path:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.experiment `
  --output runs\smoke --seeds 0 --n-hidden 8 --epochs 1 `
  --candidate-count 2 --iterations-per-t 4
```

Formal three-seed regression:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.experiment `
  --output runs\primary --seeds 0,1,2 `
  --n-hidden 64 --epochs 5 --batch-size 10 --lr 0.01 --beta 1.0
```

Stacked hidden layers with per-connection rates:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.experiment `
  --output runs\stacked --seeds 0 --hidden-sizes 128,64 `
  --lr-weights 0.01,0.01,0.005 --init-gains 1,1,0.5
```

The 1.3.1 three-seed rerun lives in `runs/primary-kaiwu-1.3.1/` (gitignored) and
matches the archived 1001 table: **38.3% ± 16.8%**.

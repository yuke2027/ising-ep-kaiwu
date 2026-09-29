# PyTorch + Kaiwu Ising Equilibrium Propagation

This repository refactors the MNIST/100 Ising equilibrium-propagation experiment
to a PyTorch core and direct Kaiwu 1.4.1 API calls. Historical `runs/` artifacts
are read-only evidence and are not overwritten by new runs.

## Environment

Kaiwu SDK 1.4.1 requires Python 3.10. Use the project-local virtual environment;
do not use the global interpreter.

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

The Kaiwu enterprise wheel is not on public PyPI. After the public dependencies,
install the official `cp310` wheel downloaded for this OS and CPU architecture.
Do not reuse a `win_amd64` filename on another platform; use the actual file
from the quantum cloud platform:

```powershell
.\.venv\Scripts\python.exe -m pip install .\vendor\<kaiwu-1.4.1-cp310-actual-wheel>.whl
```

```bash
python -m pip install ./vendor/<kaiwu-1.4.1-cp310-actual-wheel>.whl
```

Confirm the same interpreter imported Kaiwu 1.4.1:

```powershell
.\.venv\Scripts\python.exe -c "import kaiwu as kw; print(getattr(kw, '__version__', 'version unavailable'))"
```

```bash
python -c "import kaiwu as kw; print(getattr(kw, '__version__', 'version unavailable'))"
```

Create `.env` from `.env.example` and fill in `KAIWU_USER_ID` and
`KAIWU_SDK_CODE`. Do not commit `.env`, license files, or paste credential
values into artifacts, logs, tests, or reports.

## Layout

- `src/ising_ep/model.py`: `torch.nn.Module` EP parameters and autograd energy.
- `src/ising_ep/mapping.py`: pure Torch QUBO mapping and state decoding; no Kaiwu.
- `ising_ep/model.py`, `ising_ep/mapping.py`: thin compatibility imports into `src`.
- `examples/mnist_kaiwu/*`: MNIST data, official Kaiwu 1.4.1 boundary, training,
  evaluation, registration, reporting and environment gates.
- `tests/unit`: tests that do not require Kaiwu or credentials.
- `tests/integration`: optional licensed-SDK checks.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit
.\.venv\Scripts\python.exe -m pytest tests -m integration
.\.venv\Scripts\python.exe -m pylint src examples tests
.\.venv\Scripts\python.exe -m pydocstyle --convention=google src examples
```

Integration tests explicitly skip when `kaiwu` or its license is unavailable.

## Official run

The direct API path uses `kaiwu.core.qubo_matrix_to_qubo_model`,
`kaiwu.core.qubo_model_to_ising_model`, `kaiwu.core.get_sol_dict`, and
`kaiwu.classical.SimulatedAnnealingOptimizer.single_process_solve`. There is no
local solver fallback or compatibility/capability execution layer in the new
path.

Environment gate only:

```powershell
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.environment --output runs\env-check
```

One-seed smoke run (uses a new output directory):

```powershell
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.experiment `
  --output runs\torch-smoke-s0 --run-kind smoke --seeds 0 `
  --n-hidden 16 --epochs 1 --batch-size 10 --train
```

Formal three-seed regression:

```powershell
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.experiment `
  --output runs\torch-primary-h64-e5 --run-kind primary --seeds 0,1,2 `
  --n-hidden 64 --epochs 5 --batch-size 10 --lr 0.01 --beta 1.0 --train
```

Rebuild reports without training:

```powershell
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.reporting --run-dir runs\torch-primary-h64-e5
```

The old script names remain as thin wrappers (`scripts/run_primary.py`,
`scripts/aggregate_primary.py`, `scripts/verify_alignment.py`,
`scripts/build_manifest.py`). They do not expose fallback/backend switches.

There is no `kaiwu_adapter` or SDK capability matrix. New runs call
`kaiwu.core.qubo_matrix_to_qubo_model`, `kaiwu.core.qubo_model_to_ising_model`,
`kaiwu.core.get_sol_dict`, and
`kaiwu.classical.SimulatedAnnealingOptimizer.single_process_solve` directly.

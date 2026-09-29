"""Backward-compatible import path for Torch QUBO/Ising mapping utilities.

The implementation lives in ``src/ising_ep/mapping.py`` and intentionally has no
Kaiwu dependency.
"""

from src.ising_ep.mapping import (
    QuboProblem,
    build_qubo,
    clip_solver_problem,
    cost_c,
    coupling_and_fields,
    decode_aux,
    decode_aux_alt,
    energy,
    energy_equivalence_report,
    free_energy,
    hidden_field,
    iter_physical_states,
    minimizer_indices,
    output_field,
    qubo_energy,
    spins_to_bits,
    split_hidden_output,
)

__all__ = [
    "QuboProblem",
    "build_qubo",
    "clip_solver_problem",
    "cost_c",
    "coupling_and_fields",
    "decode_aux",
    "decode_aux_alt",
    "energy",
    "energy_equivalence_report",
    "free_energy",
    "hidden_field",
    "iter_physical_states",
    "minimizer_indices",
    "output_field",
    "qubo_energy",
    "spins_to_bits",
    "split_hidden_output",
]

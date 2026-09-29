"""PyTorch parameters and ±1 output encoding for the Ising EP network."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn


class IsingEPModel(nn.Module):
    """Equilibrium-propagation Ising network with input-hidden and hidden-output couplings.

    Hidden and output states are discrete ±1 values produced by a solver. They are
    constants for gradient purposes; only ``W``, ``V``, ``b_h`` and ``b_o`` are
    learnable parameters.
    """

    def __init__(
        self,
        n_in: int = 784,
        n_hidden: int = 16,
        n_classes: int = 10,
        spins_per_class: int = 4,
        seed: int = 0,
        gain_weight0: float | None = None,
        gain_weight1: float | None = None,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        """Create a float64 model with deterministic uniform Torch initialization."""
        super().__init__()
        if n_in < 1 or n_hidden < 1 or n_classes < 1 or spins_per_class < 1:
            raise ValueError("network dimensions must be positive")
        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.n_classes = int(n_classes)
        self.spins_per_class = int(spins_per_class)
        self.n_output = self.n_classes * self.spins_per_class
        self.gain_weight0 = gain_weight0
        self.gain_weight1 = gain_weight1
        self._dtype = dtype
        self._device = torch.device("cpu") if device is None else torch.device(device)

        limit_w = 1.0 / math.sqrt(self.n_in)
        limit_v = 1.0 / math.sqrt(self.n_hidden)
        if gain_weight0 is not None:
            limit_w *= float(gain_weight0)
        if gain_weight1 is not None:
            limit_v *= float(gain_weight1)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        w = torch.empty((self.n_in, self.n_hidden), dtype=dtype)
        v = torch.empty((self.n_hidden, self.n_output), dtype=dtype)
        w.uniform_(-limit_w, limit_w, generator=generator)
        v.uniform_(-limit_v, limit_v, generator=generator)
        self.W = nn.Parameter(w.to(self._device))
        self.V = nn.Parameter(v.to(self._device))
        self.b_h = nn.Parameter(torch.zeros(self.n_hidden, dtype=dtype, device=self._device))
        self.b_o = nn.Parameter(torch.zeros(self.n_output, dtype=dtype, device=self._device))

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating-point dtype used by model parameters."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Return the device containing model parameters."""
        return self._device

    def _tensor(self, value: Any, name: str) -> torch.Tensor:
        try:
            tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} cannot be converted to a {self.dtype} tensor") from exc
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
        return tensor

    def _matrix_or_vector(self, value: Any, width: int, name: str) -> tuple[torch.Tensor, bool]:
        tensor = self._tensor(value, name)
        if tensor.ndim == 1:
            if tensor.shape[0] != width:
                raise ValueError(f"{name} must have length {width}, got {tensor.shape[0]}")
            return tensor.unsqueeze(0), True
        if tensor.ndim == 2 and tensor.shape[1] == width:
            return tensor, False
        raise ValueError(f"{name} must have shape ({width},) or (batch, {width})")

    def _broadcast_batch(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = max(x.shape[0], hidden.shape[0], output.shape[0])
        for name, tensor in (("x", x), ("hidden", hidden), ("output", output)):
            if tensor.shape[0] not in (1, batch):
                raise ValueError(f"{name} batch size {tensor.shape[0]} cannot broadcast to {batch}")
        return (
            x.expand(batch, -1),
            hidden.expand(batch, -1),
            output.expand(batch, -1),
        )

    @staticmethod
    def _maybe_scalar(values: torch.Tensor, single: bool) -> torch.Tensor:
        return values[0] if single else values

    def numpy_params(self) -> dict[str, np.ndarray]:
        """Return detached parameter copies in historical NumPy layout."""
        return {
            name: value.cpu().numpy().copy()
            for name, value in self.parameter_snapshot().items()
        }

    def parameter_snapshot(self) -> dict[str, torch.Tensor]:
        """Return detached ``float64`` copies of the learnable parameters.

        Returns:
            Independent tensors ``W`` of shape ``(n_in, n_hidden)``, ``V`` of
            shape ``(n_hidden, n_output)``, and one-dimensional biases ``b_h``
            and ``b_o``.
        """
        return {name: value.detach().clone() for name, value in self.state_dict().items()}

    def load_numpy_params(self, state: dict[str, Any]) -> None:
        """Load a historical NumPy parameter mapping after shape validation."""
        expected = {
            "W": (self.n_in, self.n_hidden),
            "V": (self.n_hidden, self.n_output),
            "b_h": (self.n_hidden,),
            "b_o": (self.n_output,),
        }
        for name, shape in expected.items():
            if name not in state:
                raise ValueError(f"missing parameter {name}")
            array = np.asarray(state[name], dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"parameter {name} must be finite with shape {shape}")
        with torch.no_grad():
            for name in expected:
                getattr(self, name).copy_(
                    torch.as_tensor(np.asarray(state[name], dtype=np.float64), dtype=self.dtype, device=self.device)
                )

    def target_spins(self, labels: Any) -> torch.Tensor:
        """Encode labels as ±1 target output vectors with four spins per class.

        Args:
            labels: Scalar or one-dimensional integer labels in ``[0, n_classes)``.

        Returns:
            A ``(batch, n_output)`` tensor in which only the target class block is +1.
        """
        raw = np.asarray(labels)
        if raw.size == 0:
            raise ValueError("labels must not be empty")
        try:
            label_values = raw.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError("labels must be integers") from exc
        if not np.array_equal(raw, label_values):
            raise ValueError("labels must be integers")
        flat = label_values.reshape(-1)
        if np.any((flat < 0) | (flat >= self.n_classes)):
            raise ValueError(f"labels must be in [0, {self.n_classes})")
        encoded = torch.full(
            (flat.size, self.n_output),
            -1.0,
            dtype=self.dtype,
            device=self.device,
        )
        for row, label in enumerate(flat.tolist()):
            start = int(label) * self.spins_per_class
            encoded[row, start : start + self.spins_per_class] = 1.0
        return encoded

    def energy(self, x: Any, hidden: Any, output: Any) -> torch.Tensor:
        """Compute ``-xWh - hVo - b_h·h - b_o·o``.

        Parameters are ``float64``. Hidden and output spins lie in ``{+1, -1}``.

        Args:
            x: Input with shape ``(n_in,)`` or ``(batch, n_in)``.
            hidden: Hidden spins with shape ``(n_hidden,)`` or ``(batch, n_hidden)``.
            output: Output spins with shape ``(n_output,)`` or ``(batch, n_output)``.

        Returns:
            A ``float64`` scalar when every argument is a vector, otherwise a
            tensor of shape ``(batch,)``.

        Raises:
            ValueError: If a value is non-finite or a shape cannot broadcast.
        """
        x_batch, x_single = self._matrix_or_vector(x, self.n_in, "x")
        h_batch, h_single = self._matrix_or_vector(hidden, self.n_hidden, "hidden")
        o_batch, o_single = self._matrix_or_vector(output, self.n_output, "output")
        x_batch, h_batch, o_batch = self._broadcast_batch(x_batch, h_batch, o_batch)
        values = (
            -torch.einsum("bi,ij,bj->b", x_batch, self.W, h_batch)
            - torch.einsum("bj,jk,bk->b", h_batch, self.V, o_batch)
            - h_batch @ self.b_h
            - o_batch @ self.b_o
        )
        return self._maybe_scalar(values, x_single and h_single and o_single)

    def cost(self, output: Any, target: Any) -> torch.Tensor:
        """Compute the EP cost ``n_output - target·output``.

        Args:
            output: Output spins in ``{+1, -1}`` with shape ``(n_output,)`` or
                ``(batch, n_output)`` and dtype ``float64``.
            target: Target spins in ``{+1, -1}`` with the same width.

        Returns:
            A ``float64`` scalar or a tensor of shape ``(batch,)``.

        Raises:
            ValueError: If ``output`` and ``target`` shapes do not match.
        """
        output_batch, output_single = self._matrix_or_vector(output, self.n_output, "output")
        target_batch, target_single = self._matrix_or_vector(target, self.n_output, "target")
        if target_batch.shape[0] == 1 and output_batch.shape[0] > 1:
            target_batch = target_batch.expand(output_batch.shape[0], -1)
        if output_batch.shape != target_batch.shape:
            raise ValueError("output and target batch sizes do not match")
        values = self.n_output - (output_batch * target_batch).sum(dim=1)
        return self._maybe_scalar(values, output_single and target_single)

    def free_energy(self, x: Any, hidden: Any, output: Any, target: Any, beta: float) -> torch.Tensor:
        """Compute ``energy + beta * cost`` with gradients on the parameters.

        Args:
            x: Input with shape ``(n_in,)`` or ``(batch, n_in)`` and dtype ``float64``.
            hidden: Hidden spins in ``{+1, -1}``.
            output: Output spins in ``{+1, -1}``.
            target: Target spins in ``{+1, -1}``.
            beta: Non-negative nudging coefficient.

        Returns:
            A ``float64`` scalar or a tensor of shape ``(batch,)``.

        Raises:
            ValueError: If ``beta`` is negative or non-finite.
        """
        beta_value = float(beta)
        if not math.isfinite(beta_value) or beta_value < 0:
            raise ValueError("beta must be a non-negative finite value")
        return self.energy(x, hidden, output) + beta_value * self.cost(output, target)

    def fields(self, x: Any, target: Any, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hidden and output local fields for one solver problem.

        Args:
            x: Input with shape ``(n_in,)`` or ``(batch, n_in)`` and dtype ``float64``.
            target: Target spins in ``{+1, -1}`` with shape ``(n_output,)`` or
                ``(batch, n_output)``.
            beta: Non-negative nudging coefficient.

        Returns:
            Hidden field ``xW + b_h`` and output field ``b_o + beta * target``,
            both ``float64``. Vector inputs return vectors.

        Raises:
            ValueError: If ``beta`` is invalid or the batch sizes differ.
        """
        beta_value = float(beta)
        if not math.isfinite(beta_value) or beta_value < 0:
            raise ValueError("beta must be a non-negative finite value")
        x_batch, x_single = self._matrix_or_vector(x, self.n_in, "x")
        y_batch, y_single = self._matrix_or_vector(target, self.n_output, "target")
        if y_batch.shape[0] == 1 and x_batch.shape[0] > 1:
            y_batch = y_batch.expand(x_batch.shape[0], -1)
        if x_batch.shape[0] != y_batch.shape[0]:
            raise ValueError("x and target batch sizes do not match")
        hidden = x_batch @ self.W + self.b_h
        output = self.b_o + beta_value * y_batch
        if x_single and y_single:
            return hidden[0], output[0]
        return hidden, output

    def predict(self, hidden: Any, output: Any) -> tuple[int, bool]:
        """Predict one class from output spins, breaking ties with local fields.

        Args:
            hidden: One hidden state in ``{+1, -1}`` with shape ``(n_hidden,)``
                and dtype ``float64``.
            output: One output state in ``{+1, -1}`` with shape ``(n_output,)``
                and dtype ``float64``.

        Returns:
            The class index and whether the output-block sums tied.

        Raises:
            ValueError: If ``hidden`` or ``output`` contains more than one state.
        """
        h, _ = self._matrix_or_vector(hidden, self.n_hidden, "hidden")
        o, _ = self._matrix_or_vector(output, self.n_output, "output")
        if h.shape[0] != 1 or o.shape[0] != 1:
            raise ValueError("predict expects one hidden and one output state")
        scores = o[0].reshape(self.n_classes, self.spins_per_class).sum(dim=1)
        top = scores.max()
        tied = torch.nonzero(scores == top).reshape(-1)
        if tied.numel() == 1:
            return int(tied.item()), False
        local_fields = (h[0] @ self.V + self.b_o).reshape(self.n_classes, self.spins_per_class)
        field_scores = local_fields.sum(dim=1)
        tied_scores = field_scores[tied]
        best_tied = tied[torch.nonzero(tied_scores == tied_scores.max()).reshape(-1)]
        return int(best_tied.min().item()), True

    def predict_from_output(self, o: Any, h: Any | None = None) -> tuple[int, bool]:
        """Backward-compatible wrapper using the historical ``(output, hidden)`` order."""
        hidden = torch.zeros(self.n_hidden, dtype=self.dtype, device=self.device) if h is None else h
        return self.predict(hidden, o)


def ep_gradients(
    x: np.ndarray,
    h_free: np.ndarray,
    o_free: np.ndarray,
    h_nudged: np.ndarray,
    o_nudged: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility-only reference for the free-minus-nudged EP gradient."""
    if not np.isfinite(beta) or float(beta) <= 0:
        raise ValueError("beta must be a positive finite value")
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    h0 = np.asarray(h_free, dtype=np.float64).reshape(-1)
    o0 = np.asarray(o_free, dtype=np.float64).reshape(-1)
    hb = np.asarray(h_nudged, dtype=np.float64).reshape(-1)
    ob = np.asarray(o_nudged, dtype=np.float64).reshape(-1)
    inv = 1.0 / float(beta)
    return (
        inv * (np.outer(x, h0) - np.outer(x, hb)),
        inv * (np.outer(h0, o0) - np.outer(hb, ob)),
        inv * (h0 - hb),
        inv * (o0 - ob),
    )

"""Optional CPU/CUDA linear-solver backends for the frequency-domain DWI.

The CPU path uses SciPy's SuperLU factorisation, so the forward and adjoint
solves for a given frequency share one factorisation.  The CUDA path uses
CuPy's sparse solver and is intended for small models or machines where a
GPU is available; it is optional and imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearSolver:
    """Thin wrapper around a SciPy or CuPy sparse solver."""

    backend: str
    matrix: object
    factor: object | None = None
    cupy: object | None = None

    @classmethod
    def build(cls, matrix, backend: str = "cpu") -> "LinearSolver":
        name = str(backend).strip().lower()
        if name in {"cpu", "numpy", "scipy"}:
            from scipy.sparse import csc_matrix
            from scipy.sparse.linalg import splu

            factor = splu(csc_matrix(matrix))
            return cls(backend="cpu", matrix=matrix, factor=factor)
        if name in {"cuda", "gpu", "cupy"}:
            try:
                import cupy as cp
                import cupyx.scipy.sparse as cpsparse
            except ModuleNotFoundError as exc:  # pragma: no cover - optional
                raise ModuleNotFoundError(
                    "The CUDA backend requires CuPy. Install a matching "
                    "cupy-cudaXX package or use backend='cpu'."
                ) from exc
            csr = matrix.tocsr()
            gpu_matrix = cpsparse.csr_matrix(
                (
                    cp.asarray(csr.data, dtype=cp.complex128),
                    cp.asarray(csr.indices),
                    cp.asarray(csr.indptr),
                ),
                shape=csr.shape,
            )
            return cls(backend="cuda", matrix=gpu_matrix, cupy=cp)
        raise ValueError("backend must be 'cpu' or 'cuda'.")

    def solve(self, rhs: np.ndarray, *, trans: str = "N") -> np.ndarray:
        mode = trans.upper()
        if mode not in {"N", "T", "H"}:
            raise ValueError("trans must be 'N', 'T' or 'H'.")
        vector = np.asarray(rhs, dtype=np.complex128)
        if self.backend == "cpu":
            return np.asarray(self.factor.solve(vector, trans=mode), dtype=np.complex128)
        cp = self.cupy
        import cupyx.scipy.sparse.linalg as cpsolve

        gpu_vector = cp.asarray(vector, dtype=cp.complex128)
        if mode == "N":
            result = cpsolve.spsolve(self.matrix, gpu_vector)
        elif mode == "T":
            result = cpsolve.spsolve(self.matrix.T, gpu_vector)
        else:
            result = cpsolve.spsolve(self.matrix.conj().T, gpu_vector)
        return cp.asnumpy(result).astype(np.complex128, copy=False)


def available_backends() -> dict[str, bool]:
    result = {"cpu": True, "cuda": False}
    try:  # pragma: no cover - optional
        import cupy  # noqa: F401
        import cupyx.scipy.sparse  # noqa: F401

        result["cuda"] = True
    except ModuleNotFoundError:
        pass
    return result

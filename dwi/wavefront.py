"""Optional frequency-domain diffraction-wavefront inversion (DWI).

This is a compact 2-D scalar Helmholtz FWI refinement of the diffraction
section.  It follows the frequency-domain L2 formulation used in the
manuscript, but keeps the implementation intentionally small:

* a five-point finite-difference Helmholtz operator with a lossy sponge layer;
* one complex source and one co-located receiver per selected trace;
* a per-frequency complex source calibration (the source wavelet is unknown);
* analytic adjoint gradients and L-BFGS-B with smoothness regularization.

The CPU backend is the default.  Set ``backend='cuda'`` to use the optional
CuPy solver; no CUDA is required for the focusing inversion or the quickstart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.sparse import lil_matrix

from .backends import LinearSolver
from .config import C0, MU0, Z0, Geometry


@dataclass
class WavefrontConfig:
    """Configuration of the optional frequency-domain DWI."""

    frequencies: tuple[float, ...] | None = None
    frequency_count: int = 3
    frequency_min_ratio: float = 0.35
    frequency_max_ratio: float = 1.15
    stage_iterations: int = 4
    max_iterations: int = 12
    regularization: float = 5.0e-3
    pml_cells: int = 8
    source_depth_cells: int = 2
    shot_stride: int = 4
    epsilon_min: float = 3.0
    epsilon_max: float = 9.0
    smooth_sigma: float = 1.0
    backend: str = "cpu"
    dz: float | None = None
    max_depth: float | None = None
    time_taper_fraction: float = 0.1
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.frequency_count < 1:
            raise ValueError("frequency_count must be at least 1.")
        if self.stage_iterations < 1 or self.max_iterations < 1:
            raise ValueError("stage_iterations and max_iterations must be positive.")
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative.")
        if int(self.pml_cells) < 0:
            raise ValueError("pml_cells must be non-negative.")
        if int(self.shot_stride) < 1:
            raise ValueError("shot_stride must be at least 1.")
        if self.epsilon_min <= 0.0 or self.epsilon_max <= self.epsilon_min:
            raise ValueError("Require 0 < epsilon_min < epsilon_max.")
        if self.backend.lower() not in {"cpu", "cuda", "gpu", "cupy"}:
            raise ValueError("backend must be 'cpu' or 'cuda'.")
        if self.dz is not None and self.dz <= 0.0:
            raise ValueError("dz must be positive when set.")


@dataclass
class WavefrontResult:
    epsilon: np.ndarray
    depth: np.ndarray
    x: np.ndarray
    frequencies: np.ndarray
    history: list[dict[str, float]]
    iterations: int


def _tukey(length: int, fraction: float) -> np.ndarray:
    if length < 1:
        raise ValueError("Tukey window length must be positive.")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("time_taper_fraction must lie in [0, 1].")
    if fraction == 0.0 or length == 1:
        return np.ones(length, dtype=np.float64)
    if fraction == 1.0:
        return np.hanning(length).astype(np.float64)
    x = np.linspace(0.0, 1.0, length)
    window = np.ones(length)
    left = x < fraction / 2.0
    right = x > 1.0 - fraction / 2.0
    window[left] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[left] / fraction - 1.0)))
    window[right] = 0.5 * (
        1.0 + np.cos(np.pi * (2.0 * x[right] / fraction - 2.0 / fraction + 1.0))
    )
    return window


def _resample_model(
    epsilon: np.ndarray,
    depth_in: np.ndarray,
    depth_out: np.ndarray,
    x_in: np.ndarray,
    x_out: np.ndarray,
) -> np.ndarray:
    rows = np.interp(depth_out, depth_in, np.arange(depth_in.size))
    cols = np.interp(x_out, x_in, np.arange(x_in.size))
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    return map_coordinates(epsilon, [grid_rows, grid_cols], order=1, mode="nearest")


def _build_operator(
    epsilon: np.ndarray,
    omega: float,
    dx: float,
    dz: float,
    pml_cells: int,
) -> lil_matrix:
    nz, nx = epsilon.shape
    size = nz * nx
    matrix = lil_matrix((size, size), dtype=np.complex128)
    k2 = (omega / C0) ** 2 * np.asarray(epsilon, dtype=np.float64).ravel()
    damping = np.zeros(size, dtype=np.float64)
    if pml_cells > 0:
        profile_x = np.ones(nx, dtype=np.float64)
        profile_z = np.ones(nz, dtype=np.float64)
        for index in range(nx):
            distance = min(index, nx - 1 - index)
            if distance < pml_cells:
                profile_x[index] = (pml_cells - distance) / pml_cells
        for index in range(nz):
            distance = min(index, nz - 1 - index)
            if distance < pml_cells:
                profile_z[index] = (pml_cells - distance) / pml_cells
        sigma_x = 3.0 / (Z0 * max(dx, 1.0e-9)) * profile_x**2
        sigma_z = 3.0 / (Z0 * max(dz, 1.0e-9)) * profile_z**2
        sigma = np.maximum(sigma_x[None, :], sigma_z[:, None]).ravel()
        damping = omega * MU0 * sigma

    inv_dx2 = 1.0 / (dx * dx)
    inv_dz2 = 1.0 / (dz * dz)
    for iz in range(nz):
        for ix in range(nx):
            index = iz * nx + ix
            matrix[index, index] = -2.0 * inv_dx2 - 2.0 * inv_dz2 + k2[index] + 1j * damping[index]
            if ix > 0:
                matrix[index, index - 1] = inv_dx2
            if ix < nx - 1:
                matrix[index, index + 1] = inv_dx2
            if iz > 0:
                matrix[index, index - nx] = inv_dz2
            if iz < nz - 1:
                matrix[index, index + nx] = inv_dz2
    return matrix.tocsr()


def _roughness(epsilon: np.ndarray) -> tuple[float, np.ndarray]:
    dz = np.diff(epsilon, axis=0)
    dx = np.diff(epsilon, axis=1)
    value = 0.5 * (np.mean(dz * dz) + np.mean(dx * dx))
    gradient = np.zeros_like(epsilon)
    if dz.size:
        gradient[:-1, :] -= dz / dz.size
        gradient[1:, :] += dz / dz.size
    if dx.size:
        gradient[:, :-1] -= dx / dx.size
        gradient[:, 1:] += dx / dx.size
    return float(value), gradient


def _select_frequencies(
    available: np.ndarray,
    central: float,
    config: WavefrontConfig,
) -> np.ndarray:
    if config.frequencies is not None:
        requested = np.asarray(config.frequencies, dtype=np.float64)
        selected = []
        for value in requested:
            index = int(np.argmin(np.abs(available - value)))
            selected.append(float(available[index]))
        return np.unique(np.asarray(selected, dtype=np.float64))
    low = max(config.frequency_min_ratio * central, float(available[0]))
    high = min(config.frequency_max_ratio * central, float(available[-1]))
    if high <= low:
        return np.asarray([central], dtype=np.float64)
    candidates = available[(available >= low) & (available <= high)]
    if candidates.size <= config.frequency_count:
        return candidates
    indices = np.linspace(0, candidates.size - 1, config.frequency_count)
    return np.unique(candidates[np.round(indices).astype(int)])


def dwi_invert(
    observed: np.ndarray,
    geometry: Geometry,
    initial_epsilon: np.ndarray,
    config: WavefrontConfig | None = None,
) -> WavefrontResult:
    """Run the optional frequency-domain DWI.

    ``observed`` is the reconstructed diffraction section in ``(time, trace)``
    order.  ``initial_epsilon`` is the starting model in ``(depth, trace)``
    order.  The model is resampled to a grid with spacing ``dx`` (or
    ``config.dz``) before inversion.
    """
    config = config or WavefrontConfig()
    section = np.asarray(observed, dtype=np.float64)
    if section.ndim != 2:
        raise ValueError("observed must be a (time, trace) section.")
    samples, traces = section.shape
    start = np.asarray(initial_epsilon, dtype=np.float64)
    if start.ndim != 2:
        raise ValueError("initial_epsilon must be 2-D.")
    if start.shape[1] != traces:
        raise ValueError(
            f"initial_epsilon has {start.shape[1]} traces but the section has {traces}."
        )

    dz = float(config.dz) if config.dz is not None else geometry.dx
    if config.max_depth is None:
        max_time = geometry.t0 + (samples - 1) * geometry.dt
        max_depth = C0 * max_time / (2.0 * np.sqrt(config.epsilon_min))
    else:
        max_depth = float(config.max_depth)
    nz = max(int(np.ceil(max_depth / dz)) + 1, 3)
    depth = np.arange(nz, dtype=np.float64) * dz
    x = geometry.x_axis(traces)
    depth_in = C0 * geometry.time_axis(start.shape[0]) / (
        2.0 * np.sqrt(max(np.median(start), 1.0))
    )
    epsilon = _resample_model(start, depth_in, depth, x, x)
    epsilon = np.clip(epsilon, config.epsilon_min, config.epsilon_max)

    # Time taper and spectra.
    window = _tukey(samples, config.time_taper_fraction)
    spectrum = np.fft.rfft(section * window[:, None], axis=0)
    available = np.fft.rfftfreq(samples, geometry.dt)
    frequencies = _select_frequencies(available, geometry.frequency, config)
    frequency_indices = [int(np.argmin(np.abs(available - value))) for value in frequencies]
    frequencies = available[frequency_indices]
    observed_freq = spectrum[frequency_indices, :]

    pml = int(config.pml_cells)
    source_row = min(max(pml + int(config.source_depth_cells), 0), nz - 1)
    shot_indices = np.arange(0, traces, int(config.shot_stride), dtype=np.int64)
    receiver_indices = np.arange(traces, dtype=np.int64)

    interior = np.ones((nz, traces), dtype=bool)
    if pml > 0:
        interior[:pml, :] = False
        interior[-pml:, :] = False
        interior[:, :pml] = False
        interior[:, -pml:] = False
    if not np.any(interior):
        interior[:] = True
    active = np.flatnonzero(interior.ravel())
    epsilon_active = epsilon.ravel()[active].copy()
    bounds = [(config.epsilon_min, config.epsilon_max)] * active.size

    history: list[dict[str, float]] = []
    total_iterations = 0

    for stage, (frequency, observed_spectrum) in enumerate(
        zip(frequencies, observed_freq, strict=True), start=1
    ):
        omega = 2.0 * np.pi * float(frequency)
        normalisation = float(np.sum(np.abs(observed_spectrum) ** 2))
        normalisation = max(normalisation, np.finfo(np.float64).eps)
        k_derivative = (omega / C0) ** 2

        def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
            model = epsilon.ravel().copy()
            model[active] = vector
            model = np.clip(model.reshape(nz, traces), config.epsilon_min, config.epsilon_max)
            matrix = _build_operator(model, omega, geometry.dx, dz, pml)
            solver = LinearSolver.build(matrix, backend=config.backend)
            data_misfit = 0.0
            gradient = np.zeros((nz, traces), dtype=np.float64)
            for trace in shot_indices:
                source = np.zeros(nz * traces, dtype=np.complex128)
                source[source_row * traces + int(trace)] = 1.0 / (geometry.dx * dz)
                field = solver.solve(source, trans="N").reshape(nz, traces)
                predicted = field[source_row, receiver_indices]
                calibration = np.vdot(predicted, observed_spectrum) / max(
                    np.vdot(predicted, predicted).real, np.finfo(np.float64).eps
                )
                residual = calibration * predicted - observed_spectrum
                data_misfit += float(np.sum(np.abs(residual) ** 2))
                adjoint_source = np.zeros(nz * traces, dtype=np.complex128)
                adjoint_source[source_row * traces + receiver_indices] = (
                    np.conj(calibration) * residual
                )
                adjoint = solver.solve(adjoint_source, trans="H").reshape(nz, traces)
                gradient -= np.real(np.conj(adjoint) * field) * k_derivative
            data_misfit /= normalisation * max(len(shot_indices), 1)
            gradient /= normalisation * max(len(shot_indices), 1)
            rough_value, rough_gradient = _roughness(model)
            value = data_misfit + config.regularization * rough_value
            gradient += config.regularization * rough_gradient
            return float(value), gradient.ravel()[active]

        def callback(vector: np.ndarray) -> None:
            nonlocal epsilon_active
            epsilon_active = np.asarray(vector, dtype=np.float64).copy()

        result = minimize(
            objective,
            epsilon_active,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": min(int(config.stage_iterations), int(config.max_iterations)),
                "ftol": 1.0e-10,
                "gtol": 1.0e-6,
                "maxls": 20,
            },
        )
        epsilon_active = np.asarray(result.x, dtype=np.float64).copy()
        updated = epsilon.ravel().copy()
        updated[active] = epsilon_active
        epsilon = np.clip(updated.reshape(nz, traces), config.epsilon_min, config.epsilon_max)
        total_iterations += int(result.nit)
        history.append(
            {
                "stage": float(stage),
                "frequency_hz": float(frequency),
                "objective": float(result.fun),
                "iterations": float(result.nit),
                "status": float(result.status),
            }
        )
        if config.verbose:
            print(
                f"DWI stage {stage}/{len(frequencies)}: f={frequency / 1e6:.1f} MHz, "
                f"J={result.fun:.6g}, nit={result.nit}, {result.message}"
            )

    if config.smooth_sigma > 0:
        epsilon = gaussian_filter(epsilon, sigma=(config.smooth_sigma, config.smooth_sigma), mode="nearest")
    return WavefrontResult(
        epsilon=epsilon,
        depth=depth,
        x=x,
        frequencies=np.asarray(frequencies, dtype=np.float64),
        history=history,
        iterations=total_iterations,
    )

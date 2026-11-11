"""Diffraction-focusing inversion with isolated hyperbolic windows.

For each candidate relative permittivity the reconstructed diffraction field is
isolated with hyperbolic windows, migrated with the corresponding constant
velocity and scored by the local gradient-focusing ratios ``R1`` and ``R2`` of
Lv & Zhang (2023).  The permittivity that maximises both ratios at a candidate
diffractor is retained; all accepted points are then interpolated into a
regular dielectric model.

The default ``stolt`` migrator performs one 2-D FFT per candidate permittivity.
``kirchhoff`` is available as a slower independent check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from .config import C0, FocusingConfig, Geometry
from .migration import depth_axis, migrate


@dataclass
class FocusingResult:
    """Dielectric model and diagnostics from :func:`invert_focusing`."""

    epsilon: np.ndarray
    confidence: np.ndarray
    depth: np.ndarray
    x: np.ndarray
    point_x: np.ndarray
    point_depth: np.ndarray
    point_epsilon: np.ndarray
    point_confidence: np.ndarray
    point_energy: np.ndarray
    point_valid: np.ndarray
    epsilon_grid: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    energy: np.ndarray
    time_indices: np.ndarray
    trace_indices: np.ndarray


def _integral_image(values: np.ndarray) -> np.ndarray:
    result = np.zeros((values.shape[0] + 1, values.shape[1] + 1), dtype=np.float64)
    result[1:, 1:] = np.cumsum(np.cumsum(values, axis=0), axis=1)
    return result


def _box_sum(integral: np.ndarray, row0: int, row1: int, col0: int, col1: int) -> float:
    rows, cols = integral.shape[0] - 1, integral.shape[1] - 1
    row0 = int(np.clip(row0, 0, rows))
    row1 = int(np.clip(row1, 0, rows))
    col0 = int(np.clip(col0, 0, cols))
    col1 = int(np.clip(col1, 0, cols))
    if row1 <= row0 or col1 <= col0:
        return 0.0
    return float(
        integral[row1, col1]
        - integral[row0, col1]
        - integral[row1, col0]
        + integral[row0, col0]
    )


def _focusing_ratios(
    image: np.ndarray,
    points: np.ndarray,
    focus_u: int,
    focus_v: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``R1``, ``R2`` and centre energy ``P`` at every analysis point."""
    grad_t = np.gradient(image, axis=0)
    grad_x = np.gradient(image, axis=1)
    variation = np.abs(grad_t) + np.abs(grad_x)
    integral = _integral_image(variation)

    u = int(focus_u)
    v = int(focus_v)
    count = points.shape[0]
    r1 = np.zeros(count, dtype=np.float64)
    r2 = np.zeros(count, dtype=np.float64)
    energy = np.zeros(count, dtype=np.float64)
    for index, (row, col) in enumerate(points):
        p = _box_sum(integral, row - u, row + u + 1, col - v, col + v + 1)
        above = _box_sum(integral, row - 2 * u, row - u, col - v, col + v + 1)
        below = _box_sum(integral, row + u + 1, row + 2 * u + 1, col - v, col + v + 1)
        left = _box_sum(integral, row - u, row + u + 1, col - 2 * v, col - v)
        right = _box_sum(integral, row - u, row + u + 1, col + v + 1, col + 2 * v + 1)
        surround = above + below + left + right
        r1[index] = p / max(p + surround, np.finfo(np.float64).eps)
        r2[index] = p / max(surround, np.finfo(np.float64).eps)
        energy[index] = p
    return r1, r2, energy


def _analysis_points(
    samples: int,
    traces: int,
    geometry: Geometry,
    config: FocusingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = int(config.focus_u)
    v = int(config.focus_v)
    mute = int(np.floor(config.mute_time / geometry.dt))
    row_start = max(mute, 2 * u + 1)
    row_stop = samples - (2 * u + 1)
    col_start = max(2 * v + 1, 1)
    col_stop = traces - (2 * v + 1)
    if row_stop <= row_start or col_stop <= col_start:
        raise ValueError("Focusing analysis window is empty; reduce focus_u/focus_v or mute_time.")

    if config.max_depth is not None:
        t_max = 2.0 * float(config.max_depth) * np.sqrt(config.epsilon_max) / C0
        row_stop = min(row_stop, int(np.ceil((t_max - geometry.t0) / geometry.dt)))

    rows = np.arange(row_start, max(row_stop, row_start + 1), int(config.stride_time), dtype=np.int64)
    cols = np.arange(col_start, max(col_stop, col_start + 1), int(config.stride_trace), dtype=np.int64)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    points = np.column_stack((grid_rows.ravel(), grid_cols.ravel()))
    return points, rows, cols


def _isolation_mask(
    shape: tuple[int, int],
    points: np.ndarray,
    epsilon: float,
    geometry: Geometry,
    half_width: int,
) -> np.ndarray:
    """Union of hyperbolic windows for all analysis points.

    The mask is accumulated column by column with a difference array.  This
    avoids materialising one full hyperbola image per analysis point, which
    would dominate the run time for a dense point grid.
    """
    samples, traces = shape
    mask = np.zeros(shape, dtype=bool)
    factor = 4.0 * float(epsilon) / (C0 * C0 * geometry.dt * geometry.dt)
    rows = points[:, 0].astype(np.float64)
    cols = points[:, 1].astype(np.float64)
    width = float(half_width)
    for trace in range(traces):
        hyperbola = np.sqrt(rows * rows + factor * (float(trace) - cols) ** 2)
        low = np.clip(np.floor(hyperbola - width), 0, samples).astype(np.int64)
        high = np.clip(np.ceil(hyperbola + width) + 1, 0, samples).astype(np.int64)
        difference = np.zeros(samples + 1, dtype=np.int32)
        np.add.at(difference, low, 1)
        np.add.at(difference, high, -1)
        mask[:, trace] = np.cumsum(difference[:-1]) > 0
    return mask


def _assemble_model(
    points_x: np.ndarray,
    points_z: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    shape: tuple[int, int],
    depth: np.ndarray,
    x: np.ndarray,
    *,
    fallback: np.ndarray | None,
    epsilon_min: float,
    epsilon_max: float,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    samples, traces = shape
    if np.any(valid):
        grid_x, grid_z = np.meshgrid(x, depth)
        point_values = np.clip(values[valid], epsilon_min, epsilon_max)
        coordinates = np.column_stack((points_x[valid], points_z[valid]))
        linear = griddata(coordinates, point_values, (grid_x, grid_z), method="linear")
        nearest = griddata(coordinates, point_values, (grid_x, grid_z), method="nearest")
        model = np.where(np.isfinite(linear), linear, nearest)
    else:
        model = np.full(shape, np.nan, dtype=np.float64)

    if fallback is not None:
        fallback_values = np.asarray(fallback, dtype=np.float64)
        if fallback_values.shape != shape:
            raise ValueError(
                f"fallback model shape {fallback_values.shape} does not match {shape}."
            )
        model = np.where(np.isfinite(model), model, fallback_values)
    if not np.isfinite(model).all():
        finite = model[np.isfinite(model)]
        base = float(np.median(finite)) if finite.size else 0.5 * (epsilon_min + epsilon_max)
        model = np.where(np.isfinite(model), model, base)

    if smooth_sigma > 0:
        model = gaussian_filter(model, sigma=(smooth_sigma, smooth_sigma), mode="nearest")
    return np.clip(model, epsilon_min, epsilon_max), np.isfinite(model)


def invert_focusing(
    diffraction: np.ndarray,
    geometry: Geometry,
    config: FocusingConfig | None = None,
    *,
    initial_epsilon: np.ndarray | None = None,
) -> FocusingResult:
    """Invert a reconstructed diffraction section for relative permittivity."""
    config = config or FocusingConfig()
    section = np.asarray(diffraction, dtype=np.float64)
    if section.ndim != 2:
        raise ValueError("Diffraction section must be a (time, trace) array.")
    samples, traces = section.shape
    if initial_epsilon is not None and np.asarray(initial_epsilon).shape != section.shape:
        raise ValueError("initial_epsilon must have the same shape as the diffraction section.")

    points, rows, cols = _analysis_points(samples, traces, geometry, config)
    epsilon_values = config.epsilon_values
    r1_all = np.zeros((epsilon_values.size, points.shape[0]), dtype=np.float64)
    r2_all = np.zeros_like(r1_all)
    energy_all = np.zeros_like(r1_all)

    for index, epsilon in enumerate(epsilon_values):
        velocity = C0 / np.sqrt(float(epsilon))
        if config.isolation:
            mask = _isolation_mask(
                section.shape,
                points,
                float(epsilon),
                geometry,
                config.window_half_samples,
            )
            working = section * mask
        else:
            working = section
        image = migrate(
            working,
            geometry.dt,
            geometry.dx,
            velocity,
            method=config.migration,
            remove_mean=False,
        )
        r1, r2, energy = _focusing_ratios(image, points, config.focus_u, config.focus_v)
        r1_all[index, :] = r1
        r2_all[index, :] = r2
        energy_all[index, :] = energy

    r1_norm = r1_all / np.maximum(np.max(r1_all, axis=0, keepdims=True), np.finfo(np.float64).eps)
    r2_norm = r2_all / np.maximum(np.max(r2_all, axis=0, keepdims=True), np.finfo(np.float64).eps)
    score = np.sqrt(np.clip(r1_norm, 0.0, None) * np.clip(r2_norm, 0.0, None))
    peak_index = np.argmax(score, axis=0)
    columns = np.arange(points.shape[0])
    peak_score = score[peak_index, columns]
    background_score = np.median(score, axis=0)
    confidence = peak_score / np.maximum(background_score, np.finfo(np.float64).eps)
    peak_energy = energy_all[peak_index, columns]
    energy_threshold = np.percentile(peak_energy, config.min_energy_percentile) if peak_energy.size else 0.0
    r1_peak_index = np.argmax(r1_all, axis=0)
    r2_peak_index = np.argmax(r2_all, axis=0)
    simultaneous = (np.abs(r1_peak_index - peak_index) <= 1) & (
        np.abs(r2_peak_index - peak_index) <= 1
    )
    valid = (
        (confidence >= config.min_confidence)
        & (peak_energy >= energy_threshold)
        & simultaneous
    )
    if not np.any(valid):
        # Keep the strongest focusing points rather than returning an empty model.
        keep = max(1, min(5, points.shape[0]))
        strongest = np.argsort(confidence)[-keep:]
        valid[strongest] = True

    point_epsilon = epsilon_values[peak_index]
    point_time = geometry.t0 + points[:, 0].astype(np.float64) * geometry.dt
    point_x = geometry.x_axis(traces)[points[:, 1]]
    point_depth = C0 * point_time / (2.0 * np.sqrt(np.clip(point_epsilon, 1.0, None)))

    finite = point_epsilon[valid]
    reference_epsilon = float(np.median(finite)) if finite.size else 0.5 * (
        config.epsilon_min + config.epsilon_max
    )
    depth = C0 * geometry.time_axis(samples) / (2.0 * np.sqrt(max(reference_epsilon, 1.0)))
    x_axis = geometry.x_axis(traces)

    fallback = None if initial_epsilon is None else np.asarray(initial_epsilon, dtype=np.float64)
    model, model_valid = _assemble_model(
        point_x,
        point_depth,
        point_epsilon,
        valid,
        (samples, traces),
        depth,
        x_axis,
        fallback=fallback,
        epsilon_min=config.epsilon_min,
        epsilon_max=config.epsilon_max,
        smooth_sigma=config.smooth_sigma,
    )

    confidence_points = np.where(valid, confidence, 0.0)
    confidence_grid, _ = _assemble_model(
        point_x,
        point_depth,
        confidence_points,
        valid,
        (samples, traces),
        depth,
        x_axis,
        fallback=np.zeros((samples, traces), dtype=np.float64),
        epsilon_min=0.0,
        epsilon_max=np.inf,
        smooth_sigma=config.smooth_sigma,
    )

    return FocusingResult(
        epsilon=model,
        confidence=confidence_grid,
        depth=depth,
        x=x_axis,
        point_x=point_x,
        point_depth=point_depth,
        point_epsilon=point_epsilon,
        point_confidence=confidence,
        point_energy=peak_energy,
        point_valid=valid,
        epsilon_grid=epsilon_values,
        r1=r1_all,
        r2=r2_all,
        energy=energy_all,
        time_indices=points[:, 0],
        trace_indices=points[:, 1],
    )


def migrate_section(
    diffraction: np.ndarray,
    geometry: Geometry,
    epsilon: float,
    *,
    method: str = "stolt",
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning a migrated section and its depth axis."""
    velocity = C0 / np.sqrt(float(epsilon))
    image = migrate(diffraction, geometry.dt, geometry.dx, velocity, method=method)
    return image, depth_axis(diffraction.shape[0], geometry.dt, velocity)

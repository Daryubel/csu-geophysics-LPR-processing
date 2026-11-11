"""Diffraction velocity analysis and initial dielectric-model construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, map_coordinates

from .config import C0, Geometry, VelocityConfig


@dataclass
class VelocitySpectrum:
    """Semblance of hyperbolic diffraction moveout over a permittivity grid."""

    values: np.ndarray          # (time_pick, trace_pick, epsilon)
    epsilon: np.ndarray
    time: np.ndarray            # centre times, seconds
    x: np.ndarray               # centre positions, metres
    time_indices: np.ndarray
    trace_indices: np.ndarray

    def normalised(self) -> np.ndarray:
        peak = np.max(np.abs(self.values), axis=2, keepdims=True)
        return np.clip(self.values / np.maximum(peak, np.finfo(np.float64).eps), 0.0, 1.0)


@dataclass
class VelocityPicks:
    """Picked apparent permittivity and confidence."""

    time: np.ndarray
    x: np.ndarray
    epsilon: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    peak_index: np.ndarray
    epsilon_grid: np.ndarray


def _hyperbolic_time_samples(
    t0_samples: np.ndarray,
    offsets_m: np.ndarray,
    epsilon: float,
    dt: float,
) -> np.ndarray:
    """Two-way diffraction time in samples for a candidate permittivity."""
    factor = 4.0 * float(epsilon) / (C0 * C0 * dt * dt)
    return np.sqrt(
        t0_samples[:, None] ** 2 + factor * offsets_m[None, :] ** 2
    )


def velocity_spectrum(
    data: np.ndarray,
    geometry: Geometry,
    config: VelocityConfig | None = None,
) -> VelocitySpectrum:
    """Scan hyperbolic moveout of a diffraction section.

    For every centre time and trace, the section is stacked along a family of
    hyperbolas parameterised by relative permittivity.  The resulting
    ``(time, trace, epsilon)`` volume is the zero-offset velocity spectrum.
    """
    config = config or VelocityConfig()
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Velocity analysis expects a (time, trace) section.")
    samples, traces = values.shape
    if traces < config.aperture_traces:
        raise ValueError(
            f"Section has {traces} traces but aperture_traces={config.aperture_traces}."
        )

    mute_samples = int(np.floor(config.mute_time / geometry.dt))
    t_indices = np.arange(
        max(mute_samples, 0),
        samples,
        int(config.time_stride),
        dtype=np.int64,
    )
    half = int(config.aperture_traces) // 2
    x_indices = np.arange(
        half,
        traces - half,
        int(config.trace_stride),
        dtype=np.int64,
    )
    if t_indices.size == 0 or x_indices.size == 0:
        raise ValueError("Velocity scan grid is empty; reduce mute/stride values.")

    epsilon_values = config.epsilon_values
    spectrum = np.zeros((t_indices.size, x_indices.size, epsilon_values.size), dtype=np.float64)
    x_axis = geometry.x_axis(traces)

    for ix, trace_index in enumerate(x_indices):
        lo = trace_index - half
        hi = trace_index + half + 1
        offsets = x_axis[lo:hi] - x_axis[trace_index]
        local = values[:, lo:hi]
        trace_columns = np.broadcast_to(
            np.arange(hi - lo, dtype=np.float64)[None, :],
            (t_indices.size, hi - lo),
        )
        for ie, epsilon in enumerate(epsilon_values):
            arrival = _hyperbolic_time_samples(
                t_indices.astype(np.float64),
                offsets,
                float(epsilon),
                geometry.dt,
            )
            sampled = map_coordinates(
                local,
                [arrival, trace_columns],
                order=1,
                mode="constant",
                cval=0.0,
            )
            spectrum[:, ix, ie] = np.sum(sampled * sampled, axis=1) / (hi - lo)

    return VelocitySpectrum(
        values=spectrum,
        epsilon=epsilon_values,
        time=geometry.t0 + t_indices * geometry.dt,
        x=x_axis[x_indices],
        time_indices=t_indices,
        trace_indices=x_indices,
    )


def pick_velocity(
    spectrum: VelocitySpectrum,
    config: VelocityConfig | None = None,
) -> VelocityPicks:
    """Pick the permittivity that maximises the local stacking amplitude."""
    config = config or VelocityConfig()
    values = np.asarray(spectrum.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("Velocity spectrum must be a 3-D array.")

    peak_index = np.argmax(values, axis=2)
    peak = np.take_along_axis(values, peak_index[..., None], axis=2)[..., 0]
    background = np.median(values, axis=2)
    confidence = peak / np.maximum(background, np.finfo(np.float64).eps)
    epsilon = np.asarray(spectrum.epsilon, dtype=np.float64)[peak_index]

    epsilon = median_filter(
        epsilon,
        size=(max(int(config.smooth_time_samples), 1), max(int(config.smooth_trace_samples), 1)),
        mode="nearest",
    )
    confidence = median_filter(
        confidence,
        size=(max(int(config.smooth_time_samples), 1), max(int(config.smooth_trace_samples), 1)),
        mode="nearest",
    )

    threshold = float(np.percentile(confidence, config.confidence_percentile))
    valid = np.isfinite(epsilon) & (confidence >= threshold)
    return VelocityPicks(
        time=spectrum.time,
        x=spectrum.x,
        epsilon=epsilon,
        confidence=confidence,
        valid=valid,
        peak_index=peak_index,
        epsilon_grid=np.asarray(spectrum.epsilon, dtype=np.float64),
    )


def initial_model_from_picks(
    picks: VelocityPicks,
    geometry: Geometry,
    shape: tuple[int, int],
    *,
    epsilon_min: float = 1.0,
    epsilon_max: float = 20.0,
    depth: np.ndarray | None = None,
    max_depth: float | None = None,
    smooth_sigma: float = 1.0,
    fill_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate picked apparent permittivity onto a regular depth grid.

    The depth of a pick is ``z = c * t0 / (2 * sqrt(epsilon))``.  This is an
    apparent (stacking) model, not a Dix interval model; it is used as the
    starting model for the diffraction-focusing inversion.
    """
    samples, traces = shape
    if depth is None:
        if max_depth is None:
            max_time = float(picks.time.max()) if picks.time.size else 0.0
            max_depth = C0 * max_time / (2.0 * np.sqrt(max(epsilon_min, 1.0)))
        depth = np.arange(samples, dtype=np.float64) * float(max_depth) / max(samples - 1, 1)
    depth = np.asarray(depth, dtype=np.float64)
    if depth.shape[0] != samples:
        raise ValueError("depth must have one value per model row.")

    x_grid = geometry.x_axis(traces)
    z_grid = depth

    valid = np.asarray(picks.valid, dtype=bool)
    if valid.any():
        t0 = picks.time[:, None] * np.ones((1, picks.x.size))
        x0 = np.ones((picks.time.size, 1)) * picks.x[None, :]
        eps = np.asarray(picks.epsilon, dtype=np.float64)
        z_pick = C0 * t0 / (2.0 * np.sqrt(np.clip(eps, epsilon_min, epsilon_max)))
        points = np.column_stack((x0[valid], z_pick[valid]))
        point_values = np.clip(eps[valid], epsilon_min, epsilon_max)
        grid_x, grid_z = np.meshgrid(x_grid, z_grid)
        linear = griddata(points, point_values, (grid_x, grid_z), method="linear")
        nearest = griddata(points, point_values, (grid_x, grid_z), method="nearest")
        model = np.where(np.isfinite(linear), linear, nearest)
        confidence_points = np.asarray(picks.confidence, dtype=np.float64)[valid]
        confidence_grid = griddata(
            points, confidence_points, (grid_x, grid_z), method="linear"
        )
        confidence_grid = np.where(np.isfinite(confidence_grid), confidence_grid, 0.0)
    else:
        base = float(np.median(picks.epsilon[valid])) if valid.any() else 0.5 * (epsilon_min + epsilon_max)
        model = np.full((samples, traces), base, dtype=np.float64)
        confidence_grid = np.zeros_like(model)

    if fill_value is not None:
        model = np.where(np.isfinite(model), model, fill_value)
    model = np.clip(model, epsilon_min, epsilon_max)
    if smooth_sigma > 0:
        model = gaussian_filter(model, sigma=(smooth_sigma, smooth_sigma), mode="nearest")
        confidence_grid = gaussian_filter(
            confidence_grid, sigma=(smooth_sigma, smooth_sigma), mode="nearest"
        )
    return np.ascontiguousarray(model), np.ascontiguousarray(confidence_grid)


def dix_interval_epsilon(
    picks: VelocityPicks,
    epsilon_min: float = 1.0,
    epsilon_max: float = 20.0,
) -> np.ndarray:
    """Convert apparent stacking permittivity to a Dix interval value.

    A stable, clipped approximation is returned; the first sample keeps its
    apparent value.  Use this only when the picks follow a layered geometry.
    """
    apparent = np.asarray(picks.epsilon, dtype=np.float64)
    time = np.asarray(picks.time, dtype=np.float64)
    interval = apparent.copy()
    for index in range(1, apparent.shape[0]):
        dt = time[index] - time[index - 1]
        if dt <= 0:
            continue
        v2_now = (C0 / np.sqrt(np.clip(apparent[index], epsilon_min, epsilon_max))) ** 2
        v2_prev = (C0 / np.sqrt(np.clip(apparent[index - 1], epsilon_min, epsilon_max))) ** 2
        v2_interval = (v2_now * time[index] - v2_prev * time[index - 1]) / dt
        if v2_interval > 0.0 and np.isfinite(v2_interval):
            interval[index] = (C0 / np.sqrt(v2_interval)) ** 2
    return np.clip(interval, epsilon_min, epsilon_max)

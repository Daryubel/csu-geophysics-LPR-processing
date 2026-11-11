"""Coherence-based diffraction reconstruction.

The reflection part of a zero-offset section is modelled as a locally
coherent, plane-wave stack.  At every image point the local time shift (dip)
that maximises semblance is estimated; the aligned stack is then subtracted
from the input and the residual is retained as the diffraction field.

This is a compact implementation of the coherence-analysis idea used by
Schwarz (2019), Zhang et al. (2023) and He et al. (2023).  It is deliberately
data-driven: no velocity model is required for the reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter

from .config import CoherenceConfig, Geometry
from .io import Survey, agc, bandpass, mute_early


@dataclass
class DiffractionReconstruction:
    """Output of :func:`reconstruct_diffraction`."""

    input: np.ndarray
    diffraction: np.ndarray
    reflection: np.ndarray
    coherence: np.ndarray
    tau: np.ndarray
    alpha: np.ndarray
    geometry: Geometry

    @property
    def tau_seconds_per_metre(self) -> np.ndarray:
        return self.tau * self.geometry.dt / self.geometry.dx


def _shift_time(data: np.ndarray, shift_samples: float) -> np.ndarray:
    """Shift every trace by a constant number of samples (linear interpolation)."""
    values = np.asarray(data, dtype=np.float64)
    if shift_samples == 0.0:
        return values
    samples, traces = values.shape
    rows = np.broadcast_to(
        np.arange(samples, dtype=np.float64)[:, None] + float(shift_samples),
        values.shape,
    )
    columns = np.broadcast_to(np.arange(traces, dtype=np.float64)[None, :], values.shape)
    return map_coordinates(values, [rows, columns], order=1, mode="nearest")


def _shift_traces(data: np.ndarray, offset: int) -> np.ndarray:
    """At trace ``i`` return trace ``i + offset`` (edge values are replicated)."""
    if offset == 0:
        return data
    values = np.asarray(data, dtype=np.float64)
    pad = abs(int(offset))
    padded = np.pad(values, ((0, 0), (pad, pad)), mode="edge")
    trace_count = values.shape[1]
    if offset > 0:
        return padded[:, offset : offset + trace_count]
    return padded[:, pad + offset : pad + offset + trace_count]


def _local_ratio(numerator: np.ndarray, denominator: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    num = uniform_filter(numerator, size=size, mode="nearest")
    den = uniform_filter(denominator, size=size, mode="nearest")
    floor = np.finfo(np.float64).eps
    return num / np.maximum(den, floor)


def reconstruct_diffraction(
    survey: Survey,
    config: CoherenceConfig | None = None,
    *,
    preprocess: bool = True,
) -> DiffractionReconstruction:
    """Estimate and remove the locally coherent reflection field.

    Parameters
    ----------
    survey:
        Input B-scan in ``(time, trace)`` orientation.
    config:
        Coherence-analysis parameters.  Defaults are suitable for the
        packaged 500 MHz example.
    preprocess:
        If true, remove the trace mean, apply the optional band-pass and AGC,
        and mute the direct arrival before the analysis.
    """
    config = config or CoherenceConfig()
    geometry = survey.geometry
    values = np.asarray(survey.data, dtype=np.float64)
    if preprocess:
        if config.bandpass_low is not None or config.bandpass_high is not None:
            values = bandpass(values, geometry.dt, config.bandpass_low, config.bandpass_high)
        if config.agc_window > 0:
            values = agc(values, config.agc_window)
        values = mute_early(values, geometry.dt, config.mute_time)
        values = values - np.mean(values, axis=0, keepdims=True)

    samples, traces = values.shape
    aperture = int(config.aperture_traces)
    if traces < 3:
        raise ValueError("Coherence analysis requires at least three traces.")

    slopes = np.arange(
        -int(config.max_shift),
        int(config.max_shift) + 1,
        int(config.shift_step),
        dtype=np.float64,
    )
    if slopes.size == 0:
        slopes = np.zeros(1, dtype=np.float64)

    window = (int(config.time_window), int(aperture))
    best = np.zeros(values.shape, dtype=np.float64)
    best_tau = np.zeros(values.shape, dtype=np.float64)
    best_stack = np.zeros(values.shape, dtype=np.float64)

    offsets = range(-(aperture // 2), aperture // 2 + 1)
    for slope in slopes:
        stack = np.zeros_like(values)
        energy = np.zeros_like(values)
        for offset in offsets:
            shifted = _shift_time(values, slope * offset)
            shifted = _shift_traces(shifted, offset)
            stack += shifted
            energy += shifted * shifted
        count = float(len(list(offsets)))
        stack /= count
        energy /= count

        stack_smooth = uniform_filter(stack, size=window, mode="nearest")
        energy_smooth = uniform_filter(energy, size=window, mode="nearest")
        semblance = np.clip(
            stack_smooth * stack_smooth / np.maximum(energy_smooth, np.finfo(np.float64).eps),
            0.0,
            1.0,
        )
        update = semblance > best
        best[update] = semblance[update]
        best_tau[update] = slope
        best_stack[update] = stack_smooth[update]

    # The local amplitude ratio is retained as a diagnostic.  The reflection
    # estimate itself is the aligned coherent stack: using a hard alpha gate
    # here would leave weak side-lobe energy in the diffraction field.
    alpha = np.clip(
        _local_ratio(values * best_stack, best_stack * best_stack, window),
        0.0,
        2.0,
    )
    reflection = best_stack
    diffraction = values - reflection

    # Preserve the input amplitude convention in the returned field.
    return DiffractionReconstruction(
        input=values,
        diffraction=diffraction,
        reflection=reflection,
        coherence=best,
        tau=best_tau,
        alpha=alpha,
        geometry=geometry,
    )


def coherence_edges(
    coherence: np.ndarray,
    *,
    sigma: float = 1.0,
    percentile: float = 90.0,
) -> np.ndarray:
    """Return a simple thresholded gradient map of a coherence/tau section.

    The result is a binary map used only as a geometric guide by the
    sequential layer analysis; it is not part of the inversion objective.
    """
    from scipy.ndimage import gaussian_filter, sobel

    values = np.asarray(coherence, dtype=np.float64)
    smooth = gaussian_filter(values, sigma=float(sigma)) if sigma > 0 else values
    grad_x = sobel(smooth, axis=1, mode="nearest")
    grad_y = sobel(smooth, axis=0, mode="nearest")
    magnitude = np.hypot(grad_x, grad_y)
    threshold = float(np.percentile(magnitude, percentile))
    return magnitude >= threshold

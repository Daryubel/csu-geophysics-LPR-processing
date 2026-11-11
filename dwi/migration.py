"""Constant-velocity migration operators used by the focusing inversion.

Two complementary migrators are provided:

``stolt``
    Stolt f-k migration.  Fast, one 2-D FFT per velocity, and the default
    choice for the diffraction-focusing scan.
``kirchhoff``
    Direct zero-offset Kirchhoff/diffraction stacking.  Slower but useful as
    an independent check and for irregular or very small grids.

Both operators return an image on the same ``(time, trace)`` grid as the
input.  The vertical coordinate is the migrated two-way time converted to
depth with ``z = v * t / 2``; therefore image sample ``i`` corresponds to the
apex time sample ``i`` of the input section.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len
from scipy.ndimage import map_coordinates


def _prepare(section: np.ndarray) -> np.ndarray:
    values = np.asarray(section, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Migration input must be 2-D; got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Migration input contains NaN or infinite values.")
    return values


def stolt_migrate(
    section: np.ndarray,
    dt: float,
    dx: float,
    velocity: float,
    *,
    pad_time: float = 2.0,
    pad_space: float = 2.0,
    remove_mean: bool = True,
) -> np.ndarray:
    """Stolt constant-velocity migration of a zero-offset section.

    Parameters
    ----------
    section:
        ``(time, trace)`` array.
    dt, dx:
        Sample interval (s) and trace spacing (m).
    velocity:
        Constant migration velocity (m/s).
    pad_time, pad_space:
        Zero-padding factors before the 2-D FFT.
    remove_mean:
        Remove the per-trace mean before migration to suppress the zero-
        wavenumber artifact of causal GPR records.

    Returns
    -------
    ndarray
        Real migrated image with the same shape as ``section``.
    """
    values = _prepare(section)
    if velocity <= 0.0 or not np.isfinite(velocity):
        raise ValueError("velocity must be positive and finite.")
    if dt <= 0.0 or dx <= 0.0:
        raise ValueError("dt and dx must be positive.")
    if remove_mean:
        values = values - np.mean(values, axis=0, keepdims=True)

    samples, traces = values.shape
    n_time = next_fast_len(max(int(samples * max(pad_time, 1.0)), samples))
    n_space = next_fast_len(max(int(traces * max(pad_space, 1.0)), traces))
    padded = np.zeros((n_time, n_space), dtype=np.float64)
    padded[:samples, :traces] = values

    spectrum = np.fft.fft2(padded)
    kx = 2.0 * np.pi * np.fft.fftfreq(n_space, dx)
    dz = velocity * dt / 2.0
    kz = 2.0 * np.pi * np.fft.fftfreq(n_time, dz)
    kz_grid, kx_grid = np.meshgrid(kz, kx, indexing="ij")

    omega = 0.5 * velocity * np.sqrt(kz_grid * kz_grid + kx_grid * kx_grid)
    frequency = omega / (2.0 * np.pi)
    row_index = frequency * n_time * dt
    column_index = np.broadcast_to(np.arange(n_space, dtype=np.float64)[None, :], kz_grid.shape)

    real = map_coordinates(
        spectrum.real,
        [row_index, column_index],
        order=1,
        mode="constant",
        cval=0.0,
    )
    imag = map_coordinates(
        spectrum.imag,
        [row_index, column_index],
        order=1,
        mode="constant",
        cval=0.0,
    )
    migrated = real + 1j * imag

    radial = np.sqrt(kz_grid * kz_grid + kx_grid * kx_grid)
    obliquity = np.divide(
        np.abs(kz_grid),
        radial,
        out=np.zeros_like(radial),
        where=radial > 0.0,
    )
    migrated *= obliquity
    migrated[(frequency < 0.0) | (frequency > 0.5 / dt)] = 0.0

    # Build a Hermitian output spectrum from the positive-kz half.
    full = np.zeros_like(migrated)
    positive = n_time // 2 + 1
    full[:positive, :] = migrated[:positive, :]
    for index in range(1, positive):
        full[-index, :] = np.conj(full[index, ::-1])
    image = np.real(np.fft.ifft2(full))
    return np.ascontiguousarray(image[:samples, :traces])


def kirchhoff_migrate(
    section: np.ndarray,
    dt: float,
    dx: float,
    velocity: float,
    *,
    aperture_traces: int | None = None,
    remove_mean: bool = True,
) -> np.ndarray:
    """Zero-offset Kirchhoff migration by hyperbolic summation.

    ``aperture_traces`` limits the number of input traces used for each output
    trace.  ``None`` uses all traces.  The output has the same shape and
    vertical sampling as :func:`stolt_migrate`.
    """
    values = _prepare(section)
    if velocity <= 0.0 or not np.isfinite(velocity):
        raise ValueError("velocity must be positive and finite.")
    if remove_mean:
        values = values - np.mean(values, axis=0, keepdims=True)

    samples, traces = values.shape
    x = np.arange(traces, dtype=np.float64) * dx
    z = velocity * np.arange(samples, dtype=np.float64) * dt / 2.0
    image = np.zeros_like(values)

    if aperture_traces is None or aperture_traces >= traces:
        input_indices = np.arange(traces)
        columns = np.broadcast_to(input_indices[None, :], (traces, traces)).astype(np.float64)
        use_all = True
    else:
        half = max(int(aperture_traces) // 2, 1)
        use_all = False
        columns = None

    for iz, depth in enumerate(z):
        if use_all:
            arrival = 2.0 * np.sqrt(
                (x[None, :] - x[:, None]) ** 2 + depth * depth
            ) / velocity
            sampled = map_coordinates(
                values,
                [arrival / dt, columns],
                order=1,
                mode="constant",
                cval=0.0,
            )
            image[iz, :] = sampled.sum(axis=1) / np.sqrt(traces)
        else:
            for itrace in range(traces):
                lo = max(0, itrace - half)
                hi = min(traces, itrace + half + 1)
                source_x = x[lo:hi]
                arrival = 2.0 * np.sqrt((source_x - x[itrace]) ** 2 + depth * depth) / velocity
                row = arrival / dt
                col = np.arange(lo, hi, dtype=np.float64)
                sampled = map_coordinates(
                    values,
                    [row, col],
                    order=1,
                    mode="constant",
                    cval=0.0,
                )
                image[iz, itrace] = sampled.sum() / np.sqrt(max(hi - lo, 1))

    return image


def migrate(
    section: np.ndarray,
    dt: float,
    dx: float,
    velocity: float,
    *,
    method: str = "stolt",
    **kwargs,
) -> np.ndarray:
    name = str(method).strip().lower()
    if name == "stolt":
        return stolt_migrate(section, dt, dx, velocity, **kwargs)
    if name == "kirchhoff":
        return kirchhoff_migrate(section, dt, dx, velocity, **kwargs)
    raise ValueError("method must be 'stolt' or 'kirchhoff'.")


def depth_axis(samples: int, dt: float, velocity: float) -> np.ndarray:
    """Depth axis corresponding to the migrated image rows."""
    return velocity * np.arange(int(samples), dtype=np.float64) * float(dt) / 2.0

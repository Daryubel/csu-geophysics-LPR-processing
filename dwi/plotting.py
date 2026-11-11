"""Publication-oriented diagnostic figures for the DWI workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _clip(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite) + 1.0e-12)
    return float(lo), float(hi)


def _extent(time: np.ndarray, x: np.ndarray) -> list[float]:
    dx = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 1.0
    dt = float(np.median(np.abs(np.diff(time)))) if time.size > 1 else 1.0
    # Time is displayed in nanoseconds, distance in metres.
    return [
        x[0] - dx / 2,
        x[-1] + dx / 2,
        (time[-1] + dt / 2) * 1.0e9,
        (time[0] - dt / 2) * 1.0e9,
    ]


def plot_reconstruction(
    original: np.ndarray,
    diffraction: np.ndarray,
    reflection: np.ndarray,
    coherence: np.ndarray,
    tau: np.ndarray,
    time: np.ndarray,
    x: np.ndarray,
    path: str | Path,
) -> None:
    extent = _extent(time, x)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5), constrained_layout=True)
    panels = [
        (original, "Input B-scan", "seismic", True),
        (diffraction, "Reconstructed diffraction", "seismic", True),
        (reflection, "Coherent reflection stack", "seismic", True),
        (coherence, "Semblance", "magma", False),
        (tau, "Local time shift (samples/trace)", "coolwarm", True),
    ]
    for axis, (values, title, cmap, symmetric) in zip(axes, panels, strict=True):
        if symmetric:
            lo, hi = _clip(values)
            image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=lo, vmax=hi, extent=extent)
        else:
            lo, hi = _clip(values, 0.0, 99.0)
            image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=lo, vmax=hi, extent=extent)
        axis.set_title(title)
        axis.set_xlabel("Distance (m)")
        axis.set_ylabel("Two-way time (ns)" if axis is axes[0] else "")
        axis.set_ylim(extent[2], extent[3])
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_velocity(
    spectrum,
    picks,
    initial_epsilon: np.ndarray,
    depth: np.ndarray,
    x: np.ndarray,
    path: str | Path,
) -> None:
    values = spectrum.normalised()
    # Display a representative midpoint trace of the spectrum.
    mid = values.shape[1] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    image = axes[0].imshow(
        values[:, mid, :].T,
        aspect="auto",
        origin="lower",
        extent=[spectrum.time[0] * 1e9, spectrum.time[-1] * 1e9, spectrum.epsilon[0], spectrum.epsilon[-1]],
        cmap="viridis",
    )
    axes[0].set_title("Diffraction velocity spectrum")
    axes[0].set_xlabel("Two-way time (ns)")
    axes[0].set_ylabel("Relative permittivity")
    fig.colorbar(image, ax=axes[0], shrink=0.8)

    pick_time, _pick_x = np.meshgrid(picks.time, picks.x, indexing="ij")
    axes[1].scatter(
        pick_time[picks.valid] * 1e9,
        picks.epsilon[picks.valid],
        c=picks.confidence[picks.valid],
        s=10,
        cmap="magma",
    )
    axes[1].set_title("Picked apparent permittivity")
    axes[1].set_xlabel("Two-way time (ns)")
    axes[1].set_ylabel("Relative permittivity")

    image = axes[2].imshow(
        initial_epsilon,
        aspect="auto",
        origin="upper",
        extent=[x[0], x[-1], depth[-1], depth[0]],
        cmap="viridis",
    )
    axes[2].set_title("Initial dielectric model")
    axes[2].set_xlabel("Distance (m)")
    axes[2].set_ylabel("Depth (m)")
    fig.colorbar(image, ax=axes[2], shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_focusing(result, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    image = axes[0].imshow(
        result.epsilon,
        aspect="auto",
        origin="upper",
        extent=[result.x[0], result.x[-1], result.depth[-1], result.depth[0]],
        cmap="turbo",
    )
    axes[0].set_title("Diffraction-focusing permittivity")
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Depth (m)")
    fig.colorbar(image, ax=axes[0], shrink=0.8)

    confidence = result.confidence.copy()
    confidence[~np.isfinite(confidence)] = 0.0
    image = axes[1].imshow(
        confidence,
        aspect="auto",
        origin="upper",
        extent=[result.x[0], result.x[-1], result.depth[-1], result.depth[0]],
        cmap="magma",
    )
    axes[1].set_title("Focusing confidence")
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Depth (m)")
    fig.colorbar(image, ax=axes[1], shrink=0.8)

    valid = result.point_valid
    scatter = axes[2].scatter(
        result.point_x[valid],
        result.point_depth[valid],
        c=result.point_epsilon[valid],
        s=18,
        cmap="turbo",
    )
    axes[2].set_title("Accepted focusing points")
    axes[2].set_xlabel("Distance (m)")
    axes[2].set_ylabel("Depth (m)")
    axes[2].invert_yaxis()
    fig.colorbar(scatter, ax=axes[2], shrink=0.8, label="Relative permittivity")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_properties(result, path: str | Path) -> None:
    maps = result.maps
    fields = [
        (maps.epsilon, "Relative permittivity", "turbo"),
        (maps.density, "Bulk density (g/cm$^3$)", "viridis"),
        (maps.loss_tangent, "Loss tangent", "magma"),
        (maps.chemistry, "FeO+TiO$_2$ (%)", "cividis"),
    ]
    extent = [result.x[0], result.x[-1], result.depth[-1], result.depth[0]]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), constrained_layout=True)
    for axis, (values, title, cmap) in zip(axes, fields, strict=True):
        image = axis.imshow(values, aspect="auto", origin="upper", extent=extent, cmap=cmap)
        for boundary in result.boundaries:
            axis.axhline(boundary, color="white", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Distance (m)")
        axis.set_ylabel("Depth (m)")
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(
    original: np.ndarray,
    diffraction: np.ndarray,
    epsilon: np.ndarray,
    depth: np.ndarray,
    x: np.ndarray,
    time: np.ndarray,
    boundaries: np.ndarray,
    path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    lo, hi = _clip(original)
    axes[0].imshow(
        original,
        aspect="auto",
        cmap="seismic",
        vmin=lo,
        vmax=hi,
        extent=_extent(time, x),
    )
    axes[0].set_title("Input")
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Two-way time (ns)")
    axes[0].set_ylim(time[-1] * 1e9, time[0] * 1e9)

    lo, hi = _clip(diffraction)
    axes[1].imshow(
        diffraction,
        aspect="auto",
        cmap="seismic",
        vmin=lo,
        vmax=hi,
        extent=_extent(time, x),
    )
    axes[1].set_title("Diffraction")
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylim(time[-1] * 1e9, time[0] * 1e9)

    image = axes[2].imshow(
        epsilon,
        aspect="auto",
        origin="upper",
        extent=[x[0], x[-1], depth[-1], depth[0]],
        cmap="turbo",
    )
    for boundary in boundaries:
        axes[2].axhline(boundary, color="white", linestyle="--", linewidth=0.8)
    axes[2].set_title("Permittivity and layers")
    axes[2].set_xlabel("Distance (m)")
    axes[2].set_ylabel("Depth (m)")
    fig.colorbar(image, ax=axes[2], shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)

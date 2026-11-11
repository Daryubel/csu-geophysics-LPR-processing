"""Sequential medium-property analysis from an inverted dielectric model.

The empirical relations follow the lunar-soil correlations used in the
reference manuscript (Olhoeft & Strangway, 1975):

    rho      = ln(eps_r) / ln(1.919)
    tan(d)   = 10 ** (0.44 * rho - 2.943)
    FeO+TiO2 = (0.44 * rho - 0.312 * rho + 0.317) / 0.038

Here ``rho`` is bulk density in g/cm3, ``tan(d)`` is the loss tangent and the
last quantity is the combined FeO+TiO2 percentage.  These are empirical
lunar-soil laws, not universal terrestrial petrophysical relations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .config import PropertyConfig


@dataclass
class PropertyMaps:
    epsilon: np.ndarray
    density: np.ndarray
    loss_tangent: np.ndarray
    chemistry: np.ndarray


@dataclass
class PropertyResult:
    maps: PropertyMaps
    depth: np.ndarray
    x: np.ndarray
    boundaries: np.ndarray
    layers: list[dict[str, float]]
    epsilon_sigma: np.ndarray


def _validate_epsilon(epsilon: np.ndarray) -> np.ndarray:
    values = np.asarray(epsilon, dtype=np.float64)
    if values.ndim not in {1, 2}:
        raise ValueError(f"Relative permittivity must be 1-D or 2-D; got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Relative permittivity contains NaN or infinite values.")
    return np.clip(values, 1.000001, None)


def olhoeft_density(epsilon: np.ndarray, base: float = 1.919) -> np.ndarray:
    """Bulk density (g/cm3) from relative permittivity."""
    if base <= 1.0:
        raise ValueError("density_base must exceed 1.")
    return np.log(_validate_epsilon(epsilon)) / np.log(float(base))


def olhoeft_loss_tangent(epsilon: np.ndarray, base: float = 1.919) -> np.ndarray:
    """Loss tangent from relative permittivity."""
    rho = olhoeft_density(epsilon, base=base)
    return 10.0 ** (0.44 * rho - 2.943)


def olhoeft_chemistry(
    epsilon: np.ndarray,
    density: np.ndarray | None = None,
    base: float = 1.919,
    a: float = 0.44,
    b: float = 0.312,
    c: float = 0.317,
    d: float = 0.038,
) -> np.ndarray:
    """Combined FeO+TiO2 percentage from permittivity (and density)."""
    if d <= 0.0:
        raise ValueError("chemistry_d must be positive.")
    rho = olhoeft_density(epsilon, base=base) if density is None else np.asarray(density, dtype=np.float64)
    return ((a - b) * rho + c) / d


def properties_from_epsilon(
    epsilon: np.ndarray,
    config: PropertyConfig | None = None,
) -> PropertyMaps:
    """Convert a permittivity map to density, loss tangent and chemistry."""
    config = config or PropertyConfig()
    eps = _validate_epsilon(epsilon)
    density = olhoeft_density(eps, base=config.density_base)
    loss = 10.0 ** (
        config.loss_tangent_a * density - config.loss_tangent_b
    )
    chemistry = (
        (config.chemistry_a - config.chemistry_b) * density + config.chemistry_c
    ) / config.chemistry_d
    return PropertyMaps(epsilon=eps, density=density, loss_tangent=loss, chemistry=chemistry)


def _edge_profile(edge_map: np.ndarray) -> np.ndarray:
    values = np.asarray(edge_map, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("edge_map must be 2-D.")
    return np.mean(values, axis=1)


def detect_layer_boundaries(
    epsilon: np.ndarray,
    depth: np.ndarray,
    config: PropertyConfig | None = None,
    *,
    edge_map: np.ndarray | None = None,
) -> np.ndarray:
    """Detect layer interfaces from an edge map or from the epsilon gradient.

    Returns boundaries in metres, always starting at the first depth and
    ending at the last depth.  The detection is a geometric guide, not a
    probabilistic changepoint model.
    """
    config = config or PropertyConfig()
    eps = _validate_epsilon(epsilon)
    z = np.asarray(depth, dtype=np.float64)
    if z.ndim != 1 or z.size != eps.shape[0]:
        raise ValueError("depth must be a 1-D array with one value per model row.")
    if z.size < 3:
        return np.asarray([z[0], z[-1]], dtype=np.float64)

    if edge_map is not None:
        edge_values = np.asarray(edge_map, dtype=np.float64)
        if edge_values.shape != eps.shape:
            raise ValueError(
                f"edge_map shape {edge_values.shape} does not match epsilon {eps.shape}."
            )
        score = _edge_profile(edge_values)
    else:
        profile = np.median(eps, axis=1)
        score = np.abs(np.gradient(profile, z))

    score = gaussian_filter1d(score, sigma=max(config.edge_sigma, 0.0), mode="nearest")
    dz = float(np.median(np.abs(np.diff(z))))
    if dz <= 0.0:
        dz = 1.0
    distance = max(int(round(config.min_layer_thickness / dz)), 1)
    threshold = float(np.percentile(score, config.edge_percentile))
    peaks, properties = find_peaks(score, distance=distance, height=threshold)
    if peaks.size > config.max_layers - 1:
        order = np.argsort(properties["peak_heights"])[::-1][: config.max_layers - 1]
        peaks = np.sort(peaks[order])

    boundaries = [float(z[0])]
    for index in peaks:
        value = float(z[int(index)])
        if value - boundaries[-1] >= config.min_layer_thickness:
            boundaries.append(value)
    if z[-1] - boundaries[-1] >= config.min_layer_thickness:
        boundaries.append(float(z[-1]))
    elif len(boundaries) > 1:
        boundaries[-1] = float(z[-1])
    else:
        boundaries.append(float(z[-1]))
    return np.asarray(boundaries, dtype=np.float64)


def _monte_carlo_layer(
    epsilon_median: float,
    relative_sigma: float,
    config: PropertyConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    if relative_sigma <= 0.0:
        return {}
    samples = np.clip(
        epsilon_median * (1.0 + relative_sigma * rng.standard_normal(config.monte_carlo_samples)),
        1.000001,
        None,
    )
    density = olhoeft_density(samples, base=config.density_base)
    loss = 10.0 ** (config.loss_tangent_a * density - config.loss_tangent_b)
    chemistry = (
        (config.chemistry_a - config.chemistry_b) * density + config.chemistry_c
    ) / config.chemistry_d
    return {
        "density_p10": float(np.percentile(density, 10)),
        "density_p90": float(np.percentile(density, 90)),
        "loss_tangent_p10": float(np.percentile(loss, 10)),
        "loss_tangent_p90": float(np.percentile(loss, 90)),
        "chemistry_p10": float(np.percentile(chemistry, 10)),
        "chemistry_p90": float(np.percentile(chemistry, 90)),
    }


def layer_statistics(
    epsilon: np.ndarray,
    depth: np.ndarray,
    x: np.ndarray,
    boundaries: np.ndarray,
    config: PropertyConfig | None = None,
) -> list[dict[str, float]]:
    """Compute properties for every layer in depth order."""
    config = config or PropertyConfig()
    eps = _validate_epsilon(epsilon)
    z = np.asarray(depth, dtype=np.float64)
    lateral = np.asarray(x, dtype=np.float64)
    edges = np.asarray(boundaries, dtype=np.float64)
    if edges.size < 2:
        return []
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, float]] = []

    for layer_index in range(edges.size - 1):
        top = float(edges[layer_index])
        bottom = float(edges[layer_index + 1])
        mask = (z >= top) & (z < bottom)
        if layer_index == edges.size - 2:
            mask = (z >= top) & (z <= bottom)
        if not np.any(mask):
            continue
        values = eps[mask, :]
        flat = values.ravel()
        median = float(np.median(flat))
        properties = properties_from_epsilon(
            np.asarray([[median]], dtype=np.float64), config
        )
        row: dict[str, float] = {
            "layer": layer_index + 1,
            "top_m": top,
            "bottom_m": bottom,
            "thickness_m": bottom - top,
            "trace_count": float(values.shape[1]),
            "sample_count": float(values.size),
            "epsilon_mean": float(np.mean(flat)),
            "epsilon_median": median,
            "epsilon_std": float(np.std(flat)),
            "epsilon_p10": float(np.percentile(flat, 10)),
            "epsilon_p90": float(np.percentile(flat, 90)),
            "density_median": float(properties.density[0, 0]),
            "loss_tangent_median": float(properties.loss_tangent[0, 0]),
            "chemistry_median": float(properties.chemistry[0, 0]),
        }
        row.update(_monte_carlo_layer(median, config.relative_epsilon_sigma, config, rng))
        rows.append(row)
    return rows


def sequential_analysis(
    epsilon: np.ndarray,
    depth: np.ndarray,
    x: np.ndarray,
    config: PropertyConfig | None = None,
    *,
    edge_map: np.ndarray | None = None,
    boundaries: np.ndarray | None = None,
) -> PropertyResult:
    """Run the full property conversion and layer-by-layer interpretation.

    The layers are processed from the surface downwards; each layer is a
    contiguous depth interval with its own property distribution.  The result
    is intended for visualisation and hypothesis generation, not as a unique
    petrophysical inversion.
    """
    config = config or PropertyConfig()
    maps = properties_from_epsilon(epsilon, config)
    z = np.asarray(depth, dtype=np.float64)
    if boundaries is None:
        boundaries = detect_layer_boundaries(
            maps.epsilon, z, config, edge_map=edge_map
        )
    boundaries = np.asarray(boundaries, dtype=np.float64)
    layers = layer_statistics(maps.epsilon, z, x, boundaries, config)
    epsilon_sigma = config.relative_epsilon_sigma * maps.epsilon
    return PropertyResult(
        maps=maps,
        depth=z,
        x=np.asarray(x, dtype=np.float64),
        boundaries=boundaries,
        layers=layers,
        epsilon_sigma=epsilon_sigma,
    )

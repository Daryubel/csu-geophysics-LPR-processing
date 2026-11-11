"""End-to-end orchestration of the DWI quickstart and CLI stages."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scipy.ndimage import zoom

from .coherence import coherence_edges, reconstruct_diffraction
from .config import (
    C0,
    CoherenceConfig,
    FocusingConfig,
    Geometry,
    PropertyConfig,
    VelocityConfig,
)
from .focusing import invert_focusing
from .io import (
    ensure_run_dir,
    load_survey,
    read_json,
    save_survey_npz,
    write_json,
)
from .plotting import plot_focusing, plot_properties, plot_reconstruction, plot_summary, plot_velocity
from .properties import sequential_analysis
from .velocity import initial_model_from_picks, pick_velocity, velocity_spectrum
from .wavefront import WavefrontConfig, dwi_invert


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _copy_reference_arrays(source: str | Path, target: str | Path) -> None:
    """Keep optional reference fields alongside the saved survey."""
    source_path = Path(source)
    if source_path.suffix.lower() != ".npz" or not source_path.is_file():
        return
    wanted = (
        "diffraction_reference",
        "tau_reference",
        "coherence_reference",
        "alpha_reference",
    )
    with np.load(source_path, allow_pickle=False) as archive:
        extras = {key: archive[key] for key in wanted if key in archive}
    if not extras:
        return
    target_path = Path(target)
    with np.load(target_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload.update(extras)
    np.savez_compressed(target_path, **payload)


def _load_field(paths: Mapping[str, Path], field: str) -> np.ndarray:
    name = str(field).strip().lower()
    if name in {"reconstructed", "reconstruction", "dwi"}:
        return np.load(paths["diffraction"])
    if name in {"reference", "legacy"}:
        with np.load(paths["input"], allow_pickle=False) as archive:
            if "diffraction_reference" not in archive:
                raise KeyError(
                    "field='reference' requested but the input archive has no "
                    "diffraction_reference array."
                )
            return np.asarray(archive["diffraction_reference"], dtype=np.float64)
    raise ValueError("field must be 'reconstructed' or 'reference'.")


def _geometry_from_metadata(metadata: Mapping[str, Any]) -> Geometry:
    payload = metadata.get("geometry", metadata)
    return Geometry(
        dt=float(payload["dt"]),
        dx=float(payload["dx"]),
        x0=float(payload.get("x0", 0.0)),
        t0=float(payload.get("t0", 0.0)),
        frequency=float(payload.get("frequency", 5.0e8)),
        antenna_offset=float(payload.get("antenna_offset", 0.0)),
    )


def _update_metadata(paths: Mapping[str, Path], section: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata_path = Path(paths["metadata"])
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
    else:
        metadata = {}
    metadata.setdefault("stages", {})[section] = dict(payload)
    write_json(metadata_path, metadata)
    return metadata


def _reference_metrics(run_dir: Path, reconstructed: np.ndarray) -> dict[str, float]:
    """Compare with optional reference arrays in the input archive."""
    source = Path(run_dir) / "input.npz"
    metadata_path = Path(run_dir) / "metadata.json"
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        original = metadata.get("input", {}).get("path")
        if original and Path(original).is_file():
            source = Path(original)
    if not source.is_file():
        return {}
    metrics: dict[str, float] = {}
    with np.load(source, allow_pickle=False) as archive:
        for key, label in (
            ("diffraction_reference", "diffraction"),
            ("tau_reference", "tau"),
            ("coherence_reference", "coherence"),
        ):
            if key not in archive:
                continue
            reference = np.asarray(archive[key], dtype=np.float64)
            if reference.shape != reconstructed.shape:
                continue
            difference = reconstructed - reference
            denominator = max(float(np.sqrt(np.mean(reference * reference))), np.finfo(float).eps)
            metrics[f"{label}_relative_rms"] = float(np.sqrt(np.mean(difference * difference)) / denominator)
            left = reconstructed.ravel()
            right = reference.ravel()
            denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), np.finfo(float).eps)
            metrics[f"{label}_correlation"] = float(np.dot(left, right) / denominator)
    return metrics


def run_reconstruction(
    input_path: str | Path,
    run_dir: str | Path,
    *,
    coherence_config: CoherenceConfig | None = None,
    geometry: Geometry | None = None,
    orientation: str = "auto",
) -> dict[str, Any]:
    """Stage 1: load a B-scan and reconstruct its diffraction field."""
    config = coherence_config or CoherenceConfig()
    paths = ensure_run_dir(run_dir)
    survey = load_survey(input_path, geometry=geometry, orientation=orientation)
    save_survey_npz(paths["input"], survey)
    _copy_reference_arrays(input_path, paths["input"])
    np.save(paths["original"], survey.data)
    reconstruction = reconstruct_diffraction(survey, config)
    np.save(paths["diffraction"], reconstruction.diffraction)
    np.save(paths["reflection"], reconstruction.reflection)
    np.save(paths["coherence"], reconstruction.coherence)
    np.save(paths["tau"], reconstruction.tau)
    np.save(paths["alpha"], reconstruction.alpha)

    metadata = {
        "geometry": {
            "dt": survey.geometry.dt,
            "dx": survey.geometry.dx,
            "x0": survey.geometry.x0,
            "t0": survey.geometry.t0,
            "frequency": survey.geometry.frequency,
            "antenna_offset": survey.geometry.antenna_offset,
        },
        "input": {
            "path": str(Path(input_path).resolve()),
            "samples": survey.samples,
            "traces": survey.traces,
        },
    }
    write_json(paths["metadata"], metadata)
    metrics = _reference_metrics(paths["root"], reconstruction.diffraction)
    payload = {
        "aperture_traces": config.aperture_traces,
        "time_window": config.time_window,
        "max_shift": config.max_shift,
        "shift_step": config.shift_step,
        "coherence_power": config.coherence_power,
        "agc_window": config.agc_window,
        "bandpass_low": config.bandpass_low,
        "bandpass_high": config.bandpass_high,
        "mute_time": config.mute_time,
        "reference_metrics": metrics,
    }
    _update_metadata(paths, "reconstruction", payload)
    plot_reconstruction(
        reconstruction.input,
        reconstruction.diffraction,
        reconstruction.reflection,
        reconstruction.coherence,
        reconstruction.tau,
        survey.time,
        survey.x,
        paths["figures"] / "reconstruction.png",
    )
    return payload


def run_velocity(
    run_dir: str | Path,
    *,
    velocity_config: VelocityConfig | None = None,
    field: str = "reconstructed",
) -> dict[str, Any]:
    """Stage 2: diffraction velocity spectrum and initial dielectric model."""
    config = velocity_config or VelocityConfig()
    paths = ensure_run_dir(run_dir)
    metadata = read_json(paths["metadata"])
    geometry = _geometry_from_metadata(metadata)
    diffraction = _load_field(paths, field)
    np.save(paths["diffraction_used"], diffraction)
    spectrum = velocity_spectrum(diffraction, geometry, config)
    picks = pick_velocity(spectrum, config)
    sample_count, trace_count = diffraction.shape
    max_time = geometry.t0 + (sample_count - 1) * geometry.dt
    max_depth = C0 * max_time / (2.0 * np.sqrt(max(config.epsilon_min, 1.0)))
    depth = np.arange(sample_count, dtype=np.float64) * max_depth / max(sample_count - 1, 1)
    initial, confidence = initial_model_from_picks(
        picks,
        geometry,
        diffraction.shape,
        epsilon_min=config.epsilon_min,
        epsilon_max=config.epsilon_max,
        depth=depth,
        max_depth=max_depth,
        smooth_sigma=1.0,
    )
    np.save(paths["initial_epsilon"], initial)
    np.save(paths["root"] / "initial_confidence.npy", confidence)
    np.savez_compressed(
        paths["velocity_spectrum"],
        values=spectrum.values.astype(np.float32),
        epsilon=spectrum.epsilon,
        time=spectrum.time,
        x=spectrum.x,
        time_indices=spectrum.time_indices,
        trace_indices=spectrum.trace_indices,
    )
    rows = []
    for it in range(picks.time.size):
        for ix in range(picks.x.size):
            if not picks.valid[it, ix]:
                continue
            rows.append(
                {
                    "time_ns": float(picks.time[it] * 1.0e9),
                    "x_m": float(picks.x[ix]),
                    "epsilon": float(picks.epsilon[it, ix]),
                    "confidence": float(picks.confidence[it, ix]),
                }
            )
    _write_csv(paths["velocity_picks"], rows)
    plot_velocity(spectrum, picks, initial, depth, geometry.x_axis(trace_count), paths["figures"] / "velocity.png")
    payload = {
        "epsilon_min": config.epsilon_min,
        "epsilon_max": config.epsilon_max,
        "epsilon_step": config.epsilon_step,
        "time_stride": config.time_stride,
        "trace_stride": config.trace_stride,
        "aperture_traces": config.aperture_traces,
        "mute_time": config.mute_time,
        "confidence_percentile": config.confidence_percentile,
        "valid_picks": int(np.count_nonzero(picks.valid)),
        "max_depth_m": float(max_depth),
    }
    _update_metadata(paths, "velocity", payload)
    return payload


def run_inversion(
    run_dir: str | Path,
    *,
    method: str = "focusing",
    focusing_config: FocusingConfig | None = None,
    wavefront_config: WavefrontConfig | None = None,
    field: str = "reconstructed",
) -> dict[str, Any]:
    """Stage 3: diffraction-focusing inversion and optional DWI refinement."""
    mode = str(method).strip().lower()
    if mode not in {"focusing", "wavefront", "both"}:
        raise ValueError("method must be 'focusing', 'wavefront' or 'both'.")
    paths = ensure_run_dir(run_dir)
    metadata = read_json(paths["metadata"])
    geometry = _geometry_from_metadata(metadata)
    diffraction = _load_field(paths, field)
    np.save(paths["diffraction_used"], diffraction)
    if not paths["initial_epsilon"].is_file():
        raise FileNotFoundError("Run the velocity stage before inversion.")
    initial = np.load(paths["initial_epsilon"])
    fconfig = focusing_config or FocusingConfig()

    payload: dict[str, Any] = {"method": mode}
    focusing = None
    if mode in {"focusing", "both"}:
        focusing = invert_focusing(diffraction, geometry, fconfig, initial_epsilon=initial)
        np.save(paths["epsilon_focusing"], focusing.epsilon)
        np.save(paths["epsilon"], focusing.epsilon)
        np.save(paths["confidence"], focusing.confidence)
        np.savez_compressed(
            paths["focusing_spectrum"],
            epsilon=focusing.epsilon_grid,
            r1=focusing.r1.astype(np.float32),
            r2=focusing.r2.astype(np.float32),
            energy=focusing.energy.astype(np.float32),
            time_indices=focusing.time_indices,
            trace_indices=focusing.trace_indices,
            point_x=focusing.point_x,
            point_depth=focusing.point_depth,
            point_epsilon=focusing.point_epsilon,
            point_confidence=focusing.point_confidence,
            point_energy=focusing.point_energy,
            point_valid=focusing.point_valid,
        )
        rows = []
        for index in range(focusing.point_x.size):
            rows.append(
                {
                    "x_m": float(focusing.point_x[index]),
                    "depth_m": float(focusing.point_depth[index]),
                    "epsilon": float(focusing.point_epsilon[index]),
                    "confidence": float(focusing.point_confidence[index]),
                    "energy": float(focusing.point_energy[index]),
                    "valid": int(bool(focusing.point_valid[index])),
                }
            )
        _write_csv(paths["focusing_points"], rows)
        plot_focusing(focusing, paths["figures"] / "focusing.png")
        payload.update(
            {
                "epsilon_min": fconfig.epsilon_min,
                "epsilon_max": fconfig.epsilon_max,
                "epsilon_step": fconfig.epsilon_step,
                "window_half_samples": fconfig.window_half_samples,
                "focus_u": fconfig.focus_u,
                "focus_v": fconfig.focus_v,
                "stride_time": fconfig.stride_time,
                "stride_trace": fconfig.stride_trace,
                "isolation": fconfig.isolation,
                "migration": fconfig.migration,
                "valid_points": int(np.count_nonzero(focusing.point_valid)),
            }
        )

    if mode in {"wavefront", "both"}:
        wconfig = wavefront_config or WavefrontConfig()
        starting_model = focusing.epsilon if focusing is not None else initial
        wavefront = dwi_invert(diffraction, geometry, starting_model, wconfig)
        np.save(paths["root"] / "epsilon_wavefront.npy", wavefront.epsilon)
        np.save(paths["epsilon"], wavefront.epsilon)
        _write_csv(
            paths["wavefront_history"],
            [{key: float(value) for key, value in row.items()} for row in wavefront.history],
        )
        payload.update(
            {
                "wavefront_backend": wconfig.backend,
                "wavefront_frequency_count": len(wavefront.frequencies),
                "wavefront_iterations": wavefront.iterations,
            }
        )
    _update_metadata(paths, "inversion", payload)
    return payload


def run_analysis(
    run_dir: str | Path,
    *,
    property_config: PropertyConfig | None = None,
    boundaries: np.ndarray | None = None,
    layer_count: int | None = None,
    edge_field: str = "tau",
) -> dict[str, Any]:
    """Stage 4: sequential empirical property and layer analysis."""
    config = property_config or PropertyConfig()
    paths = ensure_run_dir(run_dir)
    metadata = read_json(paths["metadata"])
    geometry = _geometry_from_metadata(metadata)
    epsilon = np.load(paths["epsilon"])
    sample_count, trace_count = epsilon.shape
    time = geometry.time_axis(sample_count)
    reference = float(np.median(epsilon))
    depth = C0 * time / (2.0 * np.sqrt(max(reference, 1.0)))
    x = geometry.x_axis(trace_count)

    edge_map = None
    if str(edge_field).lower() in {"reference", "tau_reference", "legacy"}:
        with np.load(paths["input"], allow_pickle=False) as archive:
            tau = np.asarray(archive["tau_reference"], dtype=np.float64) if "tau_reference" in archive else None
        if tau is not None:
            edge_map = coherence_edges(tau, sigma=config.edge_sigma, percentile=config.edge_percentile)
    elif paths["tau"].is_file():
        tau = np.load(paths["tau"])
        edge_map = coherence_edges(tau, sigma=config.edge_sigma, percentile=config.edge_percentile)
    if edge_map is not None and edge_map.shape != epsilon.shape:
        factors = (
            epsilon.shape[0] / edge_map.shape[0],
            epsilon.shape[1] / edge_map.shape[1],
        )
        resized = zoom(edge_map.astype(np.float32), factors, order=1) > 0.5
        fitted = np.zeros(epsilon.shape, dtype=bool)
        rows = min(epsilon.shape[0], resized.shape[0])
        cols = min(epsilon.shape[1], resized.shape[1])
        fitted[:rows, :cols] = resized[:rows, :cols]
        edge_map = fitted
    if boundaries is not None:
        boundaries = np.asarray(boundaries, dtype=np.float64)
    elif layer_count is not None:
        if int(layer_count) < 1:
            raise ValueError("layer_count must be positive.")
        quantiles = np.linspace(0.0, 1.0, int(layer_count) + 1)
        boundaries = np.quantile(depth, quantiles)
    result = sequential_analysis(
        epsilon,
        depth,
        x,
        config,
        edge_map=edge_map,
        boundaries=boundaries,
    )
    np.savez_compressed(
        paths["properties"],
        epsilon=result.maps.epsilon,
        density=result.maps.density,
        loss_tangent=result.maps.loss_tangent,
        chemistry=result.maps.chemistry,
        epsilon_sigma=result.epsilon_sigma,
        depth=result.depth,
        x=result.x,
    )
    np.save(paths["boundaries"], result.boundaries)
    _write_csv(paths["layers"], result.layers)

    profile_rows = []
    for index, z in enumerate(result.depth):
        row = {
            "depth_m": float(z),
            "epsilon_median": float(np.median(result.maps.epsilon[index, :])),
            "density_median": float(np.median(result.maps.density[index, :])),
            "loss_tangent_median": float(np.median(result.maps.loss_tangent[index, :])),
            "chemistry_median": float(np.median(result.maps.chemistry[index, :])),
        }
        profile_rows.append(row)
    _write_csv(paths["property_profiles"], profile_rows)

    plot_properties(result, paths["figures"] / "properties.png")
    summary_field = paths["diffraction_used"] if paths["diffraction_used"].is_file() else paths["diffraction"]
    if paths["original"].is_file() and summary_field.is_file():
        plot_summary(
            np.load(paths["original"]),
            np.load(summary_field),
            result.maps.epsilon,
            result.depth,
            result.x,
            time,
            result.boundaries,
            paths["figures"] / "summary.png",
        )
    payload = {
        "layers": len(result.layers),
        "boundaries_m": result.boundaries,
        "relative_epsilon_sigma": config.relative_epsilon_sigma,
        "edge_percentile": config.edge_percentile,
    }
    _update_metadata(paths, "analysis", payload)
    return payload


def run_all(
    input_path: str | Path,
    run_dir: str | Path,
    *,
    coherence_config: CoherenceConfig | None = None,
    velocity_config: VelocityConfig | None = None,
    focusing_config: FocusingConfig | None = None,
    method: str = "focusing",
    wavefront_config: WavefrontConfig | None = None,
    property_config: PropertyConfig | None = None,
    field: str = "reconstructed",
    edge_field: str = "tau",
) -> dict[str, Any]:
    """Run reconstruction, velocity analysis, inversion and property analysis."""
    run_reconstruction(input_path, run_dir, coherence_config=coherence_config)
    run_velocity(run_dir, velocity_config=velocity_config, field=field)
    run_inversion(
        run_dir,
        method=method,
        focusing_config=focusing_config,
        wavefront_config=wavefront_config,
        field=field,
    )
    return run_analysis(run_dir, property_config=property_config, edge_field=edge_field)


def run_example(
    name: str = "ilmenite_demo",
    run_dir: str | Path = "runs/quickstart",
    **kwargs,
) -> dict[str, Any]:
    """Convenience wrapper around :func:`run_all` for the packaged example."""
    data_dir = Path(__file__).resolve().parent.parent / "examples" / "data"
    return run_all(data_dir / f"{name}.npz", run_dir, **kwargs)

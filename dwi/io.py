"""Data loading, preprocessing and run-directory helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import Geometry

try:  # optional, only needed for native gprMax .out files
    import h5py
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    h5py = None

try:
    from scipy.signal import butter, sosfiltfilt
    from scipy.ndimage import uniform_filter1d
except ModuleNotFoundError:  # pragma: no cover - SciPy is a core dependency
    butter = sosfiltfilt = uniform_filter1d = None


@dataclass
class Survey:
    """A zero-offset B-scan in ``(time, trace)`` orientation."""

    data: np.ndarray
    geometry: Geometry
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.data, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"B-scan data must be 2-D; got shape {values.shape}.")
        if not np.isfinite(values).all():
            raise ValueError("B-scan data contains NaN or infinite values.")
        self.data = np.ascontiguousarray(values)

    @property
    def samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def traces(self) -> int:
        return int(self.data.shape[1])

    @property
    def time(self) -> np.ndarray:
        return self.geometry.time_axis(self.samples)

    @property
    def x(self) -> np.ndarray:
        return self.geometry.x_axis(self.traces)

    def copy_with(self, data: np.ndarray, **metadata: Any) -> "Survey":
        merged = dict(self.metadata)
        merged.update(metadata)
        return Survey(np.asarray(data), self.geometry, merged)


def _read_text_matrix(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path)
    except ValueError as exc:
        raise ValueError(
            f"Could not read {path} as a rectangular delimited text matrix. "
            "Use .npy or .npz for large arrays."
        ) from exc


def _load_raw_array(path: str | Path, key: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    array_path = Path(path)
    if not array_path.is_file():
        raise FileNotFoundError(f"Input data file does not exist: {array_path}")
    suffix = array_path.suffix.lower()
    metadata: dict[str, Any] = {"source_path": str(array_path.resolve())}

    if suffix == ".npy":
        return np.load(array_path, allow_pickle=False), metadata
    if suffix == ".npz":
        with np.load(array_path, allow_pickle=False) as archive:
            if key is None:
                for candidate in ("data", "original", "bscan", "arr"):
                    if candidate in archive:
                        key = candidate
                        break
                if key is None:
                    keys = [name for name in archive.files if not name.startswith("meta_")]
                    if len(keys) != 1:
                        raise ValueError(
                            f"{array_path} contains multiple arrays {archive.files}; "
                            "pass --key to select one."
                        )
                    key = keys[0]
            if key not in archive:
                raise KeyError(f"Array key {key!r} not found in {array_path}.")
            values = archive[key]
            for name in archive.files:
                if name.startswith("meta_"):
                    metadata[name[5:]] = _decode_npz_scalar(archive[name])
            return values, metadata
    if suffix in {".dat", ".txt", ".csv", ".asc"}:
        return _read_text_matrix(array_path), metadata
    if suffix == ".out":
        return _load_gprmax_out(array_path), metadata
    raise ValueError(
        f"Unsupported input extension {suffix!r}. Use .npy, .npz, .dat, .txt, .csv or .out."
    )


def _decode_npz_scalar(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return item
    return array.tolist()


def _load_gprmax_out(path: Path, component: str = "Ez", receiver: str = "rx1") -> np.ndarray:
    if h5py is None:
        raise ModuleNotFoundError("Reading gprMax .out files requires h5py.")
    with h5py.File(path, "r") as handle:
        group = handle.get(f"rxs/{receiver}")
        if group is None:
            raise KeyError(f"Receiver group rxs/{receiver} not found in {path}.")
        if component not in group:
            raise KeyError(f"Component {component!r} not found in {path}.")
        data = np.asarray(group[component][:], dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"gprMax component {component!r} must be 2-D; got {data.shape}.")
    return data


def _infer_orientation(
    values: np.ndarray,
    *,
    geometry: Geometry | None,
    orientation: str,
) -> tuple[np.ndarray, bool]:
    """Return a ``(time, trace)`` array and whether it was transposed."""
    mode = orientation.strip().lower()
    if mode in {"time_trace", "time, trace", "tt"}:
        return values, False
    if mode in {"trace_time", "trace, time"}:
        return values.T, True
    if mode not in {"auto", ""}:
        raise ValueError("orientation must be 'auto', 'time_trace' or 'trace_time'.")

    # A typical GPR B-scan has more time samples than traces.
    rows, cols = values.shape
    if rows >= cols:
        return values, False
    return values.T, True


def load_survey(
    path: str | Path,
    *,
    key: str | None = None,
    geometry: Geometry | Mapping[str, Any] | None = None,
    orientation: str = "auto",
) -> Survey:
    """Load a B-scan and resolve its geometry.

    ``path`` may be ``.npy``/``.npz`` (the preferred format), a delimited text
    matrix, or a native gprMax ``.out`` file.  For ``.npz`` inputs, metadata
    stored under ``meta_*`` keys is read automatically.
    """
    values, metadata = _load_raw_array(path, key=key)
    values = np.asarray(values, dtype=np.float64)

    if isinstance(geometry, Geometry):
        geom = geometry
    else:
        payload = dict(metadata)
        if isinstance(geometry, Mapping):
            payload.update(geometry)
        if "geometry" in payload and isinstance(payload["geometry"], Mapping):
            nested = dict(payload.pop("geometry"))
            nested.update(payload)
            payload = nested
        required = [name for name in ("dt", "dx") if name not in payload]
        if required:
            raise ValueError(
                "Missing geometry fields "
                + ", ".join(required)
                + ". Supply geometry=Geometry(...), a mapping with dt/dx, or "
                "an .npz file containing meta_dt/meta_dx."
            )
        geom = Geometry(
            dt=float(payload["dt"]),
            dx=float(payload["dx"]),
            x0=float(payload.get("x0", 0.0)),
            t0=float(payload.get("t0", 0.0)),
            frequency=float(payload.get("frequency", 5.0e8)),
            antenna_offset=float(payload.get("antenna_offset", 0.0)),
        )

    data, transposed = _infer_orientation(values, geometry=geom, orientation=orientation)
    metadata = dict(metadata)
    metadata.setdefault("orientation", "trace_time" if transposed else "time_trace")
    metadata.setdefault("original_shape", list(values.shape))
    return Survey(np.ascontiguousarray(data, dtype=np.float64), geom, metadata)


def load_example(name: str = "ilmenite_demo", data_dir: str | Path | None = None) -> Survey:
    """Load one of the packaged synthetic examples."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "examples" / "data"
    path = Path(data_dir) / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Packaged example {name!r} was not found at {path}. "
            "Run examples/make_example_data.py to regenerate it."
        )
    return load_survey(path)


def agc(data: np.ndarray, window: int) -> np.ndarray:
    """Trace-wise automatic gain control along the time axis."""
    values = np.asarray(data, dtype=np.float64)
    if window <= 0:
        return values.copy()
    if uniform_filter1d is None:
        raise ModuleNotFoundError("AGC requires SciPy.")
    window = int(window)
    envelope = np.abs(values)
    scale = uniform_filter1d(envelope, size=window, axis=0, mode="nearest")
    floor = max(float(np.percentile(scale, 5)), np.finfo(np.float64).eps)
    return values / np.maximum(scale, floor)


def bandpass(data: np.ndarray, dt: float, low: float | None, high: float | None, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the time axis."""
    if low is None and high is None:
        return np.asarray(data, dtype=np.float64).copy()
    if butter is None or sosfiltfilt is None:
        raise ModuleNotFoundError("Band-pass filtering requires SciPy.")
    values = np.asarray(data, dtype=np.float64)
    nyquist = 0.5 / float(dt)
    low_norm = 0.0 if low is None else float(low) / nyquist
    high_norm = 1.0 if high is None else float(high) / nyquist
    high_norm = min(high_norm, 0.99)
    if low_norm <= 0.0:
        sos = butter(order, high_norm, btype="lowpass", output="sos")
    elif high_norm >= 0.99:
        sos = butter(order, low_norm, btype="highpass", output="sos")
    else:
        sos = butter(order, [low_norm, high_norm], btype="bandpass", output="sos")
    return sosfiltfilt(sos, values, axis=0)


def mute_early(data: np.ndarray, dt: float, mute_time: float) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64).copy()
    if mute_time <= 0.0:
        return values
    count = min(int(np.floor(float(mute_time) / dt)), values.shape[0])
    values[:count, :] = 0.0
    return values


def preprocess(
    survey: Survey,
    *,
    agc_window: int = 0,
    bandpass_low: float | None = None,
    bandpass_high: float | None = None,
    mute_time: float = 0.0,
    remove_mean: bool = True,
) -> Survey:
    """Apply the small, explicit preprocessing chain used by the example."""
    values = np.asarray(survey.data, dtype=np.float64)
    if remove_mean:
        values = values - np.mean(values, axis=0, keepdims=True)
    values = bandpass(values, survey.geometry.dt, bandpass_low, bandpass_high)
    if agc_window > 0:
        values = agc(values, agc_window)
    values = mute_early(values, survey.geometry.dt, mute_time)
    return survey.copy_with(values, preprocessed=True)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")

    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=default)
        handle.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_paths(run_dir: str | Path) -> dict[str, Path]:
    """Return the canonical file names inside a run directory."""
    root = Path(run_dir)
    return {
        "root": root,
        "input": root / "input.npz",
        "metadata": root / "metadata.json",
        "original": root / "original.npy",
        "diffraction": root / "diffraction.npy",
        "diffraction_used": root / "diffraction_used.npy",
        "reflection": root / "reflection.npy",
        "coherence": root / "coherence.npy",
        "tau": root / "tau.npy",
        "alpha": root / "alpha.npy",
        "velocity_spectrum": root / "velocity_spectrum.npz",
        "velocity_picks": root / "velocity_picks.csv",
        "initial_epsilon": root / "initial_epsilon.npy",
        "epsilon": root / "epsilon.npy",
        "epsilon_focusing": root / "epsilon_focusing.npy",
        "confidence": root / "confidence.npy",
        "focusing_points": root / "focusing_points.csv",
        "focusing_spectrum": root / "focusing_spectrum.npz",
        "wavefront_history": root / "wavefront_history.csv",
        "properties": root / "properties.npz",
        "property_profiles": root / "property_profiles.csv",
        "layers": root / "layers.csv",
        "boundaries": root / "boundaries.npy",
        "figures": root / "figures",
    }


def ensure_run_dir(run_dir: str | Path) -> dict[str, Path]:
    paths = run_paths(run_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["figures"].mkdir(parents=True, exist_ok=True)
    return paths


def save_survey_npz(path: str | Path, survey: Survey) -> None:
    """Persist a survey with self-describing metadata."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "data": np.asarray(survey.data, dtype=np.float32),
        "meta_dt": survey.geometry.dt,
        "meta_dx": survey.geometry.dx,
        "meta_x0": survey.geometry.x0,
        "meta_t0": survey.geometry.t0,
        "meta_frequency": survey.geometry.frequency,
        "meta_antenna_offset": survey.geometry.antenna_offset,
    }
    for key, value in survey.metadata.items():
        if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
            payload[f"meta_{key}"] = value
    np.savez_compressed(target, **payload)

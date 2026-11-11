"""Build the small packaged example from synthetic LPR exports.

The source directory must contain the coherence-analysis exports
``legacy_originalTrace.dat``, ``legacy_subtractedTrace.dat``,
``legacy_tauTrace.dat``, ``legacy_coherenceTrace.dat`` and
``legacy_alphaTrace.dat``.  Only a small spatial/temporal window is copied so
that the quickstart remains lightweight; the source files are never modified.

Usage
-----
python examples/make_example_data.py --source "path/to/legacy_exports"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


MODEL_LABEL = "synthetic ilmenite-bearing gprMax/LPR model"


def _load(path: Path) -> np.ndarray:
    values = np.loadtxt(path)
    # The legacy exports are stored as (trace, time).
    return np.asarray(values, dtype=np.float64).T


def build(source: Path, output: Path) -> None:
    original = _load(source / "legacy_originalTrace.dat")
    diffraction = _load(source / "legacy_subtractedTrace.dat")
    tau = _load(source / "legacy_tauTrace.dat")
    coherence = _load(source / "legacy_coherenceTrace.dat")
    alpha = _load(source / "legacy_alphaTrace.dat")

    trace_start, trace_stop = 80, 280
    sample_start, sample_stop = 250, 750
    window = (
        slice(sample_start, sample_stop),
        slice(trace_start, trace_stop),
    )
    original = original[window]
    diffraction = diffraction[window]
    tau = tau[window]
    coherence = coherence[window]
    alpha = alpha[window]

    def scale(values: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(values)))
        return (values / peak if peak > 0 else values).astype(np.float32)

    dt = 100.0e-9 / 1023.0
    dx = 0.02
    x0 = 0.1 + trace_start * dx
    t0 = sample_start * dt
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        data=scale(original),
        diffraction_reference=scale(diffraction),
        tau_reference=tau.astype(np.float32),
        coherence_reference=scale(coherence),
        alpha_reference=alpha.astype(np.float32),
        meta_dt=dt,
        meta_dx=dx,
        meta_x0=x0,
        meta_t0=t0,
        meta_frequency=5.0e8,
        meta_antenna_offset=0.0,
        meta_source=MODEL_LABEL,
        meta_citation=(
            "Synthetic gprMax model and coherence-analysis exports; see "
            "examples/data/README.md for provenance."
        ),
        meta_trace_start=trace_start,
        meta_sample_start=sample_start,
    )
    print(f"wrote {output} ({original.shape[1]} traces x {original.shape[0]} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing the legacy_*Trace.dat exports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "ilmenite_demo.npz",
    )
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()

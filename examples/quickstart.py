"""Run the packaged synthetic example end to end.

    python examples/quickstart.py

The script writes ``runs/quickstart`` and prints the main output paths.  It
uses the diffraction-focusing inversion (no CUDA required).  Add
``--method both`` to append the optional frequency-domain DWI refinement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dwi.config import CoherenceConfig, FocusingConfig, PropertyConfig, VelocityConfig  # noqa: E402
from dwi.io import load_example  # noqa: E402
from dwi.pipeline import run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/quickstart", help="Output run directory.")
    parser.add_argument("--method", choices=["focusing", "wavefront", "both"], default="focusing")
    args = parser.parse_args()

    example = load_example("ilmenite_demo")
    source = Path(__file__).resolve().parent / "data" / "ilmenite_demo.npz"
    print(f"example: {example.traces} traces x {example.samples} samples")
    print(f"dt = {example.geometry.dt:.6g} s, dx = {example.geometry.dx:g} m")
    print(f"x = {example.x[0]:.2f} .. {example.x[-1]:.2f} m")
    print(f"t = {example.time[0] * 1e9:.2f} .. {example.time[-1] * 1e9:.2f} ns")

    payload = run_all(
        source,
        args.output,
        coherence_config=CoherenceConfig(
            aperture_traces=9,
            time_window=31,
            max_shift=12,
            shift_step=1,
            agc_window=0,
            mute_time=0.0,
        ),
        velocity_config=VelocityConfig(
            epsilon_min=3.0,
            epsilon_max=9.0,
            epsilon_step=0.25,
            time_stride=4,
            trace_stride=2,
            aperture_traces=21,
            confidence_percentile=55.0,
            mute_time=1.0e-8,
        ),
        focusing_config=FocusingConfig(
            epsilon_min=3.0,
            epsilon_max=9.0,
            epsilon_step=0.25,
            window_half_samples=8,
            focus_u=4,
            focus_v=2,
            stride_time=6,
            stride_trace=4,
            isolation=True,
            migration="stolt",
            min_confidence=1.30,
            min_energy_percentile=70.0,
            smooth_sigma=2.5,
            mute_time=1.0e-8,
        ),
        method=args.method,
        property_config=PropertyConfig(
            edge_percentile=88.0,
            min_layer_thickness=0.25,
            relative_epsilon_sigma=0.03,
            monte_carlo_samples=1000,
        ),
        field="reference",
        edge_field="tau_reference",
    )

    run_dir = Path(args.output)
    print("\nfinished")
    print(f"  diffraction: {run_dir / 'diffraction.npy'}")
    print(f"  permittivity: {run_dir / 'epsilon.npy'}")
    print(f"  layers:       {run_dir / 'layers.csv'}")
    print(f"  figures:      {run_dir / 'figures'}")
    print(f"  summary:      {payload}")


if __name__ == "__main__":
    main()

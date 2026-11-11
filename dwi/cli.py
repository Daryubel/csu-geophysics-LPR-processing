"""Command-line interface for the minimal DWI workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import CoherenceConfig, FocusingConfig, PropertyConfig, VelocityConfig
from .pipeline import run_all, run_analysis, run_inversion, run_reconstruction, run_velocity
from .wavefront import WavefrontConfig


def _add_common_geometry(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dt", type=float, help="Sample interval in seconds (for .npy/.dat input).")
    parser.add_argument("--dx", type=float, help="Trace spacing in metres (for .npy/.dat input).")
    parser.add_argument("--x0", type=float, default=0.0, help="Position of the first trace (m).")
    parser.add_argument("--t0", type=float, default=0.0, help="Time of the first sample (s).")
    parser.add_argument("--frequency", type=float, default=5.0e8, help="Dominant frequency (Hz).")
    parser.add_argument(
        "--orientation",
        choices=["auto", "time_trace", "trace_time"],
        default="auto",
        help="Array orientation for .npy/.dat input.",
    )


def _coherence_config(args: argparse.Namespace) -> CoherenceConfig:
    low = high = None
    if args.bandpass is not None:
        low, high = args.bandpass
    return CoherenceConfig(
        aperture_traces=args.aperture,
        time_window=args.time_window,
        max_shift=args.max_shift,
        shift_step=args.shift_step,
        coherence_power=args.coherence_power,
        agc_window=args.agc,
        bandpass_low=low,
        bandpass_high=high,
        mute_time=args.mute,
    )


def _velocity_config(args: argparse.Namespace) -> VelocityConfig:
    return VelocityConfig(
        epsilon_min=args.eps_min,
        epsilon_max=args.eps_max,
        epsilon_step=args.eps_step,
        time_stride=args.time_stride,
        trace_stride=args.trace_stride,
        aperture_traces=args.aperture,
        mute_time=args.mute,
        confidence_percentile=args.confidence_percentile,
    )


def _focusing_config(args: argparse.Namespace) -> FocusingConfig:
    return FocusingConfig(
        epsilon_min=args.eps_min,
        epsilon_max=args.eps_max,
        epsilon_step=args.eps_step,
        window_half_samples=args.window,
        focus_u=args.focus_u,
        focus_v=args.focus_v,
        stride_time=args.stride_time,
        stride_trace=args.stride_trace,
        isolation=not args.no_isolation,
        migration=args.migration,
        min_confidence=args.min_confidence,
        min_energy_percentile=args.min_energy_percentile,
        smooth_sigma=args.smooth,
        mute_time=args.mute,
        max_depth=args.max_depth,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dwi",
        description="Minimal diffraction inversion and sequential medium-property analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reconstruct = subparsers.add_parser("reconstruct", help="Load a B-scan and reconstruct diffractions.")
    reconstruct.add_argument("--input", required=True, help="Input .npz/.npy/.dat/.out file.")
    reconstruct.add_argument("--key", help="Array key for .npz input.")
    reconstruct.add_argument("--output", required=True, help="Run directory.")
    _add_common_geometry(reconstruct)
    reconstruct.add_argument("--aperture", type=int, default=9, help="Coherent stacking aperture (traces).")
    reconstruct.add_argument("--time-window", type=int, default=31, help="Semblance time window (samples).")
    reconstruct.add_argument("--max-shift", type=int, default=12, help="Maximum local shift (samples/trace).")
    reconstruct.add_argument("--shift-step", type=int, default=1, help="Shift search step (samples/trace).")
    reconstruct.add_argument("--coherence-power", type=float, default=1.0, help="Exponent on semblance.")
    reconstruct.add_argument("--agc", type=int, default=0, help="AGC window in samples (0 disables).")
    reconstruct.add_argument("--bandpass", nargs=2, type=float, metavar=("LOW", "HIGH"), help="Band-pass in Hz.")
    reconstruct.add_argument("--mute", type=float, default=0.0, help="Early-time mute in seconds.")

    velocity = subparsers.add_parser("velocity", help="Diffraction velocity spectrum and initial model.")
    velocity.add_argument("--run", required=True, help="Run directory created by reconstruct.")
    velocity.add_argument("--eps-min", type=float, default=3.0)
    velocity.add_argument("--eps-max", type=float, default=9.0)
    velocity.add_argument("--eps-step", type=float, default=0.25)
    velocity.add_argument("--time-stride", type=int, default=4)
    velocity.add_argument("--trace-stride", type=int, default=2)
    velocity.add_argument("--aperture", type=int, default=21)
    velocity.add_argument("--mute", type=float, default=0.0)
    velocity.add_argument("--confidence-percentile", type=float, default=60.0)
    velocity.add_argument(
        "--field",
        choices=["reconstructed", "reference"],
        default="reconstructed",
        help="Use the built-in reconstruction or an archived diffraction_reference field.",
    )

    invert = subparsers.add_parser("invert", help="Diffraction-focusing inversion (and optional DWI).")
    invert.add_argument("--run", required=True, help="Run directory.")
    invert.add_argument("--method", choices=["focusing", "wavefront", "both"], default="focusing")
    invert.add_argument("--eps-min", type=float, default=3.0)
    invert.add_argument("--eps-max", type=float, default=9.0)
    invert.add_argument("--eps-step", type=float, default=0.25)
    invert.add_argument("--window", type=int, default=8, help="Hyperbolic window half-width (samples).")
    invert.add_argument("--focus-u", type=int, default=4, help="Focus template half-height (samples).")
    invert.add_argument("--focus-v", type=int, default=2, help="Focus template half-width (traces).")
    invert.add_argument("--stride-time", type=int, default=6)
    invert.add_argument("--stride-trace", type=int, default=4)
    invert.add_argument("--no-isolation", action="store_true", help="Migrate the whole section for every velocity.")
    invert.add_argument("--migration", choices=["stolt", "kirchhoff"], default="stolt")
    invert.add_argument("--min-confidence", type=float, default=1.20)
    invert.add_argument("--min-energy-percentile", type=float, default=55.0)
    invert.add_argument("--smooth", type=float, default=2.0)
    invert.add_argument("--mute", type=float, default=0.0)
    invert.add_argument("--max-depth", type=float, default=None)
    invert.add_argument(
        "--field",
        choices=["reconstructed", "reference"],
        default="reconstructed",
        help="Diffraction field used by the inversion.",
    )
    invert.add_argument("--dwi-frequencies", nargs="*", type=float, help="Explicit DWI frequencies (Hz).")
    invert.add_argument("--dwi-backend", choices=["cpu", "cuda"], default="cpu")
    invert.add_argument("--dwi-iterations", type=int, default=4, help="L-BFGS iterations per frequency.")
    invert.add_argument("--dwi-regularization", type=float, default=5.0e-3)

    analyze = subparsers.add_parser("analyze", help="Sequential medium-property and layer analysis.")
    analyze.add_argument("--run", required=True, help="Run directory.")
    analyze.add_argument("--layers", type=int, default=None, help="Force a fixed number of equal-depth layers.")
    analyze.add_argument("--boundaries", type=float, nargs="*", help="Explicit layer boundaries in metres.")
    analyze.add_argument("--edge-percentile", type=float, default=90.0)
    analyze.add_argument("--min-layer-thickness", type=float, default=0.2)
    analyze.add_argument("--relative-sigma", type=float, default=0.03)
    analyze.add_argument("--monte-carlo-samples", type=int, default=2000)
    analyze.add_argument(
        "--edge-field",
        choices=["tau", "reference"],
        default="tau",
        help="Use the reconstructed tau map or the archived tau_reference map.",
    )

    run = subparsers.add_parser("run", help="Run all stages on one input file.")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--key")
    _add_common_geometry(run)
    run.add_argument("--method", choices=["focusing", "wavefront", "both"], default="focusing")
    run.add_argument("--aperture", type=int, default=9)
    run.add_argument("--time-window", type=int, default=31)
    run.add_argument("--max-shift", type=int, default=12)
    run.add_argument("--agc", type=int, default=0)
    run.add_argument("--bandpass", nargs=2, type=float, metavar=("LOW", "HIGH"))
    run.add_argument("--mute", type=float, default=0.0)
    run.add_argument("--eps-min", type=float, default=3.0)
    run.add_argument("--eps-max", type=float, default=9.0)
    run.add_argument("--eps-step", type=float, default=0.25)
    run.add_argument(
        "--field",
        choices=["reconstructed", "reference"],
        default="reconstructed",
        help="Diffraction field used after the reconstruction stage.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command

    if command == "reconstruct":
        geometry = None
        if args.dt is not None or args.dx is not None:
            if args.dt is None or args.dx is None:
                parser.error("--dt and --dx must be supplied together.")
            from .config import Geometry

            geometry = Geometry(
                dt=args.dt,
                dx=args.dx,
                x0=args.x0,
                t0=args.t0,
                frequency=args.frequency,
            )
        payload = run_reconstruction(
            args.input,
            args.output,
            coherence_config=_coherence_config(args),
            geometry=geometry,
            orientation=args.orientation,
        )
        print(f"reconstruction: {payload}")
        return 0

    if command == "velocity":
        payload = run_velocity(
            args.run,
            velocity_config=_velocity_config(args),
            field=args.field,
        )
        print(f"velocity: {payload}")
        return 0

    if command == "invert":
        fconfig = _focusing_config(args)
        wconfig = WavefrontConfig(
            frequencies=tuple(args.dwi_frequencies) if args.dwi_frequencies else None,
            stage_iterations=args.dwi_iterations,
            regularization=args.dwi_regularization,
            backend=args.dwi_backend,
            epsilon_min=args.eps_min,
            epsilon_max=args.eps_max,
        )
        payload = run_inversion(
            args.run,
            method=args.method,
            focusing_config=fconfig,
            wavefront_config=wconfig,
            field=args.field,
        )
        print(f"inversion: {payload}")
        return 0

    if command == "analyze":
        boundaries = np.asarray(args.boundaries, dtype=np.float64) if args.boundaries else None
        config = PropertyConfig(
            edge_percentile=args.edge_percentile,
            min_layer_thickness=args.min_layer_thickness,
            relative_epsilon_sigma=args.relative_sigma,
            monte_carlo_samples=args.monte_carlo_samples,
        )
        payload = run_analysis(
            args.run,
            property_config=config,
            boundaries=boundaries,
            layer_count=args.layers,
            edge_field=args.edge_field,
        )
        print(f"analysis: {payload}")
        return 0

    if command == "run":
        from .config import Geometry

        geometry = None
        if args.dt is not None or args.dx is not None:
            if args.dt is None or args.dx is None:
                parser.error("--dt and --dx must be supplied together.")
            geometry = Geometry(
                dt=args.dt,
                dx=args.dx,
                x0=args.x0,
                t0=args.t0,
                frequency=args.frequency,
            )
        coherence_config = CoherenceConfig(
            aperture_traces=args.aperture,
            time_window=args.time_window,
            max_shift=args.max_shift,
            agc_window=args.agc,
            bandpass_low=args.bandpass[0] if args.bandpass else None,
            bandpass_high=args.bandpass[1] if args.bandpass else None,
            mute_time=args.mute,
        )
        velocity_config = VelocityConfig(
            epsilon_min=args.eps_min,
            epsilon_max=args.eps_max,
            epsilon_step=args.eps_step,
            mute_time=args.mute,
        )
        focusing_config = FocusingConfig(
            epsilon_min=args.eps_min,
            epsilon_max=args.eps_max,
            epsilon_step=args.eps_step,
            mute_time=args.mute,
        )
        payload = run_all(
            args.input,
            args.output,
            coherence_config=coherence_config,
            velocity_config=velocity_config,
            focusing_config=focusing_config,
            method=args.method,
            field=args.field,
            edge_field="reference" if args.field == "reference" else "tau",
        )
        print(f"run: {payload}")
        return 0

    parser.error(f"Unknown command {command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())

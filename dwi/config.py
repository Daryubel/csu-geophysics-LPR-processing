"""Physical constants, geometry and analysis configuration objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C0 = 299_792_458.0          # speed of light in vacuum, m/s
EPS0 = 8.854_187_812_8e-12  # vacuum permittivity, F/m
MU0 = 1.256_637_062_12e-6   # vacuum permeability, H/m
Z0 = np.sqrt(MU0 / EPS0)    # vacuum impedance, ohm


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite; got {value!r}.")
    return value


@dataclass(frozen=True)
class Geometry:
    """Acquisition geometry of a zero-offset B-scan.

    Parameters
    ----------
    dt:
        Sample interval in seconds.
    dx:
        Trace spacing in metres.
    x0:
        Position of the first trace in metres (default 0).
    t0:
        Time of the first sample in seconds (default 0).
    frequency:
        Dominant antenna frequency in Hz.
    antenna_offset:
        Tx-Rx separation in metres.  Zero for the zero-offset approximation.
    """

    dt: float
    dx: float
    x0: float = 0.0
    t0: float = 0.0
    frequency: float = 5.0e8
    antenna_offset: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dt", _positive(self.dt, "dt"))
        object.__setattr__(self, "dx", _positive(self.dx, "dx"))
        object.__setattr__(self, "frequency", _positive(self.frequency, "frequency"))
        if not np.isfinite(self.x0):
            raise ValueError("x0 must be finite.")
        if not np.isfinite(self.t0):
            raise ValueError("t0 must be finite.")
        if self.antenna_offset < 0.0 or not np.isfinite(self.antenna_offset):
            raise ValueError("antenna_offset must be finite and non-negative.")

    def time_axis(self, samples: int) -> np.ndarray:
        return self.t0 + np.arange(int(samples), dtype=np.float64) * self.dt

    def x_axis(self, traces: int) -> np.ndarray:
        return self.x0 + np.arange(int(traces), dtype=np.float64) * self.dx


@dataclass(frozen=True)
class CoherenceConfig:
    """Parameters of the semblance-based diffraction reconstruction."""

    aperture_traces: int = 9
    time_window: int = 31
    max_shift: int = 12
    shift_step: int = 1
    coherence_power: float = 1.0
    agc_window: int = 0
    bandpass_low: float | None = None
    bandpass_high: float | None = None
    mute_time: float = 0.0

    def __post_init__(self) -> None:
        if int(self.aperture_traces) < 1:
            raise ValueError("aperture_traces must be at least 1.")
        if int(self.time_window) < 1:
            raise ValueError("time_window must be at least 1.")
        if int(self.max_shift) < 0:
            raise ValueError("max_shift must be non-negative.")
        if int(self.shift_step) < 1:
            raise ValueError("shift_step must be at least 1.")
        if not np.isfinite(self.coherence_power) or self.coherence_power <= 0:
            raise ValueError("coherence_power must be positive and finite.")
        if int(self.agc_window) < 0:
            raise ValueError("agc_window must be non-negative.")
        if self.mute_time < 0.0 or not np.isfinite(self.mute_time):
            raise ValueError("mute_time must be finite and non-negative.")
        for name in ("bandpass_low", "bandpass_high"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite when set.")
        if (
            self.bandpass_low is not None
            and self.bandpass_high is not None
            and self.bandpass_high <= self.bandpass_low
        ):
            raise ValueError("bandpass_high must exceed bandpass_low.")


@dataclass(frozen=True)
class VelocityConfig:
    """Parameters of the diffraction velocity spectrum."""

    epsilon_min: float = 3.0
    epsilon_max: float = 9.0
    epsilon_step: float = 0.25
    time_stride: int = 4
    trace_stride: int = 2
    aperture_traces: int = 21
    mute_time: float = 0.0
    confidence_percentile: float = 60.0
    smooth_time_samples: int = 5
    smooth_trace_samples: int = 3

    def __post_init__(self) -> None:
        if self.epsilon_min <= 0.0 or self.epsilon_max <= self.epsilon_min:
            raise ValueError("Require 0 < epsilon_min < epsilon_max.")
        if self.epsilon_step <= 0.0:
            raise ValueError("epsilon_step must be positive.")
        if int(self.time_stride) < 1 or int(self.trace_stride) < 1:
            raise ValueError("time_stride and trace_stride must be at least 1.")
        if int(self.aperture_traces) < 3:
            raise ValueError("aperture_traces must be at least 3.")
        if self.mute_time < 0.0:
            raise ValueError("mute_time must be non-negative.")
        if not 0.0 <= self.confidence_percentile <= 100.0:
            raise ValueError("confidence_percentile must lie in [0, 100].")

    @property
    def epsilon_values(self) -> np.ndarray:
        count = int(np.floor((self.epsilon_max - self.epsilon_min) / self.epsilon_step + 1.0e-9)) + 1
        values = self.epsilon_min + np.arange(count, dtype=np.float64) * self.epsilon_step
        if values[-1] < self.epsilon_max - 1.0e-9:
            values = np.append(values, self.epsilon_max)
        return values


@dataclass(frozen=True)
class FocusingConfig:
    """Parameters of the isolated-diffraction migration-focusing inversion."""

    epsilon_min: float = 3.0
    epsilon_max: float = 9.0
    epsilon_step: float = 0.25
    window_half_samples: int = 8
    focus_u: int = 4
    focus_v: int = 2
    stride_time: int = 6
    stride_trace: int = 4
    isolation: bool = True
    migration: str = "stolt"
    min_confidence: float = 1.20
    min_energy_percentile: float = 55.0
    smooth_sigma: float = 2.0
    mute_time: float = 0.0
    max_depth: float | None = None

    def __post_init__(self) -> None:
        if self.epsilon_min <= 0.0 or self.epsilon_max <= self.epsilon_min:
            raise ValueError("Require 0 < epsilon_min < epsilon_max.")
        if self.epsilon_step <= 0.0:
            raise ValueError("epsilon_step must be positive.")
        for name in ("window_half_samples", "focus_u", "focus_v"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if int(self.stride_time) < 1 or int(self.stride_trace) < 1:
            raise ValueError("stride_time and stride_trace must be at least 1.")
        if str(self.migration).lower() not in {"stolt", "kirchhoff"}:
            raise ValueError("migration must be 'stolt' or 'kirchhoff'.")
        if self.min_confidence <= 0.0:
            raise ValueError("min_confidence must be positive.")
        if not 0.0 <= self.min_energy_percentile <= 100.0:
            raise ValueError("min_energy_percentile must lie in [0, 100].")
        if self.smooth_sigma < 0.0:
            raise ValueError("smooth_sigma must be non-negative.")
        if self.mute_time < 0.0:
            raise ValueError("mute_time must be non-negative.")
        if self.max_depth is not None and self.max_depth <= 0.0:
            raise ValueError("max_depth must be positive when set.")

    @property
    def epsilon_values(self) -> np.ndarray:
        count = int(np.floor((self.epsilon_max - self.epsilon_min) / self.epsilon_step + 1.0e-9)) + 1
        values = self.epsilon_min + np.arange(count, dtype=np.float64) * self.epsilon_step
        if values[-1] < self.epsilon_max - 1.0e-9:
            values = np.append(values, self.epsilon_max)
        return values


@dataclass(frozen=True)
class PropertyConfig:
    """Configuration for the empirical property and layer analysis."""

    edge_sigma: float = 1.0
    edge_percentile: float = 90.0
    min_layer_thickness: float = 0.20
    max_layers: int = 12
    monte_carlo_samples: int = 2000
    random_seed: int = 20260101
    relative_epsilon_sigma: float = 0.03
    density_base: float = 1.919
    loss_tangent_a: float = 0.44
    loss_tangent_b: float = 2.943
    chemistry_a: float = 0.44
    chemistry_b: float = 0.312
    chemistry_c: float = 0.317
    chemistry_d: float = 0.038

    def __post_init__(self) -> None:
        if self.edge_sigma < 0.0:
            raise ValueError("edge_sigma must be non-negative.")
        if not 0.0 <= self.edge_percentile <= 100.0:
            raise ValueError("edge_percentile must lie in [0, 100].")
        if self.min_layer_thickness <= 0.0:
            raise ValueError("min_layer_thickness must be positive.")
        if int(self.max_layers) < 1:
            raise ValueError("max_layers must be at least 1.")
        if int(self.monte_carlo_samples) < 1:
            raise ValueError("monte_carlo_samples must be at least 1.")
        if self.relative_epsilon_sigma < 0.0:
            raise ValueError("relative_epsilon_sigma must be non-negative.")
        if self.density_base <= 1.0:
            raise ValueError("density_base must exceed 1.")

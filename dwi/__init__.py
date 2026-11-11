"""Minimal diffraction inversion and sequential medium-property analysis.

The package implements a compact, reproducible subset of the processing
pipeline described in the accompanying manuscript and in the reference
papers listed in ``README.md``:

1. coherence-based diffraction reconstruction from a zero-offset GPR/LPR
   B-scan;
2. diffraction velocity analysis and initial dielectric-model construction;
3. diffraction-focusing inversion by isolated hyperbolic windows and
   migration focusing (with an optional frequency-domain DWI refinement);
4. empirical conversion of relative permittivity to bulk density, loss
   tangent and FeO+TiO2 content, followed by layer-by-layer (sequential)
   property analysis.

The code is intentionally small and dependency-light.  NumPy/SciPy are
sufficient for the quickstart.  Optional CUDA acceleration is exposed through
PyTorch/CuPy in :mod:`dwi.backends` and :mod:`dwi.wavefront`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Geometry, CoherenceConfig, VelocityConfig, FocusingConfig
from .io import Survey, load_survey, load_example
from .coherence import reconstruct_diffraction
from .velocity import velocity_spectrum, pick_velocity, initial_model_from_picks
from .focusing import invert_focusing
from .properties import properties_from_epsilon, sequential_analysis

__all__ = [
    "__version__",
    "Geometry",
    "CoherenceConfig",
    "VelocityConfig",
    "FocusingConfig",
    "Survey",
    "load_survey",
    "load_example",
    "reconstruct_diffraction",
    "velocity_spectrum",
    "pick_velocity",
    "initial_model_from_picks",
    "invert_focusing",
    "properties_from_epsilon",
    "sequential_analysis",
]

This folder contains a compact, self-contained implementation of the diffraction-inversion and property sequential analysis for article: `**Diffraction wavefront-driven inversion of Chang’E-4 Lunar Penetrating Radar data for high-resolution subsurface structure imaging**`

The workflow follows the manuscript pipeline:

1. **Coherence-based diffraction reconstruction** — estimate the local dip and semblance of the reflected wavefield, stack the coherent part and keep the residual as the diffraction section.
2. **Diffraction velocity analysis** — scan hyperbolic moveout of the reconstructed diffractions and build an apparent permittivity model.
3. **Diffraction-focusing inversion** — isolate hyperbolic windows for each candidate permittivity, migrate them, and accept the permittivity that maximises the local gradient-focusing ratios `R1` and `R2` (Lv & Zhang, 2023).
4. **Optional frequency-domain DWI** — refine the focusing model with a small Helmholtz-equation full-waveform inversion of the diffraction field.
5. **Sequential medium-property analysis** — convert relative permittivity to bulk density, loss tangent and FeO+TiO2 content with Olhoeft's lunar-soil correlations, then analyse the layers from the surface downwards.

The default quickstart uses only NumPy, SciPy and Matplotlib.  CUDA is optional and is used by the frequency-domain DWI backend when CuPy is installed.

## Layout

```text
publish_codes/
├── README.md                 this file
├── QUICKSTART.md             five-minute example
├── pyproject.toml            package metadata and console script
├── requirements.txt          core dependencies
├── dwi/
│   ├── config.py             geometry and analysis configuration
│   ├── io.py                 loaders, preprocessing, run directories
│   ├── coherence.py          semblance diffraction reconstruction
│   ├── velocity.py           hyperbolic velocity spectrum and picks
│   ├── migration.py          Stolt and Kirchhoff migrators
│   ├── focusing.py           isolated-diffraction focusing inversion
│   ├── properties.py         empirical property laws and layer statistics
│   ├── wavefront.py          optional frequency-domain DWI
│   ├── backends.py           CPU SuperLU / optional CuPy solvers
│   ├── pipeline.py           end-to-end stage orchestration
│   ├── plotting.py           diagnostic figures
│   └── cli.py                `dwi` command-line interface
├── examples/
│   ├── quickstart.py         run the packaged synthetic example
│   ├── make_example_data.py  regenerate the packaged example
│   └── data/
│       ├── ilmenite_demo.npz small synthetic example
│       └── README.md         provenance and units
└── runs/                     generated outputs (ignored by git)
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Linux / WSL
pip install -r requirements.txt
```

For the optional CUDA DWI backend, install a CuPy wheel that matches the installed CUDA toolkit (for example `cupy-cuda12x`) and pass `--dwi-backend cuda`.

## Quickstart

```bash
python examples/quickstart.py
```

The script writes `runs/quickstart` and prints the output paths.  It uses the archived coherence-analysis diffraction field of the synthetic `ilmanite_10_3_giannakis_evidence` model as the observed diffraction section, so the inversion is not limited by the small built-in reconstruction.  The reconstruction stage still runs and saves its own result for comparison.

The same workflow is available stage by stage:

```bash
python -m dwi.cli reconstruct --input examples/data/ilmenite_demo.npz --output runs/demo
python -m dwi.cli velocity  --run runs/demo --field reference
python -m dwi.cli invert    --run runs/demo --method focusing --field reference
python -m dwi.cli analyze   --run runs/demo --edge-field reference
```

Or all stages in one command:

```bash
python -m dwi.cli run --input examples/data/ilmenite_demo.npz --output runs/demo --field reference
```

See `QUICKSTART.md` for the expected figures and a description of every output.

## Input data

The loaders accept:

- `.npz` with a `data` (or `original`/`bscan`) array and optional `meta_*` geometry keys;
- `.npy` plus explicit `--dt` and `--dx`;
- delimited text matrices (`.dat`, `.txt`, `.csv`);
- native gprMax `.out` files (requires `h5py`; reads `rxs/rx1/Ez`).

Arrays are interpreted as `(time, trace)` with `time` increasing downwards. The legacy exports in the synthetic archive are stored as `(trace, time)`; the loader transposes them automatically when the shape is unambiguous, and `--orientation trace_time` is available when it is not.

Required geometry:

| symbol | meaning | example |
|---|---|---|
| `dt` | time sample interval (s) | `9.775e-11` |
| `dx` | trace spacing (m) | `0.02` |
| `x0` | first trace position (m) | `1.70` |
| `t0` | first sample time (s) | `2.444e-8` |
| `frequency` | dominant antenna frequency (Hz) | `5.0e8` |

## Method notes

### Coherence reconstruction

At every image point the local time shift `tau` (samples/trace) that maximises semblance is estimated over a coherent stacking aperture.  The aligned stack is the reflection estimate; the diffraction field is the residual.  A local amplitude ratio is also returned as a diagnostic.  The output `coherence` is the semblance and `tau` is the local time-shift field used as a geometric guide.

### Velocity spectrum

For each centre time `t0`, trace `x0` and candidate relative permittivity `epsilon`, the section is stacked along

```text
t(x) = sqrt(t0^2 + 4 * epsilon * (x - x0)^2 / c^2).
```

The resulting `(time, trace, epsilon)` energy volume is the zero-offset velocity spectrum.  The peak gives an apparent permittivity and a confidence ratio.  The initial model is an interpolation of the accepted picks.

### Focusing inversion

For each candidate permittivity the diffraction field is isolated with hyperbolic windows, migrated with the corresponding constant velocity and scored by the local gradient-focusing ratios

```text
R1 = P / (P + A + B + C + D)
R2 = P / (A + B + C + D)
```

where `P` is the gradient energy in a small template around the candidate diffractor and `A`-`D` are the surrounding template energies.  The selected permittivity is the one that maximises both ratios simultaneously.  Accepted points are interpolated into a regular `(depth, distance)` model and smoothed.

### Optional frequency-domain DWI

`dwi/wavefront.py` implements a small 2-D scalar Helmholtz FWI:

```text
(L + omega^2 / c^2 * epsilon_r) u = q,
```

with a lossy sponge layer, per-frequency complex source calibration, analytic adjoint gradients and L-BFGS-B with first-order smoothness regularization.  It is deliberately compact and intended as a refinement/demonstration, not as a replacement for a production frequency-domain solver.

### Sequential property analysis

The empirical lunar-soil relations are

```text
rho       = ln(epsilon_r) / ln(1.919)
tan(delta) = 10^(0.44 * rho - 2.943)
FeO+TiO2  = (0.44 * rho - 0.312 * rho + 0.317) / 0.038
```

with `rho` in g/cm3 and the last quantity in percent.  Layers are detected from the archived/local `tau` edge map or from the vertical permittivity gradient, then processed in depth order.  Each layer reports median, mean, standard deviation, 10th/90th percentiles and Monte-Carlo uncertainty bands.  These empirical laws are specific to lunar materials and are not universal petrophysical relations.

## Outputs

A run directory contains:

| file | content |
|---|---|
| `original.npy` | input B-scan |
| `diffraction.npy` | reconstructed diffraction section |
| `diffraction_used.npy` | diffraction field actually used by velocity/inversion |
| `reflection.npy` | coherent reflection stack |
| `coherence.npy` | semblance |
| `tau.npy` | local time shift (samples/trace) |
| `velocity_spectrum.npz` | diffraction velocity spectrum |
| `velocity_picks.csv` | accepted apparent-permittivity picks |
| `initial_epsilon.npy` | initial dielectric model |
| `epsilon.npy` | final relative-permittivity model |
| `confidence.npy` | focusing confidence |
| `focusing_points.csv` | accepted/rejected focusing points |
| `focusing_spectrum.npz` | `R1`, `R2` and energy versus permittivity |
| `properties.npz` | permittivity, density, loss tangent, chemistry |
| `property_profiles.csv` | median property profiles |
| `layers.csv` | sequential layer statistics |
| `boundaries.npy` | detected layer boundaries (m) |
| `figures/` | reconstruction, velocity, focusing, property and summary figures |
| `metadata.json` | geometry, stage settings and reference metrics |

## Relation to the reference code

This is a new, self-contained implementation.  It does not modify any upstream reference code.  The algorithms were informed by the published methods and by the authors' internal reference implementations:

- semblance-based coherence analysis and diffraction separation (He et al., 2023; Zhang et al., 2023; Schwarz, 2019);
- frequency-domain full-waveform inversion with analytic adjoints (Meles et al., 2011; Lavoué et al., 2014);
- Olhoeft's empirical lunar-soil property correlations (Olhoeft & Strangway, 1975).

The packaged example is a small crop of a synthetic gprMax LPR model distributed with the associated manuscript's data archive; see `examples/data/README.md` for exact indices, units and provenance.

## References

- He, R., Zhang, B., Lei, Y., Zhang, H., & Dai, Q. (2023). Wavefront reconstruction and diffraction-driven GPR inversion by semblance-based coherence analysis. *IEEE GRSL*.
- Zhang, B., He, R., Zhang, H., Lei, Y., & Dai, Q. (2023). Diffraction separation by a coherence analysis framework for GPR applications. *IEEE TGRS*.
- Lv, W., & Zhang, J. (2023). High-resolution permittivity estimation of GPR data by migration with isolated hyperbolic diffractions and local focusing analyses. *Frontiers in Astronomy and Space Sciences*.
- Meles, G., Greenhalgh, S., van der Kruk, J., Green, A., & Maurer, H. (2011). Taming the non-linearity problem in GPR full-waveform inversion for high contrast media. *Journal of Applied Geophysics*.
- Lavoué, F., Brossier, R., Métivier, L., Garambois, S., & Virieux, J. (2014). Two-dimensional permittivity and conductivity imaging by full waveform inversion of multioffset GPR data. *Geophysical Journal International*.
- Olhoeft, G. R., & Strangway, D. W. (1975). Dielectric properties of the first 100 meters of the Moon. *Earth and Planetary Science Letters*.
- Schwarz, B. (2019). Coherent wavefield subtraction for diffraction separation. *Geophysics*.

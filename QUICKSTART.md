# Quickstart

This example runs the complete diffraction-inversion and sequential property-analysis workflow on a small cropped synthetic LPR section.  It takes roughly one minute on a normal laptop CPU; no CUDA is required.

## 1. Install

From `publish_codes`:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or WSL use `source .venv/bin/activate` instead of the Windows activation line.

## 2. Run the example

```powershell
python examples\quickstart.py
```

The script writes `runs/quickstart`.  A successful run ends with a short summary such as:

```text
finished
  diffraction: runs\quickstart\diffraction.npy
  permittivity: runs\quickstart\epsilon.npy
  layers:       runs\quickstart\layers.csv
  figures:      runs\quickstart\figures
```

Open the figures in this order:

1. `figures/reconstruction.png` — input B-scan, reconstructed diffraction, coherent reflection stack, semblance and local time shift.
2. `figures/velocity.png` — diffraction velocity spectrum, picked apparent permittivity and the initial dielectric model.
3. `figures/focusing.png` — final permittivity, focusing confidence and accepted/rejected focusing points.
4. `figures/properties.png` — permittivity, bulk density, loss tangent and FeO+TiO2 maps with the detected layer boundaries.
5. `figures/summary.png` — compact input/diffraction/model overview.

## 3. Inspect the numbers

```powershell
python -c "import numpy as np; e=np.load('runs/quickstart/epsilon.npy'); print(e.shape, np.percentile(e,[5,50,95]))"
```

`layers.csv` contains one row per layer, in depth order, with median, percentiles and Monte-Carlo uncertainty bands for all properties. `focusing_points.csv` contains every candidate focusing point and its acceptance flag, which is useful for checking whether a layer boundary is supported by local diffractors.

## 4. Run stage by stage

```powershell
python -m dwi.cli reconstruct --input examples/data/ilmenite_demo.npz --output runs/demo
python -m dwi.cli velocity  --run runs/demo --field reference
python -m dwi.cli invert    --run runs/demo --method focusing --field reference
python -m dwi.cli analyze   --run runs/demo --edge-field reference
```

The `--field reference` option uses the archived coherence-analysis diffraction export stored in the example archive.  Omit it to use the diffraction field produced by this package:

```powershell
python -m dwi.cli velocity --run runs/demo
python -m dwi.cli invert   --run runs/demo
```

The built-in reconstruction is intentionally simple; the archived export is provided so that the inversion and property stages can be evaluated independently of the reconstruction stage.

## 5. Try your own B-scan

If your array is already `(time, trace)` and stored as `.npy`:

```powershell
python -m dwi.cli run ^
  --input my_bscan.npy ^
  --output runs/my_survey ^
  --dt 1.0e-10 ^
  --dx 0.02 ^
  --x0 0.0 ^
  --frequency 5.0e8 ^
  --eps-min 3 --eps-max 9
```

For a text export stored as `(trace, time)`, add `--orientation trace_time`.  For a `.npz`, either store `data` in `(time, trace)` order or add `meta_dt`, `meta_dx`, `meta_x0`, `meta_t0`, `meta_frequency` keys and let the loader read them.

Useful parameters:

| option | effect |
|---|---|
| `--eps-min`, `--eps-max` | search range for relative permittivity |
| `--eps-step` | velocity/focusing grid spacing |
| `--aperture` | coherence stacking aperture (traces) |
| `--max-shift` | maximum local time shift (samples/trace) |
| `--window` | hyperbolic isolation half-width (samples) |
| `--stride-time`, `--stride-trace` | focusing analysis grid |
| `--min-confidence` | reject weak focusing picks |
| `--no-isolation` | migrate the whole section for each velocity |
| `--migration kirchhoff` | use Kirchhoff instead of Stolt migration |
| `--layers N` | force N equal-depth layers in the analysis |
| `--boundaries a b c ...` | force explicit layer boundaries (m) |

## 6. Optional frequency-domain DWI

The Helmholtz DWI refinement is optional and slower:

```powershell
python -m dwi.cli invert --run runs/demo --method both --field reference --dwi-backend cpu --dwi-iterations 4
```

`--method wavefront` starts directly from the velocity-spectrum initial model; `--method both` starts from the focusing result.  For a CUDA run, install CuPy first and use `--dwi-backend cuda`.

## 7. Regenerate the packaged example

The example is a crop of the synthetic archive; see `examples/data/README.md`.  To rebuild the `.npz` from the original text exports:

```powershell
python examples/make_example_data.py --source "C:/path/to/legacy_exports"
```

No external source files are modified by this package.

# Atmospheric Radiative-Transfer Surrogate

This project builds a fast neural-network surrogate for atmospheric radiative-transfer spectra. It uses ERA5 atmospheric profiles and a digital elevation model (DEM) to construct an atmosphere at a site or across terrain, generates reference spectra with the external **AM** radiative-transfer model, and trains PyTorch models to predict:

- atmospheric transmittance (`tx`)
- sky brightness temperature (`Tb`, K)

The current data layout and example paths are configured for a Kanpur site, but the core pipeline is reusable for other locations with compatible ERA5 pressure-level data and a DEM.

## Pipeline

```text
ERA5 pressure-level profiles + DEM elevation
                 |
                 v
30-feature encoded atmospheric state
                 |
                 v
AM (.amc) radiative-transfer calculation
                 |
                 v
cached spectra + packaged .npz training dataset
                 |
                 v
PyTorch surrogate model --> transmittance and Tb spectra
```

Atmospheric states contain an interpolated surface level plus 13 temperature and water-vapour profile levels. The default AM spectrum spans 2–14 GHz in 0.01 GHz increments.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/atmosphere.py` | Creates AM-compatible `.amc` atmosphere descriptions. |
| `src/am.py` | Runs AM, parses its output, and caches spectra. |
| `src/era5.py` | Loads ERA5 arrays, interpolates surface conditions, encodes/decodes atmospheric states, and calculates PWV. |
| `src/dem.py` | Loads/crops DEMs and calculates terrain horizons and observable zenith angles. |
| `src/dataset.py` | Selects unique atmospheric states and packages AM outputs into training datasets. |
| `src/generate_dataset.py` | Command-line dataset-building workflow. |
| `src/train.py` | PyTorch models, training, evaluation, metrics, plotting, checkpoint save/load, and prediction helpers. |
| `src/infer.py` | Small inference wrapper for a preloaded joint model. |
| `src/test.py` / `src/test copy.py` | Configurable scripts for generating AM ground-truth maps over a DEM. |
| `src/*.ipynb`, `Notebooks/` | Exploratory analysis, download, visualization, and experiment notebooks. |
| `DEMs/`, `ERA5_DATABASE/` | Local input data; not included in a normal GitHub upload. |
| `datasets/`, `horizon_*`, `nn_*`, `pwv_figures/`, `yearly_*` | Generated datasets, model outputs, cached spectra, and figures. |

## Requirements

- Python 3.10+ (the scripts use standard NumPy/PyTorch tooling)
- An installed AM executable (`am.exe` on Windows) compatible with AM `.amc` input
- ERA5 pressure-level data stored as NumPy `.npz` files
- A georeferenced DEM readable by Rasterio

Install the Python dependencies in a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy torch matplotlib rasterio numba cupy-cuda12x
```

`cupy-cuda12x` is appropriate only for CUDA 12 systems. For CPU-only use, omit it or adjust `src/era5.py` and `src/dem.py`, which currently import CuPy. Install PyTorch using the command recommended for your OS/CUDA combination at [pytorch.org](https://pytorch.org/get-started/locally/).

## Input data expected by the pipeline

The dataset-generation script expects an ERA5 site directory like this:

```text
ERA5_DATABASE/
└── <site>/
    ├── metadata_pressure.npz
    ├── pressure_2000.npz
    ├── pressure_2001.npz
    └── ...
```

Each pressure file contains `valid_time`, `temperature`, `specific_humidity`, and `geopotential`. The metadata file supplies the target location, the local ERA5 grid indices, and pressure levels. A DEM path and AM executable path are also required.

The project does not include the required source-data downloads or AM executable. Obtain them separately and pass their locations explicitly in commands below.

## Build training and validation data

Run this command from `src` so the local module imports resolve:

```powershell
cd src
python generate_dataset.py `
  --dem-path "..\DEMs\Kanpur\Kanpur_90.tif" `
  --era5-dir "..\ERA5_DATABASE\site_264967_802413" `
  --am-exe "C:\path\to\am.exe" `
  --output-dir "..\datasets" `
  --train-years 2000-2022 `
  --val-years 2023-2025 `
  --num-train-samples 3000 `
  --num-val-samples 20000
```

The script encodes every available timestep, selects representative unique states, runs AM, and writes `train_dataset.npz` and `val_dataset.npz`. Per-sample AM spectra are cached, so re-running the same command reuses completed work. Add `--overwrite` to rebuild the cache and packaged datasets.

## Train a model

There is no separate training CLI; use the functions in `train.py` from a notebook or Python session:

```python
from train import AtmosphericSpectrumNet, prepare_dataloaders, save_model, train_model

train_loader, val_loader, frequency = prepare_dataloaders(
    "../datasets/train_dataset.npz",
    batch_size=64,
)

model = AtmosphericSpectrumNet(spectrum_size=len(frequency))
model, history = train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    learning_rate=1e-3,
    return_history=True,
)
save_model(model, "atmospheric_spectrum_model.pth")
```

`AtmosphericSpectrumNet` predicts both targets jointly. `SpectrumNet`, `train_tx_model`, and `train_tb_model` are available for independently trained single-target models. Use `evaluate_model` and `compute_metrics` to calculate RMSE, MAE, maximum error, R², and per-frequency RMSE.

## Run inference

```python
from train import load_model
from infer import infer

model = load_model("atmospheric_spectrum_model.pth")
tx, tb = infer(
    model,
    insert,
    surface_pressure,
    surface_temperature,
    surface_vmr,
    temperature_profile,
    vmr_profile,
)
```

The input profiles must use the same 13-level arrangement and encoding assumptions used to build the training dataset.

## Terrain and AM reference maps

`src/test.py` creates AM ground-truth maps across a cropped DEM at one timestep. `src/test copy.py` is the multi-timestep/year variant. Both run independent AM calls in a process pool and cache each pixel spectrum. Before using either script, update its path and run settings near the top of the file (`BASE`, input files, output directory, AM executable, timestep(s), crop size, and target frequencies).

## GitHub upload notes

The reusable code is small, but the local source data and generated outputs are large. Do not commit `DEMs/`, `ERA5_DATABASE/`, cached spectra, or experiment outputs unless you intentionally publish them through a data-release service. The existing `.gitignore` already excludes DEMs, ERA5 data, archives, and common temporary files; review it before staging because the generated-output directories listed above are not all currently ignored.

From this project directory, a typical first upload is:

```powershell
git init
git add README.md .gitignore src Notebooks
git status
git commit -m "Initial commit"
```

Add a license before publishing if you want others to have explicit permission to reuse the code.


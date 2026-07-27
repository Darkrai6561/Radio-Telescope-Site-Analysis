"""
generate_dataset.py

Builds and caches the neural-network training/validation datasets for the
AM radiative-transfer surrogate model.

Pipeline
--------
DEM (surface height)
   +
ERA5 pressure-level files for a set of years
   -> encode every timestep into a 30-feature atmosphere vector (era5.py)
   -> collect a representative subset of *unique* encoded atmospheres
      (dataset.collect_training_samples)
   -> run AM on each unique atmosphere, caching every spectrum to disk
      (dataset.build_training_dataset)
   -> save the final (X, frequency, transmittance, brightness_temperature)
      arrays to a single .npz file (dataset.save_training_dataset)

This script is safe to re-run:
  - Per-sample AM spectra are cached under <output_dir>/<split>_am/spectrum_*.npz
    and are NOT recomputed unless --overwrite is passed.
  - The final packaged .npz for a split is NOT rebuilt if it already exists,
    unless --overwrite is passed (it is simply loaded and reported).

Usage
-----
Basic (uses the defaults below, matching the paths used in test.py):

    python generate_dataset.py

Custom:

    python generate_dataset.py \
        --dem-path "C:\\path\\to\\Kanpur_90.tif" \
        --era5-dir "C:\\path\\to\\ERA5_DATABASE\\site_264967_802413" \
        --am-exe "C:\\path\\to\\am.exe" \
        --output-dir "C:\\path\\to\\datasets" \
        --train-years 2000-2022 \
        --val-years 2023-2025 \
        --num-train-samples 3000 \
        --sample-method stratified_time

Force a clean rebuild of everything:

    python generate_dataset.py --overwrite
"""

import argparse
from pathlib import Path

import numpy as np

from dem import load_dem, crop_dem
from era5 import (
    load_metadata,
    load_pressure_dataset,
    build_atmosphere_batch_over_time,
)
from dataset import (
    collect_training_samples,
    build_training_dataset,
    save_training_dataset,
    load_training_dataset,
)


# ---------------------------------------------------------------------------
# Defaults (matches the paths already used in test.py -- override via CLI)
# ---------------------------------------------------------------------------

DEFAULT_DEM_PATH = r"C:\Users\Darkrai\Projects\iitk proj\DEMs\Kanpur\Kanpur_90.tif"
DEFAULT_ERA5_DIR = r"C:\Users\Darkrai\Projects\iitk proj\ERA5_DATABASE\site_264967_802413"
DEFAULT_AM_EXE = r"am.exe"
DEFAULT_OUTPUT_DIR = r"C:\Users\Darkrai\Projects\iitk proj\datasets"

# Pressure files are expected to live at <era5_dir>/pressure_<year>.npz
PRESSURE_FILE_PATTERN = "pressure_{year}.npz"
METADATA_FILE_NAME = "metadata_pressure.npz"


def parse_year_range(spec):
    """
    Parse a year spec like "2000-2022" or "2023,2024,2025" into a list of ints.
    """
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


def get_surface_height(dem_path, target_lat, target_lon, half_length=4):
    """
    Compute the DEM surface elevation at the target site, exactly as done
    in test.py: crop a small window around the site and take the center
    pixel (falling back to the local median if the center pixel is nodata).
    """
    dem, transform, bounds = load_dem(dem_path)
    cropped, _, r0, c0 = crop_dem(
        dem, transform, target_lat, target_lon, half_length=half_length
    )
    center = cropped[r0, c0]
    if np.isfinite(center):
        return float(center)
    return float(np.nanmedian(cropped))


def encode_years(era5_dir, years, lat_idx, lon_idx, surface_height, pressure_levels):
    """
    Load and encode every timestep for a list of years into one big pool of
    encoded atmosphere vectors.

    Returns
    -------
    X_pool : (N, 30) float32 ndarray
        Encoded atmospheres, concatenated across all requested years.
    """
    era5_dir = Path(era5_dir)
    pools = []

    for year in years:
        pressure_path = era5_dir / PRESSURE_FILE_PATTERN.format(year=year)
        if not pressure_path.exists():
            print(f"  [skip] {pressure_path} not found")
            continue

        valid_time, temperature, q, geo = load_pressure_dataset(
            pressure_path, lat_idx, lon_idx
        )

        X, insert, ps, ts, vs = build_atmosphere_batch_over_time(
            temperature,
            q,
            geo,
            surface_height,
            pressure_levels,
        )

        print(f"  [ok]   {pressure_path.name}: {X.shape[0]} timesteps encoded")
        pools.append(X)

    if not pools:
        raise RuntimeError(f"No pressure files found for years {years} in {era5_dir}")

    return np.concatenate(pools, axis=0)


def build_split(
    split_name,
    era5_dir,
    years,
    lat_idx,
    lon_idx,
    surface_height,
    pressure_levels,
    am_exe,
    output_dir,
    num_samples,
    sample_method,
    zenith_angle,
    overwrite,
):
    """
    Build (or load, if cached) one dataset split: encode -> subsample ->
    run AM -> package -> save.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / f"{split_name}_dataset.npz"

    if final_path.exists() and not overwrite:
        print(f"\n[{split_name}] Found existing dataset at {final_path}, loading it.")
        X, frequency, transmittance, brightness_temperature = load_training_dataset(
            final_path
        )
        print(
            f"[{split_name}] Loaded {X.shape[0]} samples, "
            f"{frequency.shape[0]} frequency points. "
            f"(pass --overwrite to force a rebuild)"
        )
        return final_path

    print(f"\n[{split_name}] Encoding years {years} ...")
    X_pool = encode_years(
        era5_dir, years, lat_idx, lon_idx, surface_height, pressure_levels
    )
    print(f"[{split_name}] Encoded pool: {X_pool.shape[0]} timesteps total")

    print(
        f"[{split_name}] Selecting up to {num_samples} unique atmospheres "
        f"(method='{sample_method}') ..."
    )
    samples, original_indices = collect_training_samples(
        X_pool, num_samples=num_samples, method=sample_method
    )
    print(f"[{split_name}] Selected {samples.shape[0]} unique atmospheres")

    am_cache_dir = output_dir / f"{split_name}_am"
    print(f"[{split_name}] Running AM (cached under {am_cache_dir}) ...")
    X, frequency, transmittance, brightness_temperature = build_training_dataset(
        samples,
        pressure_levels,
        am_exe,
        am_cache_dir,
        zenith_angle=zenith_angle,
        keep_amc=False,
        overwrite=overwrite,
    )

    save_training_dataset(final_path, X, frequency, transmittance, brightness_temperature)
    print(f"[{split_name}] Saved packaged dataset -> {final_path}")

    return final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dem-path", default=DEFAULT_DEM_PATH)
    parser.add_argument("--era5-dir", default=DEFAULT_ERA5_DIR)
    parser.add_argument("--am-exe", default=DEFAULT_AM_EXE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--train-years",
        default="2000-2022",
        help="e.g. '2000-2022' or '2000,2001,2005'",
    )
    parser.add_argument(
        "--val-years",
        default="2023-2025",
        help="e.g. '2023-2025' or '2023,2024,2025'",
    )

    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=3000,
        help="Max number of unique atmospheres to run AM on for training.",
    )
    parser.add_argument(
        "--num-val-samples",
        type=int,
        default=20000,
        help=(
            "Max number of unique atmospheres to run AM on for validation. "
            "Set high (default) so validation captures essentially all "
            "distinct atmospheric states in the held-out years, not a "
            "further subsample of them."
        ),
    )
    parser.add_argument(
        "--sample-method",
        default="stratified_time",
        choices=["random", "unique_first", "unique_random", "stratified_time"],
        help="See dataset.collect_training_samples for details.",
    )
    parser.add_argument("--zenith-angle", type=float, default=0.0)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute everything (AM spectra + packaged datasets) even if cached.",
    )

    args = parser.parse_args()

    era5_dir = Path(args.era5_dir)
    metadata_path = era5_dir / METADATA_FILE_NAME

    print(f"Loading metadata from {metadata_path} ...")
    (
        place_name,
        target_lat,
        target_lon,
        nearest_lat,
        nearest_lon,
        lat_idx,
        lon_idx,
        pressure_levels,
    ) = load_metadata(metadata_path)
    print(f"Site: {place_name} ({target_lat:.4f}, {target_lon:.4f})")

    print(f"Computing surface height from DEM {args.dem_path} ...")
    surface_height = get_surface_height(args.dem_path, target_lat, target_lon)
    print(f"Surface height: {surface_height:.2f} m")

    train_years = parse_year_range(args.train_years)
    val_years = parse_year_range(args.val_years)

    train_path = build_split(
        "train",
        era5_dir,
        train_years,
        lat_idx,
        lon_idx,
        surface_height,
        pressure_levels,
        args.am_exe,
        args.output_dir,
        args.num_train_samples,
        args.sample_method,
        args.zenith_angle,
        args.overwrite,
    )

    val_path = build_split(
        "val",
        era5_dir,
        val_years,
        lat_idx,
        lon_idx,
        surface_height,
        pressure_levels,
        args.am_exe,
        args.output_dir,
        args.num_val_samples,
        args.sample_method,
        args.zenith_angle,
        args.overwrite,
    )

    print("\nDone.")
    print(f"  Training dataset:   {train_path}")
    print(f"  Validation dataset: {val_path}")
    print(
        "\nNext time, just re-run this same command -- both the per-sample "
        "AM cache and the final .npz files will be reused instead of "
        "regenerated, unless you pass --overwrite."
    )


if __name__ == "__main__":
    main()
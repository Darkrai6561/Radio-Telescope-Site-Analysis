"""
generate_am_dem_map_2025.py

AM ground-truth DEM maps for the year 2025 — which is in your held-out
validation years (2023-2025 in generate_dataset.py), so this is a
genuine out-of-sample test: the NN never saw 2025 atmospheres during
training, and neither did the model selection / early stopping.

Same machinery as generate_am_dem_map_parallel.py (stdin, per-pixel
caching, quiet logging, process-pool parallelism, decode->build_amc
so the ground truth sits on the same quantization grid the NN trained
on) — just pointed at pressure_2025.npz and generalized to loop over a
LIST of timesteps instead of a single one, since running every hour of
2025 over the whole DEM (~8760 timesteps x ~160,000 pixels) is not
computationally feasible.

Pick which hours of 2025 you actually want a ground-truth map for via
TIME_INDICES below (indices into pressure_2025.npz's time axis, i.e.
into `valid_time`) — e.g. a handful of hours spread across the year,
or specific hours you want to eyeball against the NN prediction.
"""

import os
import subprocess
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dem import load_dem, crop_dem
from era5 import (
    load_metadata,
    load_pressure_dataset,
    build_atmosphere_batch_over_heights,
    decode_atmosphere,
)
from atmosphere import build_amc
from am import parse_am_output, save_spectrum, load_spectrum


# ==========================================================
# PATHS
# ==========================================================

BASE = Path(r"C:\Users\Darkrai\Projects\iitk proj")

DEM_PATH = BASE / r"DEMs\Kanpur\Kanpur_90.tif"
ERA5_DIR = BASE / r"ERA5_DATABASE\site_264967_802413"
METADATA_PATH = ERA5_DIR / "metadata_pressure.npz"

YEAR = 2025
PRESSURE_PATH = ERA5_DIR / f"pressure_{YEAR}.npz"

AM_EXE = r"am.exe"

OUT_DIR = BASE / r"nn_surrogate_300\whole_dem_am_ground_truth" / str(YEAR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECTRUM_CACHE_DIR = OUT_DIR / "am_pixel_cache"
SPECTRUM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# SETTINGS
# ==========================================================

# Which hours of 2025 to generate a full-DEM AM map for. Indices into
# pressure_2025.npz's time axis (valid_time[i] tells you which hour
# that is). Pick a handful spread across the year, or specific hours
# you want to compare against the NN's 2025 predictions.
TIME_INDICES = [4380]

HALF_LENGTH = 100
ZENITH_ANGLE = 0.0

FREQ_TARGETS_GHZ = [10.0, 11.0, 12.0, 13.0]

OVERWRITE = False

N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# ==========================================================
# WORKER-PROCESS GLOBALS
# ==========================================================

_AM_EXE = None
_PRESSURE_LEVELS = None
_ZENITH_ANGLE = None
_CACHE_DIR = None
_LOG_FILE = None
_LOG_LOCK = None


def _init_worker(am_executable, pressure_levels, zenith_angle, cache_dir, log_file, log_lock):
    global _AM_EXE, _PRESSURE_LEVELS, _ZENITH_ANGLE, _CACHE_DIR, _LOG_FILE, _LOG_LOCK
    _AM_EXE = am_executable
    _PRESSURE_LEVELS = pressure_levels
    _ZENITH_ANGLE = zenith_angle
    _CACHE_DIR = cache_dir
    _LOG_FILE = log_file
    _LOG_LOCK = log_lock


def _log_quiet(message):
    with _LOG_LOCK:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message)


def _run_am_quiet_worker(amc_text, pixel_label):
    am_executable = Path(_AM_EXE)

    result = subprocess.run(
        [str(am_executable), "-"],
        input=amc_text,
        capture_output=True,
        text=True,
        cwd=am_executable.parent if str(am_executable.parent) else None,
    )

    try:
        frequency, transmittance, brightness_temperature = parse_am_output(
            result.stdout
        )
    except RuntimeError as e:
        _log_quiet(
            f"[{pixel_label}] AM PARSE FAILURE "
            f"(return code {result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n\n"
        )
        raise RuntimeError(f"AM failed on {pixel_label}") from e

    if result.returncode != 0:
        _log_quiet(
            f"[{pixel_label}] AM completed with warnings "
            f"(return code {result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n\n"
        )

    return frequency, transmittance, brightness_temperature


def _process_one_pixel(task):
    k, row_idx, col_idx, encoded_row, overwrite = task

    pixel_label = f"pixel r{row_idx:04d}_c{col_idx:04d}"
    spectrum_file = _CACHE_DIR / f"spectrum_r{row_idx:04d}_c{col_idx:04d}.npz"

    if spectrum_file.exists() and not overwrite:
        freq, tx, tb, _ = load_spectrum(spectrum_file)
        return k, freq, tx, tb

    (
        insert_idx,
        surface_pressure,
        surface_temperature,
        surface_vmr,
        temp_profile,
        vmr_profile,
    ) = decode_atmosphere(encoded_row)

    amc = build_amc(
        pressure_levels=_PRESSURE_LEVELS,
        temperature=temp_profile,
        vmr=vmr_profile,
        insert_idx=insert_idx,
        surface_pressure=surface_pressure,
        surface_temperature=surface_temperature,
        surface_vmr=surface_vmr,
        zenith_angle=_ZENITH_ANGLE,
    )

    freq, tx, tb = _run_am_quiet_worker(amc, pixel_label)
    save_spectrum(spectrum_file, freq, tx, tb)

    return k, freq, tx, tb


def generate_am_map_for_timestep(
    time_index,
    cropped_dem,
    temperature,
    q,
    geopotential,
    pressure_levels,
    valid_time,
):
    """
    Run the whole DEM crop through AM for one timestep, using a
    per-timestep cache subdirectory and log file so multiple timesteps
    don't collide or overwrite each other.
    """
    H, W = cropped_dem.shape
    flat_h = cropped_dem.reshape(-1)

    valid_mask = np.isfinite(flat_h)
    valid_heights = flat_h[valid_mask]
    valid_indices = np.flatnonzero(valid_mask)

    print(f"\n[t{time_index}] Encoding atmospheres (vectorized)...")
    X_valid, insert, ps, ts, vs = build_atmosphere_batch_over_heights(
        temperature=temperature,
        specific_humidity=q,
        geopotential=geopotential,
        surface_heights=valid_heights,
        pressure_levels=pressure_levels,
        time_index=time_index,
    )

    total = len(valid_indices)

    tasks = []
    for k in range(total):
        pixel_idx = int(valid_indices[k])
        row_idx, col_idx = divmod(pixel_idx, W)
        tasks.append((k, row_idx, col_idx, X_valid[k], OVERWRITE))

    cache_dir = SPECTRUM_CACHE_DIR / f"t{time_index:05d}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log_path = OUT_DIR / f"am_warnings_t{time_index:05d}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            f"AM ground-truth run — year {YEAR}, time_index={time_index}, "
            f"{total} pixels, {N_WORKERS} workers\n\n"
        )

    print(
        f"[t{time_index}] Running AM over {total} pixels with "
        f"{N_WORKERS} workers ... warnings -> {log_path}"
    )

    frequency = None
    transmittance = [None] * total
    brightness_temperature = [None] * total

    manager = multiprocessing.Manager()
    log_lock = manager.Lock()

    start_time = time.time()
    done = 0

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_init_worker,
        initargs=(AM_EXE, pressure_levels, ZENITH_ANGLE, cache_dir, log_path, log_lock),
    ) as executor:
        futures = [executor.submit(_process_one_pixel, task) for task in tasks]

        for future in as_completed(futures):
            k, freq, tx, tb = future.result()

            if frequency is None:
                frequency = freq
            elif frequency.shape != freq.shape or not np.allclose(frequency, freq):
                raise RuntimeError(f"Frequency grid mismatch at task index {k}.")

            transmittance[k] = tx
            brightness_temperature[k] = tb

            done += 1
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else 0.0
            print(
                f"\r[t{time_index}] [{done:6d}/{total}] {rate:6.2f} px/s "
                f"ETA {eta/60:6.1f} min",
                end="",
                flush=True,
            )

    print()

    n_freq = frequency.shape[0]

    Tx_map = np.full((H, W, n_freq), np.nan, dtype=np.float32)
    TB_map = np.full((H, W, n_freq), np.nan, dtype=np.float32)

    Tx_flat = Tx_map.reshape(H * W, n_freq)
    TB_flat = TB_map.reshape(H * W, n_freq)

    Tx_flat[valid_indices] = np.asarray(transmittance, dtype=np.float32)
    TB_flat[valid_indices] = np.asarray(brightness_temperature, dtype=np.float32)

    with open(log_path, "r", encoding="utf-8") as f:
        n_warnings = f.read().count("AM completed with warnings")

    print(f"[t{time_index}] Done. Pixels with AM warnings: {n_warnings} (see {log_path})")

    # -------------------- Save --------------------
    out_file = OUT_DIR / f"whole_dem_am_ground_truth_{YEAR}_t{time_index:05d}.npz"
    np.savez_compressed(
        out_file,
        valid_time=valid_time[time_index],
        frequency=frequency,
        dem=cropped_dem.astype(np.float32),
        transmittance=Tx_map.astype(np.float32),
        brightness_temperature=TB_map.astype(np.float32),
    )
    print(f"[t{time_index}] Saved: {out_file}")

    # -------------------- Plots --------------------
    freq_idxs = [int(np.argmin(np.abs(frequency - f))) for f in FREQ_TARGETS_GHZ]

    for f0, fi in zip(FREQ_TARGETS_GHZ, freq_idxs):
        tb = TB_map[:, :, fi]
        tx = Tx_map[:, :, fi]

        plt.figure(figsize=(7, 6))
        plt.imshow(tb, origin="lower")
        plt.colorbar(label="Brightness Temperature (K)")
        plt.title(
            f"AM Ground Truth {YEAR} t{time_index} — "
            f"Brightness Temperature @ {frequency[fi]:.2f} GHz"
        )
        plt.tight_layout()
        plt.savefig(
            OUT_DIR / f"am_tb_map_t{time_index:05d}_{frequency[fi]:.2f}GHz.png",
            dpi=200,
        )

        plt.figure(figsize=(7, 6))
        plt.imshow(
            tx,
            origin="lower",
            vmin=np.nanpercentile(tx, 2),
            vmax=np.nanpercentile(tx, 98),
        )
        plt.colorbar(label="Transmittance")
        plt.title(
            f"AM Ground Truth {YEAR} t{time_index} — "
            f"Transmittance @ {frequency[fi]:.2f} GHz"
        )
        plt.tight_layout()
        plt.savefig(
            OUT_DIR / f"am_tx_map_t{time_index:05d}_{frequency[fi]:.2f}GHz.png",
            dpi=200,
        )

    plt.close("all")


def main():
    print(f"Loading DEM, metadata, and ERA5 {YEAR} pressure data...")

    if not PRESSURE_PATH.exists():
        raise FileNotFoundError(
            f"Pressure file not found for year {YEAR}: {PRESSURE_PATH}\n"
            "Make sure the ERA5 pressure-level file for 2025 has been "
            "downloaded/placed alongside the other years."
        )

    dem, transform, bounds = load_dem(str(DEM_PATH))

    (
        place_name,
        target_lat,
        target_lon,
        nearest_lat,
        nearest_lon,
        lat_idx,
        lon_idx,
        pressure_levels,
    ) = load_metadata(str(METADATA_PATH))

    cropped_dem, cropped_transform, row0, col0 = crop_dem(
        dem,
        transform,
        target_lat,
        target_lon,
        half_length=HALF_LENGTH,
    )
    print("Crop shape:", cropped_dem.shape)

    valid_time, temperature, q, geopotential = load_pressure_dataset(
        str(PRESSURE_PATH),
        lat_idx,
        lon_idx,
    )

    pressure_levels = np.asarray(pressure_levels, dtype=np.float32)

    n_timesteps = temperature.shape[0]
    bad_indices = [t for t in TIME_INDICES if t < 0 or t >= n_timesteps]
    if bad_indices:
        raise ValueError(
            f"TIME_INDICES {bad_indices} out of range for {YEAR} "
            f"(pressure_{YEAR}.npz has {n_timesteps} timesteps)."
        )

    print(f"Generating AM ground truth for {len(TIME_INDICES)} timestep(s) in {YEAR}: {TIME_INDICES}")

    for time_index in TIME_INDICES:
        generate_am_map_for_timestep(
            time_index=time_index,
            cropped_dem=cropped_dem,
            temperature=temperature,
            q=q,
            geopotential=geopotential,
            pressure_levels=pressure_levels,
            valid_time=valid_time,
        )

    print("\nAll requested 2025 timesteps done.")


if __name__ == "__main__":
    main()
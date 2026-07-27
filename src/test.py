"""
generate_am_dem_map_parallel.py

Same as generate_am_dem_map.py (AM ground-truth map over the whole DEM,
fed via stdin, warnings logged to file, per-pixel caching) but runs AM
calls across a process pool instead of one pixel at a time.

Why this is safe to parallelize
--------------------------------
Each pixel's AM call is fully independent: its own stdin text, its own
subprocess, its own cached output file. Nothing is shared between pixels
except (a) the log file and (b) the cache directory, both handled safely
below (a lock around log writes; unique per-pixel cache filenames).

What this does NOT assume
--------------------------
This does not assume am.exe supports any kind of persistent/batch mode
(reading multiple atmospheres in one process). That would need to be
confirmed against AM's own docs/flags before relying on it — this script
only parallelizes independent, single-atmosphere subprocess calls, which
is safe regardless of AM's internals.

Tuning
------
Start with a modest N_WORKERS (e.g. 4) and time a small run before
cranking it up — if AM does anything with shared temp/lock files
internally, or if disk I/O from the per-pixel cache becomes the
bottleneck, throughput can stop scaling (or even regress) well before
you hit your core count.
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
PRESSURE_PATH = BASE / r"ERA5_DATABASE\site_264967_802413\pressure_2000.npz"
METADATA_PATH = BASE / r"ERA5_DATABASE\site_264967_802413\metadata_pressure.npz"

AM_EXE = r"am.exe"

OUT_DIR = BASE / r"nn_surrogate_300\whole_dem_am_ground_truth"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECTRUM_CACHE_DIR = OUT_DIR / "am_pixel_cache"
SPECTRUM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / "am_warnings.log"

# ==========================================================
# SETTINGS
# ==========================================================

TIME_INDEX = 0
HALF_LENGTH = 200
ZENITH_ANGLE = 0.0

FREQ_TARGETS_GHZ = [10.0, 11.0, 12.0, 13.0]

OVERWRITE = False

# Start conservative. Bump this up once you've confirmed AM is stable
# under concurrency and disk I/O isn't the bottleneck.
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# ==========================================================
# WORKER-PROCESS GLOBALS
#   Set once per worker by _init_worker(), not per task, so we're not
#   re-pickling pressure_levels/paths on every single pixel.
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
    """Append to the shared log file under a cross-process lock."""
    # with _LOG_LOCK:
    #     with open(_LOG_FILE, "a", encoding="utf-8") as f:
    #         f.write(message)
    pass


def _run_am_quiet_worker(amc_text, pixel_label):
    """
    Same stdin-based AM call as the serial version: am.exe is fed
    `amc_text` via subprocess `input=`, no temp files. Warnings go to
    the shared log instead of stdout.
    """
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
    """
    Runs in a worker process. Handles one DEM pixel end to end:
    cache check -> decode -> build_amc -> AM (via stdin) -> cache save.
    """
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


# ==========================================================
# MAIN
#   Everything that actually runs lives in main(), guarded by
#   `if __name__ == "__main__":` below — required on Windows, where
#   ProcessPoolExecutor uses "spawn" and re-imports this module in each
#   worker. Without the guard, workers would re-run the whole script.
# ==========================================================

def main():
    print("Loading DEM, metadata, and ERA5...")

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

    H, W = cropped_dem.shape
    flat_h = cropped_dem.reshape(-1)

    valid_mask = np.isfinite(flat_h)
    valid_heights = flat_h[valid_mask]
    valid_indices = np.flatnonzero(valid_mask)

    print("Encoding atmospheres for all valid DEM pixels (vectorized)...")
    X_valid, insert, ps, ts, vs = build_atmosphere_batch_over_heights(
        temperature=temperature,
        specific_humidity=q,
        geopotential=geopotential,
        surface_heights=valid_heights,
        pressure_levels=pressure_levels,
        time_index=TIME_INDEX,
    )

    total = len(valid_indices)

    tasks = []
    for k in range(total):
        pixel_idx = int(valid_indices[k])
        row_idx, col_idx = divmod(pixel_idx, W)
        tasks.append((k, row_idx, col_idx, X_valid[k], OVERWRITE))

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"AM ground-truth run (parallel, {N_WORKERS} workers) — "
            f"{total} pixels, time_index={TIME_INDEX}\n\n"
        )

    print(
        f"Running AM over {total} pixels with {N_WORKERS} worker "
        f"processes ...\nWarnings (if any) go to: {LOG_PATH}"
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
        initargs=(AM_EXE, pressure_levels, ZENITH_ANGLE, SPECTRUM_CACHE_DIR, LOG_PATH, log_lock),
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
                f"\r[{done:6d}/{total}] {rate:6.2f} px/s ETA {eta/60:6.1f} min",
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

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        n_warnings = f.read().count("AM completed with warnings")

    print("Transmittance map shape (Tx_map):", Tx_map.shape)
    print("Brightness temperature map shape (TB_map):", TB_map.shape)
    print(f"Pixels with AM warnings: {n_warnings} (see {LOG_PATH})")

    # ==========================================================
    # SAVE
    # ==========================================================

    out_file = OUT_DIR / f"whole_dem_am_ground_truth_t{TIME_INDEX:04d}.npz"

    np.savez_compressed(
        out_file,
        valid_time=valid_time[TIME_INDEX],
        frequency=frequency,
        dem=cropped_dem.astype(np.float32),
        transmittance=Tx_map.astype(np.float32),
        brightness_temperature=TB_map.astype(np.float32),
    )

    print("Saved:", out_file)

    # ==========================================================
    # PLOTS AT SELECTED FREQUENCIES
    # ==========================================================

    freq_idxs = [int(np.argmin(np.abs(frequency - f))) for f in FREQ_TARGETS_GHZ]

    for f0, fi in zip(FREQ_TARGETS_GHZ, freq_idxs):
        tb = TB_map[:, :, fi]
        tx = Tx_map[:, :, fi]

        plt.figure(figsize=(7, 6))
        plt.imshow(tb, origin="lower")
        plt.colorbar(label="Brightness Temperature (K)")
        plt.title(f"AM Ground Truth — Brightness Temperature @ {frequency[fi]:.2f} GHz")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"am_tb_map_{frequency[fi]:.2f}GHz.png", dpi=200)

        plt.figure(figsize=(7, 6))
        plt.imshow(
            tx,
            origin="lower",
            vmin=np.nanpercentile(tx, 2),
            vmax=np.nanpercentile(tx, 98),
        )
        plt.colorbar(label="Transmittance")
        plt.title(f"AM Ground Truth — Transmittance @ {frequency[fi]:.2f} GHz")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"am_tx_map_{frequency[fi]:.2f}GHz.png", dpi=200)

    plt.close("all")

    print("Done.")


if __name__ == "__main__":
    main()
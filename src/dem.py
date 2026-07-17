import numpy as np
import cupy as cp
import rasterio
import math
from pathlib import Path
from numba import njit, prange


def load_dem(SITE_DIR):
    dem_src = rasterio.open(SITE_DIR)
    elevation_data = dem_src.read(1)
    transform = dem_src.transform
    bounds = dem_src.bounds
    return elevation_data, transform, bounds


def crop_dem(elevation_data, transform, target_lat, target_lon, half_length=50):
    row, col = rasterio.transform.rowcol(transform, target_lon, target_lat)
    r0 = row - half_length
    r1 = row + half_length + 1
    c0 = col - half_length
    c1 = col + half_length + 1
    r0 = max(r0, 0)
    c0 = max(c0, 0)

    r1 = min(r1, elevation_data.shape[0])
    c1 = min(c1, elevation_data.shape[1])
    cropped_dem = elevation_data[r0:r1, c0:c1]
    new_transform = rasterio.windows.transform(
        rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0),
        transform,
    )

    return cropped_dem, new_transform, row - r0, col - c0

def compute_max_zenith(horizon):
    """
    Compute the maximum observable zenith angle.

    Parameters
    ----------
    horizon : ndarray
        Horizon elevation cube from compute_horizon().

    Returns
    -------
    max_zenith : ndarray
        Maximum observable zenith angle.

    best_zenith : ndarray
        Best observable zenith angle at each pixel.

    min_horizon : ndarray
        Minimum horizon elevation at each pixel.
    """

    max_zenith = 90.0 - horizon

    max_zenith = np.clip(max_zenith, 0.0, 90.0).astype(np.float32)
    best_zenith = np.nanmax(max_zenith, axis=2).astype(np.float32)
    min_horizon = np.nanmin(horizon, axis=2).astype(np.float32)

    return max_zenith, best_zenith, min_horizon


def save_horizon(output_dir, horizon, max_zenith, min_horizon, best_zenith):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "horizon_elevation_deg.npy", horizon.astype(np.float32))
    np.save(output_dir / "max_zenith_angle_deg.npy", max_zenith.astype(np.float32))
    np.save(output_dir / "min_horizon_deg.npy", min_horizon)
    np.save(output_dir / "best_zenith_deg.npy", best_zenith)

    print(f"Saved horizon products to {output_dir}")


def load_horizon(output_dir):
    """
    Load saved horizon products.
    """

    output_dir = Path(output_dir)

    horizon = np.load(output_dir / "horizon_elevation_deg.npy")
    max_zenith = np.load(output_dir / "max_zenith_angle_deg.npy")
    min_horizon = np.load(output_dir / "min_horizon_deg.npy")
    best_zenith = np.load(output_dir / "best_zenith_deg.npy")

    return horizon, max_zenith, min_horizon, best_zenith


def _build_horizon_lookup(
    H: int,
    W: int,
    pixel_size_m: float,
    n_az: int,
):
    """
    Precompute distance and azimuth-bin lookup tables for all possible
    pixel offsets in a DEM of shape (H, W).

    Offsets are indexed by:
        di in [-(H-1), ..., +(H-1)]
        dj in [-(W-1), ..., +(W-1)]

    Returns
    -------
    dist2_lookup : (2H-1, 2W-1) float32
    dist_lookup   : (2H-1, 2W-1) float32
    bin_lookup    : (2H-1, 2W-1) int32
    """
    di_min = -(H - 1)
    dj_min = -(W - 1)

    n_di = 2 * H - 1
    n_dj = 2 * W - 1

    dist2_lookup = np.empty((n_di, n_dj), dtype=np.float32)
    dist_lookup = np.empty((n_di, n_dj), dtype=np.float32)
    bin_lookup = np.empty((n_di, n_dj), dtype=np.int32)

    for di in range(di_min, H):
        for dj in range(dj_min, W):
            # Match the original code exactly:
            # dx = (tj - sj) * pixel_size_m
            # dy = (si - ti) * pixel_size_m = -(ti - si) * pixel_size_m
            dx = dj * pixel_size_m
            dy = -di * pixel_size_m

            d2 = dx * dx + dy * dy
            ii = di - di_min
            jj = dj - dj_min

            dist2_lookup[ii, jj] = d2
            dist_lookup[ii, jj] = math.sqrt(d2)

            az = math.degrees(math.atan2(dx, dy))
            if az < 0.0:
                az += 360.0

            b = int(az + 0.5)
            if b >= n_az:
                b -= n_az

            bin_lookup[ii, jj] = b

    return (
        np.ascontiguousarray(dist2_lookup),
        np.ascontiguousarray(dist_lookup),
        np.ascontiguousarray(bin_lookup),
    )


@njit(parallel=True, fastmath=True)
def _compute_horizon_with_lookup(
    elevation_data,
    dist2_lookup,
    dist_lookup,
    bin_lookup,
    pixel_size_m,
    n_az,
    earth_radius_m,
    use_curvature,
):
    H, W = elevation_data.shape
    horizon = np.zeros((H, W, n_az), dtype=np.float32)

    di_min = -(H - 1)
    dj_min = -(W - 1)

    for si in prange(H):
        for sj in range(W):
            h0 = elevation_data[si, sj]

            if np.isnan(h0):
                for a in range(n_az):
                    horizon[si, sj, a] = np.nan
                continue

            local = np.zeros(n_az, dtype=np.float32)

            for ti in range(H):
                di = ti - si
                ii = di - di_min

                for tj in range(W):
                    if ti == si and tj == sj:
                        continue

                    dj = tj - sj
                    jj = dj - dj_min

                    h = elevation_data[ti, tj]
                    if np.isnan(h):
                        continue

                    dist2 = dist2_lookup[ii, jj]
                    if dist2 <= 0.0:
                        continue

                    dz = h - h0
                    if use_curvature:
                        dz -= dist2 / (2.0 * earth_radius_m)

                    if dz <= 0.0:
                        continue

                    bin_idx = bin_lookup[ii, jj]
                    dist = dist_lookup[ii, jj]

                    slope = math.degrees(math.atan2(dz, dist))
                    if slope > local[bin_idx]:
                        local[bin_idx] = slope

            horizon[si, sj] = local

    return horizon


def compute_horizon(
    elevation_data,
    pixel_size_m=90.0,
    n_az=360,
    earth_radius_m=6371000.0,
    use_curvature=True,
):
    """
    Exact horizon computation, but with precomputed offset lookups.

    This preserves the original algorithm:
    every DEM pixel still gets compared against every other DEM pixel,
    and the result is binned by azimuth exactly as before.
    """
    H, W = elevation_data.shape

    dist2_lookup, dist_lookup, bin_lookup = _build_horizon_lookup(
        H=H, W=W, pixel_size_m=pixel_size_m, n_az=n_az
    )

    return _compute_horizon_with_lookup(
        elevation_data.astype(np.float32),
        dist2_lookup,
        dist_lookup,
        bin_lookup,
        pixel_size_m,
        n_az,
        earth_radius_m,
        use_curvature,
    )

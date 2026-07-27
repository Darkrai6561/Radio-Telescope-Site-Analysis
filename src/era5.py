import numpy as np
import cupy as cp

# Constants for the encoded atmosphere format.
G = 9.80665
EPSILON = 0.622

P_STEP = 50.0
T_STEP = 1.0
Q_STEP = 2e-4


def load_pressure_dataset(path, lat_idx, lon_idx, xp=np):
    """
    Load one ERA5 pressure dataset and return only one grid column.
    """

    ds = xp.load(path)

    valid_time = ds["valid_time"]

    temperature = ds["temperature"][:, :, lat_idx, lon_idx]
    specific_humidity = ds["specific_humidity"][:, :, lat_idx, lon_idx]
    geopotential = ds["geopotential"][:, :, lat_idx, lon_idx]

    return valid_time, temperature, specific_humidity, geopotential


def load_single_dataset(path, lat_idx, lon_idx, xp=np):
    """
    Load one ERA5 single-level dataset and return only one grid point.
    """

    ds = xp.load(path)

    valid_time = ds["valid_time"]

    surface_pressure = ds["surface_pressure"][:, lat_idx, lon_idx]
    total_column_water_vapour = ds["tcwv"][:, lat_idx, lon_idx]

    return valid_time, surface_pressure, total_column_water_vapour


def load_metadata(path):

    meta = np.load(path)

    place_name = str(meta["place_name"])

    target_lat = float(meta["target_lat"])
    target_lon = float(meta["target_lon"])

    nearest_lat = float(meta["nearest_lat"])
    nearest_lon = float(meta["nearest_lon"])

    # Global ERA5 indices
    lat_idx = int(meta["lat_idx"])
    lon_idx = int(meta["lon_idx"])

    # Start of downloaded 12×12 tile
    lat0 = int(meta["lat0"])
    lon0 = int(meta["lon0"])

    # Convert to local indices within the tile
    lat_idx -= lat0
    lon_idx -= lon0

    pressure_levels = meta["pressure_levels"]

    return (
        place_name,
        target_lat,
        target_lon,
        nearest_lat,
        nearest_lon,
        lat_idx,
        lon_idx,
        pressure_levels * 100,
    )


def vmr_from_q(q):
    """
    Convert specific humidity (kg/kg) to volume mixing ratio.
    Elementwise -- works unchanged on scalars, 1D, or 2D arrays.
    """
    return q / (EPSILON + (1.0 - EPSILON) * q)


def interpolate_surface(
    pressure_levels,
    height,
    temperature,
    vmr,
    surface_height,
):
    """
    Interpolate the atmospheric state to the surface (single sample, as before).

    Parameters
    ----------
    pressure_levels : ndarray
        Pressure levels (Pa)

    height : ndarray
        Geopotential height (m)

    temperature : ndarray
        Temperature profile (K)

    vmr : ndarray
        Water vapour mixing ratio profile

    surface_height : float
        DEM elevation (m)

    Returns
    -------
    insert : int
    surface_pressure : float
    surface_temperature : float
    surface_vmr : float
    """

    insert = np.sum(height < surface_height)

    insert = np.clip(insert, 1, len(pressure_levels) - 1)

    z0 = height[insert - 1]
    z1 = height[insert]
    if z0 == z1:
        ratio = 0
    else:
        ratio = (surface_height - z0) / (z1 - z0)

    logp0 = np.log(pressure_levels[insert - 1])
    logp1 = np.log(pressure_levels[insert])

    surface_pressure = np.exp(logp0 + ratio * (logp1 - logp0))

    surface_temperature = temperature[insert - 1] + ratio * (
        temperature[insert] - temperature[insert - 1]
    )
    surface_vmr = vmr[insert - 1] + ratio * (vmr[insert] - vmr[insert - 1])

    return insert, surface_pressure, surface_temperature, surface_vmr


def encode_atmosphere(
    insert,
    surface_pressure,
    surface_temperature,
    surface_vmr,
    temperature,
    vmr,
):
    """
    Encode an atmosphere into the 30-feature NN input vector (single sample).
    """

    row = np.empty(30, dtype=np.float32)

    row[0] = insert

    row[1] = np.round(surface_pressure / P_STEP)

    row[2] = np.round(surface_temperature / T_STEP)

    row[3] = np.round(surface_vmr / Q_STEP)

    row[4:17] = np.round(temperature / T_STEP)

    row[17:30] = np.round(vmr / Q_STEP)

    return row


def decode_atmosphere(encoded_row):
    """
    Decode an encoded atmosphere back into physical quantities.

    Parameters
    ----------
    encoded_row : (30,) ndarray
        Encoded atmosphere produced by encode_atmosphere().

    Returns
    -------
    insert_idx : int

    surface_pressure : float
        Surface pressure (Pa)

    surface_temperature : float
        Surface temperature (K)

    surface_vmr : float
        Surface water vapour VMR

    temperature : (13,) ndarray
        Temperature profile (K)

    vmr : (13,) ndarray
        Water vapour VMR profile
    """

    encoded_row = np.asarray(encoded_row)

    insert_idx = int(encoded_row[0])

    surface_pressure = float(encoded_row[1]) * P_STEP

    surface_temperature = float(encoded_row[2]) * T_STEP

    surface_vmr = float(encoded_row[3]) * Q_STEP

    temperature = encoded_row[4:17].astype(np.float32) * T_STEP

    vmr = encoded_row[17:30].astype(np.float32) * Q_STEP

    return (
        insert_idx,
        surface_pressure,
        surface_temperature,
        surface_vmr,
        temperature,
        vmr,
    )

def build_atmosphere(
    temperature,
    specific_humidity,
    geopotential,
    surface_height,
    pressure_levels,
    time_index,
):

    T = temperature[time_index]

    q = specific_humidity[time_index]

    z = geopotential[time_index] / G

    vmr = vmr_from_q(q)

    insert, ps, ts, vs = interpolate_surface(pressure_levels, z, T, vmr, surface_height)
    row = encode_atmosphere(insert, ps, ts, vs, T, vmr)

    return row, insert, ps, ts, vs


# ==========================================================
# VECTORIZED / BATCHED VERSIONS
#
# interpolate_surface() and encode_atmosphere() only do array math on
# tiny (13-element) profiles, so calling them once per sample in a Python
# loop is dominated by Python/NumPy call overhead rather than actual
# compute. The functions below do the exact same math but across an
# entire batch of N samples at once using NumPy broadcasting, which is
# typically 1-2 orders of magnitude faster than looping in Python.
#
# They support two batching modes, both of which show up in this project:
#
#   (a) Fixed surface height, many timesteps
#       -> height/temperature/vmr vary per-row (T, L), surface_height scalar
#       (used when building the training pool / test-year encodings)
#
#   (b) Fixed timestep, many surface heights (e.g. DEM pixels)
#       -> height/temperature/vmr are a single shared profile (L,),
#          surface_height varies per-row (N,)
#       (used when predicting across a whole DEM crop)
#
# Both are handled by broadcasting height/temperature/vmr to (N, L) if
# they're passed in as 1D.
# ==========================================================

def interpolate_surface_batch(pressure_levels, height, temperature, vmr, surface_heights):
    """
    Vectorized version of interpolate_surface() for N samples at once.

    Parameters
    ----------
    pressure_levels : (L,) ndarray
        Shared pressure levels (Pa).
    height, temperature, vmr : (L,) or (N, L) ndarray
        Either a single shared profile (broadcast to all N samples) or
        one profile per sample.
    surface_heights : (N,) ndarray or scalar
        Surface elevation(s) (m).

    Returns
    -------
    insert : (N,) int ndarray
    surface_pressure : (N,) ndarray
    surface_temperature : (N,) ndarray
    surface_vmr : (N,) ndarray
    """
    surface_heights = np.atleast_1d(np.asarray(surface_heights, dtype=np.float64))
    N = surface_heights.shape[0]

    pressure_levels = np.asarray(pressure_levels)
    L = pressure_levels.shape[0]

    height = np.broadcast_to(np.asarray(height), (N, L)) if np.asarray(height).ndim == 1 else np.asarray(height)
    temperature = np.broadcast_to(np.asarray(temperature), (N, L)) if np.asarray(temperature).ndim == 1 else np.asarray(temperature)
    vmr = np.broadcast_to(np.asarray(vmr), (N, L)) if np.asarray(vmr).ndim == 1 else np.asarray(vmr)

    # insert[i] = number of pressure levels whose height is below surface_height[i]
    insert = np.sum(height < surface_heights[:, None], axis=1)
    insert = np.clip(insert, 1, L - 1)

    rows = np.arange(N)
    z0 = height[rows, insert - 1]
    z1 = height[rows, insert]

    denom = z1 - z0
    safe_denom = np.where(denom == 0, 1.0, denom)
    ratio = np.where(denom == 0, 0.0, (surface_heights - z0) / safe_denom)

    logp0 = np.log(pressure_levels[insert - 1])
    logp1 = np.log(pressure_levels[insert])
    surface_pressure = np.exp(logp0 + ratio * (logp1 - logp0))

    t0 = temperature[rows, insert - 1]
    t1 = temperature[rows, insert]
    surface_temperature = t0 + ratio * (t1 - t0)

    v0 = vmr[rows, insert - 1]
    v1 = vmr[rows, insert]
    surface_vmr = v0 + ratio * (v1 - v0)

    return insert, surface_pressure, surface_temperature, surface_vmr


def encode_atmosphere_batch(insert, surface_pressure, surface_temperature, surface_vmr, temperature, vmr):
    """
    Vectorized version of encode_atmosphere() for N samples at once.

    Parameters
    ----------
    insert, surface_pressure, surface_temperature, surface_vmr : (N,) ndarray
    temperature, vmr : (L,) or (N, L) ndarray
        Either a single shared profile or one profile per sample.

    Returns
    -------
    X : (N, 4 + 2*L) float32 ndarray
        Encoded rows (30 columns when L == 13, matching the original format).
    """
    insert = np.asarray(insert)
    N = insert.shape[0]

    temperature = np.asarray(temperature)
    vmr = np.asarray(vmr)
    if temperature.ndim == 1:
        temperature = np.broadcast_to(temperature, (N, temperature.shape[0]))
    if vmr.ndim == 1:
        vmr = np.broadcast_to(vmr, (N, vmr.shape[0]))

    L = temperature.shape[1]
    X = np.empty((N, 4 + 2 * L), dtype=np.float32)

    X[:, 0] = insert
    X[:, 1] = np.round(surface_pressure / P_STEP)
    X[:, 2] = np.round(surface_temperature / T_STEP)
    X[:, 3] = np.round(surface_vmr / Q_STEP)
    X[:, 4:4 + L] = np.round(temperature / T_STEP)
    X[:, 4 + L:4 + 2 * L] = np.round(vmr / Q_STEP)

    return X


def build_atmosphere_batch_over_time(
    temperature,
    specific_humidity,
    geopotential,
    surface_height,
    pressure_levels,
    time_indices=None,
):
    """
    Vectorized replacement for calling build_atmosphere() in a loop over
    many timesteps at a FIXED surface height (e.g. building the training
    pool, or encoding a full year of hours for one site).

    Parameters
    ----------
    temperature, specific_humidity, geopotential : (T, L) ndarray
        Full time series for one grid column, as returned by
        load_pressure_dataset().
    surface_height : float
        Fixed surface elevation (m).
    pressure_levels : (L,) ndarray
    time_indices : (N,) int ndarray or None
        Which timesteps to encode. If None, encodes every timestep.

    Returns
    -------
    X : (N, 30) float32 ndarray
    insert, surface_pressure, surface_temperature, surface_vmr : (N,) ndarray
    """
    if time_indices is not None:
        time_indices = np.asarray(time_indices)
        temperature = temperature[time_indices]
        specific_humidity = specific_humidity[time_indices]
        geopotential = geopotential[time_indices]

    T_arr = np.asarray(temperature)
    q = np.asarray(specific_humidity)
    z = np.asarray(geopotential) / G
    vmr = vmr_from_q(q)

    N = T_arr.shape[0]
    surface_heights = np.full(N, surface_height, dtype=np.float64)

    insert, ps, ts, vs = interpolate_surface_batch(pressure_levels, z, T_arr, vmr, surface_heights)
    X = encode_atmosphere_batch(insert, ps, ts, vs, T_arr, vmr)

    return X, insert, ps, ts, vs


def build_atmosphere_batch_over_heights(
    temperature,
    specific_humidity,
    geopotential,
    surface_heights,
    pressure_levels,
    time_index,
):
    """
    Vectorized replacement for calling build_atmosphere() in a loop over
    many surface heights (e.g. DEM pixels or unique elevations) at a FIXED
    timestep.

    Parameters
    ----------
    temperature, specific_humidity, geopotential : (T, L) ndarray
        Full time series for one grid column.
    surface_heights : (N,) ndarray
        Surface elevations to evaluate (m), e.g. one per DEM pixel or
        one per unique elevation value.
    pressure_levels : (L,) ndarray
    time_index : int
        Which timestep to use (shared across all N rows).

    Returns
    -------
    X : (N, 30) float32 ndarray
    insert, surface_pressure, surface_temperature, surface_vmr : (N,) ndarray
    """
    T_row = np.asarray(temperature[time_index])
    q_row = np.asarray(specific_humidity[time_index])
    z_row = np.asarray(geopotential[time_index]) / G
    vmr_row = vmr_from_q(q_row)

    surface_heights = np.asarray(surface_heights, dtype=np.float64)

    insert, ps, ts, vs = interpolate_surface_batch(pressure_levels, z_row, T_row, vmr_row, surface_heights)
    X = encode_atmosphere_batch(insert, ps, ts, vs, T_row, vmr_row)

    return X, insert, ps, ts, vs


def compute_pwv(
    specific_humidity,
    pressure_levels,
    insert_idx,
    surface_pressure,
    surface_specific_humidity,
    xp=np,
):
    """
    Compute PWV for a single atmospheric column.

    Parameters
    ----------
    specific_humidity : (N,)
        Specific humidity profile.

    pressure_levels : (N,)
        Pressure levels (Pa).

    insert_idx : int
        Surface insertion index.

    surface_pressure : float
        Surface pressure (Pa).

    surface_specific_humidity : float
        Interpolated surface specific humidity.

    Returns
    -------
    pwv : float
        Precipitable Water Vapour (kg/m² = mm)
    """

    # Insert surface pressure.
    pressure = xp.insert(pressure_levels, insert_idx, surface_pressure)

    # Insert surface humidity.
    q = xp.insert(specific_humidity, insert_idx, surface_specific_humidity)

    # Integrate only above the ground.
    pressure = pressure[insert_idx:]
    q = q[insert_idx:]

    dp = xp.abs(xp.diff(pressure))

    q_avg = 0.5 * (q[:-1] + q[1:])

    pwv = xp.sum(q_avg * dp) / G

    return float(pwv)


def compute_pwv_dataset(
    specific_humidity,
    pressure_levels,
    insert_idx,
    surface_pressure,
    interpolation_ratio,
    chunk_size=100,
    xp=cp,
):
    """
    Compute PWV for an entire ERA5 dataset.

    Parameters
    ----------
    specific_humidity : (T,H,W,N)
        Specific humidity profiles.

    pressure_levels : (N,)
        Pressure levels (Pa).

    insert_idx : (T,H,W)
        Surface insertion indices.

    surface_pressure : (T,H,W)
        Surface pressure (Pa).

    interpolation_ratio : (T,H,W)
        Interpolation ratio between the two surrounding levels.

    Returns
    -------
    pwv : (T,H,W)
        PWV (kg/m² = mm)

    surface_specific_humidity : (T,H,W)
    """

    pressure_levels = xp.asarray(pressure_levels, dtype=xp.float32)

    T, H, W = insert_idx.shape

    pwv = np.empty((T, H, W), dtype=np.float32)
    surface_q = np.empty((T, H, W), dtype=np.float32)

    for start in range(0, T, chunk_size):

        end = min(start + chunk_size, T)
        t_len = end - start

        idx = insert_idx[start:end]
        ps = surface_pressure[start:end]
        q = specific_humidity[start:end]
        ratio = interpolation_ratio[start:end]

        ##################################################
        # Surface humidity
        ##################################################

        q0 = xp.take_along_axis(
            q,
            (idx - 1)[..., None],
            axis=-1,
        )[..., 0]

        q1 = xp.take_along_axis(
            q,
            idx[..., None],
            axis=-1,
        )[..., 0]

        qs = q0 + ratio * (q1 - q0)

        surface_q[start:end] = xp.asnumpy(qs)

        ##################################################
        # Insert surface level
        ##################################################

        k = xp.arange(
            pressure_levels.size + 1,
            dtype=idx.dtype,
        ).reshape(1, 1, 1, -1)

        shift = (k > idx[..., None]).astype(idx.dtype)

        src = xp.clip(
            k - shift,
            0,
            pressure_levels.size - 1,
        )

        pressure = pressure_levels[src]

        q_full = xp.take_along_axis(
            q,
            src,
            axis=-1,
        )

        xp.put_along_axis(
            pressure,
            idx[..., None],
            ps[..., None],
            axis=-1,
        )

        xp.put_along_axis(
            q_full,
            idx[..., None],
            qs[..., None],
            axis=-1,
        )

        ##################################################
        # Integrate
        ##################################################

        dp = xp.abs(
            pressure[..., 1:] - pressure[..., :-1]
        )

        q_avg = 0.5 * (
            q_full[..., 1:] + q_full[..., :-1]
        )

        k13 = xp.arange(
            pressure_levels.size,
            dtype=idx.dtype,
        ).reshape(1, 1, 1, -1)

        mask = k13 >= idx[..., None]

        integral = xp.sum(
            xp.where(mask, q_avg * dp, 0.0),
            axis=-1,
        )

        pwv[start:end] = xp.asnumpy(integral / G)

        if xp is cp:
            cp.get_default_memory_pool().free_all_blocks()

        print(f"{end}/{T}", end="\r")

    print()

    return (
        pwv.astype(np.float32),
        surface_q.astype(np.float32),
    )
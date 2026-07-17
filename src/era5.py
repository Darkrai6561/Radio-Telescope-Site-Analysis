import numpy as np
import cupy as cp

# ==========================================================
# CONSTANTS
# ==========================================================

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
    Interpolate the atmospheric state to the surface.

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
    Encode an atmosphere into the 30-feature NN input vector.
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

    # --------------------------------------------------
    # Insert surface pressure
    # --------------------------------------------------

    pressure = xp.insert(pressure_levels, insert_idx, surface_pressure)

    # --------------------------------------------------
    # Insert surface humidity
    # --------------------------------------------------

    q = xp.insert(specific_humidity, insert_idx, surface_specific_humidity)

    # --------------------------------------------------
    # Integrate only above the ground
    # --------------------------------------------------

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
    Compute PWV for an entire ERA5 dataset on the GPU.

    Parameters
    ----------
    specific_humidity : (T, N)
        Specific humidity profile.

    pressure_levels : (N,)
        Pressure levels (Pa).

    insert_idx : (T,H,W)
        Surface insertion indices.

    surface_pressure : (T,H,W)
        Interpolated surface pressure (Pa).

    interpolation_ratio : (T,H,W)
        Surface interpolation ratio.

    chunk_size : int

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

        # --------------------------------------------------
        # Surface humidity
        # --------------------------------------------------

        t_idx3 = xp.arange(t_len).reshape(t_len, 1, 1)

        q0 = q[t_idx3, idx - 1]

        q1 = q[t_idx3, idx]

        qs = q0 + ratio * (q1 - q0)

        surface_q[start:end] = xp.asnumpy(qs)

        # --------------------------------------------------
        # Insert surface level
        # --------------------------------------------------

        k = xp.arange(pressure_levels.size + 1, dtype=idx.dtype).reshape(1, 1, 1, -1)

        shift = (k > idx[..., None]).astype(idx.dtype)

        src = xp.clip(k - shift, 0, pressure_levels.size - 1)

        t_idx4 = xp.arange(t_len).reshape(t_len, 1, 1, 1)

        pressure = pressure_levels[src]

        q_full = q[t_idx4, src]

        xp.put_along_axis(pressure, idx[..., None], ps[..., None], axis=3)
        xp.put_along_axis(q_full, idx[..., None], qs[..., None], axis=3)

        # --------------------------------------------------
        # Integrate
        # --------------------------------------------------

        dp = xp.abs(pressure[..., 1:] - pressure[..., :-1])
        q_avg = 0.5 * (q_full[..., 1:] + q_full[..., :-1])
        k13 = xp.arange(pressure_levels.size, dtype=idx.dtype).reshape(1, 1, 1, -1)

        mask = k13 >= idx[..., None]

        integral = xp.sum(xp.where(mask, q_avg * dp, 0.0), axis=-1)

        pwv[start:end] = xp.asnumpy(integral / G)

        if xp is cp:
            cp.get_default_memory_pool().free_all_blocks()

        print(f"{end}/{T}", end="\r")

    print()

    return pwv.astype(np.float32), surface_q.astype(np.float32)



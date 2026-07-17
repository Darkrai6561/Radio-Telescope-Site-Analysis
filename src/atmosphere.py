import numpy as np
from pathlib import Path


DEFAULT_FREQ_START = 2.0
DEFAULT_FREQ_STOP = 14.0

# Slightly below 40 MHz to avoid unresolved spectral-line warnings in AM.
DEFAULT_FREQ_STEP = 0.039993653
def write_layer(
    pressure,
    temperature,
    vmr,
    ozone=None,
    lwp=0.0,
    iwp=0.0,
):
    """
    Build one AM atmospheric layer.

    Parameters
    ----------
    pressure : float
        Pressure at layer base (Pa)

    temperature : float
        Temperature at layer base (K)

    vmr : float
        Water vapour volume mixing ratio.

    ozone : float or None
        Ozone volume mixing ratio.
        If None, ozone is omitted.

    lwp : float
        Liquid water path (kg/m²) in this layer.

    iwp : float
        Ice water path (kg/m²) in this layer.

    Returns
    -------
    str
        AM layer block.
    """

    lines = []

    lines.append("layer")
    lines.append(f"Pbase {pressure:.2f} Pa")
    lines.append(f"Tbase {temperature:.3f} K")

    # Dry atmosphere
    lines.append("column dry_air hydrostatic")

    # Water vapour
    lines.append(f"column h2o vmr {vmr:.8e}")

    # Ozone
    if ozone is not None:
        lines.append(f"column o3 vmr {ozone:.8e}")

    # Clouds
    # Liquid cloud
    if lwp > 0:
        lines.append(
            f"column lwp_abs_Rayleigh {lwp:.8e} kg*m^-2"
        )

    # Ice cloud
    if iwp > 0:
        lines.append(
            f"column iwp_abs_Rayleigh {iwp:.8e} kg*m^-2"
        )

    lines.append("")

    return "\n".join(lines)

def build_amc(
    pressure_levels,
    temperature,
    vmr,
    insert_idx,
    surface_pressure,
    surface_temperature,
    surface_vmr,
    lwp=0.0,
    iwp=0.0,
    ozone=None,
    title="Atmosphere",
    freq_start=DEFAULT_FREQ_START,
    freq_stop=DEFAULT_FREQ_STOP,
    freq_step=DEFAULT_FREQ_STEP,
    zenith_angle=0.0
):
    """
    Build a complete AM atmosphere (.amc) as a string.

    Parameters
    ----------
    pressure_levels : (N,)
        Pressure levels (Pa)

    temperature : (N,)
        Temperature profile (K)

    vmr : (N,)
        Water vapour VMR profile

    insert_idx : int
        Surface insertion index.

    surface_pressure : float
        Surface pressure (Pa)

    surface_temperature : float
        Surface temperature (K)

    surface_vmr : float
        Surface water vapour VMR

    lwp : float
        Total liquid water path (kg/m²)

    iwp : float
        Total ice water path (kg/m²)

    ozone : ndarray or None
        Ozone VMR profile.
        If None, ozone is omitted.

    Returns
    -------
    str
        Complete AMC file.
    """

    # --------------------------------------------------
    # Insert surface level
    # --------------------------------------------------

    pressure = np.insert(
        pressure_levels,
        insert_idx,
        surface_pressure,
    )

    temperature = np.insert(
        temperature,
        insert_idx,
        surface_temperature,
    )

    vmr = np.insert(
        vmr,
        insert_idx,
        surface_vmr,
    )

    if ozone is not None:
        ozone = np.insert(
            ozone,
            insert_idx,
            ozone[insert_idx - 1],
        )

    # --------------------------------------------------
    # Reverse for AM
    # AM expects top -> surface
    # --------------------------------------------------

    pressure = pressure[insert_idx:][::-1]
    temperature = temperature[insert_idx:][::-1]
    vmr = vmr[insert_idx:][::-1]

    if ozone is not None:
        ozone = ozone[insert_idx:][::-1]

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    lines = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        f"f {freq_start:g} GHz {freq_stop:g} GHz {freq_step} GHz"
    )
    lines.append("output f GHz tx Tb K")
    lines.append("T0 2.7 K")
    lines.append(f"za {zenith_angle:g} deg")
    lines.append("")

    # --------------------------------------------------
    # Layers
    # --------------------------------------------------

    n_layers = len(pressure)

    for i in range(n_layers):

        layer_lwp = 0.0
        layer_iwp = 0.0

        # Put cloud in lowest layer by default.
        # Can be changed later if desired.
        if i == n_layers - 1:
            layer_lwp = lwp
            layer_iwp = iwp

        layer_o3 = None

        if ozone is not None:
            layer_o3 = ozone[i]

        lines.append(
            write_layer(
                pressure=pressure[i],
                temperature=temperature[i],
                vmr=vmr[i],
                ozone=layer_o3,
                lwp=layer_lwp,
                iwp=layer_iwp,
            )
        )

    return "\n".join(lines)

def save_amc(path, amc_text):
    """
    Save an AM atmosphere (.amc) file.

    Parameters
    ----------
    path : str or Path
        Output .amc filename.

    amc_text : str
        Atmosphere text returned by build_amc().
    """

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(amc_text)

    return path



def load_amc(path):
    """
    Load an AM atmosphere (.amc) file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    str
        Contents of the AMC file.
    """

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()
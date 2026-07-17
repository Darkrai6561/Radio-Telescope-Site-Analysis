import subprocess
from pathlib import Path
import numpy as np


def parse_am_output(stdout):
    """
    Parse the output produced by the Atmospheric Model (AM).

    Parameters
    ----------
    stdout : str
        Raw stdout returned by am.exe.

    Returns
    -------
    frequency : ndarray
        Frequency (GHz)

    transmittance : ndarray
        Atmospheric transmittance

    brightness_temperature : ndarray
        Brightness temperature (K)
    """

    frequency = []
    transmittance = []
    brightness_temperature = []

    for line in stdout.splitlines():

        line = line.strip()

        # Skip blank lines
        if not line:
            continue

        # Skip comments and warnings
        if line.startswith("#"):
            continue

        if line.startswith("!"):
            continue

        parts = line.split()

        if len(parts) < 3:
            continue

        try:
            frequency.append(float(parts[0]))
            transmittance.append(float(parts[1]))
            brightness_temperature.append(float(parts[2]))
        except ValueError:
            continue

    frequency = np.asarray(frequency, dtype=np.float32)
    transmittance = np.asarray(transmittance, dtype=np.float32)
    brightness_temperature = np.asarray(brightness_temperature, dtype=np.float32)

    if frequency.size == 0:
        raise RuntimeError("No spectrum found in AM output.")

    if not (len(frequency) == len(transmittance) == len(brightness_temperature)):
        raise RuntimeError("AM output columns have inconsistent lengths.")

    return frequency, transmittance, brightness_temperature


def run_am(am_executable, amc_file):
    """
    Run the Atmospheric Model (AM).

    Parameters
    ----------
    am_executable : str or Path
        Path to am.exe.

    amc_file : str or Path
        Path to the input .amc file.

    Returns
    -------
    frequency : ndarray
        Frequency (GHz)

    transmittance : ndarray
        Atmospheric transmittance

    brightness_temperature : ndarray
        Brightness temperature (K)
    """

    am_executable = Path(am_executable)
    amc_file = Path(amc_file)

    result = subprocess.run(
        [str(am_executable), str(amc_file)],
        capture_output=True,
        text=True,
        cwd=am_executable.parent,
    )

    # --------------------------------------------------
    # Try parsing the spectrum first.
    # If this succeeds, AM produced valid output.
    # --------------------------------------------------

    try:
        frequency, transmittance, brightness_temperature = parse_am_output(
            result.stdout
        )

    except RuntimeError:

        raise RuntimeError(
            f"""
AM execution failed.

Executable:
{am_executable}

AMC file:
{amc_file}

Return code:
{result.returncode}

==================== STDOUT ====================

{result.stdout}

==================== STDERR ====================

{result.stderr}
"""
        )

    # --------------------------------------------------
    # Spectrum exists.
    # Non-zero return code therefore means warning(s),
    # not a fatal error.
    # --------------------------------------------------

    if result.returncode != 0:

        print("=" * 70)
        print("AM completed with warnings")
        print("=" * 70)

        if result.stdout.strip():
            print(result.stdout)

        if result.stderr.strip():
            print(result.stderr)

        print("=" * 70)

    return frequency, transmittance, brightness_temperature


def save_spectrum(
    path,
    frequency,
    transmittance,
    brightness_temperature,
    stdout=None,
):
    """
    Save an AM spectrum.

    Parameters
    ----------
    path : str or Path
        Output .npz filename.

    frequency : ndarray
        Frequency (GHz)

    transmittance : ndarray
        Atmospheric transmittance

    brightness_temperature : ndarray
        Brightness temperature (K)

    stdout : str or None
        Raw AM stdout for debugging.

    Returns
    -------
    Path
        Saved file.
    """

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "frequency": np.asarray(frequency, dtype=np.float32),
        "transmittance": np.asarray(transmittance, dtype=np.float32),
        "brightness_temperature": np.asarray(brightness_temperature, dtype=np.float32),
    }

    if stdout is not None:
        save_dict["stdout"] = np.array(stdout)

    np.savez_compressed(path, **save_dict)

    return path


def load_spectrum(path):
    """
    Load a saved AM spectrum.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    frequency : ndarray

    transmittance : ndarray

    brightness_temperature : ndarray

    stdout : str or None
        Raw AM stdout if it was saved.
    """

    data = np.load(Path(path), allow_pickle=True)

    stdout = None

    if "stdout" in data.files:
        stdout = str(data["stdout"])

    return (
        data["frequency"],
        data["transmittance"],
        data["brightness_temperature"],
        stdout,
    )

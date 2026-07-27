import numpy as np
from atmosphere import build_amc, save_amc
from am import run_am, save_spectrum, load_spectrum
from era5 import decode_atmosphere
from pathlib import Path
import time


def collect_training_samples(
    encoded_atmospheres,
    num_samples=6000,
    method="unique_first",
    random_state=42,
):
    """
    Collect training atmospheres for AM.

    Parameters
    ----------
    encoded_atmospheres : (N,F) ndarray
        Encoded atmosphere dataset.

    num_samples : int
        Maximum number of atmospheres to collect.

    method : str

        "random"
            Random atmospheres.

        "unique_first"
            First unique atmospheres encountered.

        "unique_random"
            Shuffle dataset, then collect unique.

        "stratified_time"
            Uniformly sample throughout the dataset,
            while keeping only unique atmospheres.

    random_state : int

    Returns
    -------
    training_samples : ndarray

    original_indices : ndarray
    """

    rng = np.random.default_rng(random_state)

    N = len(encoded_atmospheres)

    # Random subset.
    if method == "random":
        n = min(num_samples, N)
        indices = rng.choice(N, size=n, replace=False)
        return encoded_atmospheres[indices].copy(), indices.astype(np.int32)

    # First unique rows.
    elif method == "unique_first":
        order = np.arange(N)

    # Unique rows after shuffling.
    elif method == "unique_random":
        order = rng.permutation(N)

    # Uniform time coverage with unique rows.
    elif method == "stratified_time":
        # Visit the dataset uniformly.
        step = max(N // num_samples, 1)
        order = np.arange(0, N, step)

        # Backfill skipped indices.
        remaining = np.setdiff1d(np.arange(N), order, assume_unique=True)
        rng.shuffle(remaining)
        order = np.concatenate((order, remaining))
    else:
        raise ValueError(f"Unknown method '{method}'.")
    # Collect unique rows in order.
    seen = set()
    training = []
    original_indices = []

    for idx in order:
        row = encoded_atmospheres[idx]
        key = row.tobytes()
        if key in seen:
            continue
        seen.add(key)
        training.append(row.copy())
        original_indices.append(idx)
        if len(training) >= num_samples:
            break

    training = np.asarray(training, dtype=encoded_atmospheres.dtype)
    original_indices = np.asarray(original_indices, dtype=np.int32)
    return training, original_indices

def build_training_dataset(
    encoded_atmospheres,
    pressure_levels,
    am_executable,
    output_dir,
    zenith_angle=0.0,
    ozone=None,
    lwp=0.0,
    iwp=0.0,
    keep_amc=False,
    overwrite=False,
):
    """
    Generate an AM training dataset from encoded atmospheres.

    Parameters
    ----------
    encoded_atmospheres : (N,30) ndarray
        Encoded atmosphere vectors.

    pressure_levels : (13,) ndarray
        Pressure levels (Pa).

    am_executable : str or Path
        Path to AM executable.

    output_dir : str or Path
        Directory used for temporary AMC files and cached spectra.

    zenith_angle : float
        Zenith angle (degrees).

    ozone : ndarray or None
        Ozone VMR profile.

    lwp : float
        Liquid water path (kg/m²).

    iwp : float
        Ice water path (kg/m²).

    keep_amc : bool
        Keep generated AMC files.

    overwrite : bool
        Recompute existing spectra.

    Returns
    -------
    X : ndarray
        Encoded atmospheres.

    frequency : ndarray

    transmittance : ndarray

    brightness_temperature : ndarray
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(encoded_atmospheres, dtype=np.float32)

    frequency = None

    transmittance = []
    brightness_temperature = []

    total = len(X)

    start = time.time()

    for i, encoded in enumerate(X):
        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
        eta = (total - i - 1) / rate if rate > 0 else 0.0
        print(
            f"\r[{i+1:5d}/{total}] "
            f"{rate:6.2f} atm/s "
            f"ETA {eta/60:6.1f} min",
            end="",
            flush=True,
        )

        spectrum_file = output_dir / f"spectrum_{i:05d}.npz"
        

        # Reuse cached spectra unless overwriting.
        if spectrum_file.exists() and not overwrite:
            freq, tx, tb, _ = load_spectrum(spectrum_file)
        else:
            (
                insert_idx,
                surface_pressure,
                surface_temperature,
                surface_vmr,
                temperature,
                vmr,
            ) = decode_atmosphere(encoded)

            amc = build_amc(
                pressure_levels=pressure_levels,
                temperature=temperature,
                vmr=vmr,
                insert_idx=insert_idx,
                surface_pressure=surface_pressure,
                surface_temperature=surface_temperature,
                surface_vmr=surface_vmr,
                lwp=lwp,
                iwp=iwp,
                ozone=ozone,
                zenith_angle=zenith_angle,
            )

            

            try:
                freq, tx, tb = run_am(am_executable,amc)
            except Exception as e:
                print(f"\nFailed atmosphere {i}")
                raise RuntimeError(f"AM failed on sample {i}") from e

            save_spectrum(spectrum_file, freq, tx, tb)

            if keep_amc:
                amc_file = output_dir / f"sample_{i:05d}.amc"
                save_amc(amc_file, amc)

        # Keep one shared frequency grid.
        if frequency is None:
            frequency = freq
        elif (frequency.shape != freq.shape or not np.allclose(frequency, freq)):
            raise RuntimeError(f"Frequency grid mismatch at sample {i}.")
        transmittance.append(tx)
        brightness_temperature.append(tb)

    print()

    return (
        X,
        frequency.astype(np.float32),
        np.asarray(transmittance, dtype=np.float32),
        np.asarray(brightness_temperature, dtype=np.float32),
    )


def save_training_dataset(
    path,
    X,
    frequency,
    transmittance,
    brightness_temperature,
):
    """
    Save a complete neural-network training dataset.

    Parameters
    ----------
    path : str or Path

    X : (N,30) ndarray
        Encoded atmospheres.

    frequency : (F,) ndarray

    transmittance : (N,F) ndarray

    brightness_temperature : (N,F) ndarray

    Returns
    -------
    Path
        Saved dataset path.
    """

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        frequency=np.asarray(frequency, dtype=np.float32),
        transmittance=np.asarray(transmittance, dtype=np.float32),
        brightness_temperature=np.asarray(brightness_temperature, dtype=np.float32),
    )

    return path


def load_training_dataset(path):
    """
    Load a neural-network training dataset.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    X : ndarray

    frequency : ndarray

    transmittance : ndarray

    brightness_temperature : ndarray
    """

    data = np.load(Path(path), allow_pickle=False)

    return (
        data["X"],
        data["frequency"],
        data["transmittance"],
        data["brightness_temperature"],
    )

def train_test_split(
    X,
    transmittance,
    brightness_temperature,
    test_size=0.2,
    shuffle=True,
    random_state=42,
):
    """
    Split the training dataset into train/test sets.

    Parameters
    ----------
    X : ndarray

    transmittance : ndarray

    brightness_temperature : ndarray

    test_size : float
        Fraction used for testing.

    shuffle : bool

    random_state : int

    Returns
    -------
    X_train
    X_test

    tx_train
    tx_test

    tb_train
    tb_test
    """

    n = len(X)

    if len(transmittance) != n or len(brightness_temperature) != n:
        raise ValueError("All arrays must have the same length.")

    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    split = int((1 - test_size) * n)
    train_idx = indices[:split]
    test_idx = indices[split:]

    return (
        X[train_idx],
        X[test_idx],
        transmittance[train_idx],
        transmittance[test_idx],
        brightness_temperature[train_idx],
        brightness_temperature[test_idx],
    )


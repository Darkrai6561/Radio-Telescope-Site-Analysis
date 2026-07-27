from torch.utils.data import TensorDataset, DataLoader
from dataset import load_training_dataset,train_test_split  
import torch
import torch.nn as nn
import copy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def prepare_dataloaders(
    dataset_path,
    batch_size=64,
    test_size=0.2,
    shuffle=True,
    random_state=42,
):
    """
    Load a training dataset and create PyTorch DataLoaders.

    Parameters
    ----------
    dataset_path : str or Path

    batch_size : int

    test_size : float

    shuffle : bool

    random_state : int

    Returns
    -------
    train_loader : DataLoader

    test_loader : DataLoader

    frequency : ndarray
        Frequency axis (GHz).
    """

    (
        X,
        frequency,
        transmittance,
        brightness_temperature,
    ) = load_training_dataset(dataset_path)

    (
        X_train,
        X_test,
        tx_train,
        tx_test,
        tb_train,
        tb_test,
    ) = train_test_split(
        X,
        transmittance,
        brightness_temperature,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(tx_train).float(),
        torch.from_numpy(tb_train).float(),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(tx_test).float(),
        torch.from_numpy(tb_test).float(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        test_loader,
        frequency,
    )

class AtmosphericSpectrumNet(nn.Module):
    """
    Predict atmospheric transmittance and
    brightness temperature spectra from an
    encoded atmosphere (single shared backbone,
    two output heads, trained jointly).
    """

    def __init__(
        self,
        input_size=30,
        hidden_size=256,
        num_hidden_layers=3,
        spectrum_size=301,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.spectrum_size = spectrum_size

        layers = []

        in_features = input_size

        for _ in range(num_hidden_layers):

            layers.extend([
                nn.Linear(
                    in_features,
                    hidden_size,
                ),
                nn.ReLU(),
            ])

            in_features = hidden_size

        self.backbone = nn.Sequential(*layers)

        self.tx_head = nn.Linear(
            hidden_size,
            spectrum_size,
        )

        self.tb_head = nn.Linear(
            hidden_size,
            spectrum_size,
        )

    def forward(self, x):

        features = self.backbone(x)

        tx = self.tx_head(features)

        tb = self.tb_head(features)

        return tx, tb


class SpectrumNet(nn.Module):
    """
    Predict a single atmospheric spectrum (either
    transmittance OR brightness temperature) from an
    encoded atmosphere. Used when training separate,
    independent models for each target instead of a
    single joint model.
    """

    def __init__(
        self,
        input_size=30,
        hidden_size=256,
        num_hidden_layers=3,
        spectrum_size=301,
        head_hidden_size=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.spectrum_size = spectrum_size
        self.head_hidden_size = head_hidden_size

        layers = []

        in_features = input_size

        for _ in range(num_hidden_layers):

            layers.extend([
                nn.Linear(
                    in_features,
                    hidden_size,
                ),
                nn.ReLU(),
            ])

            in_features = hidden_size

        self.backbone = nn.Sequential(*layers)

        if head_hidden_size is None:
            # Single linear output layer.
            self.head = nn.Linear(
                hidden_size,
                spectrum_size,
            )
        else:
            # Two-layer head: an extra hidden layer before the
            # spectrum output. This gives the model more capacity
            # right before prediction, which helps narrow-range
            # targets like transmittance.
            self.head = nn.Sequential(
                nn.Linear(hidden_size, head_hidden_size),
                nn.ReLU(),
                nn.Linear(head_hidden_size, spectrum_size),
            )

    def forward(self, x):

        features = self.backbone(x)

        out = self.head(features)

        return out


def train_model(
    model,
    train_loader,
    test_loader,
    epochs=100,
    learning_rate=1e-3,
    device=None,
    return_history=False,
    tx_weight=1.0,
    tb_weight=1.0,
):
    """
    Train an AtmosphericSpectrumNet (joint tx + tb model).

    Parameters
    ----------
    model : nn.Module

    train_loader : DataLoader

    test_loader : DataLoader

    epochs : int

    learning_rate : float

    device : torch.device or None

    return_history : bool
        If True, also return the training history.

    Returns
    -------
    model

    history : dict (optional)
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.MSELoss()

    best_state = None
    best_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(epochs):

        ###################################################
        # Training
        ###################################################

        model.train()

        running_loss = 0.0

        for X, tx_true, tb_true in train_loader:

            X = X.to(device)
            tx_true = tx_true.to(device)
            tb_true = tb_true.to(device)

            optimizer.zero_grad()

            tx_pred, tb_pred = model(X)

            loss_tx = criterion(tx_pred, tx_true)
            loss_tb = criterion(tb_pred, tb_true)

            loss = tx_weight*loss_tx + tb_weight*loss_tb

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        ###################################################
        # Validation
        ###################################################

        model.eval()

        running_loss = 0.0

        with torch.no_grad():

            for X, tx_true, tb_true in test_loader:

                X = X.to(device)
                tx_true = tx_true.to(device)
                tb_true = tb_true.to(device)

                tx_pred, tb_pred = model(X)

                loss_tx = criterion(tx_pred, tx_true)
                loss_tb = criterion(tb_pred, tb_true)

                running_loss += (loss_tx + loss_tb).item()

        val_loss = running_loss / len(test_loader)

        ###################################################
        # Save best model
        ###################################################

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f}"
        )

    ###################################################
    # Restore best model
    ###################################################

    if best_state is not None:
        model.load_state_dict(best_state)

    if return_history:
        return model, history

    return model


def _train_single_target(
    model,
    train_loader,
    test_loader,
    target,
    epochs=100,
    learning_rate=1e-3,
    device=None,
    return_history=False,
):
    """
    Shared training loop for a single-output SpectrumNet
    (used internally by train_tx_model and train_tb_model).

    Parameters
    ----------
    model : nn.Module
        A SpectrumNet instance (single output head).

    train_loader : DataLoader
        Yields (X, tx, tb) batches, same loaders produced
        by prepare_dataloaders(). Only the relevant target
        is used.

    test_loader : DataLoader

    target : str
        Either "tx" (transmittance) or "tb"
        (brightness temperature). Selects which element
        of the (X, tx, tb) batch is used as the label.

    epochs : int

    learning_rate : float

    device : torch.device or None

    return_history : bool

    Returns
    -------
    model

    history : dict (optional)
    """

    if target not in ("tx", "tb"):
        raise ValueError('target must be "tx" or "tb"')

    target_idx = 1 if target == "tx" else 2

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.MSELoss()

    best_state = None
    best_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(epochs):

        ###################################################
        # Training
        ###################################################

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            X = batch[0].to(device)
            y_true = batch[target_idx].to(device)

            optimizer.zero_grad()

            y_pred = model(X)

            loss = criterion(y_pred, y_true)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        ###################################################
        # Validation
        ###################################################

        model.eval()

        running_loss = 0.0

        with torch.no_grad():

            for batch in test_loader:

                X = batch[0].to(device)
                y_true = batch[target_idx].to(device)

                y_pred = model(X)

                loss = criterion(y_pred, y_true)

                running_loss += loss.item()

        val_loss = running_loss / len(test_loader)

        ###################################################
        # Save best model
        ###################################################

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        label = "Tx" if target == "tx" else "Tb"

        print(
            f"[{label}] Epoch {epoch+1:3d}/{epochs} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f}"
        )

    ###################################################
    # Restore best model
    ###################################################

    if best_state is not None:
        model.load_state_dict(best_state)

    if return_history:
        return model, history

    return model


def train_tx_model(
    model,
    train_loader,
    test_loader,
    epochs=100,
    learning_rate=1e-3,
    device=None,
    return_history=False,
):
    """
    Train a SpectrumNet to predict transmittance only,
    independently of the brightness temperature model.

    Parameters
    ----------
    model : nn.Module
        A SpectrumNet instance.

    train_loader : DataLoader
        Loader yielding (X, tx, tb) batches (e.g. from
        prepare_dataloaders()); only tx is used.

    test_loader : DataLoader

    epochs : int

    learning_rate : float

    device : torch.device or None

    return_history : bool

    Returns
    -------
    model

    history : dict (optional)
    """

    return _train_single_target(
        model,
        train_loader,
        test_loader,
        target="tx",
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        return_history=return_history,
    )


def train_tb_model(
    model,
    train_loader,
    test_loader,
    epochs=100,
    learning_rate=1e-3,
    device=None,
    return_history=False,
):
    """
    Train a SpectrumNet to predict brightness temperature
    only, independently of the transmittance model.

    Parameters
    ----------
    model : nn.Module
        A SpectrumNet instance.

    train_loader : DataLoader
        Loader yielding (X, tx, tb) batches (e.g. from
        prepare_dataloaders()); only tb is used.

    test_loader : DataLoader

    epochs : int

    learning_rate : float

    device : torch.device or None

    return_history : bool

    Returns
    -------
    model

    history : dict (optional)
    """

    return _train_single_target(
        model,
        train_loader,
        test_loader,
        target="tb",
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        return_history=return_history,
    )


def save_model(model, path):
    """
    Save a trained model. Works for both the joint
    AtmosphericSpectrumNet and the single-target
    SpectrumNet (used by train_tx_model / train_tb_model).

    Parameters
    ----------
    model : nn.Module

    path : str or Path
    """

    path = Path(path)

    model_type = type(model).__name__

    checkpoint = {
        "model_type": model_type,
        "state_dict": model.state_dict(),
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "num_hidden_layers": model.num_hidden_layers,
        "spectrum_size": model.spectrum_size,
        "head_hidden_size": getattr(model, "head_hidden_size", None),
    }

    torch.save(checkpoint, path)

    print(f"Model saved to {path}")


def load_model(path, device=None):
    """
    Load a saved AtmosphericSpectrumNet or SpectrumNet.
    The correct class is inferred from the checkpoint's
    "model_type" field (checkpoints saved before this
    field existed are assumed to be AtmosphericSpectrumNet).

    Parameters
    ----------
    path : str or Path

    device : torch.device or None

    Returns
    -------
    AtmosphericSpectrumNet or SpectrumNet
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    checkpoint = torch.load(
        Path(path),
        map_location=device,
    )

    model_type = checkpoint.get("model_type", "AtmosphericSpectrumNet")

    if model_type == "SpectrumNet":
        model = SpectrumNet(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_hidden_layers=checkpoint["num_hidden_layers"],
            spectrum_size=checkpoint["spectrum_size"],
            head_hidden_size=checkpoint.get("head_hidden_size", None),
        )
    else:
        model = AtmosphericSpectrumNet(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_hidden_layers=checkpoint["num_hidden_layers"],
            spectrum_size=checkpoint["spectrum_size"],
        )

    model.load_state_dict(checkpoint["state_dict"])

    model.to(device)

    model.eval()

    return model


def evaluate_model(model,data_loader,device=None,):
    """
    Evaluate a trained joint AtmosphericSpectrumNet.

    Parameters
    ----------
    model : nn.Module

    data_loader : DataLoader

    device : torch.device or None

    Returns
    -------
    tx_true : ndarray

    tx_pred : ndarray

    tb_true : ndarray

    tb_pred : ndarray
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)
    model.eval()

    tx_true = []
    tx_pred = []

    tb_true = []
    tb_pred = []

    with torch.no_grad():

        for X, tx, tb in data_loader:

            X = X.to(device)

            pred_tx, pred_tb = model(X)

            tx_true.append(tx.cpu().numpy())
            tx_pred.append(pred_tx.cpu().numpy())

            tb_true.append(tb.cpu().numpy())
            tb_pred.append(pred_tb.cpu().numpy())

    tx_true = np.concatenate(tx_true, axis=0)
    tx_pred = np.concatenate(tx_pred, axis=0)

    tb_true = np.concatenate(tb_true, axis=0)
    tb_pred = np.concatenate(tb_pred, axis=0)

    return tx_true, tx_pred, tb_true, tb_pred


def evaluate_single_model(model, data_loader, target, device=None):
    """
    Evaluate a trained single-target SpectrumNet
    (as produced by train_tx_model / train_tb_model).

    Parameters
    ----------
    model : nn.Module
        A SpectrumNet instance.

    data_loader : DataLoader
        Loader yielding (X, tx, tb) batches; only the
        relevant target is used.

    target : str
        Either "tx" or "tb". Selects which element of the
        (X, tx, tb) batch is treated as ground truth.

    device : torch.device or None

    Returns
    -------
    y_true : ndarray

    y_pred : ndarray
    """

    if target not in ("tx", "tb"):
        raise ValueError('target must be "tx" or "tb"')

    target_idx = 1 if target == "tx" else 2

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch in data_loader:

            X = batch[0].to(device)
            y = batch[target_idx]

            pred = model(X)

            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    return y_true, y_pred


def compute_metrics(tx_true, tx_pred, tb_true, tb_pred):
    """
    Compute regression metrics for atmospheric spectra.

    Parameters
    ----------
    tx_true : ndarray

    tx_pred : ndarray

    tb_true : ndarray

    tb_pred : ndarray

    Returns
    -------
    metrics : dict
    """

    metrics = {}

    ###################################################
    # Transmittance
    ###################################################

    tx_error = tx_pred - tx_true

    metrics["tx_mse"] = np.mean(tx_error**2)
    metrics["tx_rmse"] = np.sqrt(metrics["tx_mse"])
    metrics["tx_mae"] = np.mean(np.abs(tx_error))
    metrics["tx_max_error"] = np.max(np.abs(tx_error))
    ss_res = np.sum((tx_true - tx_pred) ** 2)
    ss_tot = np.sum((tx_true - np.mean(tx_true)) ** 2)
    metrics["tx_r2"] = 1.0 - ss_res / ss_tot

    ###################################################
    # Brightness temperature
    ###################################################

    tb_error = tb_pred - tb_true

    metrics["tb_mse"] = np.mean(tb_error**2)
    metrics["tb_rmse"] = np.sqrt(metrics["tb_mse"])
    metrics["tb_mae"] = np.mean(np.abs(tb_error))
    metrics["tb_max_error"] = np.max(np.abs(tb_error))
    ss_res = np.sum((tb_true - tb_pred) ** 2)
    ss_tot = np.sum((tb_true - np.mean(tb_true)) ** 2)
    metrics["tb_r2"] = 1.0 - ss_res / ss_tot
    metrics["tx_rmse_per_frequency"] = np.sqrt(
        np.mean(tx_error**2, axis=0)
    )

    metrics["tb_rmse_per_frequency"] = np.sqrt(
        np.mean(tb_error**2, axis=0)
    )

    return metrics


def compute_single_metrics(y_true, y_pred, prefix="y"):
    """
    Compute regression metrics for a single spectrum
    (transmittance OR brightness temperature), for use
    with the single-target SpectrumNet models.

    Parameters
    ----------
    y_true : ndarray

    y_pred : ndarray

    prefix : str
        Key prefix for the returned metrics dict
        (e.g. "tx" or "tb").

    Returns
    -------
    metrics : dict
    """

    metrics = {}

    error = y_pred - y_true

    metrics[f"{prefix}_mse"] = np.mean(error**2)
    metrics[f"{prefix}_rmse"] = np.sqrt(metrics[f"{prefix}_mse"])
    metrics[f"{prefix}_mae"] = np.mean(np.abs(error))
    metrics[f"{prefix}_max_error"] = np.max(np.abs(error))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics[f"{prefix}_r2"] = 1.0 - ss_res / ss_tot
    metrics[f"{prefix}_rmse_per_frequency"] = np.sqrt(
        np.mean(error**2, axis=0)
    )

    return metrics


def predict(model, encoded_atmosphere, device=None):
    """
    Predict atmospheric spectra for a single encoded
    atmosphere using the joint AtmosphericSpectrumNet.

    Parameters
    ----------
    model : nn.Module

    encoded_atmosphere : ndarray, shape (30,)
        Encoded atmosphere produced by encode_atmosphere().

    device : torch.device or None

    Returns
    -------
    transmittance : ndarray

    brightness_temperature : ndarray
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)
    model.eval()

    encoded_atmosphere = np.asarray(
        encoded_atmosphere,
        dtype=np.float32,
    )

    if encoded_atmosphere.ndim != 1:
        raise ValueError(
            "encoded_atmosphere must be a 1D array."
        )

    x = torch.from_numpy(
        encoded_atmosphere
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        tx, tb = model(x)

    tx = tx.squeeze(0).cpu().numpy()
    tb = tb.squeeze(0).cpu().numpy()

    return tx, tb


def predict_single(model, encoded_atmosphere, device=None):
    """
    Predict a single spectrum (transmittance OR brightness
    temperature) for one encoded atmosphere, using a
    single-target SpectrumNet.

    Parameters
    ----------
    model : nn.Module
        A SpectrumNet instance.

    encoded_atmosphere : ndarray, shape (30,)
        Encoded atmosphere produced by encode_atmosphere().

    device : torch.device or None

    Returns
    -------
    spectrum : ndarray
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)
    model.eval()

    encoded_atmosphere = np.asarray(
        encoded_atmosphere,
        dtype=np.float32,
    )

    if encoded_atmosphere.ndim != 1:
        raise ValueError(
            "encoded_atmosphere must be a 1D array."
        )

    x = torch.from_numpy(
        encoded_atmosphere
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        y = model(x)

    y = y.squeeze(0).cpu().numpy()

    return y


def plot_training_history(history, title="Training History"):
    """
    Plot training and validation loss.

    Parameters
    ----------
    history : dict
        Dictionary returned by train_model(), train_tx_model(),
        or train_tb_model().

    title : str
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train_loss"],
        label="Training",
        linewidth=2,
    )

    plt.plot(
        history["val_loss"],
        label="Validation",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_prediction(frequency, tx_true, tx_pred, tb_true, tb_pred, sample=0):
    """
    Plot one predicted spectrum.

    Parameters
    ----------
    frequency : ndarray

    tx_true
    tx_pred
    tb_true
    tb_pred

    sample : int
    """

    ########################################
    # Transmittance
    ########################################

    plt.figure(figsize=(10,4))

    plt.plot(frequency, tx_true[sample], label="True", linewidth=2)

    plt.plot(frequency, tx_pred[sample], "--", label="Prediction", linewidth=2)

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmittance")
    plt.title(f"Transmittance (Sample {sample})")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    ########################################
    # Brightness Temperature
    ########################################

    plt.figure(figsize=(10,4))

    plt.plot(frequency, tb_true[sample], label="True", linewidth=2)

    plt.plot(frequency, tb_pred[sample], "--", label="Prediction", linewidth=2)

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Brightness Temperature (K)")
    plt.title(f"Brightness Temperature (Sample {sample})")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_error_spectrum(frequency, tx_true, tx_pred, tb_true, tb_pred):
    """
    Plot RMSE as a function of frequency.
    """

    tx_rmse = np.sqrt(np.mean((tx_pred - tx_true) ** 2, axis=0,))
    tb_rmse = np.sqrt(np.mean((tb_pred - tb_true) ** 2, axis=0,))

    ########################################

    plt.figure(figsize=(10,4))
    plt.plot(frequency, tx_rmse, linewidth=2,)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("RMSE")
    plt.title("Transmittance RMSE")
    plt.grid(True)
    plt.tight_layout()

    ########################################

    plt.figure(figsize=(10,4))
    plt.plot(frequency, tb_rmse, linewidth=2)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("RMSE (K)")
    plt.title("Brightness Temperature RMSE")
    plt.grid(True)
    plt.tight_layout()

    plt.show()
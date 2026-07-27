from era5 import encode_atmosphere
from train import predict


def infer(
    model,
    insert,
    surface_pressure,
    surface_temperature,
    surface_vmr,
    temperature,
    vmr,
):
    """
    Predict atmospheric spectra using a preloaded model.
    """

    encoded = encode_atmosphere(
        insert,
        surface_pressure,
        surface_temperature,
        surface_vmr,
        temperature,
        vmr,
    )

    return predict(model, encoded)
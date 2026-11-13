"""Select an explicit native model backend; never replace missing assets."""
from .ouro import load_ouro, tiny_ouro


def load_backend(model_config, directory, device, mixed_precision="bf16",
                 gradient_checkpointing=True, smoke=False):
    family = model_config["family"]
    if family == "ouro":
        load, tiny = load_ouro, tiny_ouro
    elif family == "huginn":
        from .huginn import load_huginn, tiny_huginn
        load, tiny = load_huginn, tiny_huginn
    else:
        raise ValueError(f"Unsupported model family: {family}")
    if smoke:
        return tiny(directory, device), {"scope": "random tiny official architecture; no pretrained weights loaded"}
    return load(directory, device, mixed_precision, gradient_checkpointing)

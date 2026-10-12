from pathlib import Path

import torch


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    weights = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    torch.save(weights, path)
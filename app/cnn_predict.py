from functools import lru_cache

import torch
from PIL import Image

from app.cnn_data import CLASSES, PROJECT_DIR, image_transform
from app.cnn_model import AssignmentCNN


@lru_cache(maxsize=1)
def load_model():
    model = AssignmentCNN()
    model_path = PROJECT_DIR / "models" / "cnn.pt"

    weights = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(weights)
    model.eval()
    return model


def predict_image(image: Image.Image):
    model = load_model()

    image = image.convert("RGB")
    inputs = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs)
        class_id = outputs.argmax(dim=1).item()

    return {
        "class_id": class_id,
        "class_name": CLASSES[class_id],
    }
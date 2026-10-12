from app.cnn_model import AssignmentCNN


def get_model(model_name="CNN"):
    if model_name == "CNN":
        return AssignmentCNN()

    raise ValueError(f"Unsupported model name: {model_name}")
import json

import torch
from torch import nn

from app.cnn_data import CLASSES, PROJECT_DIR
from helper_lib import (
    evaluate_model,
    get_model,
    load_data,
    train_model,
)


def main():
    torch.manual_seed(42)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}", flush=True)

    batch_size = 64
    learning_rate = 0.0005
    epochs = 10

    train_loader, test_loader = load_data(batch_size=batch_size)
    model = get_model("CNN").to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    output_dir = PROJECT_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "cnn.pt"
    history = []

    model = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        history=history,
        checkpoint_path=model_path,
    )

    test_loss, test_accuracy = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    results = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": 42,
        "device": str(device),
        "classes": CLASSES,
        "history": history,
        "test_images": len(test_loader.dataset),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }

    results_path = output_dir / "training_results.json"
    results_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"Test accuracy: {test_accuracy:.2%}", flush=True)
    print(f"Model saved to: {model_path}", flush=True)
    print(f"Results saved to: {results_path}", flush=True)


if __name__ == "__main__":
    main()
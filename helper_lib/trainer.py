from .utils import save_model


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs=10,
    device="cpu",
    history=None,
    checkpoint_path=None,
):
    # Move the model to the device before creating the optimizer.
    if history is None:
        history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

            if step % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"batch {step}/{len(train_loader)}",
                    flush=True,
                )

        if total == 0:
            raise ValueError("The training data loader is empty.")

        train_loss = total_loss / total
        train_accuracy = correct / total

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        })

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss = {train_loss:.4f}, "
            f"accuracy = {train_accuracy:.2%}",
            flush=True,
        )

        if checkpoint_path is not None:
            save_model(model, checkpoint_path)

    return model
import torch


def evaluate_model(model, data_loader, criterion, device="cpu"):
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("The data loader is empty.")

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy
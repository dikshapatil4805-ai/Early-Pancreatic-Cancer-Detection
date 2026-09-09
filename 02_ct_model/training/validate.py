import torch


def validate_model(model, dataloader, criterion, device):
    """
    Validate the model on validation dataset.
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, targets, _ in dataloader:

            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            logits, _ = model(images)

            # Loss
            loss = criterion(logits, targets)

            total_loss += loss.item() * images.size(0)

            # Prediction
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy
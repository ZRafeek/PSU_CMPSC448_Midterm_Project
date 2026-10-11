import torch
from models.modelArgs import ModelType
from models.cnn import CNN

def train_model(
        model: torch.nn.Module,
        train_loader,
        val_loader,
        epochs,
        device,
        modelType
):
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 0.001
    )

    model.to(device)

    if modelType == ModelType.CNN:
        for epoch in range(epochs):
            model.train()

            total_loss = 0
            correct = 0
            total = 0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                predictions = model(x)

                loss = criterion(predictions, y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted = predictions.argmax(dim=1)

                correct += (predicted==y).sum().item()
                total += y.size(0)

            accuracy = correct/total

            print(
                f"Epoch {epoch + 1}:"
                f" loss = {total_loss:.4f}  "
                f"\taccuracy = {accuracy:.4f}"
            )
    else:
        for epoch in range(epochs):
            model.train()

            total_loss = 0
            correct = 0
            total = 0

            for x, lengths, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                predictions = model(x, lengths)

                loss = criterion(predictions, y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted = predictions.argmax(dim=1)

                correct += (predicted==y).sum().item()
                total += y.size(0)

            accuracy = correct/total

            print(
                f"Epoch {epoch + 1}:"
                f" loss = {total_loss:.4f}  "
                f"\taccuracy = {accuracy:.4f}"
            )
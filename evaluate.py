import torch
from dataset import llm_models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate(model: torch.nn.Module, loader, device):

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for x, y in loader:
            x = x.to(device)
            output = model(x)

            predicted = output.argmax(dim=1)

            predictions.extend(predicted.cpu().tolist())
            actual.extend(y.tolist())

    print(
        classification_report(
            actual,
            predictions,
            target_names=llm_models
        )
    )

    print(
        confusion_matrix(
            actual,
            predictions
        )
    )
    pass
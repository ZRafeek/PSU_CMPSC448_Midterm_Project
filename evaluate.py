import torch
from dataset import llm_models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def evaluate(model: torch.nn.Module, loader, device, filename):

    ret_value = ""
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

    ret_value +=  classification_report(actual, predictions, target_names=llm_models).__str__()
    ret_value += "\n"

    cm = confusion_matrix(actual,predictions)
    ret_value += cm.__str__()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=llm_models)
    disp.plot().figure_.savefig("results/"+filename+"_confusion_matrix.png")

    return ret_value
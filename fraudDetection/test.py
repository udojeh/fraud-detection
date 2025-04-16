from dataclasses import dataclass
from torch import no_grad
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, average_precision_score
from .device import init_device

@dataclass
class TestConfig:
    """
    Attributes:
        model_path (str): the path of the model you want to test relative to the root of this project.
    """
    model_path: str

def test_model(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.BCELoss  # Uses BCELoss for binary classification since there are only 2 classes
    ) -> None:
        device = init_device()
        model = model.to(device)
        model.eval()

        test_loss = 0.0
        y_true = []  # Stores actual labels
        y_pred = []  # Stores model predictions
        y_raw_output = [] # Raw output probabilities before thresholding

        correct = 0
        total = 0
        with no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device, dtype=torch.float32)  

                outputs = model(X_batch).squeeze()  # Get predicted probabilities
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()

                # Convert probabilities to binary predictions (0 or 1)
                predicted = (outputs > 0.5).float()  
                
                # Stores predictions and actual values
                y_true.extend(y_batch.cpu().tolist())  
                y_pred.extend(predicted.cpu().tolist())
                y_raw_output.extend(outputs.cpu().tolist())

                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        

        avg_loss = test_loss / len(data_loader)

        # Calculates Precision, Recall, and F1-score
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_raw_output)

        # Prints results
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUPRC (Area Under Precision-Recall Curve): {auprc:.4f}")
        print(f"Final Accuracy: {100 * correct / total:.2f}%")

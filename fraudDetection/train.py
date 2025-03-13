import torch
from dataclasses import dataclass
from torch import no_grad
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from .device import init_device

@dataclass
class TrainConfig:
    """
    Attributes:
        batch_size (int): the number of samples processed in one forward/backward pass during training.
        epochs (int): the maximum number of training iterations for the model.
        learning_rate (float): controls the step size at which the model updates its weights during training.
    """
    batch_size: int
    epochs: int
    learning_rate: float

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.BCELoss,   # Uses BCELoss for binary classification since there are only 2 classes
        epochs: int
    ) -> None:        
        device = init_device()
        model = model.to(device)
        best_val_loss = float('inf')
        patience = 5
        trigger_times = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            y_true_train = []
            y_pred_train = []
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device, dtype=torch.float32)  
                
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()  # Squeeze to remove unnecessary dimensions
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Store predictions and actual values for metrics
                predicted = (outputs > 0.5).float()
                y_true_train.extend(y_batch.cpu().tolist())
                y_pred_train.extend(predicted.cpu().tolist())
            
            # Calculates training precision, recall, F1-score, and accuracy
            train_precision = precision_score(y_true_train, y_pred_train)
            train_recall = recall_score(y_true_train, y_pred_train)
            train_f1 = f1_score(y_true_train, y_pred_train)
            train_accuracy = accuracy_score(y_true_train, y_pred_train)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            y_true_val = []
            y_pred_val = []
            with no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device, dtype=torch.float32)
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    # Store predictions and actual values for validation metrics
                    predicted = (outputs > 0.5).float()
                    y_true_val.extend(y_batch.cpu().tolist())
                    y_pred_val.extend(predicted.cpu().tolist())
            
            val_loss /= len(val_loader)
            
            # Calculates validation precision, recall, F1-score, and accuracy
            val_precision = precision_score(y_true_val, y_pred_val)
            val_recall = recall_score(y_true_val, y_pred_val)
            val_f1 = f1_score(y_true_val, y_pred_val)
            val_accuracy = accuracy_score(y_true_val, y_pred_val)
            
            print(f'Epoch {epoch + 1}: Validation Loss: {val_loss:.4f}')
            print(f'Training Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-score: {train_f1:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}')
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print('Early stopping!')
                    break

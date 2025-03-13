from torch import save, load, tensor
import torch.nn as nn
import os

"""
This model is a fully connected feedforward neural network (Multi-Layer Perceptron) designed for binary classification on table (.csv) data.
It consists of multiple dense layers (each neuron is connected to every neuron in the previous layer) with ReLU activations and dropout for regularization.
"""
class FraudDetectionModel(nn.Module):
    def __init__(self, input_size: int):
        super(FraudDetectionModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 128), # First dense layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), # Second dense layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # Output layer (binary classification)
            nn.Sigmoid() # Ensures output is between 0 and 1
        )

    def forward(self, x: tensor):
        return self.fc_layers(x).squeeze(dim=1)

    def export(self, name: str) -> None:
        save(self.state_dict(), os.path.join("models/", name))

    def load(self, filepath) -> bool:
        self.load_state_dict(load(filepath, weights_only=True))  # Load only weights to avoid warnings


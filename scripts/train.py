import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# For preprocessing dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import numpy as np

# For loss and optimizer
import torch.nn as nn
import torch.optim as optim

# Project code
from fraudDetection import FraudDetectionModel, train_model, TrainConfig, CreditCardFraudDataset

config = TrainConfig(
    batch_size=64,
    epochs=50,
    learning_rate=0.0005
)

if __name__ == '__main__':
    # Load the dataset and split it for validation
    dataset = CreditCardFraudDataset(file_path="dataset/creditcard_realistic.csv", train=True) # make sure dataset is correct before running
    train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Configure model
    input_size = dataset.features.shape[1]  # Determine the number of features dynamically
    model = FraudDetectionModel(input_size=input_size)

    # Loss and optimizer
    criterion = nn.BCELoss()  # Changed to BCELoss for binary classification
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=config.learning_rate
    )

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config.epochs
    )

    while True:
        ans = input('Would you like to save this model? [y/n] ')
        if ans.lower() == 'y':
            name = input("Provide a name for the model: ")
            model.export(name)
            break
        elif ans.lower() == 'n':
            break
        else:
            print(f'\'{ans}\' is not a valid response...')

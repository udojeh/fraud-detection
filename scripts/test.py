import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# For preprocessing dataset
from torch.utils.data import DataLoader

# For loss and optimizer
import torch.nn as nn

# Project code
from fraudDetection import FraudDetectionModel, test_model, TestConfig, CreditCardFraudDataset

# Define model path
config = TestConfig(
    model_path="models/real"  # Update model name appropriately, if your saved model is called cool, models/cool should be in quotation marks
)

if __name__ == '__main__':
    # Load the dataset
    dataset = CreditCardFraudDataset(file_path="dataset/creditcard_realistic.csv", train=False) # make sure dataset is correct before running
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False  # No need to shuffle test data
    )

    # Configure model
    input_size = dataset.features.shape[1]  # Dynamically determine input size
    model = FraudDetectionModel(input_size=input_size)

    # Load trained model
    model.load(config.model_path)

    # Define loss function
    criterion = nn.BCELoss()  # Use BCELoss for binary classification

    # Test the model
    test_model(
        model=model,
        data_loader=data_loader,
        criterion=criterion
    )
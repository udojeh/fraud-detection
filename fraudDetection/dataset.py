from torch.utils.data import Dataset
from torch import tensor, float32, long
import pandas as pd
import numpy as np

class CreditCardFraudDataset(Dataset):
    def __init__(self, file_path: str, train: bool, test_split: float = 0.2, random_state: int = 42):
        self.file_path = file_path
        self.train = train
        self.test_split = test_split
        self.random_state = random_state
        self.features, self.labels, self.feature_names = self._load_data()
    
    def _load_data(self):
        data = pd.read_csv(self.file_path)
        
        # Extract features and labels
        features = data.drop(columns=["id", "Class"], errors='ignore')
        labels = data["Class"].values
        
        # Normalize 'Amount' feature
        if "Amount" in features.columns:
            features["Amount"] = (features["Amount"] - features["Amount"].mean()) / features["Amount"].std()
        
        feature_names = features.columns.tolist()
        features = features.values.astype(np.float32)
        
        # Split into train/test
        indices = np.arange(len(features))
        np.random.seed(self.random_state)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - self.test_split))
        
        if self.train:
            return features[indices[:split_idx]], labels[indices[:split_idx]], feature_names
        else:
            return features[indices[split_idx:]], labels[indices[split_idx:]], feature_names
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return tensor(self.features[idx], dtype=float32), tensor(self.labels[idx], dtype=long)
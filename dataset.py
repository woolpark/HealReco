import torch
from torch.utils.data import Dataset

class DiabetesTrainDataset(Dataset):
    def __init__(self, data):
        self.patients, self.labels = self.get_dataset(data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.patients[idx], self.labels[idx]

    def get_dataset(self, data):
        labels = data['readmitted'].values
        patients = data.loc[:, data.columns != 'readmitted'].values
        return torch.tensor(labels), torch.tensor(patients)

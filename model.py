import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import dataset

class NCF(pl.LightningModule):
    def __init__(self, num_patients, data):
        super().__init__()
        self.patient_embedding = nn.Embedding(num_embeddings=num_patients, embedding_dim=16)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.data = data

    def forward(self, patient_input):
        vector = self.patient_embedding(patient_input)
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))
        output = nn.Sigmoid()(self.output(vector))
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.patient_embedding(x)
        fc1 = self.fc1(z)
        fc2 = self.fc2(fc1)
        pred_y = self.output(fc2)
        print(pred_y)
        print(y.view(-1,1))
        loss = nn.CrossEntropyLoss()(pred_y, y.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(dataset.DiabetesTrainDataset(self.data))

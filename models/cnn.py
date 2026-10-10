import torch
import torch.nn as NeuralNetwork

class CNN(NeuralNetwork.Module):
    
    def __init__(self, vocab_size, embedding_dim=128, num_classes = 7):
        super().__init__()

        self.embedding = NeuralNetwork.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = NeuralNetwork.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5)
        self.pool = NeuralNetwork.AdaptiveAvgPool1d(1)
        self.fc = NeuralNetwork.Linear(128, num_classes)

        self.to()
        pass

    def forward(self, x):
        x = self.embedding(x)

        x = x.transpose(1, 2)

        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.squeeze(2)

        return self.fc(x)
    
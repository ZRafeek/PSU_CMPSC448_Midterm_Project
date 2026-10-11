import torch
import torch.nn as NeuralNetwork
from torch.nn.utils.rnn import pack_padded_sequence

class LTSM(NeuralNetwork.Module):
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes = 7):
        super().__init__()

        self.embedding = NeuralNetwork.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.ltsm = NeuralNetwork.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = NeuralNetwork.Linear(hidden_dim, num_classes)
        
        pass

    def forward(self, x, lengths):
        
        x = self.embedding(x)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, (hidden, cell) = self.ltsm(packed)

        x = hidden[-1]

        return self.fc(x)
    
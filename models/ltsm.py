import torch
import torch.nn as NeuralNetwork

class LTSM(NeuralNetwork.Module):
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes = 7):
        super().__init__()

        self.embedding = NeuralNetwork.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.ltsm = NeuralNetwork.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = NeuralNetwork.Linear(hidden_dim, num_classes)
        
        pass

    def forward(self, x):
        
        x = self.embedding(x)

        output, (hidden, cell) = self.ltsm(x)

        x = hidden[-1]

        return self.fc(x)
    
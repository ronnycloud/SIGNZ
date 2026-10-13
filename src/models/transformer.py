import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SignTransformer(nn.Module):
    def __init__(self, num_classes, input_dim=258, d_model=128, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        # 1. Project input sequence to d_model dimensions
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Add temporal information
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. Classification Head
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # Apply transformer attention
        x = self.transformer_encoder(x)
        
        # Global Average Pooling across the temporal dimension
        x = x.mean(dim=1) 
        
        x = self.dropout(x)
        out = self.fc(x)
        return out

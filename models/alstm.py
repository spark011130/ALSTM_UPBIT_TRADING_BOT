import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, lstm_output):
        attn_scores = torch.tanh(self.attn_weights(lstm_output))  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # Normalize across time dimension
        context = torch.sum(attn_weights * lstm_output, dim=1)  # Weighted sum
        return context, attn_weights

class ALSTM(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.0):
        super(ALSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        context, attn_weights = self.attention(lstm_out)  # Apply attention
        output = self.fc(context)  # Final prediction
        return output, attn_weights
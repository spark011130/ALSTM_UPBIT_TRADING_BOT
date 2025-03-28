import torch
import torch.nn as nn

def get_model(model_name, input_dim, device):
    if model_name == "ALSTM":
        return ALSTM(input_dim=input_dim).to(device)
    elif model_name == "ALSTM_FEATURE_ATTENTION":
        return ALSTMWithFeatureAttention(input_dim=input_dim).to(device)

### ALSTM
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

### ALSTM FEATURE ATTENTION
class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        self.attn = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        score = self.attn(x)  # (batch, seq_len, input_dim)
        weights = torch.softmax(score, dim=2)  # attention across features
        weighted_x = x * weights  # (batch, seq_len, input_dim)
        return weighted_x, weights

class TimeAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TimeAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_dim)
        score = self.attn(lstm_out)  # (batch, seq_len, 1)
        weights = torch.softmax(score, dim=1)  # attention across time
        context = torch.sum(weights * lstm_out, dim=1)  # (batch, hidden_dim)
        return context, weights

class ALSTMWithFeatureAttention(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1):
        super(ALSTMWithFeatureAttention, self).__init__()
        self.feature_attn = FeatureAttention(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.time_attn = TimeAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x, feature_weights = self.feature_attn(x)  # Apply feature attention
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        context, time_weights = self.time_attn(lstm_out)  # Apply time attention
        output = self.fc(context)  # Final prediction
        return output, time_weights.squeeze(-1), feature_weights  # feature_weights: (batch, seq_len, input_dim)
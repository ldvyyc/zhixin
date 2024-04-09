import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

torch.device("mps")

# Define the Transformer model for time series forecasting
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_dim = model_dim
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout)
        self.output_linear = nn.Linear(model_dim, 1)

    def forward(self, src):
        src = self.input_linear(src)
        src = self.positional_encoding(src)
        output = self.transformer(src, src)
        output = self.output_linear(output)
        return output

# Define the Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Load your data
# For simplicity, I'm using random data here. Replace this with your actual time series data.
# data = torch.randn(100, 1, 5)  # (sequence length, batch size, features)
data = pd.read_csv('dataset-20230501.csv', header=None)
data = torch.tensor(data.values, dtype=torch.float32)



# Initialize the model
model = TimeSeriesTransformer(input_dim=5, model_dim=64, num_heads=4, num_encoder_layers=2, num_decoder_layers=2)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data[:-1])  # Use all but the last data point for training
    loss = criterion(output, data[1:])  # Predict the next data point
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Make predictions
model.eval()
predictions = model(data[:-1])  # Predict the next data point for all but the last data point

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data[1:].numpy().flatten(), label='Actual Data')
plt.plot(predictions.detach().numpy().flatten(), label='Predicted Data')
plt.legend()
plt.show()

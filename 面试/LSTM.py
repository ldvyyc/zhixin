import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load the dataset
file_path = 'dataset-20230501.csv'
df = pd.read_csv(file_path, header=None)
df.fillna(method="ffill", inplace=True)
df = df*1000
# Separate features and target
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape X for LSTM [samples, time steps, features]
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[2]
hidden_dim = 64
num_layers = 2
output_dim = 1

model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
y_pred = model(X_test).detach().numpy()
y_test_np = y_test.numpy()

mse = mean_squared_error(y_test_np, y_pred)
r_squared = r2_score(y_test_np, y_pred)
accuracy = accuracy_score((y_test_np > 0).astype(int), (y_pred > 0).astype(int))
sharpe_ratio = np.mean(y_pred) / np.std(y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r_squared}')
print(f'Prediction Accuracy: {accuracy}')
print(f'Sharpe Ratio: {sharpe_ratio}')

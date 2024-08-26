import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Example: Generate synthetic stock price data (for illustration)
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=500)
n_stocks = 5

# Generate synthetic stock data for multiple stocks with multiple features
data = {
    f'Stock_{i+1}': {
        'Open': np.cumsum(np.random.randn(500)) + 100,
        'High': np.cumsum(np.random.randn(500)) + 105,
        'Low': np.cumsum(np.random.randn(500)) + 95,
        'Close': np.cumsum(np.random.randn(500)) + 100,
        'Volume': np.random.randint(1000, 5000, size=500)
    }
    for i in range(n_stocks)
}

# Convert to a DataFrame with MultiIndex
stock_data = pd.concat(
    {k: pd.DataFrame(v, index=dates) for k, v in data.items()},
    axis=0
).unstack(level=0)

# Plot one stock's data as an example
stock_data['Stock_1'].plot(subplots=True, figsize=(12, 8), title='Stock 1 Data')
plt.show()
class StockDataset(Dataset):
    def __init__(self, data, seq_length=30):
        self.data = data
        self.seq_length = seq_length
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(data)
        
    def __len__(self):
        return len(self.data_scaled) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data_scaled[idx:idx+self.seq_length]
        y = self.data_scaled[idx+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Prepare dataset and dataloader
seq_length = 30
dataset = StockDataset(stock_data['Stock_1'].values, seq_length=seq_length)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, n_heads, n_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_layer = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, model_dim))
        encoder_layers = nn.TransformerEncoderLayer(model_dim, n_heads, dim_feedforward=512, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.output_layer = nn.Linear(model_dim, input_dim)
        
    def forward(self, x):
        x = self.input_layer(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.output_layer(x[:, -1, :])  # Use the output of the last time step
        return x

# Model parameters
input_dim = stock_data['Stock_1'].shape[1]  # Number of features
model_dim = 64
n_heads = 8
n_layers = 4

# Instantiate the model
model = TimeSeriesTransformer(input_dim, model_dim, n_heads, n_layers)
# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 50
model.train()
for epoch in range(n_epochs):
    epoch_loss = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(dataloader):.4f}')

# Switch to evaluation mode
model.eval()

# Generate predictions on the last sequence in the training set
x_test, y_test = dataset[-1]
x_test = x_test.unsqueeze(0)  # Add batch dimension
y_pred = model(x_test).detach().numpy()

# Inverse scale the prediction and ground truth
y_test = dataset.scaler.inverse_transform(y_test.numpy().reshape(1, -1))
y_pred = dataset.scaler.inverse_transform(y_pred)

# Plot the ground truth vs prediction
plt.plot(y_test.flatten(), label='Ground Truth')
plt.plot(y_pred.flatten(), label='Prediction')
plt.legend()
plt.title('Transformer Forecast')
plt.show()

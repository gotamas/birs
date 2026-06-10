"""Generate a Jupyter notebook for BIRS LSTM Forecasting."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}

cells = []

# ============================== TITLE ==============================
cells.append(nbf.v4.new_markdown_cell("""# 🧠 Hybrid Factor-LSTM Yield Curve Forecasting — BIRS

This notebook implements a hybrid time-series forecasting approach:
1. Fits the **Nelson-Siegel (NS)** cross-sectional yield curve model to Hungarian BIRS swap rates.
2. Extracts daily latent factors: $\\beta_{1,t}$ (Level), $\\beta_{2,t}$ (Slope), and $\\beta_{3,t}$ (Curvature).
3. Trains a **Long Short-Term Memory (LSTM)** neural network to forecast future factors.
4. Reconstructs the forecasted yield curve using the predicted factors.

---"""))

# ============================== IMPORTS ==============================
cells.append(nbf.v4.new_markdown_cell("""## 1. Environment Setup and Data Loading"""))

cells.append(nbf.v4.new_code_cell("""import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load data
df_factors = pd.read_pickle('data/ns_factors.pkl')
df_birs = pd.read_pickle('data/birs_flat.pkl')

# Load optimal lambda from Nelson-Siegel model parameters
if os.path.exists('data/ns_model_params.json'):
    with open('data/ns_model_params.json', 'r') as f:
        params = json.load(f)
        LAMBDA = params.get("lambda", 0.3709)
else:
    LAMBDA = 0.3709

# Extract maturities (tenors) in years
tenor_cols = [c for c in df_birs.columns if 'years' in c]
tenor_cols.sort(key=lambda x: int(x.split()[0]))
maturities = np.array([int(c.split()[0]) for c in tenor_cols], dtype=float)

# Sort and align
df_factors['date'] = pd.to_datetime(df_factors['date'])
df_factors = df_factors.sort_values('date').reset_index(drop=True)
df_birs['date of fixing'] = pd.to_datetime(df_birs['date of fixing'])
df_birs = df_birs.sort_values('date of fixing').reset_index(drop=True)

merged = pd.merge(df_factors, df_birs, left_on='date', right_on='date of fixing', how='inner')
print(f"Dataset: {len(merged):,} trading days from {merged['date'].min().date()} to {merged['date'].max().date()}")
print(f"Maturities: {maturities} years")"""))

# ============================== PREPROCESSING ==============================
cells.append(nbf.v4.new_markdown_cell("""## 2. Sliding Window Dataset Preprocessing

We scale the factors to a $[0, 1]$ range for training stability, and create sequence inputs of lookback window length $L = 20$ trading days to predict next-day factors."""))

cells.append(nbf.v4.new_code_cell("""factor_cols = ['beta1_level', 'beta2_slope', 'beta3_curvature']
data = merged[factor_cols].values

# Scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, lookback=20):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i : i + lookback])
        y.append(dataset[i + lookback])
    return np.array(X), np.array(y)

LOOKBACK = 20
X, y = create_dataset(scaled_data, LOOKBACK)

# Split into Train (80%) and Test (20%)
train_size = int(len(X) * 0.80)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

print(f"Train inputs: {X_train_t.shape} | Train labels: {y_train_t.shape}")
print(f"Test inputs:  {X_test_t.shape} | Test labels:  {y_test_t.shape}")"""))

# ============================== MODEL DEFINITION ==============================
cells.append(nbf.v4.new_markdown_cell("""## 3. PyTorch LSTM Architecture"""))

cells.append(nbf.v4.new_code_cell("""class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

INPUT_DIM = 3
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 3
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("LSTM network initialized.")
print(model)"""))

# ============================== TRAINING ==============================
cells.append(nbf.v4.new_markdown_cell("""## 4. LSTM Training Loop"""))

cells.append(nbf.v4.new_code_cell("""model.train()
losses = []

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0
    permutation = torch.randperm(X_train_t.size(0))
    for i in range(0, X_train_t.size(0), BATCH_SIZE):
        indices = permutation[i : i + BATCH_SIZE]
        batch_x, batch_y = X_train_t[indices], y_train_t[indices]
        
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * len(indices)
    
    epoch_loss /= X_train_t.size(0)
    losses.append(epoch_loss)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{EPOCHS:03d} | Train MSE Loss: {epoch_loss:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(losses, label='Train MSE Loss', color='purple')
plt.title('LSTM Training Loss curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""))

# ============================== EVALUATION ==============================
cells.append(nbf.v4.new_markdown_cell("""## 5. Forecasting Factor Dynamics"""))

cells.append(nbf.v4.new_code_cell("""model.eval()
with torch.no_grad():
    scaled_predictions = model(X_test_t).numpy()

# Inverse scaling
predictions = scaler.inverse_transform(scaled_predictions)
actuals = scaler.inverse_transform(y_test)

# Plot forecasts
test_dates = merged['date'].values[LOOKBACK + train_size:]

fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
colors = ['#4CAF50', '#F44336', '#2196F3']
for i, (col, name) in enumerate(zip(factor_cols, ["Level (β₁)", "Slope (β₂)", "Curvature (β₃)"])):
    axes[i].plot(test_dates, actuals[:, i], label='Actual', color='black', alpha=0.5, linewidth=1.2)
    axes[i].plot(test_dates, predictions[:, i], label='Predicted (LSTM)', color=colors[i], linestyle='--', linewidth=1.5)
    axes[i].set_ylabel(name)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
axes[0].set_title("LSTM Predictions vs. Actual Factors on Test Split", fontweight='bold', fontsize=14)
axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.show()

# Mean Absolute Error
factor_mae = np.mean(np.abs(predictions - actuals), axis=0)
print("Factor Mean Absolute Error (MAE):")
for name, mae in zip(factor_cols, factor_mae):
    print(f"  {name}: {mae:.4f} pp")"""))

# ============================== CURVE RECONSTRUCTION ==============================
cells.append(nbf.v4.new_markdown_cell("""## 6. Curve Reconstruction & Validation

We reconstruct the yield curves using the forecasted Nelson-Siegel factors and calculate the root mean squared error (RMSE) against actual historical BIRS rates."""))

cells.append(nbf.v4.new_code_cell("""def ns_curve(tau, beta1, beta2, beta3, lam):
    lt = lam * tau
    lt = np.maximum(lt, 1e-10)
    exp_lt = np.exp(-lt)
    loading1 = 1.0
    loading2 = (1 - exp_lt) / lt
    loading3 = loading2 - exp_lt
    return beta1 * loading1 + beta2 * loading2 + beta3 * loading3

# Actual swap rates on the test set
test_swap_rates = merged[tenor_cols].values[LOOKBACK + train_size:]

predicted_curves = []
for idx in range(len(predictions)):
    b1, b2, b3 = predictions[idx]
    curve = ns_curve(maturities, b1, b2, b3, LAMBDA)
    predicted_curves.append(curve)
predicted_curves = np.array(predicted_curves)

# Yield forecasting RMSE
curve_rmse = np.sqrt(np.mean((predicted_curves - test_swap_rates) ** 2))
print(f"Overall Yield Curve Forecasting RMSE on Test Set: {curve_rmse:.4f} pp")

# Draw curve comparison for the last test day
plt.figure(figsize=(10, 6))
sample_idx = -1
plt.plot(maturities, test_swap_rates[sample_idx], 'ko', label=f'Actual BIRS Curve ({pd.Timestamp(test_dates[sample_idx]).strftime("%Y-%m-%d")})')
plt.plot(maturities, predicted_curves[sample_idx], 'r-', linewidth=2, label='LSTM Reconstructed')
plt.xlabel('Tenor (years)')
plt.ylabel('Rate (%)')
plt.title('Sample Reconstructed Yield Curve via LSTM Forecasts', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""))

# ============================== SUMMARY ==============================
cells.append(nbf.v4.new_markdown_cell("""## 📋 Summary

### What we implemented:
1. **Dynamic Nelson-Siegel Factorization:** Extracted factor trajectories over time.
2. **LSTM Neural Network:** Modeled temporal dependency structure of Level, Slope, and Curvature.
3. **Forecasting and Mapping:** Validated next-day factor predictions and mapped factors back to smooth swap rate curves.
"""))

nb.cells = cells
nbf.write(nb, 'birs_lstm_forecast.ipynb')
print("Created create_lstm_notebook.py and generated birs_lstm_forecast.ipynb code.")

"""
Option 1: Hybrid Factor-ML Model (Dynamic Nelson-Siegel + PyTorch LSTM)
This script loads the daily Nelson-Siegel factors, trains an LSTM network to forecast
the latent factors (Level, Slope, Curvature), and reconstructs the future yield curve.
"""

import os
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

# --- 1. Load Data ---
FACTORS_PATH = os.path.join("data", "ns_factors.pkl")
BIRS_PATH = os.path.join("data", "birs_flat.pkl")
PARAMS_PATH = os.path.join("data", "ns_model_params.json")

print("Loading data...")
if not os.path.exists(FACTORS_PATH):
    raise FileNotFoundError(f"Could not find {FACTORS_PATH}. Please run create_ns_notebook.py first.")

df_factors = pd.read_pickle(FACTORS_PATH)
df_birs = pd.read_pickle(BIRS_PATH)

# Load optimal lambda
if os.path.exists(PARAMS_PATH):
    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)
        LAMBDA = params.get("lambda", 0.3709)
else:
    LAMBDA = 0.3709
print(f"Loaded Nelson-Siegel Lambda: {LAMBDA:.4f}")

# Extract maturities from BIRS columns
tenor_cols = [c for c in df_birs.columns if 'years' in c]
tenor_cols.sort(key=lambda x: int(x.split()[0]))
maturities = np.array([int(c.split()[0]) for c in tenor_cols], dtype=float)
print(f"Available maturities (tenors): {maturities} years")

# Sort and align datasets by date
df_factors['date'] = pd.to_datetime(df_factors['date'])
df_factors = df_factors.sort_values('date').reset_index(drop=True)
df_birs['date of fixing'] = pd.to_datetime(df_birs['date of fixing'])
df_birs = df_birs.sort_values('date of fixing').reset_index(drop=True)

# Merge datasets to ensure strict alignment
merged = pd.merge(df_factors, df_birs, left_on='date', right_on='date of fixing', how='inner')
print(f"Aligned dataset contains {len(merged):,} trading days.")

# --- 2. Prepare LSTM Inputs ---
# We predict [beta1_level, beta2_slope, beta3_curvature]
factor_cols = ['beta1_level', 'beta2_slope', 'beta3_curvature']
data = merged[factor_cols].values

# Scale the data to [0, 1] for stable LSTM training
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sliding windows
def create_dataset(dataset, lookback=20):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i : i + lookback])
        y.append(dataset[i + lookback])
    return np.array(X), np.array(y)

LOOKBACK = 20  # 20 trading days (~1 month) lookback
X, y = create_dataset(scaled_data, LOOKBACK)

# Split into Train (80%) and Test (20%)
train_size = int(len(X) * 0.80)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch Tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

print(f"Train shapes: X={X_train_t.shape}, y={y_train_t.shape}")
print(f"Test shapes: X={X_test_t.shape}, y={y_test_t.shape}")

# --- 3. Define the PyTorch LSTM ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the last sequence step
        out = self.fc(out[:, -1, :])
        return out

# Model Hyperparameters
INPUT_DIM = 3   # 3 beta factors
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 3  # predicting next day's 3 factors
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. Model Training ---
print("\nTraining LSTM Model...")
model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0
    # Batch learning
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
    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:03d}/{EPOCHS:03d} | Train MSE Loss: {epoch_loss:.6f}")

# --- 5. Forecast and Evaluation ---
model.eval()
with torch.no_grad():
    scaled_predictions = model(X_test_t).numpy()

# Inverse transform to original scale
predictions = scaler.inverse_transform(scaled_predictions)
actuals = scaler.inverse_transform(y_test)

# Calculate Factor Prediction Metrics (MAE)
factor_mae = np.mean(np.abs(predictions - actuals), axis=0)
print("\n--- Factor Prediction MAE ---")
for name, mae in zip(factor_cols, factor_mae):
    print(f"  {name}: {mae:.4f} pp")

# --- 6. Reconstruct Yield Curves and Evaluate Fit ---
def ns_curve(tau, beta1, beta2, beta3, lam):
    """Compute Nelson-Siegel yields on maturity grid tau."""
    lt = lam * tau
    lt = np.maximum(lt, 1e-10) # avoid division by zero
    exp_lt = np.exp(-lt)
    loading1 = 1.0
    loading2 = (1 - exp_lt) / lt
    loading3 = loading2 - exp_lt
    return beta1 * loading1 + beta2 * loading2 + beta3 * loading3

# Get the actual swap rates for the test period
# Align with sequence output (the test labels y_test correspond to days LOOKBACK + train_size to the end)
test_dates = merged['date'].values[LOOKBACK + train_size:]
test_swap_rates = merged[tenor_cols].values[LOOKBACK + train_size:]

predicted_curves = []
for idx in range(len(predictions)):
    b1, b2, b3 = predictions[idx]
    curve = ns_curve(maturities, b1, b2, b3, LAMBDA)
    predicted_curves.append(curve)
predicted_curves = np.array(predicted_curves)

# Evaluate curve reconstruction error
curve_rmse = np.sqrt(np.mean((predicted_curves - test_swap_rates) ** 2))
print(f"\nOverall Curve Forecasting RMSE (on Test Set): {curve_rmse:.4f} pp")

# --- 7. Save Visualizations ---
os.makedirs("plots", exist_ok=True)
plot_file = os.path.join("plots", "lstm_dns_predictions.png")

fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
colors = ['#4CAF50', '#F44336', '#2196F3']
for i, (col, name) in enumerate(zip(factor_cols, ["Level (beta1)", "Slope (beta2)", "Curvature (beta3)"])):
    axes[i].plot(test_dates, actuals[:, i], label='Actual Factors', color='black', alpha=0.5, linewidth=1.2)
    axes[i].plot(test_dates, predictions[:, i], label='LSTM Predicted', color=colors[i], linestyle='--', linewidth=1.5)
    axes[i].set_ylabel(name)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
axes[0].set_title("Nelson-Siegel Factors: LSTM Predicted vs. Actual on Test Set", fontweight='bold', fontsize=14)
axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig(plot_file, dpi=150)
print(f"Saved predictions plot to {plot_file}")

# Draw a sample reconstructed curve comparison
plt.figure(figsize=(10, 6))
sample_idx = -1  # last day of test set
plt.plot(maturities, test_swap_rates[sample_idx], 'ko', label=f'Actual BIRS Curve ({pd.Timestamp(test_dates[sample_idx]).strftime("%Y-%m-%d")})')
plt.plot(maturities, predicted_curves[sample_idx], 'r-', linewidth=2, label='LSTM Reconstructed Curve')
plt.xlabel('Tenor (years)')
plt.ylabel('Rate (%)')
plt.title('Sample Reconstructed Yield Curve via LSTM Predictions', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
curve_sample_file = os.path.join("plots", "lstm_sample_curve.png")
plt.savefig(curve_sample_file, dpi=150)
print(f"Saved sample curve plot to {curve_sample_file}")

print("\nOption 1 Example execution finished successfully!")

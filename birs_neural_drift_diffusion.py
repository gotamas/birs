"""
Option 4: Neural Drift-Diffusion SDE Model
This script trains neural networks to represent the drift and diffusion coefficients
of the yield curve dynamics. The dynamics are modeled as:
    dx_t = mu_theta(x_t) dt + diag(sigma_phi(x_t)) dW_t
This is discretized via Euler-Maruyama and calibrated using Maximum Likelihood.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Load Data ---
BIRS_PATH = os.path.join("data", "birs_flat.pkl")

print("Loading data...")
if not os.path.exists(BIRS_PATH):
    raise FileNotFoundError(f"Could not find {BIRS_PATH}.")

df = pd.read_pickle(BIRS_PATH)
df['date of fixing'] = pd.to_datetime(df['date of fixing'])
df = df.sort_values('date of fixing').reset_index(drop=True)

# Select a subset of tenors to model for numeric stability and clarity
# (Modeling 2Y, 5Y, 10Y, 15Y, and 20Y tenors)
tenor_cols = ['2 years', '5 years', '10 years', '15 years', '20 years']
for col in tenor_cols:
    if col not in df.columns:
        raise KeyError(f"Expected column '{col}' not found in BIRS dataset.")

# Drop rows with NaNs in our chosen tenors
df_clean = df.dropna(subset=tenor_cols).reset_index(drop=True)
data = df_clean[tenor_cols].values  # Shape: (T, D)
dates = df_clean['date of fixing'].values
T_len, D_dim = data.shape
print(f"Data shape: {T_len} days, {D_dim} tenors.")

# Time step: dt = 1 / 252 (daily time steps in annual terms)
DT = 1.0 / 252.0

# Prepare inputs (x_t) and target increments (dx_t = x_{t+1} - x_t)
x_t = data[:-1]
x_next = data[1:]
dx_t = x_next - x_t

# Convert to PyTorch tensors
x_t_t = torch.FloatTensor(x_t)
dx_t_t = torch.FloatTensor(dx_t)

# Split into Train (80%) and Test (20%)
train_size = int(len(x_t_t) * 0.80)
x_train, x_test = x_t_t[:train_size], x_t_t[train_size:]
dx_train, dx_test = dx_t_t[:train_size], dx_t_t[train_size:]

# --- 2. Define the Drift and Diffusion Neural Networks ---
class DriftNet(nn.Module):
    """Neural network parameterizing the drift vector mu(x)"""
    def __init__(self, state_dim, hidden_dim=64):
        super(DriftNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim)
        )
    def forward(self, x):
        return self.net(x)

class DiffusionNet(nn.Module):
    """Neural network parameterizing the diagonal volatility vector sigma(x)"""
    def __init__(self, state_dim, hidden_dim=64):
        super(DiffusionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim)
        )
    def forward(self, x):
        # Volatility must be positive; Softplus guarantees this, 1e-4 avoids division by zero
        return torch.softplus(self.net(x)) + 1e-4

# Create models
drift_model = DriftNet(state_dim=D_dim)
diff_model = DiffusionNet(state_dim=D_dim)

# --- 3. Calibration via Maximum Likelihood ---
# Optimization parameters
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.002

optimizer = optim.Adam(
    list(drift_model.parameters()) + list(diff_model.parameters()), 
    lr=LEARNING_RATE
)

def neural_sde_loss(x, dx, drift_net, diff_net, dt):
    """
    Computes negative log-likelihood of increments dx under the Euler-Maruyama SDE:
    dx_t ~ N( mu(x_t)*dt, sigma(x_t)^2 * dt )
    """
    # Forward passes
    pred_drift = drift_net(x)        # mu(x_t)
    pred_vol = diff_net(x)          # sigma(x_t)
    
    # Expected mean and variance of the daily increment
    mean = pred_drift * dt
    variance = (pred_vol ** 2) * dt
    
    # Gaussian negative log-likelihood
    # NLL = 0.5 * log(2 * pi * var) + (dx - mean)^2 / (2 * var)
    nll = 0.5 * torch.log(2 * np.pi * variance) + ((dx - mean) ** 2) / (2 * variance)
    return torch.mean(torch.sum(nll, dim=1)) # mean loss over batch, sum over dimensions

# Training loop
print("\nTraining Neural SDE (Drift & Diffusion networks)...")
drift_model.train()
diff_model.train()

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0
    permutation = torch.randperm(x_train.size(0))
    for i in range(0, x_train.size(0), BATCH_SIZE):
        indices = permutation[i : i + BATCH_SIZE]
        batch_x, batch_dx = x_train[indices], dx_train[indices]
        
        optimizer.zero_grad()
        loss = neural_sde_loss(batch_x, batch_dx, drift_model, diff_model, DT)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * len(indices)
        
    epoch_loss /= x_train.size(0)
    if epoch % 15 == 0 or epoch == 1:
        print(f"  Epoch {epoch:03d}/{EPOCHS:03d} | Train Neg Log-Likelihood: {epoch_loss:.4f}")

# --- 4. Evaluate on Test Set ---
drift_model.eval()
diff_model.eval()
with torch.no_grad():
    test_loss = neural_sde_loss(x_test, dx_test, drift_model, diff_model, DT).item()
print(f"\nTest Set Negative Log-Likelihood: {test_loss:.4f}")

# --- 5. Run Monte Carlo Simulations (Simulate Paths) ---
# Evolve the yield curve starting from the last day of the historical dataset
x_start = torch.FloatTensor(data[-1]).unsqueeze(0)  # Shape: (1, D)
N_PATHS = 100
N_STEPS = 60  # simulate next 60 trading days (approx 3 months)

simulated_paths = torch.zeros(N_PATHS, N_STEPS + 1, D_dim)
simulated_paths[:, 0, :] = x_start.repeat(N_PATHS, 1)

print(f"\nSimulating {N_PATHS} Monte Carlo paths for {N_STEPS} steps...")
with torch.no_grad():
    for step in range(N_STEPS):
        curr_states = simulated_paths[:, step, :]
        
        # Calculate drift and volatility for current states
        drifts = drift_model(curr_states)
        vols = diff_model(curr_states)
        
        # Standard normal random shocks
        shocks = torch.randn(N_PATHS, D_dim)
        
        # Euler step: x_{t+dt} = x_t + mu(x_t)*dt + sigma(x_t)*sqrt(dt)*Z
        next_states = curr_states + drifts * DT + vols * np.sqrt(DT) * shocks
        simulated_paths[:, step + 1, :] = next_states

# Convert simulated paths to numpy for plotting
sim_paths_np = simulated_paths.numpy()  # Shape: (Paths, Steps, D)

# --- 6. Save Plots ---
os.makedirs("plots", exist_ok=True)
plot_file = os.path.join("plots", "neural_sde_simulation.png")

# Plot simulated paths for the 5Y and 10Y rates (columns indices 1 and 2)
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
time_axis = np.arange(N_STEPS + 1)

# Index 1 = 5 years, Index 2 = 10 years
tenor_indices = [1, 2]
tenor_names = ["5 Year Tenor", "10 Year Tenor"]
colors = ['#FF9800', '#2196F3']

for i, (idx, name) in enumerate(zip(tenor_indices, tenor_names)):
    # Plot historical tail (last 60 days) to show starting context
    lookback = 60
    hist_dates = np.arange(-lookback, 1)
    axes[i].plot(hist_dates, data[-lookback-1:, idx], color='black', label='History (Last 60d)', linewidth=2)
    
    # Plot individual Monte Carlo paths
    for p in range(min(50, N_PATHS)):
        axes[i].plot(time_axis, sim_paths_np[p, :, idx], color=colors[i], alpha=0.15, linewidth=0.8)
    
    # Plot path statistics
    axes[i].plot(time_axis, np.median(sim_paths_np[:, :, idx], axis=0), color='black', linestyle='--', linewidth=2.0, label='Median Forecast')
    axes[i].plot(time_axis, np.percentile(sim_paths_np[:, :, idx], 5, axis=0), color='red', linestyle=':', linewidth=1.5, label='5th / 95th Percentile')
    axes[i].plot(time_axis, np.percentile(sim_paths_np[:, :, idx], 95, axis=0), color='red', linestyle=':', linewidth=1.5)
    
    axes[i].set_title(f"Neural SDE Monte Carlo Simulation: {name}", fontweight='bold')
    axes[i].set_ylabel("Yield Rate (%)")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(loc='upper left')

axes[-1].set_xlabel("Steps (Trading Days Ahead)")
plt.tight_layout()
plt.savefig(plot_file, dpi=150)
print(f"Saved simulation plot to {plot_file}")

print("\nOption 4 Example execution finished successfully!")

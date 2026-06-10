"""Generate a Jupyter notebook for BIRS Neural SDE (Drift-Diffusion) Calibration."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}

cells = []

# ============================== TITLE ==============================
cells.append(nbf.v4.new_markdown_cell("""# 🏗️ Continuous-Time Neural Drift-Diffusion SDE Model — BIRS

This notebook models interest rate swap curve dynamics as a multivariate continuous-time Stochastic Differential Equation (SDE):

$$d\mathbf{x}_t = \boldsymbol{\mu}_{\theta}(\mathbf{x}_t)\,dt + \boldsymbol{\sigma}_{\phi}(\mathbf{x}_t)\,d\mathbf{W}_t$$

where:
- $\mathbf{x}_t \in \mathbb{R}^D$ is the yield curve state (swap rates across $D$ selected tenors).
- $\boldsymbol{\mu}_{\theta}(\mathbf{x}_t) \in \mathbb{R}^D$ is the state-dependent drift vector parameterized by a neural network.
- $\boldsymbol{\sigma}_{\phi}(\mathbf{x}_t) \in \mathbb{R}^D$ represents the diagonal diffusion/volatility vector parameterized by another neural network.
- $\mathbf{W}_t$ is a $D$-dimensional standard Brownian motion.

Under the Euler-Maruyama discretization, daily changes are distributed as:

$$\mathbf{x}_{t+1} - \mathbf{x}_t \sim \mathcal{N}\left(\boldsymbol{\mu}_{\theta}(\mathbf{x}_t)\Delta t, \text{diag}(\boldsymbol{\sigma}_{\phi}(\mathbf{x}_t))^2 \Delta t\right)$$

This model is trained by maximizing the log-likelihood of transition increments.

---"""))

# ============================== IMPORTS ==============================
cells.append(nbf.v4.new_markdown_cell("""## 1. Imports and Data Preparation"""))

cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Load Flat BIRS data
df = pd.read_pickle('data/birs_flat.pkl')
df['date of fixing'] = pd.to_datetime(df['date of fixing'])
df = df.sort_values('date of fixing').reset_index(drop=True)

# Select a subset of tenors for high numeric stability
tenor_cols = ['2 years', '5 years', '10 years', '15 years', '20 years']
df_clean = df.dropna(subset=tenor_cols).reset_index(drop=True)
data = df_clean[tenor_cols].values
dates = df_clean['date of fixing'].values
T_len, D_dim = data.shape

# Annualized daily time step
DT = 1.0 / 252.0

# Define states (x_t) and increments (dx_t = x_{t+1} - x_t)
x_t = data[:-1]
x_next = data[1:]
dx_t = x_next - x_t

# Convert to tensors
x_t_t = torch.FloatTensor(x_t)
dx_t_t = torch.FloatTensor(dx_t)

# Split (80% Train, 20% Test)
train_size = int(len(x_t_t) * 0.80)
x_train, x_test = x_t_t[:train_size], x_t_t[train_size:]
dx_train, dx_test = dx_t_t[:train_size], dx_t_t[train_size:]

print(f"Dataset covers {T_len} days.")
print(f"Features: {tenor_cols}")
print(f"Train size: {x_train.shape[0]} | Test size: {x_test.shape[0]}")"""))

# ============================== MODEL DEFINITIONS ==============================
cells.append(nbf.v4.new_markdown_cell("""## 2. Define Drift and Volatility Networks"""))

cells.append(nbf.v4.new_code_cell("""class DriftNet(nn.Module):
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
    def __init__(self, state_dim, hidden_dim=64):
        super(DiffusionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim)
        )
    def forward(self, x):
        # Softplus ensures positive volatility bounds (>= 1e-4)
        return torch.softplus(self.net(x)) + 1e-4

drift_model = DriftNet(state_dim=D_dim)
diff_model = DiffusionNet(state_dim=D_dim)

print("Drift Network:")
print(drift_model)
print("\\nDiffusion Network:")
print(diff_model)"""))

# ============================== LOSS FUNCTION ==============================
cells.append(nbf.v4.new_markdown_cell("""## 3. Transition Log-Likelihood Loss"""))

cells.append(nbf.v4.new_code_cell("""def neural_sde_loss(x, dx, drift_net, diff_net, dt):
    \"\"\"
    Computes negative log-likelihood of increments under:
    dx_t ~ N( mu(x_t)*dt, sigma(x_t)^2 * dt )
    \"\"\"
    pred_drift = drift_net(x)
    pred_vol = diff_net(x)
    
    mean = pred_drift * dt
    variance = (pred_vol ** 2) * dt
    
    # Gaussian Negative Log-Likelihood formula
    nll = 0.5 * torch.log(2 * np.pi * variance) + ((dx - mean) ** 2) / (2 * variance)
    return torch.mean(torch.sum(nll, dim=1))"""))

# ============================== TRAINING ==============================
cells.append(nbf.v4.new_markdown_cell("""## 4. Maximum Likelihood Calibration"""))

cells.append(nbf.v4.new_code_cell("""EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.002

optimizer = optim.Adam(
    list(drift_model.parameters()) + list(diff_model.parameters()), 
    lr=LEARNING_RATE
)

drift_model.train()
diff_model.train()
losses = []

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
    losses.append(epoch_loss)
    if epoch % 15 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{EPOCHS:03d} | Train Neg Log-Likelihood: {epoch_loss:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(losses, label='Train NLL Loss', color='darkorange')
plt.xlabel('Epoch')
plt.ylabel('Negative Log-Likelihood')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""))

# ============================== EVALUATION ==============================
cells.append(nbf.v4.new_markdown_cell("""## 5. Model Validation on Test Set"""))

cells.append(nbf.v4.new_code_cell("""drift_model.eval()
diff_model.eval()
with torch.no_grad():
    test_loss = neural_sde_loss(x_test, dx_test, drift_model, diff_model, DT).item()
print(f"Test Set Negative Log-Likelihood: {test_loss:.4f}")"""))

# ============================== SIMULATION ==============================
cells.append(nbf.v4.new_markdown_cell("""## 6. Monte Carlo Paths Simulation

We simulate the calibrated SDE model forward using the Euler-Maruyama scheme starting from the final historical data point."""))

cells.append(nbf.v4.new_code_cell("""x_start = torch.FloatTensor(data[-1]).unsqueeze(0)
N_PATHS = 100
N_STEPS = 60  # predict 60 trading days (~3 months) forward

simulated_paths = torch.zeros(N_PATHS, N_STEPS + 1, D_dim)
simulated_paths[:, 0, :] = x_start.repeat(N_PATHS, 1)

with torch.no_grad():
    for step in range(N_STEPS):
        curr_states = simulated_paths[:, step, :]
        drifts = drift_model(curr_states)
        vols = diff_model(curr_states)
        shocks = torch.randn(N_PATHS, D_dim)
        
        # SDE Euler Step
        next_states = curr_states + drifts * DT + vols * np.sqrt(DT) * shocks
        simulated_paths[:, step + 1, :] = next_states

sim_paths_np = simulated_paths.numpy()

# Visualise Simulated 5Y and 10Y maturities
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
time_axis = np.arange(N_STEPS + 1)
tenor_indices = [1, 2] # 5Y and 10Y indices
tenor_names = ["5 Year BIRS", "10 Year BIRS"]
colors = ['#FF9800', '#2196F3']

for i, (idx, name) in enumerate(zip(tenor_indices, tenor_names)):
    # History tail (last 60 days)
    lookback = 60
    hist_dates = np.arange(-lookback, 1)
    axes[i].plot(hist_dates, data[-lookback-1:, idx], color='black', label='Historical', linewidth=2)
    
    # Paths
    for p in range(min(50, N_PATHS)):
        axes[i].plot(time_axis, sim_paths_np[p, :, idx], color=colors[i], alpha=0.15, linewidth=0.8)
    
    # Path Quantiles
    axes[i].plot(time_axis, np.median(sim_paths_np[:, :, idx], axis=0), color='black', linestyle='--', linewidth=2.0, label='Median Forecast')
    axes[i].plot(time_axis, np.percentile(sim_paths_np[:, :, idx], 5, axis=0), color='red', linestyle=':', linewidth=1.5, label='5% / 95% Confidence')
    axes[i].plot(time_axis, np.percentile(sim_paths_np[:, :, idx], 95, axis=0), color='red', linestyle=':')
    
    axes[i].set_title(f"Neural SDE Simulation: {name}", fontweight='bold')
    axes[i].set_ylabel("Yield Rate (%)")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

axes[-1].set_xlabel("Steps (Trading Days Ahead)")
plt.tight_layout()
plt.show()"""))

# ============================== SUMMARY ==============================
cells.append(nbf.v4.new_markdown_cell("""## 📋 Summary

### What we implemented:
1. **Continuous-Time SDE Formulation:** Expressed daily yield changes as drift and diffusion outputs.
2. **Neural Parametrization:** Substituted rigid analytical drift/volatility formulas with flexible deep networks.
3. **Euler Monte Carlo Engine:** Generated SDE forecasts using random Gaussian innovation paths.
"""))

nb.cells = cells
nbf.write(nb, 'birs_neural_drift_diffusion.ipynb')
print("Created create_neural_sde_notebook.py and generated birs_neural_drift_diffusion.ipynb code.")

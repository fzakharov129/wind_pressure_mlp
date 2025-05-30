# 📦 Final MLP model configuration (Fluctuating Pressure, May 2025)
# - Architecture: 2 hidden layers, 128 neurons each
# - Activation function: Tanh
# - Optimizer: Adam
# - Learning rate: 6e-03
# - Batch size: 32
# - Evaluation on holdout set (~55,000 samples):
#     • R²: 0.847
#     • MAE: 0.0241
#     • MSE: 0.001076
# 📁 Location: final_models/model_stddev/


# mlp_architecture.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


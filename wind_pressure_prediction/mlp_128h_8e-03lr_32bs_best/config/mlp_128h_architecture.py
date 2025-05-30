"""
Architecture used in best MLP model (May 2025)
- 2 hidden layers
- 128 hidden units
- Tanh activation
- Adam optimizer
- Learning rate: 8e-03
- Batch size: 32
- Epochs: 
- R² on holdout: 0.9784
"""

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


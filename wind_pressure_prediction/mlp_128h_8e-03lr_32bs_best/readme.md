# 🧠 MLP Final Model — 128 Hidden Units, Tanh, Adam, LR=8e-03, BS=32

This folder contains the **best-performing MLP model** for wind pressure prediction on building facades.

---

## 📌 Model Summary

- **Architecture**: Multilayer Perceptron (2 hidden layers)
- **Hidden Units**: 128
- **Activation Function**: Tanh
- **Optimizer**: Adam
- **Learning Rate**: 8e-03
- **Batch Size**: 32
- **Epochs**: 2000 (for experiments)
- **Loss Function**: MSELoss
- **Input Features**: `['X_int', 'Y_int', 'X_fac', 'Y_fac', 'Ang']`
- **Output**: Mean wind pressure (`Mean`)

---

## 📊 Holdout Set Performance

| Metric     | Value    |
|------------|----------|
| R² Score   | 0.9784   |
| MAE        | 0.0518   |
| MSE        | 0.005328 |

---

## 📁 Folder Structure

mlp_128h_8e-03lr_32bs_best/
├── config/
│ ├── config_mlp_128h_8e-03lr_32bs_2000ep_Tanh_Adam_ex.json
│ └── mlp_128h_architecture.py
├── metrics/
│ ├── holdout_eval_128h_8e-03lr_32bs_2000ep_Tanh_Adam_ex.json
│ ├── stability_mlp_128h_8e-03lr_32bs_Tanh_Adam.json
│ └── metrics_best_mlp_128h_8e-03lr_32bs_Tanh_Adam.json
├── plots/
│ ├── holdout_plot_128h_8e-03lr_32bs_2000ep_Tanh_Adam_ex.png
│ └── stability_plot_mlp_128h_8e-03lr_32bs_Tanh_Adam.png
├── results/
│ └── results_mlp_128h_8e-03lr_32bs_Tanh_Adam.csv
├── weights/
│ └── best_model_weights_mlp_128h_8e-03lr_32bs_Tanh_Adam.pt
└── README.md


---

## 🛠️ Usage Example

```python
from mlp_architecture import MLP
import torch

# Build the model
model = MLP(input_dim=5, hidden_dim=128, output_dim=1, activation_fn=torch.nn.Tanh)

# Load weights
model.load_state_dict(torch.load("weights/best_model_weights_mlp_128h_8e-03lr_32bs_Tanh_Adam.pt", map_location="cpu"))
model.eval()

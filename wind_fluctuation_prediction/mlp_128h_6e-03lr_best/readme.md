# 🧠 MLP Final Model — 128 Hidden Units, Tanh, Adam, LR=6e-03, BS=16 (Target: StdDev, Raw Data)

This folder contains the **best-performing MLP model** for predicting **fluctuating wind pressure (StdDev)** on building facades, trained and tested on **raw (non-cleaned) data**.

---

## 📌 Model Summary

- **Architecture**: Multilayer Perceptron (2 hidden layers)
- **Hidden Units**: 128
- **Activation Function**: Tanh
- **Optimizer**: Adam
- **Learning Rate**: 6e-03
- **Batch Size**: 16
- **Epochs**: 10000
- **Loss Function**: MSELoss
- **Input Features**: `['X_int', 'Y_int', 'X_fac', 'Y_fac', 'Ang']`
- **Output**: Fluctuating wind pressure (`StdDev`)

---

## 📊 Holdout Set Performance

| Metric     | Value    |
|------------|----------|
| R² Score   | 0.847    |
| MAE        | 0.0241   |
| MSE        | 0.001076 |

> Evaluation based on a separate holdout set. No data cleaning applied.

---

## 📁 Folder Structure

mlp_128h_6e-03lr_16bs_best/
├── config/
│   └── config_mlp_128h_6e-03lr_ex.json
├── metrics/
│   ├── holdout_eval_128h_6e-03lr_ex.json
│   ├── stability_mlp_128h_6e-03lr_16bs_10000ep_Tanh_Adam.json
│   └── metrics_best_mlp_128h_6e-03lr_16bs_10000ep_Tanh_Adam.json
├── plots/
│   ├── holdout_plot_128h_6e-03lr_ex.png
│   └── stability_plot_mlp_128h_6e-03lr_16bs_10000ep_Tanh_Adam.png
├── results/
│   └── results_mlp_128h_6e-03lr_16bs_10000ep_Tanh_Adam.csv
├── weights/
│   └── best_model_weights_mlp_128h_6e-03lr_16bs_10000ep_Tanh_Adam.pt
└── README.md

---

## 🛠️ Usage Example

```python
from mlp_architecture import MLP
import torch

# Build the model
model = MLP(input_dim=5, hidden_dim=128, output_dim=1, activation_fn=torch.nn.Tanh)

# Load weights
model.load_state_dict(torch.load("weights/best_model_weights_mlp_128h_6e-03lr_16bs_10000ep_Tanh_Adam.pt", map_location="cpu"))
model.eval()

# 🌬️ Wind Pressure Prediction for High-Rise Buildings  
*Prediction of Mean Pressure Coefficient and Standard Deviation of Pressure Coefficient on Facades*

---

## 📌 Project Overview

This project represents the initial stage of developing a machine learning model for predicting wind effects on high-rise buildings.  
The goal is to forecast:

- **Mean pressure coefficient (Cp)**
- **Standard deviation of pressure coefficient (StdDev)**

at specified points on a building’s facade under different wind directions and varying positions of interfering buildings.

The baseline model architecture is a **Multi-Layer Perceptron (MLP)** with two hidden layers of equal size. This structure was selected for its simplicity, interpretability, and expressive capacity suitable for the initial development phase.

Model training was based on wind tunnel experiments performed on scaled physical building models of identical height and size.

**Dataset size:**
- Total: `153,216` rows  
- After preprocessing:
  - Training set: `121,114` rows  
  - Holdout set: `30,279` rows

> For the StdDev prediction model, outlier removal was not applied. All raw values were used to preserve realistic variability and avoid distorting natural pressure fluctuations. This decision improves generalization when working with raw engineering data.

---

## 🗂️ Repository Summary

This repository includes two independently trained MLP-based models for predicting wind pressure characteristics on high-rise facades:

### 1. Model 1 — **Mean Pressure Coefficient (Cp)**
Predicts the average pressure coefficient at a given facade point under different flow conditions.

### 2. Model 2 — **Standard Deviation of Pressure Coefficient (StdDev)**
Predicts the variability of pressure using the standard deviation of the coefficient, representing turbulence intensity.

Both models use the **same input features** but were trained and validated **independently**.

Each model archive includes:
- 🔧 Model configuration (hyperparameters and architecture)
- 📜 Training and evaluation logs
- 📊 Performance metrics on the holdout set (MAE, MSE, R²)
- 🧠 Model weights
- 📈 Visualization of key metrics
- 📉 Stability analysis charts over 50 random train/test splits

---

## 📑 Dataset Description

The dataset was obtained from wind tunnel testing simulating wind flow around building groups.

Each row corresponds to a pressure measurement at a specific point on the facade, under a given wind direction and configuration of the interfering building.

**Input Features:**
- `X_int`, `Y_int`: coordinates of the interfering building
- `X_fac`, `Y_fac`: coordinates of the pressure measurement point
- `Ang`: wind direction in degrees

**Target Variables:**
- `Cp`: Mean pressure coefficient (Model 1)
- `StdDev`: Standard deviation of pressure coefficient (Model 2)

**Dataset Size:**
- Total: `153,216` rows  
- Training: `121,114` rows  
- Holdout: `30,279` rows

---

## 📊 Results

Both models achieved strong performance on the holdout set. Below are the final metrics and stability test results based on 50 randomized train/test splits.

---

### 🔹 Model 1 — Mean Pressure Coefficient (Cp)

**Holdout Metrics:**
- **R²**: 0.9784  
- **MAE**: 0.0518  
- **MSE**: 0.005328  

**Stability over 50 runs:**
- R² mean: **0.9651** ± 0.0089  
- MAE mean: **0.0644** ± 0.0082  
- MSE mean: **0.008654** ± 0.002215  

---

### 🔹 Model 2 — Standard Deviation of Pressure Coefficient (StdDev)

**Holdout Metrics:**
- **R²**: 0.847  
- **MAE**: 0.0241  
- **MSE**: 0.001076  

**Stability over 50 runs:**
- R² mean: **0.7071** ± 0.1372  
- MAE mean: **0.0316** ± 0.0072  
- MSE mean: **0.002034** ± 0.000954  

---

## 🔗 Access

Final models, training notebooks, and metric visualizations are available at:

- 📂 **GitHub Repository:**  
  [https://github.com/fzakharov129/wind_pressure_mlp](https://github.com/fzakharov129/wind_pressure_mlp)

- 📊 **Kaggle Page:**  
  [https://www.kaggle.com/datasets/fedorzakharov331/wind-pressure-mlp](https://www.kaggle.com/datasets/fedorzakharov331/wind-pressure-mlp)

---

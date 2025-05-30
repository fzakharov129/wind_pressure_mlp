Wind Pressure Prediction for High-Rise Buildings
(Prediction of Mean Pressure Coefficient and Standard Deviation of Pressure Coefficient on Facades)

Project Overview
This project represents the initial stage of developing a machine learning model for predicting wind effects on high-rise buildings.
The goal is to forecast:

•	Mean pressure coefficient (Cp)
•	Standard deviation of pressure coefficient (StdDev)

in specified points on a building’s facade under different wind directions and varying positions of interfering buildings.

The baseline model architecture is a Multi-Layer Perceptron (MLP) with two hidden layers of equal size. This structure was selected due to its simplicity, interpretability, and sufficient expressive power for the initial development phase. Model training was performed using the results of wind tunnel experiments on physical building models of identical height and dimensions.

Dataset size:
Total: 153,216 rows
After preprocessing:
•	Training set: 121,114 rows
•	Holdout set: 30,279 rows

For the model predicting the standard deviation of pressure coefficient (StdDev), the training and test sets were not filtered for outliers. All values were used in their original form to preserve realistic variability and avoid distorting the nature of pressure fluctuations. This decision was made intentionally to improve the model’s generalization ability when working with raw engineering data.

Repository Summary
This repository includes two independently trained machine learning models based on the MLP architecture, designed to predict wind pressure characteristics on high-rise building facades:
1.	Model 1: Mean Pressure Coefficient (Cp)
Predicts the average pressure coefficient at a given facade point under different flow conditions.
2.	Model 2: Standard Deviation of Pressure Coefficient (StdDev)
Predicts the variability of pressure using the standard deviation of the coefficient, representing turbulence intensity at the measurement point.
Both models use the same set of input features but were trained and validated independently.
Each model archive includes:
•	Model configuration (hyperparameters, architecture)
•	Training and evaluation logs
•	Performance metrics on the holdout set (MAE, MSE, R²)
•	Model weights
•	Visualization of metrics
•	Stability analysis charts for different subsets

Dataset Description
The dataset was obtained from wind tunnel tests simulating wind flow around building groups.
Each row represents a single pressure measurement at a specific facade point under a given wind direction and position of an interfering building.
Input features:
•	X_int, Y_int: coordinates of the interfering building
•	X_fac, Y_fac: coordinates of the pressure measurement point on the main facade
•	Ang: wind direction (degrees)
Target variables:
•	Cp: Mean pressure coefficient (for Model 1)
•	StdDev: Standard deviation of pressure coefficient (for Model 2)
Dataset size:
•	Total: 153,216 rows
•	Training set: 121,114 rows
•	Holdout set: 30,279 rows

Results
Both models demonstrated high prediction accuracy on the holdout set.
Below are the final evaluation metrics and stability test results (across 50 training set variations).
1) Model 1 — Cp (Mean Pressure Coefficient)
Holdout Metrics:
•	R²: 0.9784
•	MAE: 0.0518
•	MSE: 0.005328
 

Stability (50 runs):
•	R² mean: 0.9651 ± 0.0089
•	MAE mean: 0.0644 ± 0.0082
•	MSE mean: 0.008654 ± 0.002215
 

2) Model 2 — StdDev (Standard Deviation of Pressure Coefficient)
Holdout Metrics:
•	R²: 0.847
•	MAE: 0.0241
•	MSE: 0.001076
 

Stability (50 runs):
•	R² mean: 0.7071 ± 0.1372
•	MAE mean: 0.0316 ± 0.0072
•	MSE mean: 0.002034 ± 0.000954

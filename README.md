# Boston House Price Prediction — End-to-End Machine Learning Project

This project builds a complete end-to-end Machine Learning pipeline to predict housing prices in Boston using the classic Boston Housing Dataset.
It covers exploratory analysis, data preprocessing, model training, cross-validation, performance evaluation, and exporting the final model for production use.


# Objective

The goal of this project is to develop a regression model capable of predicting the median value of a home (MEDV) based on different neighborhood, environmental and socio-economic features, such as:
	•	number of rooms
	•	crime rate
	•	pollution levels
	•	accessibility to main roads and services
	•	education quality (student–teacher ratio)
	•	socio-economic status


# Dataset Overview

Source: Boston Housing (Kaggle)
Samples: 506
Features: 14
Target: MEDV (Median home value)

Most influential variables:
	•	RM (average number of rooms) → strong positive correlation
	•	LSTAT (% lower socio-economic status) → strong negative correlation
	•	PTRATIO, TAX, INDUS, NOX → moderate negative correlations
	•	ZN, DIS, CHAS → mild positive correlations

These relationships were confirmed through visual inspection using scatterplots and a correlation heatmap.


# Models Tested

The following models were implemented and compared:
	•	Linear Regression → baseline
	•	Ridge Regression (with cross-validation) → slight improvement
	•	Random Forest Regressor → best performance


How to Run the Project:
Install dependencies:
pip install -r requirements.txt

# Train Model:
python src/train.py

Run infenrence with the trained model:
python src/inference.py


Developed by Ariel Nuñez Valencia
Data Science & Artificial Intelligence Student
Focused on building clean, reproducible and production-oriented ML solutions.


# Heart Disease Prediction

A machine learning classification project that predicts the likelihood of heart disease using clinical patient data from the UCI Heart Disease dataset.

## Overview

This project demonstrates a complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, and model serialization. Two classification algorithms were implemented and compared to determine the best-performing model.

## Dataset

- Dataset: UCI Heart Disease Dataset
- Records: 920
- Features: 14 clinical attributes
- Target: Presence or absence of heart disease

## Workflow

- Data preprocessing
  - Missing value imputation
  - Categorical feature encoding
  - Feature scaling
- Train-test split
- Model training
  - Logistic Regression
  - Decision Tree Classifier
- Performance evaluation
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - ROC Curve
- Model serialization using Joblib

## Results

| Model | Accuracy |
|------|---------:|
| Logistic Regression | **79.89%** |
| Decision Tree | **78.26%** |

**Best Performing Model:** Logistic Regression

**ROC-AUC Score:** 0.88

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Structure

```text
Heart-Disease-Prediction/
│
├── Heart_Disease_Prediction.ipynb
├── heart_disease_uci.csv
├── requirements.txt
├── README.md
├── heart_disease_model.pkl
└── scaler.pkl
```

## Getting Started

Clone the repository

```bash
git clone https://github.com/saktronX/Heart-Disease-Prediction.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `Heart_Disease_Prediction.ipynb` and run all cells.

## Results

| Metric | Value |
|---------|-------|
| Model | Logistic Regression |
| Accuracy | 79.89% |
| ROC-AUC Score | 0.88 |

### ROC Curve

![ROC Curve](/roc_curve.png)

## Future Improvements

- Hyperparameter tuning
- Ensemble learning methods
- Cross-validation
- Streamlit web application for inference

## Author

**Saksham Verma**

GitHub: https://github.com/saktronX

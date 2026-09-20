# Loan Approval Prediction – Machine Learning Deployment

A Streamlit web application for predicting loan approval/rejection using the Loan Approval Classification project.

## Project Overview

- **Problem Type:** Supervised Machine Learning – Classification
- **Target:** `loan_status`
- **Best Model:** XGBoost
- **Best Parameters:** `learning_rate=0.1`, `max_depth=5`, `n_estimators=300`
- **Test Accuracy:** approximately **98.48%**

## Project Workflow

Business Problem → Data Understanding → EDA → Preprocessing → Model Comparison → Hyperparameter Tuning → Evaluation → Prediction → Deployment

## Files

```text
loan_approval_github/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── AI-ML Project -1(2).ipynb
└── data/
    └── .gitkeep
```

## Dataset

The application expects the file:

`loan_approval_dataset_classification.csv`

Upload the CSV using the Streamlit sidebar. The CSV is intentionally not included in this repository package because the dataset was not provided with the deployment files.

## Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, the notebook, and optionally the dataset if you are allowed to publish it.
3. Open Streamlit Community Cloud.
4. Connect your GitHub account and select this repository.
5. Select `app.py` as the main file.
6. Click **Deploy**.
7. Open the generated public app URL.
8. Upload `loan_approval_dataset_classification.csv` in the app sidebar if the dataset is not stored in the repository.

## Main App Sections

1. Overview
2. Dataset
3. EDA
4. Preprocessing
5. Model Comparison
6. Best Model
7. Evaluation
8. Feature Importance
9. Loan Prediction
10. Conclusion

## Models Compared

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
- AdaBoost
- XGBoost

## Disclaimer

This project is for educational and demonstration purposes. A machine-learning prediction should not be used as the sole basis for real-world lending decisions.

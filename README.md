📞 Telecom Customer Churn Prediction
End-to-End Machine Learning Project
📌 Project Overview

This project predicts customer churn for a telecom company using machine learning.
It includes data cleaning, EDA, preprocessing pipelines, model training, evaluation, and best-model selection.
Models used: Logistic Regression, Random Forest, XGBoost.

📂 Dataset

Source: Telco Customer Churn dataset
File: WA_Fn-UseC_-Telco-Customer-Churn.csv

Rows: 7,043
Columns: 21

Target variable:

Churn → Yes / No

🧹 Data Cleaning

✔ Converted TotalCharges from string → numeric
✔ Removed rows with missing TotalCharges
✔ Dropped customerID
✔ Encoded target (Yes → 1, No → 0)

🔍 Exploratory Data Analysis (EDA)

The notebook includes visualizations for:

📊 Churn Distribution
📈 Tenure vs Churn
📦 Monthly Charges vs Churn
📝 Contract Types
🌐 Internet Service Categories
🔥 Correlation Heatmap (Numeric Features)

Key insights:

Customers with month-to-month contracts churn more.

Lower tenure ≈ higher churn.

Higher monthly charges correlate with churn.

⚙️ Preprocessing

Preprocessing is done using scikit-learn Pipelines + ColumnTransformer:

Numeric Features:

tenure, MonthlyCharges, TotalCharges
→ Scaled with StandardScaler

Categorical Features:
→ One-Hot Encoding (ignore unknown categories)

🤖 Machine Learning Models

The project trains 3 models:

1️⃣ Logistic Regression
2️⃣ Random Forest Classifier
3️⃣ XGBoost Classifier

Each model is wrapped inside a pipeline:

Pipeline([
    ("preprocessor", ColumnTransformer),
    ("model", ML model)
])

🏆 Model Evaluation

Evaluation metrics used:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Confusion Matrix

--> Best model is selected based on ROC-AUC.

📈 Results

Metrics printed for all models.
A confusion matrix is displayed for the best-performing model.

🚀 How to Run the Project
pip install -r requirements.txt
python Customer Churn.py






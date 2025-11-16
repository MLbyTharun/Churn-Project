# 0. IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# -------------------------------
# 2. DATA PREVIEW
# -------------------------------
print("\n--- HEAD ---")
print(df.head())

print("\n--- INFO ---")
print(df.info())

print("\n--- DESCRIBE ---")
print(df.describe())
# -------------------------------
# 3. CLEANING — Fix TotalCharges
# -------------------------------
# Convert to numeric (it has empty spaces)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

print("\nMissing TotalCharges:", df["TotalCharges"].isna().sum())

# Drop missing rows
df = df.dropna(subset=["TotalCharges"])
df.reset_index(drop=True, inplace=True)

print("Shape After Cleaning:", df.shape)

# -------------------------------
# 4. CHURN DISTRIBUTION
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Churn", palette="viridis")
plt.title("Churn Distribution")
plt.show()

# -------------------------------
# 5. TENURE DISTRIBUTION vs CHURN
# -------------------------------
plt.figure(figsize=(8,4))
sns.histplot(data=df, x="tenure", hue="Churn", kde=True, palette="viridis")
plt.title("Tenure vs Churn")
plt.show()

# -------------------------------
# 6. MONTHLY CHARGES vs CHURN
# -------------------------------
plt.figure(figsize=(8,4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette="viridis")
plt.title("Monthly Charges vs Churn")
plt.show()
#------------------------------
# 7. CONTRACT TYPE vs CHURN
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Contract", hue="Churn", palette="viridis")
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.show()# -------------------------------
# 8. INTERNET SERVICE vs CHURN
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="InternetService", hue="Churn", palette="viridis")
plt.title("Internet Service vs Churn")
plt.show()

# -------------------------------
# 9. NUMERIC CORRELATION HEATMAP
# -------------------------------
plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap="viridis")
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()
# ============================================================
#                 FULL PREPROCESSING BLOCK
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------------------
# 1. DROP UNUSED COLUMNS
# -------------------------------
df_clean = df.copy()

df_clean = df_clean.drop(["customerID"], axis=1)

# -------------------------------
# 2. SEPARATE FEATURES & TARGET
# -------------------------------
X = df_clean.drop("Churn", axis=1)
y = df_clean["Churn"].map({"Yes": 1, "No": 0})   # convert to 0/1

# -------------------------------
# 3. NUMERIC & CATEGORICAL COLUMNS
# -------------------------------
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [col for col in X.columns if col not in numeric_features]

print("Numeric:", numeric_features)
print("Categorical:", categorical_features)

# -------------------------------
# 4. PREPROCESSING: ENCODING + SCALING
# -------------------------------
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -------------------------------
# 5. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ============================================================
#                 FULL MODELING BLOCK
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
)

import pickle
import matplotlib.pyplot as plt

# -------------------------------
# 1. MODELS TO TRAIN
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5, eval_metric="logloss"
    )
}

# -------------------------------
# 2. TRAIN MODELS + EVALUATE
# -------------------------------
results = {}

for name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "model": clf
    }

# -------------------------------
# 3. DISPLAY RESULTS
# -------------------------------
print("\n=== MODEL PERFORMANCE ===")
for name, metrics in results.items():
    print(f"\n{name}:")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v:.4f}")

# -------------------------------
# 4. CONFUSION MATRIX FOR BEST MODEL
# -------------------------------
best_model_name = max(results, key=lambda x: results[x]["roc_auc"])
best_model = results[best_model_name]["model"]

print(f"\nBest Model: {best_model_name}")

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6,4))
ConfusionMatrixDisplay(cm).plot(cmap="viridis")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()

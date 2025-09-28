import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Paths
DATA_PATH = r"C:\Users\eshra\heart-disease-prediction\data\framingham.csv"  # adjust if needed
MODEL_DIR = "../model"

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Features & target
    FEATURES = [
        'age','male','currentSmoker','cigsPerDay','totChol','sysBP',
        'diaBP','BMI','heartRate','glucose','diabetes','prevalentHyp'
    ]
    TARGET = 'TenYearCHD'

    df = df[FEATURES + [TARGET]].dropna()

    X = df[FEATURES].values
    y = df[TARGET].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.30, random_state=42, stratify=y
    )

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model & scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "logistic_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print("✅ Model and scaler saved in", MODEL_DIR)

if __name__ == "__main__":
    main()

# ==============================
# MODEL MODULE (FIXED VERSION)
# ==============================

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except:
    print("XGBoost not installed. Skipping...")
    xgb_available = False


# ==============================
# MAIN FUNCTION (IMPORTANT)
# ==============================

def run_models(X_train, X_test, y_train, y_test):

    print("🚀 MODEL TRAINING STARTED\n")

    # ==============================
    # LOGISTIC REGRESSION
    # ==============================
    print("==== LOGISTIC REGRESSION ====")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)

    y_pred_log = log_model.predict(X_test_scaled)

    log_acc = accuracy_score(y_test, y_pred_log)

    print(f"Accuracy: {log_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_log))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_log))

    if len(np.unique(y_train)) == 2:
        y_prob = log_model.predict_proba(X_test_scaled)[:, 1]

        # ======================================
        # SAVE MODEL PREDICTIONS
        # ======================================

        predictions_df = X_test.copy()

        predictions_df["actual_result"] = y_test.values
        predictions_df["predicted_result"] = y_pred_log
        predictions_df["home_win_probability"] = y_prob

        # Create outputs folder
        os.makedirs("outputs", exist_ok=True)

        # Save CSV
        predictions_df.to_csv(
            "outputs/predictions_with_probabilities.csv",
            index=False
        )

        print("\n✅ Prediction output file saved successfully!")

        auc = roc_auc_score(y_test, y_prob)
        print(f"AUC Score: {auc:.4f}")

    # ==============================
    # RANDOM FOREST
    # ==============================
    print("\n==== RANDOM FOREST ====")

    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)

    rf_acc = accuracy_score(y_test, y_pred_rf)

    print(f"Accuracy: {rf_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    # Feature Importance
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": rf_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    os.makedirs("outputs", exist_ok=True)
    feature_importance.to_csv("outputs/feature_importance.csv", index=False)

    # ==============================
    # XGBOOST
    # ==============================
    if xgb_available:
        print("\n==== XGBOOST ====")

        xgb_model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric='logloss'
        )

        xgb_model.fit(X_train, y_train)

        y_pred_xgb = xgb_model.predict(X_test)

        xgb_acc = accuracy_score(y_test, y_pred_xgb)

        print(f"Accuracy: {xgb_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_xgb))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_xgb))
    else:
        xgb_acc = None

    # ==============================
    # MODEL COMPARISON
    # ==============================
    print("\n==== MODEL COMPARISON ====")

    print(f"Logistic Regression Accuracy: {log_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    if xgb_acc:
        print(f"XGBoost Accuracy: {xgb_acc:.4f}")

    best_model = "Logistic Regression"
    best_score = log_acc

    if rf_acc > best_score:
        best_model = "Random Forest"
        best_score = rf_acc

    if xgb_acc and xgb_acc > best_score:
        best_model = "XGBoost"
        best_score = xgb_acc

    print(f"\n🏆 Best Model: {best_model} ({best_score:.4f})")

    print("\n✅ MODEL TRAINING COMPLETED")

    return best_model, best_score
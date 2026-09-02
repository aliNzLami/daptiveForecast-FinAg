import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import shap
import lime
import lime.lime_tabular

warnings.filterwarnings('ignore')

def load_data():
    base_dir = os.getcwd()
    transitions_path = os.path.join(base_dir, "dataset", "transition_points.csv")
    windows_path = os.path.join(base_dir, "dataset", "window_features.csv")
    main_path = os.path.join(base_dir, "dataset", "US_Agriculture_Weather_2010_2024.csv")

    if not os.path.exists(transitions_path):
        raise FileNotFoundError(f"transition_points.csv not found at {transitions_path}")
    if not os.path.exists(windows_path):
        raise FileNotFoundError(f"window_features.csv not found at {windows_path}")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"US_Agriculture_Weather_2010_2024.csv not found at {main_path}")

    df_transitions = pd.read_csv(transitions_path)
    df_transitions["transition_date"] = pd.to_datetime(df_transitions["transition_date"])
    df_transitions["window_start"] = pd.to_datetime(df_transitions["window_start"])
    df_transitions["window_end"] = pd.to_datetime(df_transitions["window_end"])

    df_windows = pd.read_csv(windows_path)
    df_windows["window_start"] = pd.to_datetime(df_windows["window_start"])
    df_windows["window_end"] = pd.to_datetime(df_windows["window_end"])
    df_windows["window_center"] = pd.to_datetime(df_windows["window_center"])

    df_main = pd.read_csv(main_path)
    df_main["Date"] = pd.to_datetime(df_main["Date"])
    df_main = df_main.sort_values("Date").reset_index(drop=True)

    return df_transitions, df_windows, df_main

def compute_regime_for_window(row, price_dict):
    end_date = row["window_end"]
    future_end = end_date + pd.Timedelta(days=30)
    if future_end not in price_dict:
        return None
    current_price = row["corn_mean"]
    future_price = price_dict[future_end]
    if current_price <= 0:
        return None
    ret = (future_price - current_price) / current_price
    if ret > 0.03:
        return "Bullish"
    elif ret < -0.03:
        return "Bearish"
    else:
        return "Neutral"

def extract_unique_windows(df_transitions, df_windows, days_window=30):
    selected_windows = []
    for _, trans in df_transitions.iterrows():
        trans_date = trans["transition_date"]
        mask = (df_windows["window_center"] >= trans_date - pd.Timedelta(days=days_window)) & \
               (df_windows["window_center"] <= trans_date + pd.Timedelta(days=days_window))
        nearby = df_windows[mask].copy()
        if not nearby.empty:
            nearby["transition_date"] = trans_date
            selected_windows.append(nearby)

    if not selected_windows:
        return pd.DataFrame()

    df_all = pd.concat(selected_windows, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["window_start", "window_end"])
    df_all = df_all.sort_values("window_start").reset_index(drop=True)
    return df_all

def extract_days_from_windows(df_windows, df_main):
    price_dict = dict(zip(df_main["Date"], df_main["Corn_Price_USD"]))
    all_days = []

    for idx, window_row in df_windows.iterrows():
        start = window_row["window_start"]
        end = window_row["window_end"]
        mask = (df_main["Date"] >= start) & (df_main["Date"] <= end)
        daily_chunk = df_main[mask].copy()
        if daily_chunk.empty:
            continue

        regime = compute_regime_for_window(window_row, price_dict)
        if regime is None:
            continue

        daily_chunk["window_start"] = start
        daily_chunk["window_end"] = end
        daily_chunk["regime"] = regime
        daily_chunk["transition_date"] = window_row.get("transition_date", None)

        all_days.append(daily_chunk)

    if not all_days:
        return pd.DataFrame()

    df_days = pd.concat(all_days, ignore_index=True)
    df_days = df_days.drop_duplicates(subset=["Date", "window_start"]).reset_index(drop=True)
    df_days = df_days.sort_values("Date").reset_index(drop=True)

    return df_days

def prepare_features(df_days):
    feature_cols = ["Max_Temp_C", "Min_Temp_C", "Precipitation_mm"]
    X = df_days[feature_cols].values
    y = df_days["regime"].values
    return X, y, feature_cols

def main():
    print("=" * 70)
    print("TRANSITION POINT ANALYSIS - DAILY DATA APPROACH")
    print("=" * 70)

    print("\n[1] Loading data...")
    df_transitions, df_windows, df_main = load_data()
    print(f"    Transitions: {len(df_transitions)}")
    print(f"    Windows: {len(df_windows)}")
    print(f"    Daily records: {len(df_main)}")

    print("\n[2] Extracting unique windows around transition points...")
    df_unique_windows = extract_unique_windows(df_transitions, df_windows, days_window=30)
    print(f"    Unique windows found: {len(df_unique_windows)}")

    if df_unique_windows.empty:
        print("    No windows found. Exiting.")
        return

    print("\n[3] Extracting all days from these windows...")
    df_days = extract_days_from_windows(df_unique_windows, df_main)
    print(f"    Total days extracted: {len(df_days)}")

    if df_days.empty:
        print("    No days extracted. Exiting.")
        return

    print("\n[4] Class distribution:")
    class_dist = df_days["regime"].value_counts()
    for cls, count in class_dist.items():
        print(f"    {cls}: {count} ({count/len(df_days)*100:.1f}%)")

    X, y, feature_cols = prepare_features(df_days)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n[5] Training Random Forest classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_scaled, y_enc)
    y_pred_rf = rf_model.predict(X_scaled)
    acc_rf = accuracy_score(y_enc, y_pred_rf)
    print(f"    Accuracy: {acc_rf:.4f}")
    print("\n    Classification Report:")
    print(classification_report(y_enc, y_pred_rf, target_names=le.classes_))

    print("\n[6] Training KNN classifier...")
    k = min(5, max(1, len(X_scaled) - 1))
    knn_model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn_model.fit(X_scaled, y_enc)
    y_pred_knn = knn_model.predict(X_scaled)
    acc_knn = accuracy_score(y_enc, y_pred_knn)
    print(f"    Accuracy: {acc_knn:.4f}")
    print("\n    Classification Report:")
    print(classification_report(y_enc, y_pred_knn, target_names=le.classes_))

    print("\n[7] Computing SHAP values...")
    try:
        explainer_shap = shap.TreeExplainer(rf_model)
        shap_values = explainer_shap.shap_values(X_scaled)

        if isinstance(shap_values, list):
            shap_importance_per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
            shap_importance_mean = np.mean(shap_importance_per_class, axis=0)
        else:
            shap_importance_mean = np.abs(shap_values).mean(axis=0)

        if shap_importance_mean.shape[0] != len(feature_cols):
            raise ValueError(f"SHAP importance shape mismatch: {shap_importance_mean.shape[0]} vs {len(feature_cols)}")

        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "shap_importance": shap_importance_mean
        }).sort_values("shap_importance", ascending=False)

        shap_df["shap_percentage"] = (shap_df["shap_importance"] / shap_df["shap_importance"].sum()) * 100

        print("\n    SHAP Feature Importance:")
        print(shap_df.to_string(index=False))

    except Exception as e:
        print(f"    SHAP failed: {e}")
        shap_df = pd.DataFrame({"feature": feature_cols, "shap_importance": [0]*len(feature_cols)})

    print("\n[8] Computing LIME explanations...")
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_scaled,
            feature_names=feature_cols,
            class_names=le.classes_,
            mode='classification',
            verbose=False,
            discretize_continuous=False
        )

        sample_size = min(10, len(X_scaled))
        sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)

        lime_importance = {}
        lime_results = []

        for idx in sample_indices:
            exp = lime_explainer.explain_instance(
                X_scaled[idx],
                rf_model.predict_proba,
                num_features=len(feature_cols)
            )
            exp_dict = exp.as_list()
            lime_results.append({
                "sample_index": int(idx),
                "true_regime": le.inverse_transform([y_enc[idx]])[0],
                "predicted_regime": le.inverse_transform([rf_model.predict([X_scaled[idx]])[0]])[0],
                "explanations": exp_dict
            })

            for feature, weight in exp_dict:
                if feature not in lime_importance:
                    lime_importance[feature] = []
                lime_importance[feature].append(weight)

        for feature in lime_importance:
            lime_importance[feature] = np.mean(lime_importance[feature])

        lime_df = pd.DataFrame({
            "feature": list(lime_importance.keys()),
            "lime_importance": list(lime_importance.values())
        }).sort_values("lime_importance", ascending=False)

        lime_df["lime_percentage"] = (lime_df["lime_importance"] / lime_df["lime_importance"].sum()) * 100

        print("\n    LIME Feature Importance:")
        print(lime_df.to_string(index=False))

    except Exception as e:
        print(f"    LIME failed: {e}")
        lime_df = pd.DataFrame({"feature": feature_cols, "lime_importance": [0]*len(feature_cols)})
        lime_results = []

    print("\n[9] Saving outputs...")
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    shap_output = os.path.join(output_dir, "shap_importance_daily_transitions.csv")
    shap_df.to_csv(shap_output, index=False)
    print(f"    SHAP importance: {shap_output}")

    lime_output = os.path.join(output_dir, "lime_importance_daily_transitions.csv")
    lime_df.to_csv(lime_output, index=False)
    print(f"    LIME importance: {lime_output}")

    combined_df = pd.merge(shap_df, lime_df, on="feature", how="outer").fillna(0)
    if "shap_importance" in combined_df.columns and "lime_importance" in combined_df.columns:
        combined_df["shap_rank"] = combined_df["shap_importance"].rank(ascending=False)
        combined_df["lime_rank"] = combined_df["lime_importance"].rank(ascending=False)
        combined_df["avg_rank"] = (combined_df["shap_rank"] + combined_df["lime_rank"]) / 2
        combined_df = combined_df.sort_values("avg_rank")

        combined_output = os.path.join(output_dir, "feature_importance_combined_daily.csv")
        combined_df.to_csv(combined_output, index=False)
        print(f"    Combined importance: {combined_output}")

    if lime_results:
        lime_json_output = os.path.join(output_dir, "lime_explanations_daily.json")
        with open(lime_json_output, "w") as f:
            json.dump(lime_results, f, indent=2, default=str)
        print(f"    LIME explanations: {lime_json_output}")

    df_output = df_days.copy()
    df_output["predicted_regime_rf"] = le.inverse_transform(y_pred_rf)
    df_output["predicted_regime_knn"] = le.inverse_transform(y_pred_knn)
    df_output["rf_correct"] = df_output["regime"] == df_output["predicted_regime_rf"]
    df_output["knn_correct"] = df_output["regime"] == df_output["predicted_regime_knn"]

    full_output = os.path.join(output_dir, "daily_data_with_predictions.csv")
    df_output.to_csv(full_output, index=False)
    print(f"    Full predictions: {full_output}")

    report = {
        "total_transitions": len(df_transitions),
        "unique_windows_analyzed": len(df_unique_windows),
        "total_days_analyzed": len(df_days),
        "class_distribution": dict(class_dist),
        "random_forest_accuracy": float(acc_rf),
        "knn_accuracy": float(acc_knn),
        "shap_feature_importance": shap_df.to_dict(orient="records"),
        "lime_feature_importance": lime_df.to_dict(orient="records")
    }

    report_path = os.path.join(output_dir, "analysis_report_daily.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"    Report: {report_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total transitions: {len(df_transitions)}")
    print(f"Unique windows: {len(df_unique_windows)}")
    print(f"Daily samples: {len(df_days)}")
    print(f"Random Forest accuracy: {acc_rf:.4f}")
    print(f"KNN accuracy: {acc_knn:.4f}")
    print("\nTop 5 features (SHAP):")
    print(shap_df.head(5).to_string(index=False))
    print("\nTop 5 features (LIME):")
    print(lime_df.head(5).to_string(index=False))
    print("\nDone.")

if __name__ == "__main__":
    main()

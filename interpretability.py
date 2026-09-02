import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
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

def compute_regime_labels(df_windows, df_main):
    price_dict = dict(zip(df_main["Date"], df_main["Corn_Price_USD"]))
    df_windows["future_return"] = None

    for idx, row in df_windows.iterrows():
        end_date = row["window_end"]
        future_end = end_date + pd.Timedelta(days=30)
        if future_end in price_dict:
            current_price = row["corn_mean"]
            future_price = price_dict[future_end]
            if current_price > 0:
                df_windows.loc[idx, "future_return"] = (future_price - current_price) / current_price

    df_windows = df_windows.dropna(subset=["future_return"])

    def classify_regime(ret):
        if ret > 0.03:
            return "Bullish"
        elif ret < -0.03:
            return "Bearish"
        else:
            return "Neutral"

    df_windows["regime"] = df_windows["future_return"].apply(classify_regime)
    return df_windows

def extract_unique_windows(df_transitions, df_windows):
    selected_windows = []
    for _, trans in df_transitions.iterrows():
        trans_date = trans["transition_date"]
        mask = (df_windows["window_center"] >= trans_date - pd.Timedelta(days=30)) & \
               (df_windows["window_center"] <= trans_date + pd.Timedelta(days=30))
        nearby = df_windows[mask].copy()
        nearby["transition_date"] = trans_date
        selected_windows.append(nearby)

    if not selected_windows:
        return pd.DataFrame()

    df_all = pd.concat(selected_windows, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["window_start", "window_end"])
    df_all = df_all.sort_values("window_start").reset_index(drop=True)

    return df_all

def main():
    print("Loading data...")
    df_transitions, df_windows, df_main = load_data()

    print(f"Found {len(df_transitions)} transition points.")

    print("Extracting unique windows around transitions...")
    df_unique_windows = extract_unique_windows(df_transitions, df_windows)

    if df_unique_windows.empty:
        print("No windows found around transitions.")
        return

    print(f"Found {len(df_unique_windows)} unique windows.")

    print("Computing regime labels for each window...")
    df_unique_windows = compute_regime_labels(df_unique_windows, df_main)

    df_unique_windows = df_unique_windows.dropna(subset=["regime"])

    if df_unique_windows.empty:
        print("No windows with valid regime labels.")
        return

    print(f"Windows with valid regime labels: {len(df_unique_windows)}")

    feature_cols = [
        "corn_mean", "corn_std", "corn_skew", "corn_kurtosis",
        "temp_mean", "temp_std", "temp_skew", "temp_kurtosis",
        "volatility", "corr_price_temp", "precip_mean"
    ]

    X = df_unique_windows[feature_cols].values
    y = df_unique_windows["regime"].values

    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nTraining Random Forest classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y_enc)
    y_pred_rf = rf_model.predict(X_scaled)
    acc_rf = accuracy_score(y_enc, y_pred_rf)
    print(f"Random Forest accuracy: {acc_rf:.4f}")
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_enc, y_pred_rf, target_names=le.classes_))

    print("\nTraining KNN classifier...")
    n_samples = len(X_scaled)
    k = min(5, max(1, n_samples - 1))
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_scaled, y_enc)
    y_pred_knn = knn_model.predict(X_scaled)
    acc_knn = accuracy_score(y_enc, y_pred_knn)
    print(f"KNN accuracy: {acc_knn:.4f}")
    print("\nClassification Report (KNN):")
    print(classification_report(y_enc, y_pred_knn, target_names=le.classes_))

    print("\nComputing SHAP values for Random Forest...")
    explainer_shap = shap.TreeExplainer(rf_model)
    shap_values = explainer_shap.shap_values(X_scaled)

    # پردازش صحیح shap_values
    if isinstance(shap_values, list):
        shap_importance_per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
        shap_importance_mean = np.mean(shap_importance_per_class, axis=0)
    else:
        shap_importance_mean = np.abs(shap_values).mean(axis=0)

    shap_importance_mean = np.array(shap_importance_mean).flatten()

    if len(shap_importance_mean) != len(feature_cols):
        raise ValueError(f"Length mismatch: shap_importance_mean ({len(shap_importance_mean)}) != feature_cols ({len(feature_cols)})")

    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_importance": shap_importance_mean
    }).sort_values("shap_importance", ascending=False)

    shap_df["shap_percentage"] = (shap_df["shap_importance"] / shap_df["shap_importance"].sum()) * 100

    print("\nComputing LIME explanations...")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_scaled,
        feature_names=feature_cols,
        class_names=le.classes_,
        mode='classification',
        verbose=False
    )

    lime_results = []
    sample_indices = np.random.choice(len(X_scaled), min(10, len(X_scaled)), replace=False)

    for idx in sample_indices:
        exp = lime_explainer.explain_instance(
            X_scaled[idx],
            rf_model.predict_proba,
            num_features=5
        )
        exp_dict = exp.as_list()
        lime_results.append({
            "sample_index": idx,
            "window_start": df_unique_windows.iloc[idx]["window_start"],
            "window_end": df_unique_windows.iloc[idx]["window_end"],
            "true_regime": le.inverse_transform([y_enc[idx]])[0],
            "predicted_regime": le.inverse_transform([rf_model.predict([X_scaled[idx]])[0]])[0],
            "explanations": exp_dict
        })

    lime_importance = {}
    for res in lime_results:
        for feature, weight in res["explanations"]:
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

    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    shap_output = os.path.join(output_dir, "shap_importance_transition_windows.csv")
    shap_df.to_csv(shap_output, index=False)
    print(f"\nSHAP importance saved to {shap_output}")

    lime_output = os.path.join(output_dir, "lime_importance_transition_windows.csv")
    lime_df.to_csv(lime_output, index=False)
    print(f"LIME importance saved to {lime_output}")

    combined_df = pd.merge(shap_df, lime_df, on="feature", how="outer").fillna(0)
    combined_df["shap_rank"] = combined_df["shap_importance"].rank(ascending=False)
    combined_df["lime_rank"] = combined_df["lime_importance"].rank(ascending=False)
    combined_df["avg_rank"] = (combined_df["shap_rank"] + combined_df["lime_rank"]) / 2
    combined_df = combined_df.sort_values("avg_rank")

    combined_output = os.path.join(output_dir, "feature_importance_combined_transition_windows.csv")
    combined_df.to_csv(combined_output, index=False)
    print(f"Combined feature importance saved to {combined_output}")

    lime_json_output = os.path.join(output_dir, "lime_explanations_transition_windows.json")
    with open(lime_json_output, "w") as f:
        json.dump(lime_results, f, indent=2, default=str)
    print(f"LIME explanations saved to {lime_json_output}")

    df_output = df_unique_windows.copy()
    df_output["predicted_regime_rf"] = le.inverse_transform(y_pred_rf)
    df_output["predicted_regime_knn"] = le.inverse_transform(y_pred_knn)
    df_output["rf_correct"] = df_output["regime"] == df_output["predicted_regime_rf"]
    df_output["knn_correct"] = df_output["regime"] == df_output["predicted_regime_knn"]

    full_output = os.path.join(output_dir, "transition_windows_with_predictions.csv")
    df_output.to_csv(full_output, index=False)
    print(f"Full predictions saved to {full_output}")

    report = {
        "total_transitions": len(df_transitions),
        "unique_windows_analyzed": len(df_unique_windows),
        "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
        "random_forest_accuracy": acc_rf,
        "knn_accuracy": acc_knn,
        "top_5_features_shap": shap_df.head(5)[["feature", "shap_importance"]].to_dict(orient="records"),
        "top_5_features_lime": lime_df.head(5)[["feature", "lime_importance"]].to_dict(orient="records"),
        "combined_feature_ranking": combined_df[["feature", "avg_rank"]].to_dict(orient="records")
    }

    report_path = os.path.join(output_dir, "analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Analysis report saved to {report_path}")

    print("\n" + "="*60)
    print("SUMMARY OF FEATURE IMPORTANCE")
    print("="*60)
    print("\nTop 5 features by SHAP:")
    print(shap_df.head(5).to_string(index=False))

    print("\nTop 5 features by LIME:")
    print(lime_df.head(5).to_string(index=False))

    print("\nTop 5 features by average rank:")
    print(combined_df[["feature", "shap_importance", "lime_importance", "avg_rank"]].head(5).to_string(index=False))

    print(f"\nTotal transitions: {len(df_transitions)}")
    print(f"Unique windows analyzed: {len(df_unique_windows)}")
    print(f"Random Forest accuracy: {acc_rf:.4f}")
    print(f"KNN accuracy: {acc_knn:.4f}")

    print("\nDone.")

if __name__ == "__main__":
    main()

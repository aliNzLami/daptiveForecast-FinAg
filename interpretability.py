import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

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

def calculate_classification_metrics(y_true, y_pred, model_name, class_labels):
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')

    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "classification_report": report,
        "confusion_matrix": cm,
        "class_labels": class_labels
    }

def transition_detection_accuracy(df, model_name, pred_col):
    true_labels = df["regime"]
    pred_labels = df[pred_col]
    df["regime_shift"] = true_labels != true_labels.shift(1)

    transition_days = df[df["regime_shift"]]
    if len(transition_days) == 0:
        return None

    correct = (transition_days["regime"] == transition_days[pred_col]).sum()
    total = len(transition_days)
    return correct / total if total > 0 else 0

def weather_based_accuracy(df, model_name, pred_col):
    # Temperature groups
    temp_bins = [-float('inf'), 0, 15, 25, float('inf')]
    temp_labels = ['<0°C', '0-15°C', '15-25°C', '>25°C']
    df["temp_group"] = pd.cut(df["Max_Temp_C"], bins=temp_bins, labels=temp_labels)

    temp_acc = {}
    for group in df["temp_group"].unique():
        if pd.isna(group):
            continue
        subset = df[df["temp_group"] == group]
        correct = (subset["regime"] == subset[pred_col]).sum()
        total = len(subset)
        temp_acc[group] = correct / total if total > 0 else 0

    # Precipitation groups
    df["rain_group"] = df["Precipitation_mm"].apply(lambda x: "Rainy" if x > 0 else "Dry")
    rain_acc = {}
    for group in df["rain_group"].unique():
        if pd.isna(group):
            continue
        subset = df[df["rain_group"] == group]
        correct = (subset["regime"] == subset[pred_col]).sum()
        total = len(subset)
        rain_acc[group] = correct / total if total > 0 else 0

    return {
        "temperature": temp_acc,
        "precipitation": rain_acc
    }

def regime_stability_index(df, model_name, pred_col):
    true_labels = df["regime"]
    pred_labels = df[pred_col]

    df["regime_block"] = (true_labels != true_labels.shift(1)).cumsum()
    block_accuracies = []

    for block_id in df["regime_block"].unique():
        block_data = df[df["regime_block"] == block_id]
        if len(block_data) == 0:
            continue
        correct = (block_data["regime"] == block_data[pred_col]).sum()
        total = len(block_data)
        block_accuracies.append(correct / total if total > 0 else 0)

    if not block_accuracies:
        return 0

    return np.mean(block_accuracies)

def main():
    print("=" * 70)
    print("TRANSITION WINDOWS ANALYSIS - DAILY DATA APPROACH")
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

    print("\n[6] Training KNN classifier...")
    k = min(5, max(1, len(X_scaled) - 1))
    knn_model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn_model.fit(X_scaled, y_enc)
    y_pred_knn = knn_model.predict(X_scaled)
    acc_knn = accuracy_score(y_enc, y_pred_knn)
    print(f"    Accuracy: {acc_knn:.4f}")

    print("\n[7] Calculating classification metrics...")
    results = {}

    # RF metrics
    rf_metrics = calculate_classification_metrics(
        y_enc, y_pred_rf, "rf", le.classes_
    )
    results["rf"] = rf_metrics

    # KNN metrics
    knn_metrics = calculate_classification_metrics(
        y_enc, y_pred_knn, "knn", le.classes_
    )
    results["knn"] = knn_metrics

    # Print RF report
    print(f"\n    Random Forest Classification Report:")
    print(f"      Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"      F1-Macro: {rf_metrics['f1_macro']:.4f}")
    print(f"      F1-Weighted: {rf_metrics['f1_weighted']:.4f}")
    print(f"      Precision-Macro: {rf_metrics['precision_macro']:.4f}")
    print(f"      Recall-Macro: {rf_metrics['recall_macro']:.4f}")

    print(f"\n    KNN Classification Report:")
    print(f"      Accuracy: {knn_metrics['accuracy']:.4f}")
    print(f"      F1-Macro: {knn_metrics['f1_macro']:.4f}")
    print(f"      F1-Weighted: {knn_metrics['f1_weighted']:.4f}")
    print(f"      Precision-Macro: {knn_metrics['precision_macro']:.4f}")
    print(f"      Recall-Macro: {knn_metrics['recall_macro']:.4f}")

    # Feature importance from Random Forest
    print("\n[8] Feature importance analysis...")
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n    Random Forest Feature Importance:")
    for idx, row in feature_importance_df.iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")

    # Add predictions to dataframe for further analysis
    df_days["predicted_regime_rf"] = le.inverse_transform(y_pred_rf)
    df_days["predicted_regime_knn"] = le.inverse_transform(y_pred_knn)
    df_days["rf_correct"] = df_days["regime"] == df_days["predicted_regime_rf"]
    df_days["knn_correct"] = df_days["regime"] == df_days["predicted_regime_knn"]

    # Transition detection accuracy
    print("\n[9] Transition detection accuracy...")
    for model, pred_col in [("rf", "predicted_regime_rf"), ("knn", "predicted_regime_knn")]:
        acc = transition_detection_accuracy(df_days, model, pred_col)
        if acc is not None:
            print(f"    {model.upper()} on transition days: {acc:.4f}")
            results[model]["transition_accuracy"] = acc

    # Weather-based accuracy (temperature + precipitation)
    print("\n[10] Weather-based accuracy analysis...")
    for model, pred_col in [("rf", "predicted_regime_rf"), ("knn", "predicted_regime_knn")]:
        weather_acc = weather_based_accuracy(df_days, model, pred_col)
        if weather_acc:
            print(f"\n    {model.upper()} accuracy by temperature group:")
            for group, acc in weather_acc["temperature"].items():
                print(f"      {group}: {acc:.4f}")
            print(f"\n    {model.upper()} accuracy by precipitation group:")
            for group, acc in weather_acc["precipitation"].items():
                print(f"      {group}: {acc:.4f}")
            results[model]["weather_accuracy"] = weather_acc

    # Regime stability index
    print("\n[11] Regime stability index...")
    for model, pred_col in [("rf", "predicted_regime_rf"), ("knn", "predicted_regime_knn")]:
        stability = regime_stability_index(df_days, model, pred_col)
        if stability is not None:
            print(f"    {model.upper()} stability index: {stability:.4f}")
            results[model]["stability_index"] = stability

    print("\n[12] Saving outputs...")

    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Daily predictions with all analysis columns
    temp_bins = [-float('inf'), 0, 15, 25, float('inf')]
    temp_labels = ['<0°C', '0-15°C', '15-25°C', '>25°C']
    df_days["temp_group"] = pd.cut(df_days["Max_Temp_C"], bins=temp_bins, labels=temp_labels)
    df_days["rain_group"] = df_days["Precipitation_mm"].apply(lambda x: "Rainy" if x > 0 else "Dry")
    df_days["regime_shift"] = df_days["regime"] != df_days["regime"].shift(1)

    full_output = os.path.join(output_dir, "daily_data_with_predictions.csv")
    df_days.to_csv(full_output, index=False)
    print(f"    Full dataset with predictions: {full_output}")

    # 2. Feature importance
    feature_output = os.path.join(output_dir, "feature_importance.csv")
    feature_importance_df.to_csv(feature_output, index=False)
    print(f"    Feature importance: {feature_output}")

    # 3. Classification metrics summary
    summary_rows = []
    for model, metrics in results.items():
        summary_rows.append({
            "model": model.upper(),
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "transition_accuracy": metrics.get("transition_accuracy", None),
            "stability_index": metrics.get("stability_index", None)
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_output = os.path.join(output_dir, "classification_summary.csv")
    summary_df.to_csv(summary_output, index=False)
    print(f"    Classification summary: {summary_output}")

    # 4. Per-class detailed reports
    class_reports = []
    for model, metrics in results.items():
        report = metrics["classification_report"]
        for class_name, class_metrics in report.items():
            if class_name in ["accuracy", "macro avg", "weighted avg"]:
                continue
            class_reports.append({
                "model": model.upper(),
                "class": class_name,
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1-score": class_metrics["f1-score"],
                "support": class_metrics["support"]
            })
    class_report_df = pd.DataFrame(class_reports)
    class_report_output = os.path.join(output_dir, "per_class_metrics.csv")
    class_report_df.to_csv(class_report_output, index=False)
    print(f"    Per-class metrics: {class_report_output}")

    # 5. Confusion matrices
    for model, metrics in results.items():
        cm = metrics["confusion_matrix"]
        labels_cm = metrics["class_labels"]
        cm_df = pd.DataFrame(cm, index=labels_cm, columns=labels_cm)
        cm_output = os.path.join(output_dir, f"confusion_matrix_{model}.csv")
        cm_df.to_csv(cm_output)
        print(f"    Confusion matrix ({model.upper()}): {cm_output}")

    # 6. Weather-based accuracy (both temperature and precipitation)
    weather_rows = []
    for model, metrics in results.items():
        if "weather_accuracy" in metrics:
            # Temperature groups
            for temp_group, acc in metrics["weather_accuracy"]["temperature"].items():
                weather_rows.append({
                    "model": model.upper(),
                    "weather_type": "temperature",
                    "group": temp_group,
                    "accuracy": acc
                })
            # Precipitation groups
            for rain_group, acc in metrics["weather_accuracy"]["precipitation"].items():
                weather_rows.append({
                    "model": model.upper(),
                    "weather_type": "precipitation",
                    "group": rain_group,
                    "accuracy": acc
                })
    if weather_rows:
        weather_df = pd.DataFrame(weather_rows)
        weather_output = os.path.join(output_dir, "weather_based_accuracy.csv")
        weather_df.to_csv(weather_output, index=False)
        print(f"    Weather-based accuracy: {weather_output}")

    # 7. Final JSON report
    report = {
        "total_transitions": len(df_transitions),
        "unique_windows_analyzed": len(df_unique_windows),
        "total_days_analyzed": len(df_days),
        "class_distribution": class_dist.to_dict(),
        "feature_importance": feature_importance_df.to_dict(orient="records"),
        "models": results
    }
    report_path = os.path.join(output_dir, "final_analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"    Analysis report: {report_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for model, metrics in results.items():
        print(f"\n{model.upper()} Model:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
        print(f"  Transition Accuracy: {metrics.get('transition_accuracy', 0):.4f}")
        print(f"  Stability Index: {metrics.get('stability_index', 0):.4f}")

    print("\nFeature Importance (Random Forest):")
    for idx, row in feature_importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    print("\nDone.")

if __name__ == "__main__":
    main()

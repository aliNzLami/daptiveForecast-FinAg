import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

def get_model_instance(model_name):
    if model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "KNeighborsTimeSeries":
        return KNeighborsRegressor(n_neighbors=5)
    else:
        return None

def extract_window_features(df_windows, transition_date, window_days=30):
    before = transition_date - pd.Timedelta(days=window_days)
    after = transition_date + pd.Timedelta(days=window_days)
    window_rows = df_windows[
        (df_windows["window_center"] >= before) &
        (df_windows["window_center"] <= after)
    ]
    return window_rows

def compute_shap_and_lime(model, X_train, y_train, X_test, feature_names, class_names, model_name):
    results = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    model.fit(X_train_scaled, y_train_enc)

    y_pred_enc = model.predict(X_test_scaled)
    y_pred = le.inverse_transform(y_pred_enc.astype(int))

    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(X_test_scaled)

    shap_importance = np.abs(shap_values).mean(axis=0)
    results["shap_importance"] = dict(zip(feature_names, shap_importance))

    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_train_scaled,
        feature_names=feature_names,
        class_names=le.classes_,
        mode='classification',
        verbose=False
    )

    lime_explanations = []
    for i in range(min(5, len(X_test_scaled))):
        exp = explainer_lime.explain_instance(
            X_test_scaled[i],
            model.predict_proba,
            num_features=5
        )
        lime_explanations.append(exp.as_list())

    results["lime_explanations"] = lime_explanations

    lime_importance = {}
    for exp_list in lime_explanations:
        for feature, weight in exp_list:
            if feature not in lime_importance:
                lime_importance[feature] = []
            lime_importance[feature].append(weight)

    for feature in lime_importance:
        lime_importance[feature] = np.mean(lime_importance[feature])

    results["lime_importance"] = lime_importance

    return results

def main():
    print("Loading data...")
    df_transitions, df_windows, df_main = load_data()

    print("Computing regime labels...")
    df_windows = compute_regime_labels(df_windows, df_main)

    feature_cols = [
        "corn_mean", "corn_std", "corn_skew", "corn_kurtosis",
        "temp_mean", "temp_std", "temp_skew", "temp_kurtosis",
        "volatility", "corr_price_temp", "precip_mean"
    ]

    all_results = []

    for idx, trans_row in df_transitions.iterrows():
        trans_date = trans_row["transition_date"]
        from_model = trans_row["from_model"]
        to_model = trans_row["to_model"]

        window_rows = extract_window_features(df_windows, trans_date)

        if window_rows.empty:
            continue

        X = window_rows[feature_cols].values
        y_regime = window_rows["regime"].values

        if len(np.unique(y_regime)) < 2:
            continue

        from_model_instance = get_model_instance(from_model)
        to_model_instance = get_model_instance(to_model)

        if from_model_instance is None or to_model_instance is None:
            continue

        results_entry = {
            "transition_date": str(trans_date),
            "from_model": from_model,
            "to_model": to_model,
            "window_count": len(window_rows)
        }

        print(f"Processing transition at {trans_date} ({from_model} -> {to_model})")

        try:
            from_results = compute_shap_and_lime(
                from_model_instance, X, y_regime, X,
                feature_cols, np.unique(y_regime), from_model
            )
            results_entry["from_model_shap"] = from_results["shap_importance"]
            results_entry["from_model_lime"] = from_results["lime_importance"]
        except Exception as e:
            results_entry["from_model_error"] = str(e)
            print(f"  Failed for {from_model}: {e}")

        try:
            to_results = compute_shap_and_lime(
                to_model_instance, X, y_regime, X,
                feature_cols, np.unique(y_regime), to_model
            )
            results_entry["to_model_shap"] = to_results["shap_importance"]
            results_entry["to_model_lime"] = to_results["lime_importance"]
        except Exception as e:
            results_entry["to_model_error"] = str(e)
            print(f"  Failed for {to_model}: {e}")

        all_results.append(results_entry)

    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "transition_analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"Analysis complete. Results saved to {output_path}")
    print(f"Total transitions processed: {len(all_results)}")

    summary_rows = []
    for res in all_results:
        if "from_model_shap" in res and "to_model_shap" in res:
            if "error" not in str(res.get("from_model_shap", {})) and "error" not in str(res.get("to_model_shap", {})):
                from_imp = res["from_model_shap"]
                to_imp = res["to_model_shap"]
                if isinstance(from_imp, dict) and isinstance(to_imp, dict):
                    common_features = set(from_imp.keys()) & set(to_imp.keys())
                    for feat in common_features:
                        summary_rows.append({
                            "transition_date": res["transition_date"],
                            "from_model": res["from_model"],
                            "to_model": res["to_model"],
                            "feature": feat,
                            "from_model_importance": from_imp.get(feat, 0),
                            "to_model_importance": to_imp.get(feat, 0),
                            "agreement": "high" if abs(from_imp.get(feat, 0) - to_imp.get(feat, 0)) < 0.1 else "low"
                        })

    if summary_rows:
        summary_path = os.path.join(output_dir, "feature_agreement_analysis.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Feature agreement analysis saved to {summary_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()

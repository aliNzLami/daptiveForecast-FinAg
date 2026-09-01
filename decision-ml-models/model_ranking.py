import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def get_model_instance(model_name):
    model_name = model_name.strip()
    if model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "KNeighborsTimeSeries":
        return KNeighborsRegressor(n_neighbors=5)
    elif model_name == "XGBoost":
        return XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    elif model_name == "LightGBM":
        return LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    elif model_name == "Hidden Markov Model":
        return DummyRegressor(strategy="mean")
    else:
        return None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(base_dir, "..", "dataset", "US_Agriculture_Weather_2010_2024.csv")
    windows_path = os.path.join(base_dir, "..", "dataset", "window_features.csv")
    rankings_path = os.path.join(base_dir, "..", "dataset", "model_rankings.csv")
    output_path = os.path.join(base_dir, "..", "dataset", "top_models_performance.csv")

    print(f"Base directory: {base_dir}")
    print(f"Main data path: {main_path}")
    print(f"Windows path: {windows_path}")
    print(f"Rankings path: {rankings_path}")

    if not os.path.exists(main_path):
        print("Error: Main dataset not found.")
        sys.exit(1)
    if not os.path.exists(windows_path):
        print("Error: Window features not found.")
        sys.exit(1)
    if not os.path.exists(rankings_path):
        print("Error: Model rankings not found.")
        sys.exit(1)

    df_main = pd.read_csv(main_path)
    df_main["Date"] = pd.to_datetime(df_main["Date"])
    df_main = df_main.sort_values("Date").reset_index(drop=True)

    df_windows = pd.read_csv(windows_path)
    df_windows["window_start"] = pd.to_datetime(df_windows["window_start"])
    df_windows["window_end"] = pd.to_datetime(df_windows["window_end"])

    df_rankings = pd.read_csv(rankings_path)
    df_rankings["window_start"] = pd.to_datetime(df_rankings["window_start"])
    df_rankings["window_end"] = pd.to_datetime(df_rankings["window_end"])

    price_dict = dict(zip(df_main["Date"], df_main["Corn_Price_USD"]))
    features = ["Max_Temp_C", "Min_Temp_C", "Precipitation_mm"]

    results = []

    for idx, row in df_rankings.iterrows():
        start = row["window_start"]
        end = row["window_end"]
        future_end = end + pd.Timedelta(days=30)

        if future_end not in price_dict:
            continue

        train_data = df_main[(df_main["Date"] >= start) & (df_main["Date"] <= end)]
        test_data = df_main[(df_main["Date"] > end) & (df_main["Date"] <= future_end)]

        if len(train_data) < 10 or len(test_data) < 3:
            continue

        X_train = train_data[features].values
        y_train = train_data["Corn_Price_USD"].values
        X_test = test_data[features].values
        y_test = test_data["Corn_Price_USD"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        result_row = {
            "window_start": start,
            "window_end": end,
            "best_model": row["best_model"],
            "second_model": row["second_model"]
        }

        best_model = get_model_instance(row["best_model"])
        if best_model is not None:
            best_model.fit(X_train_scaled, y_train)
            y_pred_best = best_model.predict(X_test_scaled)
            result_row["best_rmse"] = np.sqrt(mean_squared_error(y_test, y_pred_best))
            result_row["best_mae"] = mean_absolute_error(y_test, y_pred_best)
        else:
            result_row["best_rmse"] = None
            result_row["best_mae"] = None

        second_model = get_model_instance(row["second_model"])
        if second_model is not None:
            second_model.fit(X_train_scaled, y_train)
            y_pred_second = second_model.predict(X_test_scaled)
            result_row["second_rmse"] = np.sqrt(mean_squared_error(y_test, y_pred_second))
            result_row["second_mae"] = mean_absolute_error(y_test, y_pred_second)
        else:
            result_row["second_rmse"] = None
            result_row["second_mae"] = None

        if result_row["best_rmse"] is not None and result_row["second_rmse"] is not None:
            result_row["rmse_gap"] = result_row["best_rmse"] - result_row["second_rmse"]
        else:
            result_row["rmse_gap"] = None

        results.append(result_row)

    if not results:
        print("No results generated. Check data.")
        sys.exit(1)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"Performance metrics saved to {output_path}")
    print(f"Total windows evaluated: {len(df_results)}")

    valid = df_results.dropna(subset=["best_rmse", "second_rmse"])
    if len(valid) > 0:
        print(f"\nAverage RMSE - Best Model: {valid['best_rmse'].mean():.4f}")
        print(f"Average RMSE - Second Model: {valid['second_rmse'].mean():.4f}")
        best_wins = (valid['best_rmse'] < valid['second_rmse']).sum()
        second_wins = (valid['second_rmse'] < valid['best_rmse']).sum()
        print(f"Best model wins in {best_wins} windows")
        print(f"Second model wins in {second_wins} windows")

if __name__ == "__main__":
    main()

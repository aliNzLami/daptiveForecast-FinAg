import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format='mixed')
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def compute_window_features(df, window_days=90, step_days=30):
    features_list = []
    
    for start_idx in range(0, len(df) - window_days + 1, step_days):
        window_df = df.iloc[start_idx:start_idx + window_days]
        
        if len(window_df) < window_days:
            break
        
        def get_stats(series):
            return {
                "mean": series.mean(),
                "std": series.std(),
                "skew": skew(series.dropna()),
                "kurtosis": kurtosis(series.dropna())
            }
        
        corn_stats = get_stats(window_df["Corn_Price_USD"])
        
        temp_avg = (window_df["Max_Temp_C"] + window_df["Min_Temp_C"]) / 2
        temp_stats = get_stats(temp_avg)
        
        returns = window_df["Corn_Price_USD"].pct_change().dropna()
        volatility = returns.std()
        
        corr_price_temp = window_df["Corn_Price_USD"].corr(temp_avg)
        
        features = {
            "window_start": window_df["Date"].iloc[0],
            "window_end": window_df["Date"].iloc[-1],
            "window_center": window_df["Date"].iloc[len(window_df)//2],
            "corn_mean": corn_stats["mean"],
            "corn_std": corn_stats["std"],
            "corn_skew": corn_stats["skew"],
            "corn_kurtosis": corn_stats["kurtosis"],
            "temp_mean": temp_stats["mean"],
            "temp_std": temp_stats["std"],
            "temp_skew": temp_stats["skew"],
            "temp_kurtosis": temp_stats["kurtosis"],
            "volatility": volatility,
            "corr_price_temp": corr_price_temp,
            "precip_mean": window_df["Precipitation_mm"].mean()
        }
        features_list.append(features)
    
    return pd.DataFrame(features_list)

if __name__ == "__main__":
    input_file = "dataset/US_Agriculture_Weather_2010_2024.csv"
    output_file = "window_features.csv"
    
    df = load_and_clean_data(input_file)
    print(f"Cleaned rows: {len(df)}")
    
    features_df = compute_window_features(df)
    print(f"Windows extracted: {len(features_df)}")
    
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

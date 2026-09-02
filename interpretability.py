import pandas as pd
import os

def main():
    # Load data
    transitions = pd.read_csv("dataset/transition_points.csv")
    windows = pd.read_csv("dataset/window_features.csv")
    
    transitions["transition_date"] = pd.to_datetime(transitions["transition_date"])
    windows["window_start"] = pd.to_datetime(windows["window_start"])
    windows["window_end"] = pd.to_datetime(windows["window_end"])
    windows["window_center"] = pd.to_datetime(windows["window_center"])
    
    # For each transition, find windows within 30 days before and after
    selected_windows = []
    for _, trans in transitions.iterrows():
        trans_date = trans["transition_date"]
        mask = (windows["window_center"] >= trans_date - pd.Timedelta(days=30)) & \
               (windows["window_center"] <= trans_date + pd.Timedelta(days=30))
        nearby = windows[mask].copy()
        nearby["transition_date"] = trans_date
        selected_windows.append(nearby)
    
    # Combine all
    if selected_windows:
        df_all = pd.concat(selected_windows, ignore_index=True)
    else:
        print("No windows found.")
        return
    
    # Remove duplicates based on window_start and window_end
    df_unique = df_all.drop_duplicates(subset=["window_start", "window_end"])
    
    # Sort by window_start
    df_unique = df_unique.sort_values("window_start").reset_index(drop=True)
    
    # Output
    output_path = "selected_windows_for_transitions.csv"
    df_unique.to_csv(output_path, index=False)
    
    print(f"Total transitions: {len(transitions)}")
    print(f"Total windows selected (before dedup): {len(df_all)}")
    print(f"Unique windows after dedup: {len(df_unique)}")
    print(f"Output saved to {output_path}")
    
    # Also show per transition window count
    print("\nWindows per transition (before dedup):")
    for _, trans in transitions.iterrows():
        trans_date = trans["transition_date"]
        count = len(df_all[df_all["transition_date"] == trans_date])
        print(f"{trans_date}: {count} windows")

if __name__ == "__main__":
    main()

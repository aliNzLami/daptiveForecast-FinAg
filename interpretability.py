import pandas as pd
import os

def main():
    base_dir = os.getcwd()
    transitions_path = os.path.join(base_dir, "dataset", "transition_points.csv")
    windows_path = os.path.join(base_dir, "dataset", "window_features.csv")
    output_path = os.path.join(base_dir, "selected_windows_for_transitions.csv")

    if not os.path.exists(transitions_path):
        print(f"Error: {transitions_path} not found.")
        return
    if not os.path.exists(windows_path):
        print(f"Error: {windows_path} not found.")
        return

    transitions = pd.read_csv(transitions_path)
    windows = pd.read_csv(windows_path)

    transitions["transition_date"] = pd.to_datetime(transitions["transition_date"])
    windows["window_start"] = pd.to_datetime(windows["window_start"])
    windows["window_end"] = pd.to_datetime(windows["window_end"])
    windows["window_center"] = pd.to_datetime(windows["window_center"])

    selected_windows = []

    for _, trans in transitions.iterrows():
        trans_date = trans["transition_date"]
        mask = (windows["window_center"] >= trans_date - pd.Timedelta(days=30)) & \
               (windows["window_center"] <= trans_date + pd.Timedelta(days=30))
        nearby = windows[mask].copy()
        nearby["transition_date"] = trans_date
        selected_windows.append(nearby)

    if not selected_windows:
        print("No windows found for any transition.")
        return

    df_all = pd.concat(selected_windows, ignore_index=True)

    df_unique = df_all.drop_duplicates(subset=["window_start", "window_end"])
    df_unique = df_unique.sort_values("window_start").reset_index(drop=True)

    df_unique.to_csv(output_path, index=False)

    print(f"Total transitions: {len(transitions)}")
    print(f"Total windows selected (before dedup): {len(df_all)}")
    print(f"Unique windows after dedup: {len(df_unique)}")
    print(f"Output saved to {output_path}")

    print("\nWindows per transition (before dedup):")
    for _, trans in transitions.iterrows():
        trans_date = trans["transition_date"]
        count = len(df_all[df_all["transition_date"] == trans_date])
        print(f"{trans_date.strftime('%Y-%m-%d')}: {count} windows")

    print("\nSample of unique windows:")
    print(df_unique[["window_start", "window_end", "window_center"]].head(10))

if __name__ == "__main__":
    main()

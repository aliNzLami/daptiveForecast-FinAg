import pandas as pd
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "dataset", "top_models_performance.csv")
    output_path = os.path.join(base_dir, "..", "dataset", "final_model_per_window.csv")

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    df = pd.read_csv(input_path)

    df["window_start"] = pd.to_datetime(df["window_start"])
    df["window_end"] = pd.to_datetime(df["window_end"])

    def select_final_model(row):
        if pd.isna(row["best_rmse"]) and pd.isna(row["second_rmse"]):
            return "Unknown"
        if pd.isna(row["best_rmse"]):
            return row["second_model"]
        if pd.isna(row["second_rmse"]):
            return row["best_model"]
        if row["best_rmse"] <= row["second_rmse"]:
            return row["best_model"]
        else:
            return row["second_model"]

    df["final_model"] = df.apply(select_final_model, axis=1)

    df.to_csv(output_path, index=False)
    print(f"Final model per window saved to {output_path}")

if __name__ == "__main__":
    main()

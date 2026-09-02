import pandas as pd
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "dataset", "final_model_per_window.csv")
    output_path = os.path.join(base_dir, "dataset", "transition_points.csv")

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    df = pd.read_csv(input_path)
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start").reset_index(drop=True)

    transitions = []
    previous_model = None
    first_occurrence = None

    for idx, row in df.iterrows():
        current_model = row["final_model"]

        if previous_model is None:
            previous_model = current_model
            first_occurrence = row["window_start"]
            continue

        if current_model != previous_model:
            transitions.append({
                "transition_date": row["window_start"],
                "from_model": previous_model,
                "to_model": current_model,
                "window_start": row["window_start"],
                "window_end": row["window_end"]
            })
            previous_model = current_model
            first_occurrence = row["window_start"]

    if not transitions:
        print("No transitions found.")
        return

    df_transitions = pd.DataFrame(transitions)
    df_transitions.to_csv(output_path, index=False)

    print(f"Transitions saved to {output_path}")
    print(f"Number of transitions: {len(df_transitions)}")
    print(df_transitions[["transition_date", "from_model", "to_model"]].head(10))

if __name__ == "__main__":
    main()

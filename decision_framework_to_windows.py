import pandas as pd
import numpy as np
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def compute_context_from_window(row):
    V = np.log10(90) / 6
    N = 0.3 * (row["corn_std"] / 50) + 0.3 * (row["volatility"] / 0.05) + 0.4 * (1 - abs(row["corr_price_temp"]))
    N = min(1.0, max(0.0, N))
    G = 0.5
    rho = 15 / 90
    E = 0.5
    return np.array([V, N, G, rho, E])

def compute_requirements(ctx, a1, a2, a3, a4):
    V, N, G, rho, E = ctx
    b1, b2, b3, b4 = 1 - a1, 1 - a2, 1 - a3, 1 - a4
    r_interp = a1 * (1 - sigmoid(10 * (E - 0.5))) + b1 * rho
    r_robust = a2 * sigmoid(12 * (N - 0.35)) + b2 * tanh(2 * rho)
    r_scal = a3 * tanh(3 * V) + b3 * G
    r_rep = a4 * G + b4 * E
    return np.array([r_interp, r_robust, r_scal, r_rep])

def get_model_profiles():
    return {
        "Random Forest": np.array([0.50, 0.80, 0.50, 0.80]),
        "XGBoost": np.array([0.30, 0.80, 0.80, 0.90]),
        "LightGBM": np.array([0.30, 0.80, 0.90, 0.90]),
        "Hidden Markov Model": np.array([0.80, 0.50, 0.30, 0.50]),
        "KNeighborsTimeSeries": np.array([0.80, 0.40, 0.50, 0.50])
    }

def manhattan_score(req, cap, weights):
    return 1.0 - np.sum(weights * np.abs(req - cap))

def get_requirement_weights():
    alpha, gamma, zeta, theta = 0.60, 0.55, 0.50, 0.85
    total = alpha + gamma + zeta + theta
    return np.array([alpha, gamma, zeta, theta]) / total

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "dataset", "window_features.csv")
    output_path = os.path.join(base_dir, "..", "model_recommendations.csv")

    df_windows = pd.read_csv(input_path)
    df_windows["window_start"] = pd.to_datetime(df_windows["window_start"])
    df_windows["window_end"] = pd.to_datetime(df_windows["window_end"])

    model_profiles = get_model_profiles()
    req_weights = get_requirement_weights()

    recommendations = []
    scores = []
    for idx, row in df_windows.iterrows():
        ctx = compute_context_from_window(row)
        req = compute_requirements(ctx, 0.60, 0.55, 0.50, 0.85)

        score_dict = {}
        for name, cap in model_profiles.items():
            score_dict[name] = manhattan_score(req, cap, req_weights)

        best_model = max(score_dict, key=score_dict.get)
        best_score = score_dict[best_model]

        recommendations.append(best_model)
        scores.append(round(best_score, 4))

    df_windows["recommended_model"] = recommendations
    df_windows["compatibility_score"] = scores

    df_windows.to_csv(output_path, index=False)
    print(f"model_recommendations.csv generated successfully at: {output_path}")

if __name__ == "__main__":
    main()

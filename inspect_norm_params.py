import numpy as np

params = np.load("norm_params.npz", allow_pickle=True)

feature_cols = params["feature_cols"].tolist()
mean = params["mean"].flatten()
std  = params["std"].flatten()

print(f"{'Feature':<15} {'Mean':>12} {'Std':>12}")
print("-" * 41)
for name, m, s in zip(feature_cols, mean, std):
    print(f"{name:<15} {m:>12.6f} {s:>12.6f}")

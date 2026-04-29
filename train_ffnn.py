"""
FFNN for predicting system time from QA, QB, QC, Route.
Dataset: DataSet/TrainingData_Experiment1_Scenario1_Rep*.xlsx
Run with: conda activate egr608 && python train_ffnn.py

Output files:
  ffnn_systemtime_model.keras  — Keras model (includes normalization)
  ffnn_systemtime_model.onnx   — ONNX model (single input, pre-normalized)
  norm_params.npz              — mean/std for pre-normalization at inference
  training_history.png
  prediction_error_hist.png
  actual_vs_predicted.png
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
# 1. Load all xlsx files
# ─────────────────────────────────────────
DATA_DIR = "DataSet"
files = sorted(glob.glob(os.path.join(DATA_DIR, "TrainingData_Experiment1_Scenario1_Rep*.xlsx")))

if not files:
    raise FileNotFoundError(f"No xlsx files found in {DATA_DIR}/")

print(f"Loading {len(files)} files...")
frames = [pd.read_excel(f, header=None) for f in files]
df = pd.concat(frames, ignore_index=True)
print(f"Total rows loaded: {len(df)}")

# ─────────────────────────────────────────
# 2. Column assignment
#    col 0,1  → ignore
#    col 2    → QA   (input)
#    col 3    → QB   (input)
#    col 4    → QC   (input)
#    col 5    → Route (input, categorical)
#    col 6    → filter: drop rows where > 0
#    col 7    → SystemTime (label)
# ─────────────────────────────────────────
df = df.iloc[:, 2:]
df.columns = ["QA", "QB", "QC", "Route", "FilterCol", "SystemTime"]

# ─────────────────────────────────────────
# 3. Filter: discard rows where FilterCol > 0
# ─────────────────────────────────────────
before = len(df)
df = df[df["FilterCol"] <= 0].copy()
print(f"Rows after filter (FilterCol <= 0): {len(df)}  (dropped {before - len(df)})")

df = df.drop(columns=["FilterCol"])
df = df.dropna()
print(f"Rows after dropna: {len(df)}")

# ─────────────────────────────────────────
# 4. One-hot encode Route (categorical: 1, 2, 3)
# ─────────────────────────────────────────
df = pd.get_dummies(df, columns=["Route"], prefix="Route")
print(f"Columns after one-hot encoding: {list(df.columns)}")

# ─────────────────────────────────────────
# 5. Split features / label
# ─────────────────────────────────────────
label_col = "SystemTime"
feature_cols = [c for c in df.columns if c != label_col]

X = df[feature_cols].astype(np.float32).values
y = df[label_col].astype(np.float32).values

print(f"\nFeature shape: {X.shape}")
print(f"Label  shape: {y.shape}")
print(f"Label  range: [{y.min():.4f}, {y.max():.4f}],  mean={y.mean():.4f}")

# ─────────────────────────────────────────
# 6. Train / test split  (80 / 20)
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain samples: {len(X_train)},  Test samples: {len(X_test)}")

# ─────────────────────────────────────────
# 7. Manual normalization (fit on train only)
#    Saves mean/std to norm_params.npz for inference use.
# ─────────────────────────────────────────
norm_mean = X_train.mean(axis=0, keepdims=True)
norm_std  = X_train.std(axis=0, keepdims=True) + 1e-8

X_train_n = (X_train - norm_mean) / norm_std
X_test_n  = (X_test  - norm_mean) / norm_std

np.savez("norm_params.npz", mean=norm_mean, std=norm_std, feature_cols=feature_cols)
print(f"Normalization params saved: norm_params.npz")

# ─────────────────────────────────────────
# 8. Build model (no Normalization layer —
#    inputs are already normalized)
# ─────────────────────────────────────────
input_dim = X_train_n.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])
model.compile(
    loss="mean_absolute_error",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)
model.summary()

# ─────────────────────────────────────────
# 9. Train
# ─────────────────────────────────────────
EPOCHS = 100

history = model.fit(
    X_train_n, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    verbose=1
)

# ─────────────────────────────────────────
# 10. Plot training history
# ─────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"],     label="Train MAE")
plt.plot(history.history["val_loss"], label="Val MAE")
plt.xlabel("Epoch")
plt.ylabel("Error [SystemTime]")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.show()
print("Saved: training_history.png")

# ─────────────────────────────────────────
# 11. Evaluate on test set
# ─────────────────────────────────────────
y_pred = model.predict(X_test_n).flatten()

mse  = mean_squared_error(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n{'='*40}")
print(f"Test MAE  : {mae:.6f}")
print(f"Test RMSE : {rmse:.6f}")
print(f"Test MSE  : {mse:.6f}")
print(f"{'='*40}")

# ─────────────────────────────────────────
# 12. Prediction error histogram
# ─────────────────────────────────────────
error = y_pred - y_test

plt.figure(figsize=(8, 4))
plt.hist(error, bins=40, edgecolor="k", alpha=0.8)
plt.xlabel("Prediction Error [SystemTime]")
plt.ylabel("Count")
plt.title("Prediction Error Distribution")
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_error_hist.png", dpi=150)
plt.show()
print("Saved: prediction_error_hist.png")

# ─────────────────────────────────────────
# 13. Actual vs Predicted scatter
# ─────────────────────────────────────────
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3, s=8)
lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual SystemTime")
plt.ylabel("Predicted SystemTime")
plt.title("Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()
print("Saved: actual_vs_predicted.png")

# ─────────────────────────────────────────
# 14. Sample prediction table (first 10)
# ─────────────────────────────────────────
results = pd.DataFrame({
    "Actual":    y_test[:10],
    "Predicted": y_pred[:10],
    "Error":     error[:10]
})
print("\nSample predictions (first 10 test rows):")
print(results.to_string(index=False))

# ─────────────────────────────────────────
# 15. Save Keras model
# ─────────────────────────────────────────
model.save("ffnn_systemtime_model.keras")
print("\nModel saved: ffnn_systemtime_model.keras")

# ─────────────────────────────────────────
# 16. Export to ONNX
#     Input:  "input"  shape=[N, 6]  (already normalized)
#     Output: "output" shape=[N, 1]
# ─────────────────────────────────────────
import tf2onnx

onnx_path = "ffnn_systemtime_model.onnx"

input_signature = [tf.TensorSpec((None, input_dim), tf.float32, name="input")]

@tf.function(input_signature=input_signature)
def model_fn(x):
    return {"output": model(x)}

tf2onnx.convert.from_function(
    model_fn,
    input_signature=input_signature,
    opset=13,
    output_path=onnx_path
)
print(f"ONNX model saved: {onnx_path}")

# ─────────────────────────────────────────
# 17. Verify ONNX output matches Keras
# ─────────────────────────────────────────
import onnxruntime as ort

sess      = ort.InferenceSession(onnx_path)
sample    = X_test_n[:5]
keras_out = model(sample).numpy().flatten()
onnx_out  = sess.run(["output"], {"input": sample})[0].flatten()

print("\nONNX verification (first 5 predictions):")
print(f"  Keras : {keras_out}")
print(f"  ONNX  : {onnx_out}")
print(f"  Max diff: {np.max(np.abs(keras_out - onnx_out)):.2e}")

print("\nDone.")

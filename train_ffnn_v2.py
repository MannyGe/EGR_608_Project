"""
FFNN v2: residual connections + dropout
Dataset: DataSet/TrainingData_Experiment1_Scenario1_Rep*.xlsx
Run with: conda activate egr608 && python train_ffnn_v2.py
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
# 1. Load & clean data  (same as v1)
# ─────────────────────────────────────────
DATA_DIR = "DataSet"
files = sorted(glob.glob(os.path.join(DATA_DIR, "TrainingData_Experiment1_Scenario1_Rep*.xlsx")))
if not files:
    raise FileNotFoundError(f"No xlsx files found in {DATA_DIR}/")

print(f"Loading {len(files)} files...")
df = pd.concat([pd.read_excel(f, header=None) for f in files], ignore_index=True)
print(f"Total rows: {len(df)}")

df = df.iloc[:, 2:]
df.columns = ["QA", "QB", "QC", "Route", "FilterCol", "SystemTime"]

before = len(df)
df = df[df["FilterCol"] <= 0].copy()
print(f"After filter: {len(df)}  (dropped {before - len(df)})")
df = df.drop(columns=["FilterCol"]).dropna()

df = pd.get_dummies(df, columns=["Route"], prefix="Route")

label_col    = "SystemTime"
feature_cols = [c for c in df.columns if c != label_col]

X = df[feature_cols].astype(np.float32).values
y = df[label_col].astype(np.float32).values

# ─────────────────────────────────────────
# 2. Train / test split
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)},  Test: {len(X_test)}")

# ─────────────────────────────────────────
# 3. Manual normalization
# ─────────────────────────────────────────
norm_mean = X_train.mean(axis=0, keepdims=True)
norm_std  = X_train.std(axis=0, keepdims=True) + 1e-8

X_train_n = (X_train - norm_mean) / norm_std
X_test_n  = (X_test  - norm_mean) / norm_std

np.savez("norm_params.npz", mean=norm_mean, std=norm_std, feature_cols=feature_cols)

# ─────────────────────────────────────────
# 4. Build model: residual blocks + dropout
# ─────────────────────────────────────────
UNITS       = 64
DROPOUT     = 0.2
input_dim   = X_train_n.shape[1]

def residual_block(x, units, dropout_rate):
    skip = x
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(units)(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)
    return x

inputs = keras.Input(shape=(input_dim,))
x      = layers.Dense(UNITS)(inputs)          # projection: 6 → 64
x      = residual_block(x, UNITS, DROPOUT)
x      = residual_block(x, UNITS, DROPOUT)
output = layers.Dense(1)(x)

model = keras.Model(inputs, output, name="ffnn_v2_residual")
model.compile(
    loss="mean_absolute_error",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)
model.summary()

# ─────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────
EPOCHS = 100

history = model.fit(
    X_train_n, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    verbose=1
)

# ─────────────────────────────────────────
# 6. Plot training history
# ─────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"],     label="Train MAE")
plt.plot(history.history["val_loss"], label="Val MAE")
plt.xlabel("Epoch")
plt.ylabel("Error [SystemTime]")
plt.title("Training History (v2: Residual + Dropout)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_history_v2.png", dpi=150)
plt.show()
print("Saved: training_history_v2.png")

# ─────────────────────────────────────────
# 7. Evaluate
# ─────────────────────────────────────────
y_pred = model.predict(X_test_n).flatten()
error  = y_pred - y_test

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse  = mean_squared_error(y_test, y_pred)

print(f"\n{'='*40}")
print(f"Test MAE  : {mae:.6f}")
print(f"Test RMSE : {rmse:.6f}")
print(f"Test MSE  : {mse:.6f}")
print(f"{'='*40}")

# ─────────────────────────────────────────
# 8. Prediction error histogram
# ─────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.hist(error, bins=40, edgecolor="k", alpha=0.8)
plt.xlabel("Prediction Error [SystemTime]")
plt.ylabel("Count")
plt.title("Prediction Error Distribution (v2)")
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_error_hist_v2.png", dpi=150)
plt.show()
print("Saved: prediction_error_hist_v2.png")

# ─────────────────────────────────────────
# 9. Actual vs Predicted scatter
# ─────────────────────────────────────────
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3, s=8)
lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual SystemTime")
plt.ylabel("Predicted SystemTime")
plt.title("Actual vs Predicted (v2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_v2.png", dpi=150)
plt.show()
print("Saved: actual_vs_predicted_v2.png")

# ─────────────────────────────────────────
# 10. Sample predictions
# ─────────────────────────────────────────
results = pd.DataFrame({
    "Actual":    y_test[:10],
    "Predicted": y_pred[:10],
    "Error":     error[:10]
})
print("\nSample predictions (first 10):")
print(results.to_string(index=False))

# ─────────────────────────────────────────
# 11. Save Keras model
# ─────────────────────────────────────────
model.save("ffnn_v2_model.keras")
print("\nModel saved: ffnn_v2_model.keras")

# ─────────────────────────────────────────
# 12. Export to ONNX
# ─────────────────────────────────────────
import tf2onnx

onnx_path = "ffnn_v2_model.onnx"

input_signature = [tf.TensorSpec((None, input_dim), tf.float32, name="input")]

@tf.function(input_signature=input_signature)
def model_fn(x):
    return {"output": model(x, training=False)}

tf2onnx.convert.from_function(
    model_fn,
    input_signature=input_signature,
    opset=13,
    output_path=onnx_path
)
print(f"ONNX model saved: {onnx_path}")

# ─────────────────────────────────────────
# 13. Verify ONNX matches Keras
# ─────────────────────────────────────────
import onnxruntime as ort

sess      = ort.InferenceSession(onnx_path)
sample    = X_test_n[:5]
keras_out = model(sample, training=False).numpy().flatten()
onnx_out  = sess.run(["output"], {"input": sample})[0].flatten()

print("\nONNX verification (first 5):")
print(f"  Keras : {keras_out}")
print(f"  ONNX  : {onnx_out}")
print(f"  Max diff: {np.max(np.abs(keras_out - onnx_out)):.2e}")

print("\nDone.")

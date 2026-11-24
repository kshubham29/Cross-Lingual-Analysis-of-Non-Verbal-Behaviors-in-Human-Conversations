# Step 12 (simple version): our own Transformer on the 25 frames of each second.
# Gives the same numbers as 12_transformer.py. Run from the project folder: python src/12_transformer_simple.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # hide TensorFlow messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # same maths every run

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

keras.utils.set_random_seed(42)


def ccc(y_true, y_pred):
    # Concordance Correlation Coefficient: 1 = perfect agreement
    covariance = ((y_true - y_true.mean()) * (y_pred - y_pred.mean())).mean()
    return (2 * covariance) / (y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2 + 1e-9)


# ---------- data: 25 frames x 802 features per second ----------
y = pd.read_parquet("data/features/noxi_engagement.parquet", columns=["engagement"])["engagement"].to_numpy("float32")
X = np.load("data/features/noxi_sequences.npy")

# random split 60 / 20 / 20 (development stage)
rows = np.arange(len(y))
rest, test = train_test_split(rows, test_size=0.2, random_state=42)
train, val = train_test_split(rest, test_size=0.25, random_state=42)

# scale every frame with a scaler fitted on the training frames only
scaler = StandardScaler().fit(X[train].reshape(-1, 802))


def prepare(idx):
    return scaler.transform(X[idx].reshape(-1, 802)).reshape(len(idx), 25, 802).astype("float32")


X_train, X_val, X_test = prepare(train), prepare(val), prepare(test)
y_train, y_val, y_test = y[train], y[val], y[test]
print("train", len(X_train), "| val", len(X_val), "| test", len(X_test))


# ---------- the Transformer ----------
class PositionalEncoding(layers.Layer):
    # attention does not know the order of the frames, so a learned vector is added for each of the 25 positions
    def build(self, input_shape):
        self.pos = self.add_weight(shape=(25, 64), initializer="random_normal", name="pos")

    def call(self, x):
        return x + self.pos


def encoder_block(x):
    # self-attention (4 heads) + feed-forward, each followed by Add & Norm
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attention]))
    ff = layers.Dense(128, activation="relu")(x)
    ff = layers.Dropout(0.1)(ff)
    ff = layers.Dense(64)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ff]))


keras.utils.set_random_seed(42)
inp = layers.Input(shape=(25, 802))
x = layers.Dense(64)(inp)                 # each frame: 802 -> 64 numbers
x = PositionalEncoding()(x)               # add the position of each frame
x = encoder_block(x)                      # encoder block 1
x = encoder_block(x)                      # encoder block 2
x = layers.GlobalAveragePooling1D()(x)    # average the 25 frames
x = layers.Dropout(0.1)(x)
x = layers.Dense(64, activation="relu")(x)
out = layers.Dense(1)(x)                  # engagement
model = keras.Model(inp, out, name="Transformer")
model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
model.summary()

# ---------- train 25 epochs (5 stages of 5, printing the validation CCC) ----------
for stage in range(5):
    model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=0)
    p = model.predict(X_val, verbose=0).flatten()
    print(f"  epoch {(stage + 1) * 5:3d}  val CCC = {ccc(y_val, p):.3f}")

# ---------- results ----------
val_ccc = ccc(y_val, model.predict(X_val, verbose=0).flatten())
pred = model.predict(X_test, verbose=0).flatten()
result = {"model": "Transformer", "params": model.count_params(), "val_ccc": val_ccc,
          "test_ccc": ccc(y_test, pred), "pearson": np.corrcoef(y_test, pred)[0, 1],
          "mae": mean_absolute_error(y_test, pred)}
for key, value in result.items():
    print(f"  {key:10s}: {value}")
pd.DataFrame([result]).to_csv("reports/transformer_results_simple.csv", index=False)

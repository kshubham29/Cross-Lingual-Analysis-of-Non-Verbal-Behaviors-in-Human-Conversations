"""Step 32: latency and model size of the final models (supervisor's review, E15).
Measures only the PREDICTION stage on this laptop CPU, after the features exist. The time to compute the input features
(eGeMAPS, OpenFace, w2v-BERT 2.0, VideoMAE) is NOT measured here.
Neural networks: parameters, weight size (float32, 4 bytes each) and latency are architecture properties, so untrained networks
with the final architectures are timed. Tree models and Linear Regression are fitted once on all windows (needed for the timing).
Latency = time to predict ONE one-second window (median and 95th percentile over many repeats); throughput = windows per second
for a large batch. One window is one second of data, so a prediction faster than 1000 ms keeps up with live data.
Run: python src/32_latency_size.py   ->  reports/latency_size.csv"""
import os, sys, time, pickle, platform, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


df = pd.read_parquet("data/features/noxi_engagement.parquet")
cols = [c for c in df.columns if c.startswith("voice_")] + [f"face_{i}" for i in range(5, 714)]
X = StandardScaler().fit_transform(df[cols].fillna(0).to_numpy("float32")).clip(-5, 5).astype("float32")
y = df["engagement"].to_numpy("float32")
log(f"data: {X.shape[0]:,} windows x {X.shape[1]} features | CPU: {platform.processor()[:60]} | logical cores: {os.cpu_count()}")


def timeit(fn, repeats, warm=10):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(repeats):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1000)
    return float(np.median(t)), float(np.percentile(t, 95))


rows = []


def record(name, params, size_mb, one, batch_fn, n_batch, repeats, note=""):
    med, p95 = timeit(one, repeats)
    s = time.perf_counter(); batch_fn(); per_s = n_batch / (time.perf_counter() - s)
    rows.append(dict(model=name, parameters=params, size_mb=size_mb, latency_ms_median=med, latency_ms_p95=p95,
                     windows_per_second=per_s, real_time_margin=1000 / med, note=note))
    log(f"=== {name}: params {params if params else '-'} | size {size_mb:.2f} MB | 1 window: median {med:.2f} ms, p95 {p95:.2f} ms "
        f"| {per_s:,.0f} windows/s | {1000 / med:,.0f}x faster than the 1 s of data")


one_x, big_x = X[:1], X[:4096]

# ---------------------------------------------------------------- classic models (fitted once on all windows)
log("fitting Linear Regression, Random Forest (100 trees, sqrt features) and XGBoost on all windows (needed for the timing) ...")
lr = LinearRegression().fit(X, y)
record("Linear Regression", X.shape[1] + 1, len(pickle.dumps(lr)) / 1e6, lambda: lr.predict(one_x), lambda: lr.predict(big_x), len(big_x), 300, "797 features")
log("   Linear Regression done")
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=8).fit(X, y)
rf.set_params(n_jobs=1)                                    # one window: no thread start-up cost
record("Random Forest (100 trees)", None, len(pickle.dumps(rf)) / 1e6, lambda: rf.predict(one_x), lambda: rf.predict(big_x), len(big_x), 100,
       "single thread; size = pickled model")
log("   Random Forest done")
xg = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=8).fit(X, y)
xg.set_params(n_jobs=1)
record("XGBoost (300 trees, depth 6)", None, len(xg.get_booster().save_raw("ubj")) / 1e6, lambda: xg.predict(one_x), lambda: xg.predict(big_x), len(big_x), 200,
       "single thread; size = saved booster")
log("   XGBoost done")

# ---------------------------------------------------------------- neural networks (final architectures, untrained)
from tensorflow import keras
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    def __init__(self, n_steps, d_model, **kw):
        super().__init__(**kw); self.n_steps, self.d_model = n_steps, d_model

    def build(self, input_shape):
        self.pos = self.add_weight(shape=(self.n_steps, self.d_model), initializer="random_normal", name="pos")

    def call(self, x):
        return x + self.pos


def dense_net(n, sizes):
    seq = [layers.Input(shape=(n,))]
    for u in sizes:
        seq += [layers.Dense(u, activation="relu"), layers.Dropout(0.1)]
    return keras.Sequential(seq + [layers.Dense(1)])


def encoder_block(x):
    att = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, att]))
    ff = layers.Dense(64)(layers.Dropout(0.1)(layers.Dense(128, activation="relu")(x)))
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ff]))


def transformer(steps, n):
    inp = layers.Input(shape=(steps, n))
    x = PositionalEncoding(steps, 64)(layers.Dense(64)(inp)); x = encoder_block(x); x = encoder_block(x)
    x = layers.Dense(64, activation="relu")(layers.Dropout(0.1)(layers.GlobalAveragePooling1D()(x)))
    return keras.Model(inp, layers.Dense(1)(x))


rng = np.random.default_rng(0)
specs = [("ANN (512-256), 1 window of 797 features", dense_net(797, [512, 256]), (1, 797), (1024, 797), ""),
         ("Dense head, w2v-BERT + VideoMAE (2,432 inputs)", dense_net(2432, [256, 128]), (1, 2432), (1024, 2432), ""),
         ("Dense head + context (7,296 inputs)", dense_net(7296, [256, 128]), (1, 7296), (1024, 7296), "best model; context = running means, cheap to update"),
         ("Transformer, 10 s (10 x 797 inputs)", transformer(10, 797), (1, 10, 797), (256, 10, 797), "")]
for name, m, shp1, shpb, note in specs:
    x1 = rng.standard_normal(shp1).astype("float32"); xb = rng.standard_normal(shpb).astype("float32")
    record(name, m.count_params(), m.count_params() * 4 / 1e6, lambda m=m, x1=x1: m(x1, training=False).numpy(),
           lambda m=m, xb=xb: m.predict(xb, verbose=0, batch_size=len(xb)), len(xb), 200, note or "float32 weights")
    log(f"   {name} done")

out = pd.DataFrame(rows)
out.to_csv("reports/latency_size.csv", index=False)
log("saved reports/latency_size.csv")
log("NOT measured: time to extract the input features (OpenFace, eGeMAPS, w2v-BERT 2.0, VideoMAE) from audio/video")
log("GROUP latency DONE")

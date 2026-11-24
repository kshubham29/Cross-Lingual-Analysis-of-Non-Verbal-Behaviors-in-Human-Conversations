"""Step 29: final models on all 24 conversations, one pipeline (supervisor's review: E2, E3, problem 6).
Same speaker-independent 6-fold CV and leave-one-language-out (train on 2 languages, test on the 3rd) as step 28.
Neural models: 5 seeds each, reported as mean +- std over seeds. Classic models are deterministic (one run).
Every model is also scored frame-level (each frame gets its second's prediction), as in the MultiMediate challenge.
Run one group at a time:  python src/29_final_models.py <group>
  classic   Linear Regression, Random Forest, XGBoost
  ann       ANN (MLP 512-256, tuned in step 07)
  ae        autoencoder (797 -> 32 code) + dense head and + XGBoost
  cnn       CNN, VGG style (step 11)            rnn   SimpleRNN       lstm  LSTM      gru  GRU
  tr1       Transformer, 1 s (25 frames)        tr10  Transformer, 10 s (last 10 seconds)
  dense     dense head on frozen w2v-BERT 2.0 + VideoMAE, without and with past context
Results: reports/final_models.csv (one row per model, updated after every group) and reports/final_per_session.csv"""
import os, sys, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from splits import person_groups

GROUP = sys.argv[1]
SEEDS = [1, 2, 3, 4, 5]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ccc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64); y_pred = np.asarray(y_pred, dtype=np.float64)
    cov = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    return 2 * cov / (y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2 + 1e-9)


# ---------------------------------------------------------------- data and protocol (same as step 28)
df = pd.read_parquet("data/features/noxi_engagement.parquet")
y = df["engagement"].to_numpy("float32")
session = df["session"].to_numpy(); language = df["language"].to_numpy()
speaker = (df["session"] + "_" + df["role"]).to_numpy()
languages = ["English", "French", "German"]
sessions = sorted(df["session"].unique())
group = person_groups(sessions)
row_group = np.array([group[s] for s in session])
cv_folds = [np.isin(np.arange(len(y)), te) for _, te in GroupKFold(n_splits=6).split(np.zeros(len(y)), groups=row_group)]
splits = [("cv", k, te) for k, te in enumerate(cv_folds)] + [("lolo", l, language == l) for l in languages]
spk_rows = [np.where(speaker == s)[0] for s in pd.unique(speaker)]
flat_cols = [c for c in df.columns if c.startswith("voice_")] + [f"face_{i}" for i in range(5, 714)]   # 797, no bookkeeping


def frame_labels():
    """per-frame engagement labels (25 per window, NaN where missing), cached in data/features/frame_labels.npy"""
    path = "data/features/frame_labels.npy"
    if os.path.exists(path):
        return np.load(path)
    out = np.full((len(df), 25), np.nan, dtype=np.float32)
    for (s, r), idx in df.groupby(["session", "role"]).indices.items():
        lab = pd.to_numeric(pd.read_csv(f"data/noxi/{s}/{r}.engagement.annotation.csv", header=None)[0],
                            errors="coerce").to_numpy("float32")
        for i, w in zip(idx, df["second"].to_numpy()[idx]):
            out[i] = lab[w * 25:(w + 1) * 25]
    np.save(path, out)
    return out


Yf = frame_labels()


def frame_ccc(test, pred):
    lab = Yf[test]; rep = np.repeat(pred[:, None], 25, axis=1); ok = ~np.isnan(lab)
    return ccc(lab[ok], rep[ok])


def past_mean(X, k):
    out = np.empty(X.shape, dtype=np.float32)
    for r in spk_rows:
        c = np.cumsum(X[r].astype(np.float64), axis=0)
        prev = np.vstack([np.zeros((k, X.shape[1])), c[:-k]]) if len(r) > k else np.zeros_like(c)
        out[r] = ((c - prev[:len(r)]) / np.minimum(np.arange(1, len(r) + 1), k)[:, None]).astype(np.float32)
    return out


# ---------------------------------------------------------------- inputs
def flat_input():
    return df[flat_cols].fillna(0).to_numpy("float32")


def scale_flat(X, train, test):
    sc = StandardScaler().fit(X[train])
    return (np.clip(sc.transform(X[train]), -5, 5).astype("float32"), np.clip(sc.transform(X[test]), -5, 5).astype("float32"))


SEQ = np.load("data/features/noxi_sequences.npy", mmap_mode="r")      # windows x 25 x 802 (read from disk)
KEEP = np.r_[0:88, 93:802]                                              # drop the 5 OpenFace bookkeeping values -> 797


def scale_seq(train, test, chunk=4000):
    """scaler fitted on training frames (in chunks), clipped to [-5, 5]; memory-safe"""
    sc = StandardScaler()
    tr_idx, te_idx = np.where(train)[0], np.where(test)[0]
    for i in range(0, len(tr_idx), chunk):
        sc.partial_fit(SEQ[tr_idx[i:i + chunk]][:, :, KEEP].reshape(-1, 797))

    def prep(idx):
        out = np.empty((len(idx), 25, 797), dtype=np.float32)
        for i in range(0, len(idx), chunk):
            part = SEQ[idx[i:i + chunk]][:, :, KEEP].reshape(-1, 797)
            out[i:i + chunk] = np.clip(sc.transform(part), -5, 5).reshape(-1, 25, 797)
        return out
    return prep(tr_idx), prep(te_idx)


def last_10_seconds(Z):
    out = np.zeros((len(Z), 10, Z.shape[1]), dtype="float32")
    for g in spk_rows:
        for lag in range(10):
            out[g[lag:], 9 - lag] = Z[g[:len(g) - lag]]
    return out


# ---------------------------------------------------------------- models
def keras_models():
    from tensorflow import keras
    from tensorflow.keras import layers

    class PositionalEncoding(layers.Layer):
        def __init__(self, n_steps, d_model, **kw):
            super().__init__(**kw); self.n_steps, self.d_model = n_steps, d_model

        def build(self, input_shape):
            self.pos = self.add_weight(shape=(self.n_steps, self.d_model), initializer="random_normal", name="pos")

        def call(self, x):
            return x + self.pos

    def head(x):
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        return layers.Dense(1)(x)

    def ann(n):
        return keras.Sequential([layers.Input(shape=(n,)), layers.Dense(512, activation="relu"), layers.Dropout(0.1),
                                 layers.Dense(256, activation="relu"), layers.Dropout(0.1), layers.Dense(1)])

    def dense_head(n):
        return keras.Sequential([layers.Input(shape=(n,)), layers.Dense(256, activation="relu"), layers.Dropout(0.1),
                                 layers.Dense(128, activation="relu"), layers.Dropout(0.1), layers.Dense(1)])

    def cnn(steps, n):
        inp = layers.Input(shape=(steps, n))
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(inp)
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        return keras.Model(inp, head(layers.Flatten()(x)))

    def rnn(kind):
        def build(steps, n):
            inp = layers.Input(shape=(steps, n))
            cell = {"rnn": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU}[kind](128)
            return keras.Model(inp, head(cell(inp)))
        return build

    def encoder_block(x):
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attention]))
        ff = layers.Dense(128, activation="relu")(x)
        ff = layers.Dropout(0.1)(ff)
        ff = layers.Dense(64)(ff)
        return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ff]))

    def transformer(steps, n):
        inp = layers.Input(shape=(steps, n))
        x = PositionalEncoding(steps, 64)(layers.Dense(64)(inp))
        x = encoder_block(x); x = encoder_block(x)
        return keras.Model(inp, head(layers.GlobalAveragePooling1D()(x)))

    return keras, layers, dict(ann=ann, dense=dense_head, cnn=cnn, rnn=rnn("rnn"), lstm=rnn("lstm"), gru=rnn("gru"),
                               tr1=transformer, tr10=transformer)


def fit_keras(build, X_tr, y_tr, X_te, seed, epochs):
    keras, _, _ = keras_models()
    keras.utils.set_random_seed(seed)
    m = build(*X_tr.shape[1:]) if X_tr.ndim == 3 else build(X_tr.shape[1])
    m.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
    m.fit(X_tr, y_tr, epochs=epochs, batch_size=128, verbose=0)
    return m.predict(X_te, verbose=0, batch_size=1024).flatten(), m.count_params()


# ---------------------------------------------------------------- evaluation
def evaluate(name, family, predict_fn, seeds):
    """predict_fn(train, test, seed) -> (pred, params). Every split x seed; returns one summary row.
    Resume: each finished run is appended to reports/checkpoints/<model>.jsonl. If the script stops (crash, heat, shutdown),
    run the same command again and finished runs are skipped. The checkpoint is deleted when the model is complete."""
    import json, re
    os.makedirs("reports/checkpoints", exist_ok=True)
    ck = f"reports/checkpoints/{GROUP}_{re.sub('[^A-Za-z0-9]+', '_', name).strip('_')}.jsonl"
    saved = {}
    if os.path.exists(ck):
        for line in open(ck, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); saved[(r["kind"], str(r["key"]), r["seed"])] = r
        log(f"   {name}: resuming, {len(saved)} runs already finished (checkpoint {ck})")
    total, done = len(splits) * len(seeds), 0
    res = {seed: {"cv": [], "cv_frame": [], "lolo": {}} for seed in seeds}
    per_session = {}
    params = None
    for kind, key, test in splits:
        for seed in seeds:
            r = saved.get((kind, str(key), seed))
            if r is None:
                pred, p = predict_fn(~test, test, seed)
                r = dict(kind=kind, key=key, seed=seed, ccc=float(ccc(y[test], pred)), params=None if p is None else int(p))
                if kind == "cv":
                    r["frame"] = float(frame_ccc(test, pred))
                    r["sess"] = {s: float(ccc(y[test][session[test] == s], pred[session[test] == s])) for s in np.unique(session[test])}
                with open(ck, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
                tag = ""
            else:
                tag = " [from checkpoint]"
            params = r["params"] if r["params"] is not None else params
            if kind == "cv":
                res[seed]["cv"].append(r["ccc"]); res[seed]["cv_frame"].append(r["frame"])
                for s, v in r["sess"].items():
                    per_session.setdefault(s, []).append(v)
            else:
                res[seed]["lolo"][key] = r["ccc"]
            done += 1
            log(f"   {name}: {done}/{total} models ({kind} {key}, seed {seed}){tag}")
    cv = [np.mean(res[s]["cv"]) for s in seeds]
    cvf = [np.mean(res[s]["cv_frame"]) for s in seeds]
    lolo = [np.mean(list(res[s]["lolo"].values())) for s in seeds]
    row = dict(model=name, family=family, seeds=len(seeds), params=params,
               cv_ccc=np.mean(cv), cv_std_over_seeds=np.std(cv, ddof=1) if len(seeds) > 1 else 0.0,
               cv_std_over_folds=np.mean([np.std(res[s]["cv"], ddof=1) for s in seeds]),
               frame_level_cv_ccc=np.mean(cvf), lolo_ccc=np.mean(lolo),
               lolo_std_over_seeds=np.std(lolo, ddof=1) if len(seeds) > 1 else 0.0,
               **{f"lolo_{l}": np.mean([res[s]["lolo"][l] for s in seeds]) for l in languages})
    log(f"=== {name}: CV {row['cv_ccc']:.3f} +- {row['cv_std_over_seeds']:.3f} (seeds) | frame-level {row['frame_level_cv_ccc']:.3f} "
        f"| train 2 languages -> 3rd {row['lolo_ccc']:.3f}")
    save(row, {s: np.mean(v) for s, v in per_session.items()})
    os.remove(ck)


def save(row, per_session):
    path = "reports/final_models.csv"
    t = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    t = pd.concat([t[t["model"] != row["model"]] if len(t) else t, pd.DataFrame([row])], ignore_index=True)
    t.to_csv(path, index=False)
    path = "reports/final_per_session.csv"
    p = pd.read_csv(path, dtype={"session": str}) if os.path.exists(path) else pd.DataFrame()
    new = pd.DataFrame([dict(model=row["model"], session=s, ccc=v) for s, v in per_session.items()])
    p = pd.concat([p[p["model"] != row["model"]] if len(p) else p, new], ignore_index=True)
    p.to_csv(path, index=False)


# ---------------------------------------------------------------- groups
log(f"group '{GROUP}': {len(df):,} windows, 24 conversations, 6 speaker-independent folds + 3 leave-one-language-out runs")
if GROUP == "classic":
    X = flat_input()
    models = {"Linear Regression": lambda: LinearRegression(),
              "Random Forest": lambda: RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=8),
              "XGBoost": lambda: XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                              colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=8)}
    for name, make in models.items():
        def predict(train, test, seed, make=make):
            sc = StandardScaler().fit(X[train])
            return make().fit(sc.transform(X[train]), y[train]).predict(sc.transform(X[test])), None
        evaluate(name, "classic ML", predict, [42])

elif GROUP in ("ann", "dense"):
    _, _, B = keras_models()
    if GROUP == "ann":
        X = flat_input()
        inputs = [("ANN (MLP 512-256)", X, "ann")]
    else:
        pre = np.hstack([np.load("data/features/pretrained_w2vbert2.npy"), np.load("data/features/pretrained_videomae.npy")])
        inputs = [("dense head: w2v-BERT 2.0 + VideoMAE", pre, "dense"),
                  ("dense head: w2v-BERT 2.0 + VideoMAE + context", np.hstack([pre, past_mean(pre, 5), past_mean(pre, 10)]), "dense")]
    for name, X, kind in inputs:
        def predict(train, test, seed, X=X, kind=kind):
            Xtr, Xte = scale_flat(X, train, test)
            return fit_keras(B[kind], Xtr, y[train], Xte, seed, epochs=30)
        evaluate(name, "ANN" if GROUP == "ann" else "pretrained Transformer + dense head", predict, SEEDS)

elif GROUP == "ae":
    keras, layers, _ = keras_models()
    X = flat_input()
    cache = {}

    def codes(train, test, seed):
        key = (seed, int(np.where(test)[0][0]), int(test.sum()))
        if key not in cache:
            Xtr, Xte = scale_flat(X, train, test)
            keras.utils.set_random_seed(seed)
            inp = layers.Input(shape=(797,))
            code = layers.Dense(32, activation="relu")(layers.Dense(256, activation="relu")(inp))
            out = layers.Dense(797)(layers.Dense(256, activation="relu")(code))
            ae = keras.Model(inp, out); enc = keras.Model(inp, code)
            ae.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
            ae.fit(Xtr, Xtr, epochs=30, batch_size=128, verbose=0)
            cache.clear(); cache[key] = (Xtr, Xte, enc)
        return cache[key]

    def predict_dense(train, test, seed):
        Xtr, Xte, enc = codes(train, test, seed)
        enc.trainable = False
        inp = layers.Input(shape=(797,))
        x = layers.Dropout(0.1)(layers.Dense(64, activation="relu")(enc(inp)))
        head = keras.Model(inp, layers.Dense(1)(x))
        head.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
        head.fit(Xtr, y[train], epochs=30, batch_size=128, verbose=0)
        return head.predict(Xte, verbose=0, batch_size=1024).flatten(), None

    def predict_xgb(train, test, seed):
        Xtr, Xte, enc = codes(train, test, seed)
        m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", random_state=42, n_jobs=8)
        m.fit(enc.predict(Xtr, verbose=0, batch_size=1024), y[train])
        return m.predict(enc.predict(Xte, verbose=0, batch_size=1024)), None

    # the autoencoder of each split and seed is trained once and used by both heads
    total = len(splits) * len(SEEDS); done = 0
    rows = {"Autoencoder + dense head": predict_dense, "Autoencoder + XGBoost": predict_xgb}
    results = {n: {s: {"cv": [], "cv_frame": [], "lolo": {}} for s in SEEDS} for n in rows}
    per_session = {n: {} for n in rows}
    for kind, key, test in splits:
        for seed in SEEDS:
            for n, fn in rows.items():
                pred, _ = fn(~test, test, seed)
                if kind == "cv":
                    results[n][seed]["cv"].append(ccc(y[test], pred)); results[n][seed]["cv_frame"].append(frame_ccc(test, pred))
                    for s in np.unique(session[test]):
                        m = session[test] == s
                        per_session[n].setdefault(s, []).append(ccc(y[test][m], pred[m]))
                else:
                    results[n][seed]["lolo"][key] = ccc(y[test], pred)
            done += 1
            log(f"   autoencoder: {done}/{total} splits ({kind} {key}, seed {seed})")
    for n in rows:
        r = results[n]
        cv = [np.mean(r[s]["cv"]) for s in SEEDS]; lolo = [np.mean(list(r[s]["lolo"].values())) for s in SEEDS]
        row = dict(model=n, family="Autoencoder", seeds=5, params=None, cv_ccc=np.mean(cv), cv_std_over_seeds=np.std(cv, ddof=1),
                   cv_std_over_folds=np.mean([np.std(r[s]["cv"], ddof=1) for s in SEEDS]),
                   frame_level_cv_ccc=np.mean([np.mean(r[s]["cv_frame"]) for s in SEEDS]), lolo_ccc=np.mean(lolo),
                   lolo_std_over_seeds=np.std(lolo, ddof=1), **{f"lolo_{l}": np.mean([r[s]["lolo"][l] for s in SEEDS]) for l in languages})
        log(f"=== {n}: CV {row['cv_ccc']:.3f} +- {row['cv_std_over_seeds']:.3f} | train 2 languages -> 3rd {row['lolo_ccc']:.3f}")
        save(row, {s: np.mean(v) for s, v in per_session[n].items()})

elif GROUP in ("cnn", "rnn", "lstm", "gru", "tr1"):
    _, _, B = keras_models()
    names = {"cnn": ("CNN (VGG style)", "CNN"), "rnn": ("SimpleRNN", "RNN"), "lstm": ("LSTM", "RNN"), "gru": ("GRU", "RNN"),
             "tr1": ("Transformer trained from scratch, 1 s", "Transformer")}
    state = {}

    def predict(train, test, seed):
        key = (int(np.where(test)[0][0]), int(test.sum()))
        if state.get("key") != key:                    # scale each split once, reuse for all 5 seeds
            state.clear(); state["key"] = key; state["data"] = scale_seq(train, test)
        Xtr, Xte = state["data"]
        return fit_keras(B[GROUP], Xtr, y[train], Xte, seed, epochs=20)
    evaluate(*names[GROUP], predict, SEEDS)

elif GROUP == "tr10":
    _, _, B = keras_models()
    X = flat_input()

    def predict(train, test, seed):
        sc = StandardScaler().fit(X[train])
        S = last_10_seconds(np.clip(sc.transform(X), -5, 5).astype("float32"))
        return fit_keras(B["tr10"], S[train], y[train], S[test], seed, epochs=20)
    evaluate("Transformer trained from scratch, 10 s", "Transformer", predict, SEEDS)

log(f"GROUP {GROUP} DONE")

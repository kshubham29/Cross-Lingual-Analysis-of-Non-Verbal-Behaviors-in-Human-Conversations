"""Step 36 (optional check, one seed): the same Transformer as step 29 (group tr10) but its input is the frozen pretrained embeddings
(w2v-BERT 2.0 1,024 + VideoMAE 1,408 = 2,432 per second) for the last 10 seconds, instead of the 797 voice + face features.
Same speaker-independent 6 folds + 3 leave-one-language-out runs, same training settings (Adam 0.001, MSE, batch 128, 20 epochs).
ONE seed only, so it is a check and not a final result. It does not touch reports/final_models.csv.
Run:  python src/36_transformer_pretrained_check.py          (full check, 9 models, resumable)
      python src/36_transformer_pretrained_check.py smoke    (fold 0, 1 epoch, to measure the time)
Output: reports/transformer_pretrained_check.csv, checkpoint reports/checkpoints/tr_pretrained_seed1.jsonl"""
import os, sys, json, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from splits import person_groups

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SEED, EPOCHS, STEPS = 1, (1 if SMOKE else 20), 10


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ccc(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return 2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-9)


df = pd.read_parquet("data/features/noxi_engagement.parquet")
y = df["engagement"].to_numpy("float32")
session = df["session"].to_numpy(); language = df["language"].to_numpy()
speaker = (df["session"] + "_" + df["role"]).to_numpy()
languages = ["English", "French", "German"]
group = person_groups(sorted(df["session"].unique()))
row_group = np.array([group[s] for s in session])
cv_folds = [np.isin(np.arange(len(y)), te) for _, te in GroupKFold(n_splits=6).split(np.zeros(len(y)), groups=row_group)]
splits = [("cv", k, te) for k, te in enumerate(cv_folds)] + [("lolo", l, language == l) for l in languages]
if SMOKE:
    splits = splits[:1]
N = len(y)
Z_raw = np.hstack([np.load("data/features/pretrained_w2vbert2.npy"), np.load("data/features/pretrained_videomae.npy")]).astype("float32")
assert Z_raw.shape == (N, 2432), Z_raw.shape

# for every row: the row itself and the 9 earlier rows of the same speaker (N = a zero row when the recording has not started yet)
IDX = np.full((N, STEPS), N, dtype=np.int64)
for s in pd.unique(speaker):
    g = np.where(speaker == s)[0]
    for lag in range(STEPS):
        IDX[g[lag:], STEPS - 1 - lag] = g[:len(g) - lag]

from tensorflow import keras
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    def __init__(self, n_steps, d_model, **kw):
        super().__init__(**kw); self.n_steps, self.d_model = n_steps, d_model

    def build(self, input_shape):
        self.pos = self.add_weight(shape=(self.n_steps, self.d_model), initializer="random_normal", name="pos")

    def call(self, x):
        return x + self.pos


def encoder_block(x):
    att = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, att]))
    ff = layers.Dense(64)(layers.Dropout(0.1)(layers.Dense(128, activation="relu")(x)))
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ff]))


def build(steps, n):
    inp = layers.Input(shape=(steps, n))
    x = PositionalEncoding(steps, 64)(layers.Dense(64)(inp))
    x = encoder_block(encoder_block(x))
    x = layers.Dense(64, activation="relu")(layers.Dropout(0.1)(layers.GlobalAveragePooling1D()(x)))
    return keras.Model(inp, layers.Dense(1)(x))


class Batches(keras.utils.PyDataset):
    """builds the 10-second sequences batch by batch, so the full (windows x 10 x 2432) array never sits in memory"""
    def __init__(self, Zpad, rows, target=None, batch=128, shuffle=False, seed=0, **kw):
        super().__init__(**kw)
        self.Z, self.rows, self.t, self.b, self.shuffle = Zpad, rows, target, batch, shuffle
        self.rng = np.random.RandomState(seed); self.order = np.arange(len(rows))
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.rows) / self.b))

    def __getitem__(self, i):
        o = self.order[i * self.b:(i + 1) * self.b]
        X = self.Z[IDX[self.rows[o]]]
        return (X, self.t[o]) if self.t is not None else (X,)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def run_split(test):
    train = ~test
    sc = StandardScaler().fit(Z_raw[train])
    Z = np.clip(sc.transform(Z_raw), -5, 5).astype("float32")
    Zpad = np.vstack([Z, np.zeros((1, Z.shape[1]), dtype="float32")])
    tr_rows, te_rows = np.where(train)[0], np.where(test)[0]
    keras.utils.set_random_seed(SEED)
    m = build(STEPS, Z.shape[1])
    m.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
    m.fit(Batches(Zpad, tr_rows, y[tr_rows], shuffle=True, seed=SEED), epochs=EPOCHS, verbose=0)
    pred = m.predict(Batches(Zpad, te_rows, batch=1024), verbose=0).flatten()
    return pred, m.count_params()


CK = "reports/checkpoints/tr_pretrained_seed1.jsonl"
os.makedirs("reports/checkpoints", exist_ok=True)
saved = {}
if os.path.exists(CK) and not SMOKE:
    for line in open(CK, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); saved[(r["kind"], str(r["key"]))] = r
    log(f"resuming: {len(saved)} runs already finished")
log(f"Transformer on pretrained embeddings, last {STEPS} s, 2432 inputs, seed {SEED}, {EPOCHS} epochs, {len(splits)} runs")
rows = []
for n, (kind, key, test) in enumerate(splits, 1):
    r = saved.get((kind, str(key)))
    if r is None:
        t0 = time.time()
        pred, params = run_split(test)
        r = dict(kind=kind, key=key, ccc=float(ccc(y[test], pred)), params=int(params), seconds=round(time.time() - t0))
        if not SMOKE:
            with open(CK, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
    rows.append(r)
    log(f"   {n}/{len(splits)} ({kind} {key}): CCC {r['ccc']:.3f}  [{r.get('seconds', '?')} s]  params {r['params']:,}")
if SMOKE:
    log("smoke test finished (1 epoch, fold 0); nothing saved")
    sys.exit()
cv = [r["ccc"] for r in rows if r["kind"] == "cv"]
lolo = {r["key"]: r["ccc"] for r in rows if r["kind"] == "lolo"}
out = dict(model="Transformer on pretrained embeddings (w2v-BERT 2.0 + VideoMAE), last 10 s, one seed", seed=SEED, params=rows[0]["params"],
           cv_ccc=np.mean(cv), cv_std_over_folds=np.std(cv, ddof=1), lolo_ccc=np.mean(list(lolo.values())),
           **{f"lolo_{l}": lolo[l] for l in languages}, **{f"fold{i + 1}": c for i, c in enumerate(cv)})
pd.DataFrame([out]).to_csv("reports/transformer_pretrained_check.csv", index=False)
log(f"=== CV {out['cv_ccc']:.3f} (std over folds {out['cv_std_over_folds']:.3f}) | train 2 languages -> 3rd {out['lolo_ccc']:.3f} "
    f"({', '.join(f'{l} {lolo[l]:.3f}' for l in languages)})")
log("for comparison (5-seed means): Transformer 797 features 0.654 / 0.600; dense head + context 0.767 / 0.733; dense head 0.676 / 0.627")

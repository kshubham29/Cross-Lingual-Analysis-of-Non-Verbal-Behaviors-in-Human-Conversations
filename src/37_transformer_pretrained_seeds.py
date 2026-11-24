"""Step 37: seeds 2-5 of the Transformer on the frozen pretrained embeddings (step 36 ran seed 1), then the 5-seed summary and a paired test against the best model.
Same folds, same settings as step 36 (last 10 seconds of w2v-BERT 2.0 + VideoMAE = 2,432 inputs per second; 2 encoder blocks, 4 heads; Adam 0.001, MSE, batch 128, 20 epochs).
Per-session CCC is saved for seeds 2-5 (step 36 did not save it), so the paired test uses the mean of seeds 2-5 per session.
Run:  python src/37_transformer_pretrained_seeds.py          (36 models, resumable; about 1.5-2 hours)
Needs reports/transformer_pretrained_seed1_runs.jsonl (the seed-1 runs of step 36).
Output: reports/transformer_pretrained_5seeds.csv, transformer_pretrained_per_session.csv, transformer_pretrained_vs_best.csv
Does not touch reports/final_models.csv."""
import os, sys, json, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from splits import person_groups

SEEDS, EPOCHS, STEPS = [2, 3, 4, 5], 20, 10


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
N = len(y)
Z_raw = np.hstack([np.load("data/features/pretrained_w2vbert2.npy"), np.load("data/features/pretrained_videomae.npy")]).astype("float32")
assert Z_raw.shape == (N, 2432), Z_raw.shape
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


def run_split(test, seed):
    train = ~test
    sc = StandardScaler().fit(Z_raw[train])
    Z = np.clip(sc.transform(Z_raw), -5, 5).astype("float32")
    Zpad = np.vstack([Z, np.zeros((1, Z.shape[1]), dtype="float32")])
    tr_rows, te_rows = np.where(train)[0], np.where(test)[0]
    keras.utils.set_random_seed(seed)
    m = build(STEPS, Z.shape[1])
    m.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
    m.fit(Batches(Zpad, tr_rows, y[tr_rows], shuffle=True, seed=seed), epochs=EPOCHS, verbose=0)
    pred = m.predict(Batches(Zpad, te_rows, batch=1024), verbose=0).flatten()
    return pred, m.count_params()


CK = "reports/checkpoints/tr_pretrained_seeds2_5.jsonl"
os.makedirs("reports/checkpoints", exist_ok=True)
done = {}
if os.path.exists(CK):
    for line in open(CK, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); done[(r["seed"], r["kind"], str(r["key"]))] = r
    log(f"resuming: {len(done)} of {len(SEEDS) * len(splits)} runs already finished")
log(f"Transformer on pretrained embeddings, seeds {SEEDS}, {EPOCHS} epochs, {len(SEEDS) * len(splits)} runs")
n = 0
for seed in SEEDS:
    for kind, key, test in splits:
        n += 1
        r = done.get((seed, kind, str(key)))
        if r is None:
            t0 = time.time()
            pred, params = run_split(test, seed)
            r = dict(seed=seed, kind=kind, key=key, ccc=float(ccc(y[test], pred)), params=int(params), seconds=round(time.time() - t0))
            if kind == "cv":
                r["sess"] = {s: float(ccc(y[test][session[test] == s], pred[session[test] == s])) for s in np.unique(session[test])}
            with open(CK, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            done[(seed, kind, str(key))] = r
        log(f"   {n}/{len(SEEDS) * len(splits)} (seed {seed}, {kind} {key}): CCC {r['ccc']:.3f}  [{r.get('seconds', '?')} s]")

# ---------------- summary over the 5 seeds (seed 1 from step 36)
runs = [dict(json.loads(l), seed=1) for l in open("reports/transformer_pretrained_seed1_runs.jsonl", encoding="utf-8") if l.strip()] + list(done.values())
per_seed = {}
for seed in [1] + SEEDS:
    rs = [r for r in runs if r["seed"] == seed]
    cv = [r["ccc"] for r in rs if r["kind"] == "cv"]; lo = {r["key"]: r["ccc"] for r in rs if r["kind"] == "lolo"}
    per_seed[seed] = dict(cv=np.mean(cv), cv_fold_std=np.std(cv, ddof=1), lolo=np.mean(list(lo.values())), **{l: lo[l] for l in languages})
cvs = [v["cv"] for v in per_seed.values()]; los = [v["lolo"] for v in per_seed.values()]
row = dict(model="Transformer on pretrained embeddings (w2v-BERT 2.0 + VideoMAE), last 10 s", seeds=5, params=runs[0]["params"],
           cv_ccc=np.mean(cvs), cv_std_over_seeds=np.std(cvs, ddof=1), cv_std_over_folds=np.mean([v["cv_fold_std"] for v in per_seed.values()]),
           lolo_ccc=np.mean(los), lolo_std_over_seeds=np.std(los, ddof=1), **{f"lolo_{l}": np.mean([v[l] for v in per_seed.values()]) for l in languages},
           **{f"cv_seed{s}": v["cv"] for s, v in per_seed.items()})
pd.DataFrame([row]).to_csv("reports/transformer_pretrained_5seeds.csv", index=False)
log(f"=== 5 seeds: CV {row['cv_ccc']:.3f} +- {row['cv_std_over_seeds']:.3f} | train 2 languages -> 3rd {row['lolo_ccc']:.3f} +- {row['lolo_std_over_seeds']:.3f} "
    f"({', '.join(f'{l} {row[f'lolo_{l}']:.3f}' for l in languages)}) | per seed CV: {', '.join(f'{c:.3f}' for c in cvs)}")

# ---------------- per-session CCC (mean of seeds 2-5) and paired test against the best model
sess = {}
for r in done.values():
    if r["kind"] == "cv":
        for s, v in r["sess"].items():
            sess.setdefault(s, []).append(v)
ps = pd.DataFrame([dict(session=s, ccc=np.mean(v), seeds=len(v)) for s, v in sess.items()]).sort_values("session")
ps.to_csv("reports/transformer_pretrained_per_session.csv", index=False)
best = pd.read_csv("reports/final_per_session.csv", dtype={"session": str})
best = best[best.model == "dense head: w2v-BERT 2.0 + VideoMAE + context"].set_index("session").ccc
mine = ps.set_index("session").ccc
common = sorted(set(mine.index) & set(best.index))
diff = (mine[common] - best[common]).to_numpy()
rng = np.random.RandomState(0)
boot = [diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(10000)]
out = dict(comparison="Transformer on pretrained embeddings minus dense head + context (per-session CCC; Transformer = mean of seeds 2-5)", sessions=len(diff),
           mean_diff=diff.mean(), ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
           transformer_better=f"{int((diff > 0).sum())}/{len(diff)}", p_wilcoxon=wilcoxon(diff).pvalue)
pd.DataFrame([out]).to_csv("reports/transformer_pretrained_vs_best.csv", index=False)
log(f"=== vs best model over {len(diff)} sessions: mean diff {out['mean_diff']:+.3f} (95% CI {out['ci_low']:.3f} to {out['ci_high']:.3f}), better in {out['transformer_better']}, Wilcoxon p = {out['p_wilcoxon']:.3f}")
log("DONE")

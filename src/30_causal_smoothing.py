"""Step 30: causal smoothing audit (supervisor's review, E4: leakage audit of context and smoothing).
The model predicts each second separately. Smoothing averages each prediction with its neighbours of the SAME speaker:
  raw       no smoothing
  causal    mean of the current and the previous k-1 predictions (past only; the only version that is allowed)
  centered  mean of k predictions around the current one (uses FUTURE predictions; shown only as a reference, not allowed)
Predictions come from the input features only (no labels are used anywhere in the smoothing).
Same 6 speaker-independent folds as step 29; 5 seeds. A test fold contains whole speakers, so all neighbours are test predictions.
Run:  python src/30_causal_smoothing.py <dense|densectx> [number of seeds]
  dense     dense head on w2v-BERT 2.0 + VideoMAE (1 s)                 -> about 30 min
  densectx  the same + past context (best model of step 29)             -> about 3.5 min per model
  selftest  checks the smoothing functions on a tiny example (no data, a few seconds)
Result: reports/smoothing_causal.csv (one row per model and smoothing setting)"""
import os, sys, time, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
GROUP = sys.argv[1]
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEEDS = list(range(1, N_SEEDS + 1))
KS = [3, 5, 10]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ccc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64); y_pred = np.asarray(y_pred, dtype=np.float64)
    cov = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    return 2 * cov / (y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2 + 1e-9)


def causal_mean(p, k):
    """out[i] = mean(p[i-k+1 .. i]) (fewer values at the start). Uses only the past and the present."""
    c = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    i = np.arange(len(p))
    lo = np.maximum(0, i - k + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def centered_mean(p, k):
    """out[i] = mean(p[i-k//2 .. i+k//2]); uses future predictions -> reference only"""
    c = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    i = np.arange(len(p))
    lo, hi = np.maximum(0, i - k // 2), np.minimum(len(p), i + k // 2 + 1)
    return (c[hi] - c[lo]) / (hi - lo)


if GROUP == "selftest":
    p = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    assert np.allclose(causal_mean(p, 3), [1, 2, 3, 5, 7]), causal_mean(p, 3)
    assert np.allclose(centered_mean(p, 3), [2, 3, 5, 7, 8]), centered_mean(p, 3)
    p2 = p.copy(); p2[4] = 100.0                         # changing the LAST value must not change earlier causal outputs
    assert np.allclose(causal_mean(p2, 3)[:4], causal_mean(p, 3)[:4])
    assert not np.allclose(centered_mean(p2, 3)[:4], centered_mean(p, 3)[:4])   # centered does look at the future
    log("selftest OK: causal smoothing never uses future values, centered smoothing does")
    sys.exit(0)

sys.path.insert(0, "src")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from splits import person_groups

df = pd.read_parquet("data/features/noxi_engagement.parquet", columns=["engagement", "session", "role", "language"])
y = df["engagement"].to_numpy("float32")
session = df["session"].to_numpy()
speaker = (df["session"] + "_" + df["role"]).to_numpy()
group = person_groups(sorted(df["session"].unique()))
row_group = np.array([group[s] for s in session])
cv_folds = [np.isin(np.arange(len(y)), te) for _, te in GroupKFold(n_splits=6).split(np.zeros(len(y)), groups=row_group)]
spk_rows = [np.where(speaker == s)[0] for s in pd.unique(speaker)]           # rows of one speaker, in time order


def past_mean(X, k):
    out = np.empty(X.shape, dtype=np.float32)
    for r in spk_rows:
        c = np.cumsum(X[r].astype(np.float64), axis=0)
        prev = np.vstack([np.zeros((k, X.shape[1])), c[:-k]]) if len(r) > k else np.zeros_like(c)
        out[r] = ((c - prev[:len(r)]) / np.minimum(np.arange(1, len(r) + 1), k)[:, None]).astype(np.float32)
    return out


pre = np.hstack([np.load("data/features/pretrained_w2vbert2.npy"), np.load("data/features/pretrained_videomae.npy")])
if GROUP == "densectx":
    X = np.hstack([pre, past_mean(pre, 5), past_mean(pre, 10)])
    NAME = "dense head: w2v-BERT 2.0 + VideoMAE + context"
elif GROUP == "dense":
    X, NAME = pre, "dense head: w2v-BERT 2.0 + VideoMAE"
else:
    sys.exit("group must be dense, densectx or selftest")


def fit_predict(train, test, seed):
    from tensorflow import keras
    from tensorflow.keras import layers
    sc = StandardScaler().fit(X[train])
    Xtr, Xte = (np.clip(sc.transform(X[m]), -5, 5).astype("float32") for m in (train, test))
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([layers.Input(shape=(Xtr.shape[1],)), layers.Dense(256, activation="relu"), layers.Dropout(0.1),
                          layers.Dense(128, activation="relu"), layers.Dropout(0.1), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
    m.fit(Xtr, y[train], epochs=30, batch_size=128, verbose=0)
    return m.predict(Xte, verbose=0, batch_size=1024).flatten()


def smoothed_scores(test, pred):
    """CCC of the raw predictions and of every smoothing variant, smoothing inside each speaker (labels are never used)"""
    idx = np.where(test)[0]
    pos = {r: i for i, r in enumerate(idx)}
    variants = {"raw": lambda p: p}
    for k in KS:
        variants[f"causal k={k}"] = lambda p, k=k: causal_mean(p, k)
    variants["centered k=5 (uses future, reference only)"] = lambda p: centered_mean(p, 5)
    out = {}
    for name, fn in variants.items():
        sm = np.empty(len(idx))
        for r in spk_rows:
            if not np.isin(r, idx).all():
                continue                                   # speaker not (fully) in this test fold
            ii = [pos[v] for v in r]
            sm[ii] = fn(pred[ii])
        out[name] = float(ccc(y[test], sm))
    return out


os.makedirs("reports/checkpoints", exist_ok=True)
ck = f"reports/checkpoints/smoothing_{GROUP}.jsonl"
saved = {}
if os.path.exists(ck):
    for line in open(ck, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); saved[(r["fold"], r["seed"])] = r["scores"]
    log(f"resuming: {len(saved)} runs already finished")
total, done = len(cv_folds) * len(SEEDS), 0
log(f"{NAME}: {len(cv_folds)} speaker-independent folds x {len(SEEDS)} seeds = {total} models")
for k, test in enumerate(cv_folds):
    for seed in SEEDS:
        if (k, seed) not in saved:
            saved[(k, seed)] = smoothed_scores(test, fit_predict(~test, test, seed))
            with open(ck, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(fold=k, seed=seed, scores=saved[(k, seed)])) + "\n")
        done += 1
        s = saved[(k, seed)]
        log(f"   {NAME}: {done}/{total} models (cv {k}, seed {seed}) raw {s['raw']:.3f} causal k=5 {s['causal k=5']:.3f}")

rows = []
variants = list(next(iter(saved.values())).keys())
raw_by_seed = [np.mean([saved[(k, s)]["raw"] for k in range(len(cv_folds))]) for s in SEEDS]
for v in variants:
    by_seed = [np.mean([saved[(k, s)][v] for k in range(len(cv_folds))]) for s in SEEDS]
    gain = [b - r for b, r in zip(by_seed, raw_by_seed)]
    rows.append(dict(model=NAME, smoothing=v, uses_future=v.startswith("centered"), seeds=len(SEEDS),
                     cv_ccc=np.mean(by_seed), cv_std_over_seeds=np.std(by_seed, ddof=1) if len(SEEDS) > 1 else 0.0,
                     gain_vs_raw=np.mean(gain), gain_std_over_seeds=np.std(gain, ddof=1) if len(SEEDS) > 1 else 0.0,
                     seeds_with_gain=int(sum(g > 0 for g in gain))))
path = "reports/smoothing_causal.csv"
t = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
t = pd.concat([t[t["model"] != NAME] if len(t) else t, pd.DataFrame(rows)], ignore_index=True)
t.to_csv(path, index=False)
for r in rows:
    log(f"=== {r['smoothing']}: CV {r['cv_ccc']:.3f} | gain vs raw {r['gain_vs_raw']:+.3f} +- {r['gain_std_over_seeds']:.3f} "
        f"({r['seeds_with_gain']}/{len(SEEDS)} seeds improve)")
os.remove(ck)
log(f"GROUP {GROUP} DONE")

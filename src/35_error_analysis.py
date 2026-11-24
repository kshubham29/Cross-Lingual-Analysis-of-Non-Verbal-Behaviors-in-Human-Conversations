"""Step 35: error analysis, predicted vs true engagement (supervisor's review, E16).
Retrains the best final model (dense head on w2v-BERT 2.0 + VideoMAE embeddings + past 5 s / 10 s context; same settings as step 29, ONE seed = 1)
on the 6 speaker-independent folds and keeps the out-of-fold predictions (every window is predicted by a model that never saw that person).
Outputs:
  reports/error_analysis_sessions.csv   per session: CCC, MAE, bias (mean prediction minus mean truth), language
  reports/error_analysis_levels.csv     error by engagement level (low / middle / high true engagement) and the slope of truth on prediction
  charts/error_analysis_timelines.png   true and predicted engagement over time, the 2 best and 2 worst sessions (expert and novice)
  charts/error_analysis_scatter.png     predicted vs true, all windows
  data/features/oof_dense_context_seed1.npy   the predictions (derived data, stays local)
Run: python src/35_error_analysis.py  (about 15-25 min: 6 models)"""
import os, sys, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from splits import person_groups

SEED = 1


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ccc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-9)


df = pd.read_parquet("data/features/noxi_engagement.parquet", columns=["engagement", "session", "role", "language", "second"])
y = df["engagement"].to_numpy("float32"); session = df["session"].to_numpy(); role = df["role"].to_numpy(); language = df["language"].to_numpy()
speaker = (df["session"] + "_" + df["role"]).to_numpy(); spk_rows = [np.where(speaker == s)[0] for s in pd.unique(speaker)]
group = person_groups(sorted(df["session"].unique())); row_group = np.array([group[s] for s in session])
folds = [te for _, te in GroupKFold(n_splits=6).split(np.zeros(len(y)), groups=row_group)]


def past_mean(X, k):                                       # same causal context as step 29: mean of the current and the previous k-1 seconds
    out = np.empty(X.shape, dtype=np.float32)
    for r in spk_rows:
        c = np.cumsum(X[r].astype(np.float64), axis=0)
        prev = np.vstack([np.zeros((k, X.shape[1])), c[:-k]]) if len(r) > k else np.zeros_like(c)
        out[r] = ((c - prev[:len(r)]) / np.minimum(np.arange(1, len(r) + 1), k)[:, None]).astype(np.float32)
    return out


pre = np.hstack([np.load("data/features/pretrained_w2vbert2.npy"), np.load("data/features/pretrained_videomae.npy")])
X = np.hstack([pre, past_mean(pre, 5), past_mean(pre, 10)])
log(f"input: {X.shape[0]:,} windows x {X.shape[1]} values (embeddings + past 5 s + past 10 s)")

from tensorflow import keras
from tensorflow.keras import layers

pred = np.zeros(len(y), dtype=np.float32)
for k, te_idx in enumerate(folds):
    tr = np.ones(len(y), bool); tr[te_idx] = False
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = np.clip(sc.transform(X[tr]), -5, 5).astype("float32"), np.clip(sc.transform(X[te_idx]), -5, 5).astype("float32")
    keras.utils.set_random_seed(SEED)
    m = keras.Sequential([layers.Input(shape=(Xtr.shape[1],)), layers.Dense(256, activation="relu"), layers.Dropout(0.1),
                          layers.Dense(128, activation="relu"), layers.Dropout(0.1), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
    m.fit(Xtr, y[tr], epochs=30, batch_size=128, verbose=0)
    pred[te_idx] = m.predict(Xte, verbose=0, batch_size=1024).flatten()
    log(f"   fold {k + 1}/6: CCC {ccc(y[te_idx], pred[te_idx]):.3f}")
np.save("data/features/oof_dense_context_seed1.npy", pred)
log(f"all folds: mean fold CCC {np.mean([ccc(y[f], pred[f]) for f in folds]):.3f} (final-model table, 5 seeds: 0.767)")

# ---------------------------------------------------------------- per session
rows = []
for s in sorted(np.unique(session)):
    m_ = session == s
    rows.append(dict(session=s, language=language[m_][0], windows=int(m_.sum()), ccc=ccc(y[m_], pred[m_]), mae=np.abs(y[m_] - pred[m_]).mean(),
                     bias=float(pred[m_].mean() - y[m_].mean()), mean_true=float(y[m_].mean()), std_true=float(y[m_].std()), std_pred=float(pred[m_].std())))
sess = pd.DataFrame(rows).sort_values("ccc", ascending=False).reset_index(drop=True); sess.to_csv("reports/error_analysis_sessions.csv", index=False)

# ---------------------------------------------------------------- by engagement level
bins = [(0.0, 0.4, "low (true < 0.4)"), (0.4, 0.7, "middle (0.4 to 0.7)"), (0.7, 1.01, "high (true >= 0.7)")]
lv = [dict(level=n, windows=int(((y >= a) & (y < b)).sum()), mean_true=float(y[(y >= a) & (y < b)].mean()), mean_pred=float(pred[(y >= a) & (y < b)].mean()),
           mae=float(np.abs(y - pred)[(y >= a) & (y < b)].mean()), bias=float((pred - y)[(y >= a) & (y < b)].mean())) for a, b, n in bins]
slope = float(np.polyfit(pred, y, 1)[0])
lv.append(dict(level=f"all (slope of truth on prediction = {slope:.2f}; 1.0 would be ideal)", windows=len(y), mean_true=float(y.mean()), mean_pred=float(pred.mean()),
               mae=float(np.abs(y - pred).mean()), bias=float((pred - y).mean())))
pd.DataFrame(lv).to_csv("reports/error_analysis_levels.csv", index=False)

# ---------------------------------------------------------------- charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("charts", exist_ok=True)
chosen = list(sess.session[:2]) + list(sess.session[-2:])
fig, axes = plt.subplots(len(chosen), 2, figsize=(14, 2.6 * len(chosen)), sharey=True)
for i, s in enumerate(chosen):
    srow = sess[sess.session == s].iloc[0]
    for j, r in enumerate(["expert", "novice"]):
        idx = np.where((session == s) & (role == r))[0]; t = df["second"].to_numpy()[idx]
        axes[i, j].plot(t, y[idx], color="black", lw=1.2, label="true"); axes[i, j].plot(t, pred[idx], color="#888888", lw=1.2, label="predicted")
        axes[i, j].set_title(f"{'BEST' if i < 2 else 'WORST'}: session {s} ({srow.language}), {r}; session CCC {srow.ccc:.2f}", fontsize=9)
        axes[i, j].set_ylim(0, 1)
        if i == len(chosen) - 1: axes[i, j].set_xlabel("second")
axes[0, 0].legend(loc="lower left", fontsize=8); axes[0, 0].set_ylabel("engagement")
plt.tight_layout(); plt.savefig("charts/error_analysis_timelines.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.hist2d(y, pred, bins=50, range=[[0, 1], [0, 1]], cmap="Greys", cmin=1); ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--")
ax.set_xlabel("true engagement"); ax.set_ylabel("predicted engagement"); ax.set_title(f"All windows, out-of-fold (seed {SEED})")
plt.tight_layout(); plt.savefig("charts/error_analysis_scatter.png", dpi=130); plt.close()

log("=== sessions (CCC): best " + ", ".join(f"{r.session} {r.ccc:.2f}" for r in sess.head(3).itertuples()) + " | worst " + ", ".join(f"{r.session} {r.ccc:.2f}" for r in sess.tail(3).itertuples()))
log("=== by level: " + "; ".join(f"{r['level'].split(' (')[0]} bias {r['bias']:+.3f}, MAE {r['mae']:.3f}" for r in lv[:3]) + f"; slope of truth on prediction {slope:.2f}")
log("saved reports/error_analysis_sessions.csv, error_analysis_levels.csv, charts/error_analysis_timelines.png, error_analysis_scatter.png")
log("GROUP error DONE")

"""Step 28: experiments for the supervisor's review, on all 24 labelled EN/FR/DE conversations, one pipeline.
Stage 1 (python src/28_review_experiments.py 1): Gradient Boosting on 10 inputs
  - speaker-independent 6-fold CV (no person in both train and test)          -> problems 1, 4, E1, E2
  - leave-one-language-out: train on 2 languages, test on the 3rd             -> E2
  - within vs cross language with MATCHED training size                         -> problem 5, E2
  - held-out test set (6 sessions) used once; model choice made on the other 18 -> problem 4c, E2
  - baselines: training mean, session mean, persistence                       -> problem 7, E5
  - face tests: AUs+gaze+pose only, per-speaker z-normalisation, detection per site -> problems 2, 3, E6
  - per-session CCC, bootstrap 95% CI and Wilcoxon over sessions               -> E3
  Context and smoothing use only past input features of the same speaker (no labels)  -> problem 4a, E4
Stage 2 (python src/28_review_experiments.py 2): neural models with 5 seeds (ANN, dense head, Transformer 10 s) -> E3, problem 6
Results: reports/review/*.csv"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from splits import person_groups

OUT = "reports"
os.makedirs(OUT, exist_ok=True)
STAGE = sys.argv[1] if len(sys.argv) > 1 else "1"


def ccc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    cov = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    return 2 * cov / (y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2 + 1e-9)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- data
df = pd.read_parquet("data/features/noxi_engagement.parquet")
y = df["engagement"].to_numpy("float32")
session = df["session"].to_numpy()
language = df["language"].to_numpy()
speaker = (df["session"] + "_" + df["role"]).to_numpy()
languages = ["English", "French", "German"]
sessions = sorted(df["session"].unique())
group = person_groups(sessions)                                  # sessions sharing a person -> same group
row_group = np.array([group[s] for s in session])
lang_of = df.groupby("session")["language"].first().to_dict()

# OpenFace 2.0 column order (checked on the data): 0-4 bookkeeping, 5-12 gaze, 293-298 head pose,
# 679-695 action-unit intensities, 696-713 action-unit presence; the rest are 2D/3D landmark coordinates
voice_cols = [f"voice_{i}" for i in range(88)]
face_cols = [f"face_{i}" for i in range(5, 714)]
aux_idx = list(range(5, 13)) + list(range(293, 299)) + list(range(679, 714))
face_aux_cols = [f"face_{i}" for i in aux_idx]                   # 49 features: gaze + head pose + AUs

# speaker index ranges (rows of each speaker are consecutive and in time order)
spk_rows = [np.where(speaker == s)[0] for s in pd.unique(speaker)]


def past_mean(X, k):
    """mean of the same speaker's current and previous k-1 seconds (only the past; no labels)"""
    out = np.empty(X.shape, dtype=np.float32)
    for r in spk_rows:
        c = np.cumsum(X[r].astype(np.float64), axis=0)
        prev = np.vstack([np.zeros((k, X.shape[1])), c[:-k]]) if len(r) > k else np.zeros_like(c)
        n = np.minimum(np.arange(1, len(r) + 1), k)[:, None]
        out[r] = ((c - prev[:len(r)]) / n).astype(np.float32)
    return out


def with_context(X):
    return np.hstack([X, past_mean(X, 5), past_mean(X, 10)])


def speaker_norm(X):
    """z-score every feature with the speaker's own mean and std (label-free)"""
    out = np.empty(X.shape, dtype=np.float32)
    for r in spk_rows:
        mu, sd = X[r].mean(axis=0), X[r].std(axis=0)
        out[r] = (X[r] - mu) / np.where(sd > 0, sd, 1)
    return out


def get(cols):
    return df[cols].fillna(0).to_numpy("float32")


def make_input(name):
    if name == "voice": return get(voice_cols)
    if name == "face (all 709)": return get(face_cols)
    if name == "face (AU+gaze+pose, 49)": return get(face_aux_cols)
    if name == "face (all 709), speaker-normalised": return speaker_norm(get(face_cols))
    if name == "face (AU+gaze+pose), speaker-normalised": return speaker_norm(get(face_aux_cols))
    if name == "voice + face": return get(voice_cols + face_cols)
    if name == "voice + face (AU+gaze+pose)": return get(voice_cols + face_aux_cols)
    if name == "voice + face + context": return with_context(get(voice_cols + face_cols))
    fused = np.hstack([get(voice_cols), np.load("data/features/pretrained_w2vbert2.npy"),
                       np.load("data/features/pretrained_videomae.npy")])
    if name == "fusion": return fused
    if name == "fusion + context (best)": return with_context(fused)
    raise KeyError(name)


INPUTS = ["voice", "face (all 709)", "face (AU+gaze+pose, 49)", "face (all 709), speaker-normalised",
          "face (AU+gaze+pose), speaker-normalised", "voice + face", "voice + face (AU+gaze+pose)",
          "voice + face + context", "fusion", "fusion + context (best)"]


PROGRESS = {"part": "", "done": 0, "total": 0}


def progress(part, total):
    """start a new part; a progress line is printed after every few models"""
    PROGRESS.update(part=part, done=0, total=total)


def gb(X, train, test):
    scaler = StandardScaler()
    model = HistGradientBoostingRegressor(max_iter=300, random_state=42)
    model.fit(scaler.fit_transform(X[train]), y[train])
    pred = model.predict(scaler.transform(X[test]))
    PROGRESS["done"] += 1
    every = max(1, PROGRESS["total"] // 12)
    if PROGRESS["total"] and (PROGRESS["done"] % every == 0 or PROGRESS["done"] == PROGRESS["total"]):
        log(f"   {PROGRESS['part']}: {PROGRESS['done']}/{PROGRESS['total']} models")
    return pred


# ---------------------------------------------------------------- protocol
# speaker-independent 6-fold CV over person groups (GroupKFold keeps every group in one fold)
cv_folds = [np.isin(np.arange(len(y)), te) for _, te in GroupKFold(n_splits=6).split(np.zeros(len(y)), groups=row_group)]
# held-out test set used once: 2 sessions per language, person-disjoint from the other 18
TEST_SESSIONS = ["062", "063", "007", "009", "072", "073"]
is_test = np.isin(session, TEST_SESSIONS)
dev_groups = row_group[~is_test]
dev_rows = np.where(~is_test)[0]
dev_folds = [dev_rows[te] for _, te in GroupKFold(n_splits=4).split(dev_rows, groups=dev_groups)]
assert not set(row_group[is_test]) & set(dev_groups), "test and development share a person"


def session_cccs(pred, rows):
    return {s: ccc(y[rows][session[rows] == s], pred[session[rows] == s]) for s in np.unique(session[rows])}


if STAGE == "1":
    # ------------------------------------------------ site / face-detection statistics (E6c)
    site = df.groupby("language").agg(windows=("engagement", "size"), mean_engagement=("engagement", "mean"),
                                      face_confidence=("face_3", "mean"), face_success_rate=("face_4", "mean"))
    site.to_csv(os.path.join(OUT, "site_face_detection.csv"))
    log("site statistics\n" + site.round(3).to_string())

    # ------------------------------------------------ baselines (E5): computed on the same CV folds
    base = []
    for name in ["training mean", "session mean (uses test labels, reference)", "persistence: previous second (uses labels, reference)"]:
        fold_scores, per_session = [], {}
        for te in cv_folds:
            tr = ~te
            if name.startswith("training"):
                p = np.full(te.sum(), y[tr].mean())
            elif name.startswith("session"):
                p = df.loc[te, "engagement"].groupby(session[te]).transform("mean").to_numpy()
            else:
                p = pd.Series(y[te]).groupby(speaker[te]).shift(1).bfill().to_numpy()
            fold_scores.append(ccc(y[te], p))
            per_session.update(session_cccs(p, np.where(te)[0]))
        base.append(dict(model=name, cv_ccc=np.mean(fold_scores), cv_std=np.std(fold_scores, ddof=1),
                         median_session_ccc=np.median(list(per_session.values()))))
    pd.DataFrame(base).to_csv(os.path.join(OUT, "baselines.csv"), index=False)
    log("baselines\n" + pd.DataFrame(base).round(3).to_string(index=False))

    # ------------------------------------------------ Gradient Boosting on every input
    rows, per_session_rows, lolo_rows, matched_rows, dev_rows_out = [], [], [], [], []
    for name in INPUTS:
        X = make_input(name)
        log(f"=== {name}  ({X.shape[1]} features)")
        # 1) speaker-independent CV
        progress("speaker-independent CV", len(cv_folds))
        pred = np.zeros(len(y), dtype=np.float64); fold_scores = []
        for te in cv_folds:
            pred[te] = gb(X, ~te, te); fold_scores.append(ccc(y[te], pred[te]))
        ps = session_cccs(pred, np.arange(len(y)))
        per_session_rows += [dict(input=name, session=s, ccc=v) for s, v in ps.items()]
        # 2) leave-one-language-out
        progress("train on 2 languages -> 3rd", len(languages))
        lolo = {}
        for l in languages:
            te = language == l
            lolo[l] = ccc(y[te], gb(X, ~te, te))
        lolo_rows.append(dict(input=name, **{f"test_{l}": lolo[l] for l in languages}, mean=np.mean(list(lolo.values()))))
        # 3) within vs cross language with matched training size (person-disjoint)
        progress("within vs cross (matched size)", 3 * len(sessions))
        rng = np.random.default_rng(42)
        within, cross = [], []
        for s in sessions:
            l = lang_of[s]
            same = [t for t in sessions if lang_of[t] == l and group[t] != group[s]]
            te = session == s
            within.append(ccc(y[te], gb(X, np.isin(session, same), te)))
            for other in languages:
                if other == l:
                    continue
                pool = [t for t in sessions if lang_of[t] == other]
                pick = rng.choice(pool, size=min(len(same), len(pool)), replace=False)
                cross.append(ccc(y[te], gb(X, np.isin(session, pick), te)))
        w, c = np.mean(within), np.mean(cross)
        matched_rows.append(dict(input=name, within=w, cross=c, drop_abs=w - c, drop_pct=100 * (w - c) / w))
        # 4) model choice on the 18 development sessions (4 speaker-independent folds)
        progress("model choice on 18 sessions", len(dev_folds))
        dev_scores = [ccc(y[te], gb(X, np.setdiff1d(dev_rows, te), te)) for te in dev_folds]
        dev_rows_out.append(dict(input=name, dev_ccc=np.mean(dev_scores), dev_std=np.std(dev_scores, ddof=1)))
        rows.append(dict(input=name, features=X.shape[1], cv_ccc=np.mean(fold_scores), cv_std=np.std(fold_scores, ddof=1),
                         median_session_ccc=np.median(list(ps.values())), lolo_mean=lolo_rows[-1]["mean"],
                         within_matched=w, cross_matched=c, drop_abs=w - c, drop_pct=100 * (w - c) / w,
                         dev_ccc=dev_rows_out[-1]["dev_ccc"]))
        log(f"   CV {rows[-1]['cv_ccc']:.3f} +- {rows[-1]['cv_std']:.3f} | LOLO {lolo_rows[-1]['mean']:.3f} | "
            f"within {w:.3f} cross {c:.3f} drop {100 * (w - c) / w:.0f}% | dev {rows[-1]['dev_ccc']:.3f}")
        for nm, data in [("gb_all_inputs", rows), ("gb_per_session", per_session_rows), ("gb_lolo", lolo_rows),
                         ("gb_within_cross_matched", matched_rows), ("gb_dev_selection", dev_rows_out)]:
            pd.DataFrame(data).to_csv(os.path.join(OUT, nm + ".csv"), index=False)
        del X

    # ------------------------------------------------ held-out test set, used once (choice made on dev only)
    dev = pd.DataFrame(dev_rows_out)
    chosen = dev.sort_values("dev_ccc", ascending=False).iloc[0]["input"]
    final = []
    progress("held-out test (used once)", len(dict.fromkeys(["voice + face", "voice + face + context", chosen])))
    for name in dict.fromkeys(["voice + face", "voice + face + context", chosen]):
        X = make_input(name)
        p = gb(X, ~is_test, is_test)
        final.append(dict(input=name, chosen_on_dev=(name == chosen), test_ccc=ccc(y[is_test], p),
                          **{f"test_{s}": v for s, v in session_cccs(p, np.where(is_test)[0]).items()}))
        del X
    pd.DataFrame(final).to_csv(os.path.join(OUT, "heldout_test_once.csv"), index=False)
    log(f"held-out test (chosen on dev: {chosen})\n" + pd.DataFrame(final).round(3).to_string(index=False))

    # ------------------------------------------------ statistics over sessions (E3)
    ps = pd.DataFrame(per_session_rows).pivot(index="session", columns="input", values="ccc")
    rng = np.random.default_rng(42)
    pairs = [("fusion + context (best)", "voice + face"), ("voice + face + context", "voice + face"),
             ("fusion + context (best)", "voice + face + context"), ("voice", "face (all 709)"),
             ("face (AU+gaze+pose, 49)", "face (all 709)")]
    stats = []
    for a, b in pairs:
        diff = (ps[a] - ps[b]).to_numpy()
        boot = rng.choice(diff, (10000, len(diff))).mean(axis=1)
        stats.append(dict(comparison=f"{a} vs {b}", mean_diff=diff.mean(), ci_low=np.percentile(boot, 2.5),
                          ci_high=np.percentile(boot, 97.5), better=f"{(diff > 0).sum()}/{len(diff)}",
                          p_wilcoxon=wilcoxon(diff).pvalue))
    st = pd.DataFrame(stats)
    order = st["p_wilcoxon"].sort_values().index
    running, holm = 0, {}
    for rank, i in enumerate(order):
        running = max(running, min(1, (len(st) - rank) * st.loc[i, "p_wilcoxon"]))
        holm[i] = running
    st["p_holm"] = pd.Series(holm)
    st.to_csv(os.path.join(OUT, "session_statistics.csv"), index=False)
    log("statistics over 24 sessions\n" + st.round(4).to_string(index=False))
    log("STAGE 1 DONE")

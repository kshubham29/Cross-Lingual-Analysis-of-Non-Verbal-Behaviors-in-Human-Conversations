"""Step 33: which features matter (supervisor's review, E11), with TreeSHAP of XGBoost.
For each of the 6 speaker-independent folds an XGBoost model (same settings as step 29) is trained on the training people and explained on
unseen test people: up to 600 random windows per language and fold. SHAP value = how much a feature pushed one prediction up or down.
Importance = mean |SHAP| over the explained windows, reported overall and separately for English, French and German; features are also
summed into groups (voice, gaze, head pose, action-unit intensity / presence, other OpenFace values). The data has no feature names, so
single features are shown by position (voice_k = k-th eGeMAPS value, face_k = k-th OpenFace column) and groups come from column positions.
Also: do the three languages agree on which features matter (Spearman rank correlation, overlap of the top 20)?
Run: python src/33_shap_features.py   ->  reports/shap_features.csv, shap_groups.csv, shap_language_agreement.csv, charts/shap_features.png"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from splits import person_groups

PER_LANG = 600


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ccc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-9)


df = pd.read_parquet("data/features/noxi_engagement.parquet")
cols = [c for c in df.columns if c.startswith("voice_")] + [f"face_{i}" for i in range(5, 714)]
X = df[cols].fillna(0).to_numpy("float32"); y = df["engagement"].to_numpy("float32")
language = df["language"].to_numpy(); session = df["session"].to_numpy()
languages = ["English", "French", "German"]
group = person_groups(sorted(df["session"].unique())); row_group = np.array([group[s] for s in session])
folds = [te for _, te in GroupKFold(n_splits=6).split(np.zeros(len(y)), groups=row_group)]


def feature_group(name):
    if name.startswith("voice_"):
        return "voice (eGeMAPS)"
    i = int(name.split("_")[1])
    if 5 <= i <= 12: return "face: gaze"
    if 293 <= i <= 298: return "face: head pose"
    if 679 <= i <= 695: return "face: action-unit intensity"
    if 696 <= i <= 713: return "face: action-unit presence"
    return "face: other OpenFace values (landmarks, shape)"


groups = np.array([feature_group(c) for c in cols])
sum_abs = {l: np.zeros(len(cols)) for l in languages}; n_win = {l: 0 for l in languages}
corr_acc = np.zeros(len(cols)); corr_n = 0
rng = np.random.default_rng(0)
for k, te_idx in enumerate(folds):
    tr = np.ones(len(y), bool); tr[te_idx] = False
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=8)
    model.fit(X[tr], y[tr])
    fold_ccc = ccc(y[te_idx], model.predict(X[te_idx]))
    pick = []
    for l in languages:
        idx = te_idx[language[te_idx] == l]
        if len(idx):
            take = rng.choice(idx, size=min(PER_LANG, len(idx)), replace=False)
            pick.append((l, take))
    for l, take in pick:
        contrib = model.get_booster().predict(xgb.DMatrix(X[take]), pred_contribs=True)[:, :-1]     # last column is the bias term
        sum_abs[l] += np.abs(contrib).sum(axis=0); n_win[l] += len(take)
        xs = X[take]; sd = xs.std(axis=0); ok = sd > 0                                               # direction: does a larger value push the prediction up?
        cs = contrib - contrib.mean(axis=0)
        c = np.zeros(len(cols)); c[ok] = (cs[:, ok] * (xs[:, ok] - xs[:, ok].mean(axis=0))).mean(axis=0) / (cs[:, ok].std(axis=0) * sd[ok] + 1e-12)
        corr_acc += c; corr_n += 1
    log(f"   fold {k + 1}/6: model CCC {fold_ccc:.3f}; explained windows so far: " + ", ".join(f"{l} {n_win[l]}" for l in languages))

mean_abs = {l: sum_abs[l] / n_win[l] for l in languages}
overall = sum(sum_abs[l] for l in languages) / sum(n_win.values())
tab = pd.DataFrame({"feature": cols, "group": groups, "mean_abs_shap_all": overall, **{f"mean_abs_shap_{l}": mean_abs[l] for l in languages},
                    "direction_corr": corr_acc / corr_n})
tab["share_pct"] = 100 * tab.mean_abs_shap_all / tab.mean_abs_shap_all.sum()
tab = tab.sort_values("mean_abs_shap_all", ascending=False).reset_index(drop=True); tab["rank"] = np.arange(1, len(tab) + 1)
tab["direction"] = np.where(tab.direction_corr > 0.05, "higher value -> higher engagement", np.where(tab.direction_corr < -0.05, "higher value -> lower engagement", "unclear / non-linear"))
tab.to_csv("reports/shap_features.csv", index=False)

g = tab.groupby("group").agg(n_features=("feature", "size"), share_all=("mean_abs_shap_all", "sum"), **{f"share_{l}": (f"mean_abs_shap_{l}", "sum") for l in languages})
for c in ["share_all"] + [f"share_{l}" for l in languages]:
    g[c] = 100 * g[c] / g[c].sum()
g = g.sort_values("share_all", ascending=False).reset_index(); g.to_csv("reports/shap_groups.csv", index=False)

agree = []
for i, a in enumerate(languages):
    for b in languages[i + 1:]:
        rho = spearmanr(mean_abs[a], mean_abs[b]).correlation
        ta, tb = set(np.argsort(-mean_abs[a])[:20]), set(np.argsort(-mean_abs[b])[:20])
        agree.append(dict(language_a=a, language_b=b, spearman_all_797=rho, top20_overlap=len(ta & tb)))
pd.DataFrame(agree).to_csv("reports/shap_language_agreement.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("charts", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
gg = g.iloc[::-1]
ax[0].barh(gg["group"], gg["share_all"], color="#555555"); ax[0].set_xlabel("share of total importance (%)"); ax[0].set_title("Importance by feature group")
top = tab.head(15).iloc[::-1]; y_pos = np.arange(len(top)); w = 0.27
for j, (l, sh) in enumerate(zip(languages, ["#222222", "#777777", "#bbbbbb"])):
    ax[1].barh(y_pos + (j - 1) * w, top[f"mean_abs_shap_{l}"], height=w, color=sh, label=l)
ax[1].set_yticks(y_pos); ax[1].set_yticklabels(top["feature"]); ax[1].set_xlabel("mean |SHAP|"); ax[1].legend(); ax[1].set_title("Top 15 single features, by language")
plt.tight_layout(); plt.savefig("charts/shap_features.png", dpi=130); plt.close()

log("=== importance by group (% of total): " + "; ".join(f"{r.group} {r.share_all:.1f}" for r in g.itertuples()))
log("=== top 10 features: " + ", ".join(f"{r.feature} ({r.share_pct:.1f}%)" for r in tab.head(10).itertuples()))
log("=== language agreement: " + "; ".join(f"{r['language_a']}-{r['language_b']} rho {r['spearman_all_797']:.2f}, top-20 overlap {r['top20_overlap']}" for r in agree))
log("saved reports/shap_features.csv, shap_groups.csv, shap_language_agreement.csv, charts/shap_features.png")
log("GROUP shap DONE")

"""Step 34: role analysis, expert vs novice (supervisor's review, E12; partner features are NOT part of this).
Questions: (1) is engagement different for experts and novices? (2) is the model more accurate for one role? (3) how much of the score
could come from simply knowing the role?
Model: XGBoost on the 797 voice + face features (same settings as step 29), 6 speaker-independent folds, out-of-fold predictions.
 - Engagement by role and language (mean, std, windows).
 - CCC per role: pooled CCC inside each test fold and role, averaged over folds; also the mean of the per-session CCC (24 sessions per role).
 - Paired test over the 24 sessions: expert CCC minus novice CCC (bootstrap 95% CI, Wilcoxon).
 - Role-mean baseline: predict the training mean of the window's role (no features at all): how much does the role alone explain?
Run: python src/34_role_analysis.py -> reports/role_analysis.csv, reports/role_stats.csv"""
import sys, time, warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from splits import person_groups


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ccc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-9)


df = pd.read_parquet("data/features/noxi_engagement.parquet")
cols = [c for c in df.columns if c.startswith("voice_")] + [f"face_{i}" for i in range(5, 714)]
X = df[cols].fillna(0).to_numpy("float32"); y = df["engagement"].to_numpy("float32")
role = df["role"].to_numpy(); language = df["language"].to_numpy(); session = df["session"].to_numpy()
roles, languages = ["expert", "novice"], ["English", "French", "German"]
group = person_groups(sorted(df["session"].unique())); row_group = np.array([group[s] for s in session])
folds = [te for _, te in GroupKFold(n_splits=6).split(np.zeros(len(y)), groups=row_group)]

pred = np.zeros(len(y)); base = np.zeros(len(y))
for k, te in enumerate(folds):
    tr = np.ones(len(y), bool); tr[te] = False
    m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=8)
    pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
    rm = {r: y[tr & (role == r)].mean() for r in roles}
    base[te] = np.array([rm[r] for r in role[te]])
    log(f"   fold {k + 1}/6: model CCC {ccc(y[te], pred[te]):.3f} | role-mean baseline CCC {ccc(y[te], base[te]):.3f}")

rows = []


def fold_mean(mask, p):
    v = [ccc(y[te & mask], p[te & mask]) for te in (np.isin(np.arange(len(y)), f) for f in folds) if (te & mask).sum() > 50 and y[te & mask].std() > 0]
    return float(np.mean(v)) if v else np.nan


def sess_scores(mask):
    out = {}
    for s in np.unique(session[mask]):
        mm = mask & (session == s)
        if y[mm].std() > 0:
            out[s] = ccc(y[mm], pred[mm])
    return out


for r in roles + ["both"]:
    for l in languages + ["all"]:
        mask = np.ones(len(y), bool)
        if r != "both": mask &= role == r
        if l != "all": mask &= language == l
        ss = sess_scores(mask)
        rows.append(dict(role=r, language=l, windows=int(mask.sum()), mean_engagement=y[mask].mean(), std_engagement=y[mask].std(),
                         model_ccc_fold_mean=fold_mean(mask, pred), model_ccc_session_mean=np.mean(list(ss.values())) if ss else np.nan,
                         role_mean_baseline_ccc=fold_mean(mask, base) if r == "both" else np.nan))
tab = pd.DataFrame(rows); tab.to_csv("reports/role_analysis.csv", index=False)

se, sn = sess_scores(role == "expert"), sess_scores(role == "novice")
common = sorted(set(se) & set(sn)); d = np.array([se[s] - sn[s] for s in common])
rng = np.random.default_rng(0); boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
stat = pd.DataFrame([dict(comparison="expert CCC - novice CCC (per session, XGBoost voice+face)", sessions=len(d), mean_diff=d.mean(),
                          ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5), expert_better=f"{int((d > 0).sum())}/{len(d)}",
                          p_wilcoxon=wilcoxon(d).pvalue)])
eng_e = df.groupby(["session", "role"])["engagement"].mean().unstack()
d2 = (eng_e["expert"] - eng_e["novice"]).to_numpy()
boot2 = [d2[rng.integers(0, len(d2), len(d2))].mean() for _ in range(10000)]
stat = pd.concat([stat, pd.DataFrame([dict(comparison="expert mean engagement - novice mean engagement (per session)", sessions=len(d2), mean_diff=d2.mean(),
                                           ci_low=np.percentile(boot2, 2.5), ci_high=np.percentile(boot2, 97.5), expert_better=f"{int((d2 > 0).sum())}/{len(d2)}",
                                           p_wilcoxon=wilcoxon(d2).pvalue)])], ignore_index=True)
stat.to_csv("reports/role_stats.csv", index=False)

pd.set_option("display.width", 220)
log("=== role x language table:")
print(tab.round(3).to_string(index=False), flush=True)
log("=== paired tests over sessions:")
print(stat.round(4).to_string(index=False), flush=True)
log("GROUP role DONE")

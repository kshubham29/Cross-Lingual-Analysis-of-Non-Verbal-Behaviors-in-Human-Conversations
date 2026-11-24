"""Step 31: language-ID probe (supervisor's review, E8).
Question: how easily can the LANGUAGE (English / French / German) of a one-second window be decoded from each feature set?
If the face reveals language (= recording site, camera, room) much more easily than the voice, the face carries a language/site
signature, which can explain why it transfers worse. In our data language and recording site go together, so the probe cannot
separate the two.
Same speaker-independent 6 folds as steps 28-30 (no person in both training and test), so the classifier cannot memorise people.
Classifier: XGBoost (a strong non-linear probe), 100 trees, depth 5. Score: balanced accuracy over all windows (out-of-fold
predictions), chance = 1/3; also macro-F1, recall per language and the spread over the 6 folds.
Run:  python src/31_language_probe.py          (all 5 inputs, about 15-25 min)
      python src/31_language_probe.py smoke    (tiny test on 3,000 rows, a few seconds, writes nothing)
Result: reports/language_probe.csv"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from splits import person_groups

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


df = pd.read_parquet("data/features/noxi_engagement.parquet")
languages = ["English", "French", "German"]
lang = df["language"].map({l: i for i, l in enumerate(languages)}).to_numpy()
session = df["session"].to_numpy()
group = person_groups(sorted(df["session"].unique()))
row_group = np.array([group[s] for s in session])
folds = [te for _, te in GroupKFold(n_splits=6).split(np.zeros(len(df)), groups=row_group)]

fc = lambda idx: [f"face_{i}" for i in idx]
au_gaze_pose = list(range(5, 13)) + list(range(293, 299)) + list(range(679, 714))          # 8 + 6 + 35 = 49
inputs = {
    "voice (eGeMAPS, 88)": df[[c for c in df.columns if c.startswith("voice_")]].fillna(0).to_numpy("float32"),
    "face (all 709)": df[fc(range(5, 714))].fillna(0).to_numpy("float32"),
    "face (AU+gaze+pose, 49)": df[fc(au_gaze_pose)].fillna(0).to_numpy("float32"),
    "w2v-BERT 2.0 (voice embedding, 1024)": np.load("data/features/pretrained_w2vbert2.npy"),
    "VideoMAE (video embedding, 1408)": np.load("data/features/pretrained_videomae.npy"),
}
assert inputs["face (AU+gaze+pose, 49)"].shape[1] == 49

if SMOKE:                                                    # tiny test: 3,000 random rows, 2 folds' worth of logic
    rng = np.random.default_rng(0); keep = np.sort(rng.choice(len(df), 3000, replace=False))
    inputs = {k: v[keep] for k, v in inputs.items()}
    lang, row_group = lang[keep], row_group[keep]
    folds = [np.where(np.isin(keep, f))[0] for f in folds]
    folds = [f for f in folds if len(f)][:2]

rows = []
for name, X in inputs.items():
    pred = np.full(len(lang), -1)
    fold_acc = []
    for k, te in enumerate(folds):
        tr = np.ones(len(lang), bool); tr[te] = False
        clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=8, random_state=42)
        classes = np.unique(lang[tr])
        remap = {c: i for i, c in enumerate(classes)}
        clf.fit(X[tr], np.vectorize(remap.get)(lang[tr]))
        pred[te] = classes[clf.predict(X[te])]
        fold_acc.append(balanced_accuracy_score(lang[te], pred[te]))
        log(f"   {name}: fold {k + 1}/{len(folds)} balanced accuracy {fold_acc[-1]:.3f}")
    ok = pred >= 0
    rec = recall_score(lang[ok], pred[ok], labels=[0, 1, 2], average=None, zero_division=0)
    rows.append(dict(input=name, features=X.shape[1], classifier="XGBoost (100 trees, depth 5)",
                     balanced_accuracy=balanced_accuracy_score(lang[ok], pred[ok]), macro_f1=f1_score(lang[ok], pred[ok], average="macro"),
                     fold_mean=np.mean(fold_acc), fold_std=np.std(fold_acc, ddof=1) if len(fold_acc) > 1 else 0.0,
                     recall_English=rec[0], recall_French=rec[1], recall_German=rec[2], chance=1 / 3))
    log(f"=== {name}: balanced accuracy {rows[-1]['balanced_accuracy']:.3f} (chance 0.333) | macro-F1 {rows[-1]['macro_f1']:.3f} "
        f"| recall EN/FR/DE {rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f}")

if SMOKE:
    log("smoke test OK (nothing written)")
else:
    pd.DataFrame(rows).to_csv("reports/language_probe.csv", index=False)
    log("saved reports/language_probe.csv")
    log("GROUP probe DONE")

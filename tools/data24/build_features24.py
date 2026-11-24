"""Builds the feature table for all conversations in data/noxi24/ (same method as steps 02 and 19):
one row per 1-second window: 88 eGeMAPS + 714 OpenFace values (averaged over the 25 frames), engagement,
language, session, role, second; plus the w2v-BERT 2.0 and VideoMAE embeddings averaged over the same windows.
Each session is cached in data/features24/cache/, so new sessions can be added later without redoing the others.
Output: data/features24/noxi24_engagement.parquet, pretrained_w2vbert2.npy, pretrained_videomae.npy
Run from the project folder: python tools/data24/build_features24.py"""
import os
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")                  # "mean of empty slice" for windows without labels
SRC = os.path.join("data", "noxi")
OUT = os.path.join("data", "features")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)


def read_stream(path):
    # same as step 02: XML header says numbers per frame and their type; the .stream~ file holds the numbers
    info = ET.parse(path).getroot().find("info")
    dim = int(info.get("dim"))
    dtype = np.float32 if info.get("type") == "FLOAT" else np.float64
    data = np.fromfile(path + "~", dtype=dtype)
    frames = data.size // dim
    return data[: frames * dim].reshape(frames, dim)


def build_session(session):
    folder = os.path.join(SRC, session)
    language = open(os.path.join(folder, "language.annotation.csv")).read().strip().split(";")[2]
    parts = []
    for role in ["expert", "novice"]:
        voice = read_stream(os.path.join(folder, f"{role}.audio.egemapsv2.stream"))
        face = read_stream(os.path.join(folder, f"{role}.openface2.stream"))
        label = pd.to_numeric(pd.read_csv(os.path.join(folder, f"{role}.engagement.annotation.csv"), header=None)[0],
                              errors="coerce").to_numpy("float32")
        w2v = read_stream(os.path.join(folder, f"{role}.audio.w2vbert2_embeddings.stream"))
        vmae = read_stream(os.path.join(folder, f"{role}.videomae.stream"))
        n = min(len(voice), len(face), len(label))
        feats, ys, seconds, w2v_rows, vmae_rows = [], [], [], [], []
        for w in range(n // 25):
            start, end = w * 25, (w + 1) * 25
            engagement = np.nanmean(label[start:end])
            if np.isnan(engagement):                # no valid label: skip this second (as in step 02)
                continue
            feats.append(np.concatenate([np.nanmean(voice[start:end], axis=0), np.nanmean(face[start:end], axis=0)]))
            ys.append(engagement)
            seconds.append(w)
            for E, store in ((w2v, w2v_rows), (vmae, vmae_rows)):   # as in step 19
                if start < len(E):
                    store.append(np.nanmean(E[start:end], axis=0))
                else:
                    store.append(np.full(E.shape[1], np.nan, np.float32))
        parts.append(dict(role=role, feats=np.array(feats, dtype=np.float64), y=np.array(ys, dtype=np.float64),
                          second=np.array(seconds), w2v=np.nan_to_num(np.stack(w2v_rows)).astype(np.float32),
                          vmae=np.nan_to_num(np.stack(vmae_rows)).astype(np.float32)))
    np.savez(os.path.join(CACHE, session + ".npz"), language=language,
             **{f"{p['role']}_{k}": v for p in parts for k, v in p.items() if k != "role"})
    return language, sum(len(p["y"]) for p in parts)


sessions = sorted(s for s in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, s)))
for s in sessions:
    if os.path.exists(os.path.join(CACHE, s + ".npz")):
        continue
    language, n = build_session(s)
    print(f"{s}  {language:8s} {n:5d} windows", flush=True)

# ---------- combine all sessions in sorted order (expert, then novice), like step 02 ----------
cols = [f"voice_{i}" for i in range(88)] + [f"face_{i}" for i in range(714)]
tables, w2v_all, vmae_all = [], [], []
for s in sessions:
    c = np.load(os.path.join(CACHE, s + ".npz"))
    for role in ["expert", "novice"]:
        t = pd.DataFrame(c[f"{role}_feats"], columns=cols)
        t["engagement"] = c[f"{role}_y"]
        t["language"] = str(c["language"])
        t["session"] = s
        t["role"] = role
        t["second"] = c[f"{role}_second"]
        tables.append(t)
        w2v_all.append(c[f"{role}_w2v"])
        vmae_all.append(c[f"{role}_vmae"])
df = pd.concat(tables, ignore_index=True)
df.to_parquet(os.path.join(OUT, "noxi_engagement.parquet"))
np.save(os.path.join(OUT, "pretrained_w2vbert2.npy"), np.concatenate(w2v_all))
np.save(os.path.join(OUT, "pretrained_videomae.npy"), np.concatenate(vmae_all))
print(f"\n{len(sessions)} conversations, {len(df):,} windows")
print(df.groupby("language")["session"].nunique().to_string())
print(df["language"].value_counts().to_string())

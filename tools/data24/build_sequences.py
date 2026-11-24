"""Builds data/features/noxi_sequences.npy for all conversations in data/noxi/ (same method as step 02):
for every 1-second window (same windows and order as noxi_engagement.parquet) the 25 raw frames x 802 values
(88 eGeMAPS + 714 OpenFace), NaN -> 0, float32. Written straight to disk (memory-mapped), so it fits in RAM.
Then checks that the 9 original conversations are identical to the old file (data/features_9conv/).
Run from the project folder: python tools/data24/build_sequences.py"""
import os, time, warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SRC = os.path.join("data", "noxi")
OUT = os.path.join("data", "features", "noxi_sequences.npy")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_stream(path):
    info = ET.parse(path).getroot().find("info")
    dim = int(info.get("dim"))
    dtype = np.float32 if info.get("type") == "FLOAT" else np.float64
    data = np.fromfile(path + "~", dtype=dtype)
    frames = data.size // dim
    return data[: frames * dim].reshape(frames, dim)


df = pd.read_parquet(os.path.join("data", "features", "noxi_engagement.parquet"), columns=["session", "role", "second"])
seq = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float32, shape=(len(df), 25, 802))
sessions = sorted(df["session"].unique())
row = 0
for k, s in enumerate(sessions, 1):
    for role in ["expert", "novice"]:
        folder = os.path.join(SRC, s)
        voice = read_stream(os.path.join(folder, f"{role}.audio.egemapsv2.stream"))
        face = read_stream(os.path.join(folder, f"{role}.openface2.stream"))
        seconds = df["second"].to_numpy()[(df["session"] == s).to_numpy() & (df["role"] == role).to_numpy()]
        for w in seconds:                                   # the same windows as the feature table
            start, end = w * 25, (w + 1) * 25
            frames = np.concatenate([voice[start:end], face[start:end]], axis=1)
            seq[row] = np.nan_to_num(frames).astype(np.float32)
            row += 1
    log(f"session {s} done ({k}/{len(sessions)}), rows so far {row:,}")
assert row == len(df)
seq.flush()
log(f"saved {OUT}: shape {seq.shape}")

# ---- check: the 9 original conversations must be identical to the old file
old = np.load(os.path.join("data", "features_9conv", "noxi_sequences.npy"), mmap_mode="r")
old_sessions = pd.read_parquet(os.path.join("data", "features_9conv", "noxi_engagement.parquet"), columns=["session"])["session"]
keep = np.where(df["session"].isin(old_sessions.unique()).to_numpy())[0]
same = all(np.array_equal(old[i:i + 2000], seq[keep[i:i + 2000]]) for i in range(0, len(keep), 2000))
log(f"check: 9 original conversations identical to the old file: {same} ({len(keep):,} rows)")

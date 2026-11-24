"""Copies the 19 needed files of every session into data/noxi24/<session>/ (straight from the zips).
data/noxi/ (the 9 original conversations) is not touched. Run from the project folder."""
import os, zipfile
DL = r"C:\Users\SHUBHAM KUMAR\Downloads"
OUT = os.path.join("data", "noxi")
STREAMS = ["audio.egemapsv2", "openface2", "audio.w2vbert2_embeddings", "videomae"]
NEEDED = ["language.annotation.csv"]
for role in ("expert", "novice"):
    NEEDED.append(f"{role}.engagement.annotation.csv")
    for s in STREAMS:
        NEEDED += [f"{role}.{s}.stream", f"{role}.{s}.stream~"]          # header + data
assert len(NEEDED) == 19

source = {s: "train.zip" for s in ["001", "013", "021", "028", "030", "070", "072", "077"]}
source["027"] = "027.zip"
for s in ["007", "009", "011", "014", "020", "026", "029", "058", "062", "063", "068", "073", "074", "075", "076"]:
    source[s] = f"{s}.zip"

for session, zname in sorted(source.items()):
    z = zipfile.ZipFile(os.path.join(DL, zname))
    names = set(z.namelist())
    folder = os.path.join(OUT, session); os.makedirs(folder, exist_ok=True)
    copied, missing = 0, []
    for f in NEEDED:
        member = f"{session}/{f}"
        target = os.path.join(folder, f)
        if member not in names:
            missing.append(f); continue
        if not os.path.exists(target) or os.path.getsize(target) != z.getinfo(member).file_size:
            with z.open(member) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
        copied += 1
    lang = open(os.path.join(folder, "language.annotation.csv")).read().strip().split(";")[2] if "language.annotation.csv" not in missing else "?"
    size = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)) / 1e6
    print(f"{session}  {lang:8s} {copied}/19 files  {size:6.0f} MB" + (f"  MISSING: {missing}" if missing else ""), flush=True)

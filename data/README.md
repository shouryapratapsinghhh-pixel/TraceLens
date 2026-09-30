# Data

Nothing here is committed (`data/raw/` is gitignored). Tests use synthetic videos and a
fake MOT-format sequence built on the fly; they need no downloads.

## MOT17 (real footage)

```bash
mkdir -p data/raw && cd data/raw
curl -L -o MOT17.zip https://motchallenge.net/data/MOT17.zip   # large: several GB
unzip -q MOT17.zip
ls MOT17/train        # MOT17-02-FRCNN, MOT17-04-FRCNN, ... (plus -DPM / -SDP variants)
```

Each sequence comes in three variants (-DPM, -FRCNN, -SDP) with the same video and ground
truth but different public detections. The evaluation uses the -FRCNN variants by default.
If the unzip nests folders differently, point `--root` at the folder that directly contains
the `MOT17-XX-FRCNN` sequence folders.

Use only public research datasets.

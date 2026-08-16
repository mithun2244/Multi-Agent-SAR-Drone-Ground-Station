# data/

Everything the detectors are trained and measured on. **Scripts live here and
are committed; the data they fetch or produce is not** — see the `data/` block
in `.gitignore`, which tracks `*.py`, `*.sh`, `*.md` and `.gitkeep` and ignores
everything else. Nothing in this tree is required to run the ground station: the
system reaches every detector through an injected callable, so an empty `data/`
means stubs, not a broken build.

```
data/
├── visdrone/        RGB training data
│   ├── download_visdrone.sh    fetch DET-train and DET-val
│   └── convert_to_yolo.py      VisDrone annotations -> YOLO, people only
├── lidar/           range imagery for the LiDAR detector
│   └── project_to_rangeimg.py  point cloud (.ply/.las/.bin) -> range PNG
├── paired/          the same scene captured by both sensors, at the same moment
├── splits/          fixed train/val/test manifests
│   └── generate_splits.py      deterministic splits from a seed
├── recordings/      raw sorties: RGB, LiDAR and telemetry together
└── ground_truth/    where subjects actually were, for the critic and the harness
```

## Why the splits are files

Phase 1 exists so that every later exit test refers to the same numbers. A split
regenerated on the fly is a different validation set every run, and two mAP
figures measured against two different sets are not comparable — which is the
whole point of having a harness. `generate_splits.py` writes a JSON manifest,
the manifest is what training and evaluation read, and the seed that produced it
is recorded inside it.

## Why `paired/` is separate

RGB and LiDAR search-and-rescue datasets are independent of each other, so
nothing off the shelf tells you whether fusing two sensors beats one. That needs
a small set of frames where both sensors saw the same scene at the same moment.
Until `paired/` has real content, the Weighted Box Fusion merge is measured
against projected stubs — real end to end, but simulated imagery.

## Usage

```bash
# 1. RGB
bash data/visdrone/download_visdrone.sh
python data/visdrone/convert_to_yolo.py            # -> train/ val/ + dataset yaml

# 2. LiDAR
python data/lidar/project_to_rangeimg.py --demo    # synthetic, no data needed
python data/lidar/project_to_rangeimg.py --input cloud.bin --output range.png

# 3. Splits
python data/splits/generate_splits.py --dataset visdrone --seed 0

# every script checks itself, offline, with no data present
python data/visdrone/convert_to_yolo.py --selfcheck
python data/lidar/project_to_rangeimg.py --selfcheck
python data/splits/generate_splits.py --selfcheck
```

## Classes

VisDrone labels ten categories; the converter keeps **two**:

| VisDrone | meaning | YOLO class |
|---|---|---|
| 1 | pedestrian — a person standing or walking | 0 |
| 2 | people — a person otherwise posed, grouped or seated | 1 |

The other eight are traffic. Training on them spends capacity on objects a
search does not care about, and every one is a chance to report a van as a find.
The two are kept apart rather than merged because a casualty is far more likely
to be lying or sitting than walking, and that is exactly the distinction
VisDrone's `people` class carries — a model trained on `pedestrian` alone learns
the upright case a search is *least* interested in. Merge them by rewriting the
leading `1` to `0` in every label file if a single-class detector is wanted.

`score = 0` rows are VisDrone's ignored regions and are dropped, never treated
as background or as objects.

## Weights

Trained weights go in `config/weights/`, which is committed empty. `*.pt` is
ignored everywhere: weights worth keeping belong in a release, not in git.

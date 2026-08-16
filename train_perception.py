"""Train the RGB detector: YOLO11m on VisDrone, configured as a fast smoke run.

    python train_perception.py

This is the "real detector weights" line item Phase 2 deferred — the stub in
`src/perception/detectors.py` models the *shape* of a detector's behaviour, and
this is what eventually replaces it. The settings below are a smoke test, not a
training recipe: a tenth of the data for ten epochs proves the pipeline runs end
to end and produces weights, it does not produce a detector worth flying.

VisDrone is drone-captured aerial imagery, which is the right domain — a model
trained on ground-level photographs sees a person from an angle no search drone
ever gets. `data="VisDrone.yaml"` is the config shipped with ultralytics; it
downloads the dataset (~2 GB) on first run.

Classes are filtered to the three that matter for search:

    0  pedestrian
    1  people      (VisDrone splits standing/walking from grouped or seated)
    3  car

The other seven — bicycle, van, truck, tricycle, awning-tricycle, bus, motor —
are traffic. Training on them spends capacity on objects a search does not care
about, and every one of them is a chance to report a van as a find.

Whatever this produces is measured by the Phase 1 harness before it is believed:

    python -m src.evaluation.harness
"""

from ultralytics import YOLO

MODEL = "yolo11m.pt"          # Medium, per the standardised RGB detector
DATA = "VisDrone.yaml"        # ultralytics' config; downloads on first use

# Search-relevant classes only. See the module docstring for why.
CLASSES = [0, 1, 3]


def main():
    model = YOLO(MODEL)
    results = model.train(
        data=DATA,
        epochs=10,
        batch=8,
        imgsz=640,
        fraction=0.1,         # a tenth of the training set: smoke run, not a recipe
        classes=CLASSES,
        # No `project=`: ultralytics already roots runs at its own runs_dir, and
        # a relative project nests under it — the first pass landed in
        # runs/detect/runs/perception/. `name` alone is enough.
        name="yolo11m_visdrone_smoke",
    )
    print(f"\nweights: {results.save_dir}")
    return results


if __name__ == "__main__":
    # Required on Windows: the dataloader spawns worker processes, and without
    # this guard each one re-imports the module and starts its own training run.
    main()

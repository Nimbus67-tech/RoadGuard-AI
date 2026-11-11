# RoadGuard AI 🛣️

**Highway helmet-safety monitoring and analytics — a human-reviewable research prototype.**

RoadGuard AI is a decision-support concept for turning authorized highway-camera observations into transparent road-safety insights. It is not an automatic challan or enforcement system. It does not use face recognition or number-plate recognition, and model outputs must never be the sole basis for penalties or adverse decisions.

## What is included

- **Safety overview:** interactive charts and observation metrics using clearly labelled synthetic demo data.
- **Dataset audit:** upload a YOLO-format dataset ZIP and inspect image/label coverage, malformed annotations, class counts, image readability and exact duplicate files.
- **Responsible-use guidance:** explicit uncertainty, human review, privacy and licensing safeguards.

> The dashboard preview uses synthetic records. It is not real highway data, model output, or evidence of helmet compliance. No detector has been trained and no real dataset has yet been validated in this repository.

## Run locally

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Nimbus67-tech/RoadGuard-AI.git
cd RoadGuard-AI
pip install -r requirements.txt
pip install -e .
streamlit run app/streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Choose **Create app**.
3. Select Nimbus67-tech/RoadGuard-AI, branch main, and main file app/streamlit_app.py.
4. Select **Deploy**.

## Dataset audit

The audit expects YOLO text rows in this format:

```text
class_id x_center y_center width height
```

Coordinates must be normalized to [0, 1]. Example folder layout:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Run the CLI:

```bash
python -m roadguard.dataset_audit --root /path/to/dataset \
  --images images/train images/val \
  --labels labels/train labels/val \
  --names BikeWithRider NoHelmet Helmet \
  --out reports/dataset_audit
```

The audit generates summary.json, images.csv, annotations.csv, class_counts.csv, and audit_report.md. It detects byte-identical files only; it does not assess semantic annotation correctness or near-duplicates.

## Evaluation principles

- Split by source video, camera, location or collection session before extracting frames. Never randomly split adjacent video frames across train and test.
- Report precision, recall, F1, AP50 and AP50:95 on a held-out test set.
- Report tracking metrics only when ground-truth identities are available.
- Keep UNKNOWN observations separate; do not force uncertain cases into helmet/no-helmet classes.
- State the denominator for every rate and do not call frame observations unique riders without a validated counting protocol.

## Privacy and responsible use

- Use only footage you are authorized to process.
- Do not commit private footage, identifiable evidence images, datasets, model weights or secrets.
- Apply access controls and retention limits to raw video.
- Treat model output as a review cue, not proof of a violation.
- Publish aggregate results and privacy-safe examples only.

## Candidate datasets

Potential sources and access/licensing caveats are documented in docs/DATASET_REGISTER.md. Verify current dataset terms and provenance before use.

## Development status

**Phase 2 — interactive prototype and dataset engineering.** The app and YOLO annotation audit are implemented. Real-data validation, detector training, video tracking and live-camera inference are future milestones.

## License

Original project code is released under the MIT License. Third-party datasets and model weights retain their own terms.

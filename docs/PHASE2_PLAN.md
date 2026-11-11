# Development plan

## Phase 2 — dataset engineering and prototype UI
- [x] YOLO label parser and structural validation
- [x] Image readability, label coverage and exact SHA-256 duplicate audit
- [x] CSV, JSON and Markdown audit reports
- [x] Streamlit dashboard prototype with explicitly synthetic preview data
- [x] ZIP upload workflow for YOLO dataset audits

## Next milestones
1. Obtain a dataset with verified access rights and provenance.
2. Audit annotation structure and manually inspect a sample of boxes.
3. Split data by source video, camera, location or collection session.
4. Train a baseline detector and document exact configuration and seeds.
5. Evaluate on an untouched test set; report per-class precision, recall, F1, AP50 and AP50:95.
6. Add rider/helmet association and video tracking only with suitable annotations.
7. Add human-review workflow and evidence-retention controls.

No model accuracy or highway performance claim should be made until the relevant experiment has been completed and independently reproduced.

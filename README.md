# Shelf Gap Detection for Restock Flags (YOLOv8 on SKU-110K)

Finds every product facing on a densely packed retail shelf photo, groups the facings into shelf rows,
flags empty runs wider than one facing, and serves the result behind FastAPI as a bay-level gap count,
restock flag and restock task.

**Live page:** https://abhayjuloori.github.io/shelf-gap-detection-yolov8-sku110k/

## Problem

Store associates walk aisles to spot empty facings. A photo of a bay should come back with "how many
gaps, where, and is this worth a restock task now". That needs a detector that copes with 100+ near
identical products per image, and a rule layer that turns boxes into gaps.

## Data

[SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) (Goldman et al., CVPR 2019): 11,762 shelf photos,
about 1.7M annotated product boxes, one class. Official splits via the Ultralytics layout:
8,185 train / 584 val / 2,920 test photos (all usable, no label problems found). The dataset has no gap
labels; see evaluation below for how that is handled. Used under its non-commercial licence.

## Method

1. **Detector.** YOLOv8s from COCO weights, fine-tuned at 640 px on 2x T4 (Kaggle, DDP), seed 0, 6.5 h
   time cap (197 epochs). `scripts/kaggle/train_eval.py`.
2. **Dense NMS settings.** `max_det` and NMS IoU swept on the validation split (default `max_det=300`
   and especially 100 lose recall on 150-product photos). Test-time size 640 vs 1024 also chosen on val.
3. **Shelf rows.** Detections are chained into rows by mutual nearest neighbours with vertical overlap,
   which follows sloped shelves in angled photos; rows with fewer than 3 boxes are dropped.
   `src/shelfgap/gaps.py`.
4. **Gaps.** An interior horizontal gap in a row wider than `gap_factor` x the row's median facing
   width is a gap. Thin slivers are filtered. Edge gaps are not flagged (the photo frame cuts rows).
5. **Operating point.** Box confidence and `gap_factor` chosen on val by gap F1
   (`src/shelfgap/evaluate_gaps.py`), stored in `configs/operating_point.json`, applied once to test.
6. **Service.** `POST /predict` returns facings, rows, gaps, `restock_flag`, and a `task` block
   (`type`, `bay_id`, `priority`, `est_missing_facings`) in the shape the store task planner consumes.
   Priority is high at 3+ gaps.

## Results

All numbers are from one seeded run on the official test split; nothing was tuned on test.

### Detector (test, 2,920 photos)

| Setting | mAP@.5:.95 | mAP@.5 | mAP@.75 | Precision | Recall | ms/img (T4) |
|---|---|---|---|---|---|---|
| imgsz 640 | 0.608 | 0.939 | 0.701 | 0.910 | 0.886 | 8.1 |
| imgsz 1024 (chosen on val) | 0.608 | 0.944 | 0.699 | 0.905 | 0.889 | 23.1 |
| Goldman et al. 2019 (paper) | 0.492 | | | | | |

The spec's reference range for modern YOLO models on SKU-110K is about 0.49 to 0.60 mAP@.5:.95; this run
lands at the top of it (0.608). Val sweep: NMS IoU 0.5 beat 0.6 / 0.7 / 0.8; `max_det=100` drops val
recall from 0.886 to 0.620.

### Gap flags

**Box-derived reference (full test split).** Running the same row and gap rules on the annotated boxes
gives 14,459 reference gaps. Predicted gaps match a reference gap at gap IoU >= 0.3.

| Test split | ref gaps | pred gaps | Precision | Recall | F1 [95% bootstrap CI] | count MAE | count r |
|---|---|---|---|---|---|---|---|
| conf 0.25, gap factor 1.0 | 14,459 | 15,964 | 0.569 | 0.628 | 0.597 [0.584, 0.610] | 2.10 | 0.72 |

The val grid search picked conf 0.25 / factor 1.0, which happens to equal the untuned default.

Restock flag at 3+ gaps (high priority): accuracy 80.6% vs 64.8% for always predicting the majority
class; precision 0.83, recall 0.88. A 1+ gap flag is nearly useless as a signal here: 91% of test photos
have at least one reference gap.

**Hand audit (25 photos, 152 flagged gaps).** The reference above only measures agreement with the
annotated layout, so I also looked at every flagged gap on 25 test photos (sampled with seed 0):

| Photos | Flagged | Real empty slot | Precision | Obvious gaps missed |
|---|---|---|---|---|
| 25 | 152 | 41 | 27% | 8 |

Precision is low and concentrated in a few scene types: glass cooler doors (frames read as gaps,
1 of 22 real), flat boxes stacked at a steep angle (row grouping breaks, 1 of 30), densely hung
toothbrush packs (thin overlapping facings missed by the detector), and clip strips or signage in front
of shelves. On plain shelves and pegboards most flags are real (excluding the 3 worst photos: 43%).
Recall looks high (roughly 41 of 49 visible gaps found), so as a triage signal it over-flags rather
than misses. Details: `docs/data/hand_audit.json`.

### API

`POST /predict?bay_id=A12-03` on `test_970.jpg` with the trained weights: 137 facings, 9 rows, 3 gaps
(3.3, 1.5 and 1.2 facings wide), `restock_flag: true`, task priority high, 5 missing facings estimated.
Warm latency about 0.15 to 0.3 s per photo at 1024 px on an M4 MacBook Air. Full response:
`docs/data/api_example.json`.

## Limitations

- No gap ground truth exists for SKU-110K. The box-derived reference measures layout agreement, not true
  out-of-stock; the hand audit is one reviewer on 25 photos, not a labelled benchmark.
- Hand-audit precision (27%) is far below what a store would accept without scene filtering. Cooler
  doors, angled stacked boxes and hanging displays would need their own handling or a gap classifier.
- Only interior gaps are found; empty shelf at a photo's left or right edge is invisible to the method.
- Photos come from a few retailers, mostly handheld and front-on. A deployment would use its own fixed
  camera images.
- Single seed, single model size (YOLOv8s), because of free GPU limits.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,model]"
pytest -q

# re-score gaps from the committed predictions (no GPU needed)
python -m shelfgap.evaluate_gaps --data data_release --out docs/data/gap_report.json

# train + evaluate on Kaggle (2x T4)
./scripts/kaggle/push.sh            # full run
./scripts/kaggle/push.sh --smoke    # 1-epoch check

# serve (weights from the Kaggle run, copied to models/best.pt)
SHELFGAP_WEIGHTS=models/best.pt uvicorn shelfgap.api:app --port 8000
curl -F file=@shelf.jpg "localhost:8000/predict?bay_id=A12-03"
```

`data_release/` holds val and test predictions (conf >= 0.15) and ground truth so CI re-runs the gap
evaluation and checks it matches `docs/data/gap_report.json`.

## Layout

```
src/shelfgap/gaps.py            row chaining, gap rules, gap matching
src/shelfgap/evaluate_gaps.py   val grid search, test scoring, bootstrap CI
src/shelfgap/api.py             FastAPI service
scripts/kaggle/train_eval.py    training, NMS sweep, test mAP, prediction export
scripts/build_report.py         site data and charts
scripts/render_overlay.py       overlays used for the hand audit
docs/                           GitHub Pages site
```

Code: MIT. Data: SKU-110K, non-commercial research licence.

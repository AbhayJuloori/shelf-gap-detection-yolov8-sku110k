"""Gap evaluation: predicted-box gaps vs gaps derived from the ground-truth boxes.

SKU-110K has no gap labels. The reference ("gold") gap set is built by running the same row/gap
rules on the annotated product boxes with fixed settings (gap wider than one median facing). The
detector side has two knobs, box confidence threshold and gap factor, which are tuned on the
validation split and then applied once to the official test split.

Usage: python -m shelfgap.evaluate_gaps --data data/kaggle --out docs/data/gap_report.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from shelfgap.gaps import analyse, match_gaps, to_pixels

GOLD = {"row_overlap": 0.5, "gap_factor": 1.0, "min_row_boxes": 3}
CONF_GRID = [0.15, 0.25, 0.35, 0.45, 0.55]
FACTOR_GRID = [0.6, 0.8, 1.0, 1.2, 1.5]
IOU_THR = 0.3
SEED = 0


def load(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def gold_reports(gt: dict) -> dict:
    return {k: analyse(to_pixels(v["boxes"], v["size"]), **GOLD) for k, v in gt.items()}


def pred_reports(preds: dict, gt: dict, conf: float, factor: float) -> dict:
    out = {}
    for k, v in gt.items():
        boxes = [b for b in preds.get(k, []) if b[4] >= conf]
        out[k] = analyse(to_pixels(boxes, v["size"]), GOLD["row_overlap"], factor, GOLD["min_row_boxes"])
    return out


def score(pred: dict, gold: dict, keys=None) -> dict:
    keys = list(gold) if keys is None else keys
    per = np.array([match_gaps(pred[k].gaps, gold[k].gaps, IOU_THR) for k in keys], dtype=float)
    tp, fp, fn = per.sum(0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    pc = np.array([pred[k].gap_count for k in keys])
    gc = np.array([gold[k].gap_count for k in keys])
    pf, gf = pc >= 1, gc >= 1
    ftp = int((pf & gf).sum())
    return {
        "images": len(keys), "gold_gaps": int(gc.sum()), "pred_gaps": int(pc.sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "count_mae": round(float(np.abs(pc - gc).mean()), 3),
        "count_corr": round(float(np.corrcoef(pc, gc)[0, 1]), 4) if pc.std() and gc.std() else None,
        "flag": {
            "gold_rate": round(float(gf.mean()), 4),
            "accuracy": round(float((pf == gf).mean()), 4),
            "precision": round(ftp / max(1, int(pf.sum())), 4),
            "recall": round(ftp / max(1, int(gf.sum())), 4),
        },
        "_per": per,
    }


def bootstrap_f1(per: np.ndarray, n: int = 1000, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    f1s = []
    for _ in range(n):
        tp, fp, fn = per[rng.integers(0, len(per), len(per))].sum(0)
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return [round(float(np.percentile(f1s, 2.5)), 4), round(float(np.percentile(f1s, 97.5)), 4)]


def clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def run(data: Path) -> dict:
    out = {"gold_settings": GOLD, "match_iou": IOU_THR, "conf_grid": CONF_GRID, "factor_grid": FACTOR_GRID}
    val_gt, val_pr = load(data / "val_gt.json.gz"), load(data / "val_preds.json.gz")
    gold_v = gold_reports(val_gt)
    grid = []
    for c in CONF_GRID:
        for f in FACTOR_GRID:
            s = score(pred_reports(val_pr, val_gt, c, f), gold_v)
            grid.append({"conf": c, "gap_factor": f, **clean(s)})
    best = max(grid, key=lambda d: d["f1"])
    out["val_grid"] = grid
    out["chosen"] = {"conf": best["conf"], "gap_factor": best["gap_factor"], "val_f1": best["f1"]}

    test_gt, test_pr = load(data / "test_gt.json.gz"), load(data / "test_preds.json.gz")
    gold_t = gold_reports(test_gt)
    s = score(pred_reports(test_pr, test_gt, best["conf"], best["gap_factor"]), gold_t)
    out["test"] = {**clean(s), "f1_ci95": bootstrap_f1(s["_per"])}
    # untuned default (conf .25, factor 1.0) for reference
    s0 = score(pred_reports(test_pr, test_gt, 0.25, 1.0), gold_t)
    out["test_default"] = {**clean(s0), "f1_ci95": bootstrap_f1(s0["_per"])}
    rows = [len(r.rows) for r in gold_t.values()]
    out["test_gold_stats"] = {
        "rows_per_image_median": float(np.median(rows)),
        "images_with_gap": int(sum(r.gap_count > 0 for r in gold_t.values())),
        "gap_width_facings_median": round(float(np.median(
            [g.width_in_facings for r in gold_t.values() for g in r.gaps] or [0])), 3),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/kaggle", type=Path)
    ap.add_argument("--out", default="docs/data/gap_report.json", type=Path)
    a = ap.parse_args()
    rep = run(a.data)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("chosen", "test", "test_default", "test_gold_stats")}, indent=1))


if __name__ == "__main__":
    main()

"""Draw ground-truth (or predicted) boxes, shelf rows and gaps on test images for visual audit.

Usage: python scripts/render_overlay.py --data data/kaggle --images data/images --out data/overlays [--pred]
"""
import argparse
import gzip
import json
from pathlib import Path

import cv2

from shelfgap.evaluate_gaps import GOLD
from shelfgap.gaps import analyse, to_pixels

COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]


def render(img, boxes, report, max_side=1100):
    out = img.copy()
    for r_i, row in enumerate(report.rows):
        c = COLORS[r_i % len(COLORS)]
        for x1, y1, x2, y2 in row.boxes.astype(int):
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 3)
    for k, g in enumerate(report.gaps):
        cv2.rectangle(out, (int(g.x1), int(g.y1)), (int(g.x2), int(g.y2)), (0, 0, 255), 8)
        cv2.putText(out, str(k), (int(g.x1) + 5, int(g.y1) + 60), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
    s = max_side / max(out.shape[:2])
    return cv2.resize(out, None, fx=s, fy=s) if s < 1 else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/kaggle"))
    ap.add_argument("--images", type=Path, default=Path("data/images"))
    ap.add_argument("--out", type=Path, default=Path("data/overlays"))
    ap.add_argument("--pred", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--gap-factor", type=float, default=GOLD["gap_factor"])
    a = ap.parse_args()
    gt = json.load(gzip.open(a.data / "test_gt.json.gz", "rt"))
    preds = json.load(gzip.open(a.data / "test_preds.json.gz", "rt")) if a.pred else None
    a.out.mkdir(parents=True, exist_ok=True)
    for p in sorted(a.images.glob("*.jpg")):
        if p.name not in gt:
            continue
        img = cv2.imread(str(p))
        if preds is not None:
            src = [b for b in preds.get(p.name, []) if b[4] >= a.conf]
        else:
            src = gt[p.name]["boxes"]
        boxes = to_pixels(src, gt[p.name]["size"])
        rep = analyse(boxes, GOLD["row_overlap"], a.gap_factor, GOLD["min_row_boxes"])
        tag = "pred" if a.pred else "gt"
        cv2.imwrite(str(a.out / f"{p.stem}_{tag}.jpg"), render(img, boxes, rep), [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(p.name, tag, "rows", len(rep.rows), "gaps", rep.gap_count,
              [round(g.width_in_facings, 1) for g in rep.gaps])


if __name__ == "__main__":
    main()

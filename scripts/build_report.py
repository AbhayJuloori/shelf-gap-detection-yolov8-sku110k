"""Collect the Kaggle run outputs and the gap evaluation into docs/data/ for the README and the site.

Usage: python scripts/build_report.py --kaggle data/kaggle
Writes docs/data/site.json, docs/img/*.png, and data_release/ (val/test predictions at
conf >= 0.15 + ground truth, small enough to commit so CI can re-score gaps).
"""
import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MIN_CONF = 0.15
EXAMPLES = [
    {"src": "img/example_2329.jpg", "alt": "Deodorant aisle with three flagged gaps",
     "caption": "test_2329: all 3 flagged gaps are empty slots (hand audit 3/3)."},
    {"src": "img/example_970.jpg", "alt": "Deodorant aisle with an empty section top right",
     "caption": "test_970: a 3.3-facing empty section and two single slots, all real (3/3)."},
    {"src": "img/example_1928.jpg", "alt": "Toothpaste boxes stacked flat at an angle",
     "caption": "Failure mode, test_1928: flat stacked boxes at a steep angle break row grouping (1 of 30 real)."},
]
INK, MUTED, ACCENT, ACCENT2 = "#1f2933", "#7b8794", "#2f6fdf", "#e0702b"


def training_curves(results_csv: Path, out: Path) -> list[dict]:
    rows = list(csv.DictReader(open(results_csv)))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    ep = [int(float(r["epoch"])) for r in rows]
    m = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    m50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    ax.plot(ep, m50, color=ACCENT2, lw=1.8, label="mAP@.5 (val)")
    ax.plot(ep, m, color=ACCENT, lw=1.8, label="mAP@.5:.95 (val)")
    ax.axhline(0.492, color=MUTED, ls="--", lw=1)
    ax.text(ep[0], 0.497, "paper AP .492 (test)", color=MUTED, fontsize=8)
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return [{"epoch": e, "map50_95": round(a, 4), "map50": round(b, 4)} for e, a, b in zip(ep, m, m50, strict=True)]


def filter_preds(src: Path, dst: Path):
    with gzip.open(src, "rt") as f:
        d = json.load(f)
    d = {k: [b for b in v if b[4] >= MIN_CONF] for k, v in d.items()}
    with gzip.open(dst, "wt") as f:
        json.dump(d, f, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", type=Path, default=Path("data/kaggle"))
    a = ap.parse_args()
    k = a.kaggle
    Path("docs/img").mkdir(parents=True, exist_ok=True)
    Path("data_release").mkdir(exist_ok=True)
    metrics = json.load(open(k / "metrics.json"))
    prep = json.load(open(k / "prep_report.json"))
    curves = training_curves(k / "runs/y8s640/results.csv", Path("docs/img/training_curves.png"))
    for split in ("val", "test"):
        filter_preds(k / f"{split}_preds.json.gz", Path(f"data_release/{split}_preds.json.gz"))
        shutil.copy(k / f"{split}_gt.json.gz", f"data_release/{split}_gt.json.gz")
    shutil.copy(k / "runs/y8s640/args.yaml", "docs/data/train_args.yaml")
    gap = json.loads(Path("docs/data/gap_report.json").read_text())
    audit_path = Path("docs/data/hand_audit.json")
    site = {
        "detector": {k2: metrics[k2] for k2 in metrics if k2 not in ("counts",)},
        "data_counts": prep["counts"],
        "data_issues": prep["issues"],
        "epochs_trained": len(curves),
        "curves": curves,
        "gaps": {k2: gap[k2] for k2 in ("gold_settings", "match_iou", "chosen", "test", "test_default",
                                        "test_gold_stats")},
        "hand_audit": json.loads(audit_path.read_text()) if audit_path.exists() else None,
        "examples": EXAMPLES,
        "api_example": json.loads(Path("docs/data/api_example.json").read_text())
        if Path("docs/data/api_example.json").exists() else None,
    }
    Path("docs/data/site.json").write_text(json.dumps(site, indent=1))
    print(json.dumps({"epochs": len(curves), "test": metrics.get("test"), "gaps": site["gaps"]["test"]}, indent=1))


if __name__ == "__main__":
    main()

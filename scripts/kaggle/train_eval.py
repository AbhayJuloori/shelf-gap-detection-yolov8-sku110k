"""Kaggle GPU kernel: fine-tune YOLOv8s on SKU-110K, tune NMS on val, score the official test split,
and export val/test predictions + ground truth for the local gap analysis.

Runs on 2x T4. Inputs: Kaggle dataset thedatasith/sku110k-annotations (images + YOLO txt labels).
"""
import gzip
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.40"], check=True)
# Kaggle's preinstalled ray breaks the Ultralytics raytune callback (also inside DDP workers)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "ray", "wandb", "mlflow", "comet_ml",
                "clearml", "dvclive", "neptune"], check=False)

T0 = time.time()
OUT = Path("/kaggle/working")
WORK = Path("/tmp/sku")
MAX_SIDE = 1024
SEED = 0
TRAIN_HOURS = 6.5
EXPORT_ONLY = False  # set True by push.sh --export: reuse best.pt from the train kernel output, only export predictions
SMOKE = False  # set True by scripts/kaggle/push.sh --smoke: tiny subset, 1 epoch, end-to-end check


def log(msg):
    print(f"[{time.time() - T0:8.0f}s] {msg}", flush=True)


def find_root():
    for p in Path("/kaggle/input").rglob("SKU110K_fixed"):
        if (p / "images").is_dir():
            return p
    raise SystemExit("SKU110K_fixed/images not found under /kaggle/input")


def resize_one(args):
    from PIL import Image

    src, dst_img, src_lbl, dst_lbl = args
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = min(1.0, MAX_SIDE / max(w, h))
            if s < 1.0:
                im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
            im.save(dst_img, quality=92)
    except Exception as e:  # corrupt files exist in SKU-110K
        return src.name, None, None, f"bad image: {e}"
    if not src_lbl.exists():
        return src.name, w, h, "missing label"
    rows = [r.split() for r in src_lbl.read_text().splitlines() if r.strip()]
    bad = [r for r in rows if len(r) != 5 or r[0] != "0"]
    dst_lbl.write_text("\n".join(" ".join(r) for r in rows) + "\n")
    return src.name, w, h, f"{len(bad)} malformed rows" if bad else None


def prepare():
    root = find_root()
    log(f"dataset root {root}")
    sizes, issues, counts = {}, {}, {}
    for split in (("val", "test") if EXPORT_ONLY else ("train", "val", "test")):
        (WORK / "images" / split).mkdir(parents=True, exist_ok=True)
        (WORK / "labels" / split).mkdir(parents=True, exist_ok=True)
        imgs = sorted((root / "images" / split).glob("*.jpg"))
        if SMOKE:
            imgs = imgs[: 300 if split == "train" else 40]
        jobs = [
            (p, WORK / "images" / split / p.name, root / "labels" / split / (p.stem + ".txt"),
             WORK / "labels" / split / (p.stem + ".txt"))
            for p in imgs
        ]
        with ProcessPoolExecutor(4) as ex:
            res = list(ex.map(resize_one, jobs, chunksize=32))
        ok = 0
        for name, w, h, issue in res:
            if w is not None:
                sizes[f"{split}/{name}"] = [w, h]
            if issue:
                issues[f"{split}/{name}"] = issue
                if issue.startswith("bad image"):
                    (WORK / "images" / split / name).unlink(missing_ok=True)
                    (WORK / "labels" / split / (Path(name).stem + ".txt")).unlink(missing_ok=True)
            if w is not None and issue != "missing label":
                ok += 1
        counts[split] = {"images_in_source": len(imgs), "usable": ok}
        log(f"{split}: {counts[split]}")
    (WORK / "sku.yaml").write_text(
        f"path: {WORK}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: object\n"
    )
    json.dump({"counts": counts, "issues": issues}, open(OUT / "prep_report.json", "w"), indent=1)
    return sizes, counts


def read_labels(split, sizes):
    gt = {}
    for lbl in sorted((WORK / "labels" / split).glob("*.txt")):
        name = lbl.stem + ".jpg"
        if f"{split}/{name}" not in sizes:
            continue
        boxes = []
        for r in lbl.read_text().split("\n"):
            p = r.split()
            if len(p) == 5:
                x, y, w, h = map(float, p[1:])
                boxes.append([round(x - w / 2, 5), round(y - h / 2, 5), round(x + w / 2, 5), round(y + h / 2, 5)])
        gt[name] = {"size": sizes[f"{split}/{name}"], "boxes": boxes}
    return gt


def metrics_dict(m):
    return {
        "map50_95": float(m.box.map),
        "map50": float(m.box.map50),
        "map75": float(m.box.map75),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "speed_ms": {k: float(v) for k, v in m.speed.items()},
    }


def export_preds(model, split, iou, imgsz, conf=0.05):
    out = {}
    files = sorted(str(p) for p in (WORK / "images" / split).glob("*.jpg"))
    # One file per call: a list source is run as a single batch in ultralytics 8.3 and OOMs a T4
    for fp in files:
        r = model.predict(fp, imgsz=imgsz, conf=conf, iou=iou, max_det=1000, half=True, device=0, verbose=False)[0]
        b = r.boxes
        xyxyn = b.xyxyn.cpu().numpy().round(5).tolist()
        cf = b.conf.cpu().numpy().round(4).tolist()
        out[Path(r.path).name] = [x + [c] for x, c in zip(xyxyn, cf)]
    return out


def main():
    from ultralytics import YOLO, settings

    settings.update({k: False for k in ("raytune", "wandb", "mlflow", "comet", "clearml", "dvc", "neptune",
                                        "tensorboard") if k in settings})
    sizes, counts = prepare()
    if EXPORT_ONLY:
        return export_only(sizes)
    yaml = str(WORK / "sku.yaml")
    report = {"counts": counts, "seed": SEED, "train_hours_cap": TRAIN_HOURS, "model": "yolov8s.pt", "imgsz": 640}

    model = YOLO("yolov8s.pt")
    kw = dict(data=yaml, epochs=1 if SMOKE else 100, time=None if SMOKE else TRAIN_HOURS, imgsz=640, batch=32, workers=4, seed=SEED,
              deterministic=True, project=str(OUT / "runs"), name="y8s640", exist_ok=True, max_det=1000,
              patience=100, plots=True, cache=False)
    try:
        model.train(device=[0, 1], **kw)
        report["devices"] = "0,1 (DDP)"
    except Exception as e:  # DDP can fail inside kernels; fall back to one GPU
        log(f"DDP failed ({e!r}); retrying on one GPU")
        model = YOLO("yolov8s.pt")
        model.train(device=0, **{**kw, "batch": 16})
        report["devices"] = "0"
    report["train_seconds"] = time.time() - T0
    best = OUT / "runs" / "y8s640" / "weights" / "best.pt"
    log(f"training done, best={best}")
    model = YOLO(str(best))

    # NMS IoU + max_det sweep on VAL only (test stays untouched for selection)
    sweep = []
    for iou in (0.5, 0.6, 0.7, 0.8):
        m = model.val(data=yaml, split="val", imgsz=640, batch=16, conf=0.001, iou=iou, max_det=1000,
                      device=0, plots=False, verbose=False)
        sweep.append({"iou": iou, "max_det": 1000, **metrics_dict(m)})
        log(f"val iou={iou}: {sweep[-1]['map50_95']:.4f}")
    best_iou = max(sweep, key=lambda d: d["map50_95"])["iou"]
    m = model.val(data=yaml, split="val", imgsz=640, batch=16, conf=0.001, iou=best_iou, max_det=300,
                  device=0, plots=False, verbose=False)
    sweep.append({"iou": best_iou, "max_det": 300, **metrics_dict(m)})
    m = model.val(data=yaml, split="val", imgsz=640, batch=16, conf=0.001, iou=best_iou, max_det=100,
                  device=0, plots=False, verbose=False)
    sweep.append({"iou": best_iou, "max_det": 100, **metrics_dict(m)})
    m = model.val(data=yaml, split="val", imgsz=1024, batch=8, conf=0.001, iou=best_iou, max_det=1000,
                  device=0, plots=False, verbose=False)
    sweep.append({"iou": best_iou, "max_det": 1000, "imgsz": 1024, **metrics_dict(m)})
    report["val_sweep"] = sweep
    report["chosen_iou"] = best_iou
    v640 = next(d for d in sweep if d["iou"] == best_iou and d["max_det"] == 1000 and "imgsz" not in d)
    best_imgsz = 1024 if sweep[-1]["map50_95"] > v640["map50_95"] else 640
    report["chosen_imgsz"] = best_imgsz
    json.dump(report, open(OUT / "metrics.json", "w"), indent=1)

    # Official test split: report both sizes, the chosen one is the headline
    report["test"] = {}
    for sz in (640, 1024):
        m = model.val(data=yaml, split="test", imgsz=sz, batch=8, conf=0.001, iou=best_iou, max_det=1000,
                      device=0, plots=(sz == best_imgsz), project=str(OUT / "runs"), name=f"test{sz}",
                      exist_ok=True, verbose=False)
        report["test"][str(sz)] = metrics_dict(m)
        log(f"test imgsz={sz}: {report['test'][str(sz)]}")
    json.dump(report, open(OUT / "metrics.json", "w"), indent=1)

    export_all(model, sizes, best_iou, best_imgsz)
    report["total_seconds"] = time.time() - T0
    json.dump(report, open(OUT / "metrics.json", "w"), indent=1)
    log("done")


def export_all(model, sizes, iou, imgsz):
    for split in ("val", "test"):
        preds = export_preds(model, split, iou, imgsz)
        with gzip.open(OUT / f"{split}_preds.json.gz", "wt") as f:
            json.dump(preds, f)
        with gzip.open(OUT / f"{split}_gt.json.gz", "wt") as f:
            json.dump(read_labels(split, sizes), f)
        log(f"exported {split}: {len(preds)} images")


def export_only(sizes):
    """Second pass used after the train kernel ran out of GPU memory while exporting predictions."""
    from ultralytics import YOLO

    src = next(Path("/kaggle/input").rglob("metrics.json"))
    report = json.load(open(src))
    best = next(Path("/kaggle/input").rglob("y8s640/weights/best.pt"))
    log(f"export-only: weights {best}, iou {report['chosen_iou']}, imgsz {report['chosen_imgsz']}")
    export_all(YOLO(str(best)), sizes, report["chosen_iou"], report["chosen_imgsz"])
    log("done")


if __name__ == "__main__":
    main()

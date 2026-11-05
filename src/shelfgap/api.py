"""FastAPI service: one shelf photo in, bay-level gap count and restock flag out.

The response carries a ``task`` block in the same shape the store task planner consumes
(type, bay, priority, units estimate), so a flagged bay can be queued as a restock task.

Run: SHELFGAP_WEIGHTS=models/best.pt uvicorn shelfgap.api:app --port 8000
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from shelfgap.gaps import ShelfReport, analyse

CONFIG_PATH = Path(os.environ.get("SHELFGAP_CONFIG", "configs/operating_point.json"))
DEFAULTS = {"conf": 0.25, "gap_factor": 1.0, "row_overlap": 0.5, "min_row_boxes": 3, "nms_iou": 0.7,
            "imgsz": 640, "max_det": 1000, "high_priority_gaps": 3}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    return cfg


class Detector(Protocol):
    def __call__(self, image: np.ndarray) -> np.ndarray:  # (n, 5) pixel xyxy + conf
        ...


class YoloDetector:
    def __init__(self, weights: str, cfg: dict):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.cfg = cfg

    def __call__(self, image: np.ndarray) -> np.ndarray:
        r = self.model.predict(image, imgsz=self.cfg["imgsz"], conf=self.cfg["conf"], iou=self.cfg["nms_iou"],
                               max_det=self.cfg["max_det"], verbose=False)[0]
        b = r.boxes
        return np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]])


def build_response(report: ShelfReport, n_boxes: int, size: tuple[int, int], bay_id: str, cfg: dict) -> dict:
    flag = report.restock_flag()
    missing = int(sum(round(g.width_in_facings) for g in report.gaps))
    return {
        "bay_id": bay_id,
        "image_size": list(size),
        "facings_detected": n_boxes,
        "shelf_rows": len(report.rows),
        "gap_count": report.gap_count,
        "gaps": [{"row": g.row, "box": [round(v, 1) for v in g.as_list()],
                  "width_in_facings": round(g.width_in_facings, 2)} for g in report.gaps],
        "restock_flag": flag,
        "task": None if not flag else {
            "type": "restock",
            "bay_id": bay_id,
            "priority": "high" if report.gap_count >= cfg["high_priority_gaps"] else "normal",
            "est_missing_facings": max(1, missing),
            "source": "shelf_gap_detector",
        },
    }


def draw(image: np.ndarray, boxes: np.ndarray, report: ShelfReport) -> np.ndarray:
    out = image.copy()
    for x1, y1, x2, y2 in boxes[:, :4].astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 180, 75), 2)
    for g in report.gaps:
        cv2.rectangle(out, (int(g.x1), int(g.y1)), (int(g.x2), int(g.y2)), (0, 0, 230), 4)
    return out


def create_app(detector: Detector | None = None, cfg: dict | None = None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(title="Shelf gap detector", version="1.0")
    state = {"detector": detector}

    def get_detector() -> Detector:
        if state["detector"] is None:
            weights = os.environ.get("SHELFGAP_WEIGHTS", "models/best.pt")
            if not Path(weights).exists():
                raise HTTPException(503, f"weights not found: {weights}")
            state["detector"] = YoloDetector(weights, cfg)
        return state["detector"]

    @app.get("/health")
    def health():
        return {"status": "ok", "config": cfg}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), bay_id: str = Query("unknown"),
                      overlay: bool = Query(False)):
        raw = np.frombuffer(await file.read(), np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "could not decode image")
        dets = np.asarray(get_detector()(image), dtype=float).reshape(-1, 5)
        dets = dets[dets[:, 4] >= cfg["conf"]]
        report = analyse(dets[:, :4], cfg["row_overlap"], cfg["gap_factor"], cfg["min_row_boxes"])
        if overlay:
            ok, png = cv2.imencode(".png", draw(image, dets, report))
            return Response(png.tobytes(), media_type="image/png")
        h, w = image.shape[:2]
        return build_response(report, len(dets), (w, h), bay_id, cfg)

    return app


app = create_app()

import cv2
import numpy as np
from fastapi.testclient import TestClient

from shelfgap.api import create_app

FIXED_DETECTIONS = np.array(
    [
        [5, 10, 15, 30, 0.9],
        [15, 10, 25, 30, 0.9],
        [45, 10, 55, 30, 0.9],
        [55, 10, 65, 30, 0.9],
    ],
    dtype=float,
)
CONFIG = {
    "conf": 0.25,
    "gap_factor": 1.0,
    "row_overlap": 0.5,
    "min_row_boxes": 3,
    "high_priority_gaps": 1,
}


class FakeDetector:
    def __call__(self, image):
        assert image.shape == (40, 80, 3)
        return FIXED_DETECTIONS.copy()


def png_bytes():
    ok, encoded = cv2.imencode(".png", np.zeros((40, 80, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_predict_returns_gap_and_high_priority_task():
    client = TestClient(create_app(detector=FakeDetector(), cfg=CONFIG))

    response = client.post(
        "/predict",
        params={"bay_id": "bay-17"},
        files={"file": ("shelf.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["bay_id"] == "bay-17"
    assert body["gap_count"] == 1
    assert body["restock_flag"] is True
    assert body["task"] == {
        "type": "restock",
        "bay_id": "bay-17",
        "priority": "high",
        "est_missing_facings": 2,
        "source": "shelf_gap_detector",
    }


def test_predict_overlay_returns_png():
    client = TestClient(create_app(detector=FakeDetector(), cfg=CONFIG))

    response = client.post(
        "/predict",
        params={"overlay": "true"},
        files={"file": ("shelf.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_predict_rejects_undecodable_bytes():
    client = TestClient(create_app(detector=FakeDetector(), cfg=CONFIG))

    response = client.post(
        "/predict",
        files={"file": ("not-an-image.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "could not decode image"}


def test_health():
    client = TestClient(create_app(detector=FakeDetector(), cfg=CONFIG))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "config": CONFIG}

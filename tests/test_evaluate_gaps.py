import gzip
import json

import pytest

from shelfgap.evaluate_gaps import gold_reports, pred_reports, run, score

GAP_BOXES = [
    [0.0, 0.1, 0.1, 0.3],
    [0.1, 0.1, 0.2, 0.3],
    [0.4, 0.1, 0.5, 0.3],
    [0.5, 0.1, 0.6, 0.3],
]


def write_gzip_json(path, value):
    with gzip.open(path, "wt") as stream:
        json.dump(value, stream)


def predictions(boxes, confidence=0.9):
    return [box + [confidence] for box in boxes]


@pytest.fixture
def evaluation_data(tmp_path):
    gt = {"shelf.png": {"size": [100, 100], "boxes": GAP_BOXES}}
    preds = {"shelf.png": predictions(GAP_BOXES)}
    for split in ("val", "test"):
        write_gzip_json(tmp_path / f"{split}_gt.json.gz", gt)
        write_gzip_json(tmp_path / f"{split}_preds.json.gz", preds)
    return tmp_path


def test_run_perfect_predictions_have_perfect_gap_f1(evaluation_data):
    report = run(evaluation_data)

    assert report["chosen"]["val_f1"] == 1.0
    assert report["test"]["f1"] == 1.0
    assert report["test"]["precision"] == 1.0
    assert report["test"]["recall"] == 1.0


def test_run_selects_parameters_from_validation_only(evaluation_data):
    before = run(evaluation_data)
    write_gzip_json(evaluation_data / "test_preds.json.gz", {"shelf.png": []})

    after = run(evaluation_data)

    assert after["chosen"] == before["chosen"]
    assert before["test"]["f1"] == 1.0
    assert after["test"]["f1"] == 0.0


def test_dropping_edge_facing_lowers_facing_recall_but_preserves_gap_metrics():
    gt = {"shelf.png": {"size": [100, 100], "boxes": GAP_BOXES}}
    preds = {"shelf.png": predictions(GAP_BOXES[:-1])}
    gold = gold_reports(gt)
    predicted = pred_reports(preds, gt, conf=0.25, factor=1.0)

    gold_facings = sum(len(row.boxes) for row in gold["shelf.png"].rows)
    pred_facings = sum(len(row.boxes) for row in predicted["shelf.png"].rows)
    metrics = score(predicted, gold)

    assert pred_facings / gold_facings == pytest.approx(0.75)
    assert metrics["gold_gaps"] == 1
    assert metrics["pred_gaps"] == 1
    assert metrics["f1"] == 1.0


def test_pred_reports_filters_low_confidence_boxes():
    gt = {"shelf.png": {"size": [100, 100], "boxes": GAP_BOXES}}
    bridge = [0.2, 0.1, 0.4, 0.3, 0.1]
    preds = {"shelf.png": predictions(GAP_BOXES) + [bridge]}

    filtered = pred_reports(preds, gt, conf=0.25, factor=1.0)["shelf.png"]
    unfiltered = pred_reports(preds, gt, conf=0.05, factor=1.0)["shelf.png"]

    assert sum(len(row.boxes) for row in filtered.rows) == 4
    assert filtered.gap_count == 1
    assert sum(len(row.boxes) for row in unfiltered.rows) == 5
    assert unfiltered.gap_count == 0

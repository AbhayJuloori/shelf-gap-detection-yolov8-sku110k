import numpy as np
import pytest

from shelfgap.gaps import Gap, find_gaps, gap_iou, group_rows, match_gaps, to_pixels


def boxes_for_rows(n_rows: int) -> np.ndarray:
    return np.array(
        [
            [x, row * 20, x + 10, row * 20 + 10]
            for row in range(n_rows)
            for x in (0, 10, 20, 30)
        ],
        dtype=float,
    )


@pytest.mark.parametrize("n_rows", [2, 3])
def test_group_rows_equal_boxes(n_rows):
    rows = group_rows(boxes_for_rows(n_rows))

    assert len(rows) == n_rows
    assert [len(row.boxes) for row in rows] == [4] * n_rows
    assert [row.y1 for row in rows] == [row * 20 for row in range(n_rows)]


def test_group_rows_tall_box_does_not_merge_shelves():
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [10, 0, 20, 10],
            [20, 0, 30, 24],  # overlaps only 4/10 of the next shelf row
            [0, 20, 10, 30],
            [10, 20, 20, 30],
            [20, 20, 30, 30],
        ]
    )

    rows = group_rows(boxes)

    assert len(rows) == 2
    assert [len(row.boxes) for row in rows] == [3, 3]
    assert (rows[0].y1, rows[0].y2) == (0, 10)
    assert (rows[1].y1, rows[1].y2) == (20, 30)


def test_group_rows_empty_input():
    assert group_rows([]) == []


def test_find_gaps_contiguous_row_has_no_gap():
    rows = group_rows([[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]])

    assert find_gaps(rows) == []


def test_find_gaps_detects_two_facing_interior_gap():
    rows = group_rows(
        [[0, 0, 10, 10], [10, 0, 20, 10], [40, 0, 50, 10], [50, 0, 60, 10]]
    )

    gaps = find_gaps(rows)

    assert len(gaps) == 1
    assert (gaps[0].x1, gaps[0].x2) == (20, 40)
    assert gaps[0].width_in_facings == pytest.approx(2.0)


def test_find_gaps_ignores_space_outside_first_and_last_box():
    # The row starts at x=20 and ends at x=50; only spaces between boxes are evidence.
    rows = group_rows([[20, 0, 30, 10], [30, 0, 40, 10], [40, 0, 50, 10]])

    assert find_gaps(rows) == []


def test_find_gaps_overlapping_boxes_do_not_create_fake_gap():
    rows = group_rows([[0, 0, 10, 10], [8, 0, 18, 10], [16, 0, 26, 10]])

    assert find_gaps(rows) == []


def test_find_gaps_skips_rows_below_minimum_box_count():
    rows = group_rows([[0, 0, 10, 10], [40, 0, 50, 10]])

    assert find_gaps(rows, min_row_boxes=3) == []


def test_find_gaps_uses_strict_gap_factor_threshold():
    rows = group_rows([[0, 0, 10, 10], [10, 0, 20, 10], [30, 0, 40, 10]])

    assert find_gaps(rows, gap_factor=1.0) == []
    assert len(find_gaps(rows, gap_factor=0.99)) == 1


def test_to_pixels_scales_coordinates_and_ignores_confidence():
    boxes = to_pixels([[0.1, 0.2, 0.5, 0.8, 0.9]], (200, 100))

    np.testing.assert_allclose(boxes, [[20, 20, 100, 80]])


def make_gap(x1, x2, y1=0, y2=10):
    return Gap(x1=x1, x2=x2, y1=y1, y2=y2, row=0, width_in_facings=1)


def test_gap_iou():
    a = make_gap(0, 10)

    assert gap_iou(a, make_gap(0, 10)) == pytest.approx(1.0)
    assert gap_iou(a, make_gap(5, 15)) == pytest.approx(1 / 3)
    assert gap_iou(a, make_gap(20, 30)) == 0.0


def test_match_gaps_is_one_to_one():
    gold = [make_gap(0, 10)]
    pred = [make_gap(0, 10), make_gap(0, 10)]

    assert match_gaps(pred, gold) == (1, 1, 0)


def test_match_gaps_respects_iou_threshold():
    pred = [make_gap(0, 10)]
    gold = [make_gap(5, 15)]

    assert match_gaps(pred, gold, iou_thr=0.30) == (1, 0, 0)
    assert match_gaps(pred, gold, iou_thr=0.34) == (0, 1, 1)


def test_sloped_shelf_stays_one_row_and_keeps_its_gap():
    # perspective shot: each facing sits 6 px lower than its left neighbour (height 40)
    boxes = np.array([[i * 30, i * 6, i * 30 + 28, i * 6 + 40] for i in range(12) if i not in (5, 6)], float)
    rows = group_rows(boxes)
    assert len(rows) == 1
    gaps = find_gaps(rows, gap_factor=1.0)
    assert len(gaps) == 1
    assert gaps[0].x1 == 4 * 30 + 28 and gaps[0].x2 == 7 * 30


def test_stacked_facings_form_separate_rows():
    top = [[i * 30, 0, i * 30 + 28, 20] for i in range(6)]
    bottom = [[i * 30, 21, i * 30 + 28, 41] for i in range(6)]
    assert len(group_rows(np.array(top + bottom, float))) == 2


def test_sliver_between_misaligned_neighbours_is_not_a_gap():
    row = [[0, 0, 28, 40], [30, 0, 58, 40], [60, 0, 88, 40], [150, 35, 178, 75], [180, 35, 208, 75]]
    rows = group_rows(np.array(row, float), row_overlap=0.1)
    assert len(rows) == 1
    assert find_gaps(rows, gap_factor=1.0) == []

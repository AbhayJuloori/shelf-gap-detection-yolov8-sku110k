"""Group product boxes into shelf rows and flag empty gaps between neighbouring facings.

All functions work on pixel boxes ``[x1, y1, x2, y2]``. No learned parameters: the two knobs
(``row_overlap`` and ``gap_factor``) are chosen on the validation split, never on test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Row:
    boxes: np.ndarray  # (n, 4) sorted by x1
    y1: float
    y2: float

    @property
    def median_width(self) -> float:
        return float(np.median(self.boxes[:, 2] - self.boxes[:, 0]))


@dataclass
class Gap:
    x1: float
    x2: float
    y1: float
    y2: float
    row: int
    width_in_facings: float

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class ShelfReport:
    rows: list[Row] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)

    @property
    def gap_count(self) -> int:
        return len(self.gaps)

    def restock_flag(self, min_gaps: int = 1) -> bool:
        return self.gap_count >= min_gaps


def vertical_overlap(a: np.ndarray, b_y1: float, b_y2: float) -> float:
    """Overlap of box a with band [b_y1, b_y2], as a share of the smaller height."""
    inter = max(0.0, min(a[3], b_y2) - max(a[1], b_y1))
    return inter / max(1e-9, min(a[3] - a[1], b_y2 - b_y1))


def group_rows(boxes: np.ndarray, row_overlap: float = 0.5, max_link: float = 8.0) -> list[Row]:
    """Assign boxes to shelf rows by chaining left-right neighbours.

    Each box links to its nearest box to the right that overlaps it vertically by at least
    ``row_overlap`` of the smaller height and starts within ``max_link`` box widths. Links are kept
    only when they are mutual (i's nearest right neighbour is j and j's nearest left neighbour is
    i), then rows are the connected components. Chaining neighbours instead of fitting one
    horizontal band follows shelves that slope in perspective shots, and the mutual rule stops a
    tall box from bridging two shelves.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    n = len(boxes)
    if n == 0:
        return []
    w = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-9)
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-9)
    xc = (boxes[:, 0] + boxes[:, 2]) / 2
    # pairwise vertical overlap share and horizontal spacing (j to the right of i)
    inter = np.clip(np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1]),
                    0, None)
    vov = inter / np.minimum(h[:, None], h[None, :])
    space = boxes[None, :, 0] - boxes[:, None, 2]
    ok = (vov >= row_overlap) & (xc[None, :] > xc[:, None]) & (space < max_link * w[:, None])
    np.fill_diagonal(ok, False)
    cost = np.where(ok, np.maximum(space, -w[:, None]) , np.inf)
    right = np.where(ok.any(1), cost.argmin(1), -1)
    left_cost = np.where(ok, cost, np.inf)
    left = np.where(ok.any(0), left_cost.argmin(0), -1)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        j = right[i]
        if j >= 0 and left[j] == i:
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    rows = []
    for idx in groups.values():
        b = boxes[idx]
        rows.append(Row(boxes=b[np.argsort(b[:, 0])], y1=float(np.median(b[:, 1])), y2=float(np.median(b[:, 3]))))
    rows.sort(key=lambda r: (r.y1 + r.y2) / 2)
    return rows


def find_gaps(rows: list[Row], gap_factor: float = 1.0, min_row_boxes: int = 3) -> list[Gap]:
    """Flag horizontal gaps inside a row wider than ``gap_factor`` x the row's median facing width.

    Only interior gaps are flagged: the photo frame cuts rows at both ends, so empty space before
    the first or after the last facing is not evidence of a missing product. Rows with fewer than
    ``min_row_boxes`` boxes are skipped (too little evidence for a facing width). The gap band is
    the vertical overlap of the two neighbouring boxes, so it follows a sloping shelf; bands shorter
    than half the row's median box height are dropped.
    """
    gaps = []
    for r_i, row in enumerate(rows):
        if len(row.boxes) < min_row_boxes:
            continue
        w = row.median_width
        min_h = 0.5 * float(np.median(row.boxes[:, 3] - row.boxes[:, 1]))
        # running right edge, so an overlapping wide box does not open a fake gap
        right, prev = row.boxes[0, 2], row.boxes[0]
        for b in row.boxes[1:]:
            space = b[0] - right
            if space > gap_factor * w:
                y1, y2 = max(prev[1], b[1]), min(prev[3], b[3])
                if y2 <= y1:
                    y1, y2 = min(prev[1], b[1]), max(prev[3], b[3])
                if y2 - y1 >= min_h:  # a sliver between mismatched neighbours is not a facing slot
                    gaps.append(Gap(float(right), float(b[0]), float(y1), float(y2), r_i, float(space / w)))
            if b[2] >= right:
                right, prev = b[2], b
    return gaps


def analyse(boxes: np.ndarray, row_overlap: float = 0.5, gap_factor: float = 1.0,
            min_row_boxes: int = 3) -> ShelfReport:
    rows = group_rows(boxes, row_overlap)
    return ShelfReport(rows=rows, gaps=find_gaps(rows, gap_factor, min_row_boxes))


def to_pixels(norm_boxes, size) -> np.ndarray:
    w, h = size
    b = np.asarray([list(x)[:4] for x in norm_boxes], dtype=float).reshape(-1, 4)
    return b * np.array([w, h, w, h])


def gap_iou(a: Gap, b: Gap) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / ua if ua > 0 else 0.0


def match_gaps(pred: list[Gap], gold: list[Gap], iou_thr: float = 0.3) -> tuple[int, int, int]:
    """Greedy one-to-one matching by IoU. Returns (tp, fp, fn)."""
    pairs = sorted(((gap_iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold)),
                   reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for iou, i, j in pairs:
        if iou < iou_thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return tp, len(pred) - tp, len(gold) - tp

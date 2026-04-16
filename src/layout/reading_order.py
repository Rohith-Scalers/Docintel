"""XY-cut reading order algorithm for document layout regions.

Recursively bisects the page along the largest whitespace gap, alternating
between horizontal and vertical cuts, to produce a correct linear reading
order for single-column, two-column, and multi-column document layouts.
"""
from __future__ import annotations

from src.models import RawRegion

_GAP_THRESHOLD_PX: int = 20


def _find_largest_h_gap(regions: list[RawRegion]) -> float | None:
    """Find the y-coordinate of the largest horizontal whitespace gap.

    Projects all regions onto the Y axis and finds the largest gap between
    the end of one span and the start of the next. Returns None when no gap
    of at least _GAP_THRESHOLD_PX is found.

    Returns:
        float | None: midpoint y-coordinate of the largest gap, or None.
    """
    events: list[tuple[float, bool]] = []
    for r in regions:
        events.append((r.bbox.y0, True))
        events.append((r.bbox.y1, False))
    events.sort(key=lambda e: e[0])

    active: int = 0
    last_end: float = 0.0
    best_gap: float = 0.0
    best_mid: float | None = None

    for y, is_start in events:
        if active == 0 and last_end > 0:
            gap = y - last_end
            if gap > best_gap and gap >= _GAP_THRESHOLD_PX:
                best_gap = gap
                best_mid = (last_end + y) / 2.0
        if is_start:
            active += 1
        else:
            active -= 1
            last_end = max(last_end, y)

    return best_mid


def _find_largest_v_gap(regions: list[RawRegion]) -> float | None:
    """Find the x-coordinate of the largest vertical whitespace gap.

    Projects all regions onto the X axis and finds the largest gap between
    the end of one span and the start of the next. Returns None when no gap
    of at least _GAP_THRESHOLD_PX is found.

    Returns:
        float | None: midpoint x-coordinate of the largest gap, or None.
    """
    events: list[tuple[float, bool]] = []
    for r in regions:
        events.append((r.bbox.x0, True))
        events.append((r.bbox.x1, False))
    events.sort(key=lambda e: e[0])

    active: int = 0
    last_end: float = 0.0
    best_gap: float = 0.0
    best_mid: float | None = None

    for x, is_start in events:
        if active == 0 and last_end > 0:
            gap = x - last_end
            if gap > best_gap and gap >= _GAP_THRESHOLD_PX:
                best_gap = gap
                best_mid = (last_end + x) / 2.0
        if is_start:
            active += 1
        else:
            active -= 1
            last_end = max(last_end, x)

    return best_mid


def _xy_cut_recursive(regions: list[RawRegion], cut_horizontal_first: bool) -> list[RawRegion]:
    """Recursively partition regions using XY-cut and return in reading order.

    Alternates between horizontal (page-level) and vertical (column-level)
    cuts. Leaf groups (no further partition possible) are sorted by centroid
    (y, x).

    Returns:
        list[RawRegion]: regions in reading order.
    """
    if len(regions) <= 1:
        return regions

    if cut_horizontal_first:
        mid = _find_largest_h_gap(regions)
        if mid is not None:
            top = [r for r in regions if r.bbox.cy <= mid]
            bottom = [r for r in regions if r.bbox.cy > mid]
            if top and bottom:
                ordered_top = _xy_cut_recursive(top, cut_horizontal_first=False)
                ordered_bottom = _xy_cut_recursive(bottom, cut_horizontal_first=False)
                return ordered_top + ordered_bottom

        mid = _find_largest_v_gap(regions)
        if mid is not None:
            left = [r for r in regions if r.bbox.cx <= mid]
            right = [r for r in regions if r.bbox.cx > mid]
            if left and right:
                ordered_left = _xy_cut_recursive(left, cut_horizontal_first=True)
                ordered_right = _xy_cut_recursive(right, cut_horizontal_first=True)
                return ordered_left + ordered_right
    else:
        mid = _find_largest_v_gap(regions)
        if mid is not None:
            left = [r for r in regions if r.bbox.cx <= mid]
            right = [r for r in regions if r.bbox.cx > mid]
            if left and right:
                ordered_left = _xy_cut_recursive(left, cut_horizontal_first=True)
                ordered_right = _xy_cut_recursive(right, cut_horizontal_first=True)
                return ordered_left + ordered_right

        mid = _find_largest_h_gap(regions)
        if mid is not None:
            top = [r for r in regions if r.bbox.cy <= mid]
            bottom = [r for r in regions if r.bbox.cy > mid]
            if top and bottom:
                ordered_top = _xy_cut_recursive(top, cut_horizontal_first=False)
                ordered_bottom = _xy_cut_recursive(bottom, cut_horizontal_first=False)
                return ordered_top + ordered_bottom

    return sorted(regions, key=lambda r: (r.bbox.cy, r.bbox.cx))


def xy_cut_order(regions: list[RawRegion]) -> list[RawRegion]:
    """Return regions in correct reading order using the recursive XY-cut algorithm.

    XY-cut bisects the page along the largest whitespace gap alternating
    between horizontal and vertical cuts. This correctly handles single-column,
    two-column, and multi-column layouts, including sidebars.

    The algorithm:
    1. Find the largest horizontal whitespace gap (gap between projected Y spans).
    2. Split regions into top-half and bottom-half at that gap.
    3. Recursively process each half with vertical cuts first.
    4. For vertical cuts: find largest vertical whitespace gap, split left/right.
    5. Leaf groups (no more cuts possible) sorted by centroid (y, x).

    Updates region.region_index to reflect correct reading order (0-based).

    Returns:
        list[RawRegion]: same regions re-indexed and returned in reading order.
    """
    if not regions:
        return regions

    ordered = _xy_cut_recursive(regions, cut_horizontal_first=True)

    for idx, region in enumerate(ordered):
        region.region_index = idx

    return ordered

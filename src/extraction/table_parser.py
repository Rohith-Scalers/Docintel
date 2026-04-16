"""Parse VLM table extraction output into structured TableData objects."""

from __future__ import annotations

import re

from src.models import DocumentChunk, RegionType, TableCell, TableData

# Markdown table separator: a line consisting only of pipes, dashes, colons,
# and spaces (e.g. |---|:---:|---| ).
_MD_SEPARATOR_RE = re.compile(r"^\|?[\s\-:]+(?:\|[\s\-:]+)*\|?\s*$")

# Detect at least two pipe characters on a line to identify a table row.
_MD_ROW_RE = re.compile(r"\|")

# Terminal row keywords indicating a table is complete rather than truncated.
_TABLE_TERMINAL_KEYWORDS = re.compile(
    r"\b(total|grand total|subtotal|sum|average)\b",
    re.IGNORECASE,
)


def parse_table_response(response: dict) -> TableData:
    """Parse the VLM JSON response for a TABLE region into a TableData object.

    Handles three response shapes:

    1. Full: {"markdown": "...", "headers": [...], "rows": [[...]], "cells": [...]}
    2. Markdown only: {"markdown": "..."} — parse markdown table to extract
       headers and rows.
    3. Raw text fallback: {"raw_text": "..."} — attempt markdown extraction from
       the raw text string.

    For spanning cells: if the response contains cells with row_span or
    col_span > 1 these are preserved in TableData.cells.

    Returns:
        TableData: populated to the extent the response allows.
    """
    markdown: str = response.get("markdown", "").strip()
    raw_text: str = response.get("raw_text", "").strip()
    headers: list[str] = []
    rows: list[list[str]] = []
    cells: list[TableCell] = []

    # Path 1: structured response with explicit headers/rows.
    if response.get("headers") is not None and response.get("rows") is not None:
        headers = [str(h) for h in response["headers"]]
        rows = [[str(v) for v in row] for row in response["rows"]]
        if not markdown:
            markdown = _build_markdown(headers, rows)

    # Path 2: markdown-only or fallback from raw_text.
    elif markdown or raw_text:
        source = markdown or raw_text
        headers, rows = parse_markdown_table(source)
        if not markdown:
            markdown = source

    # Path 3: nothing usable.
    if not headers and not rows and not markdown:
        return TableData(markdown=raw_text or "")

    # Build cell list from explicit cells in response, or synthesise from rows.
    raw_cells: list[dict] = response.get("cells", []) or []
    if raw_cells:
        for c in raw_cells:
            try:
                cells.append(
                    TableCell(
                        row=int(c["row"]),
                        col=int(c["col"]),
                        row_span=int(c.get("row_span", 1)),
                        col_span=int(c.get("col_span", 1)),
                        value=str(c.get("value", "")),
                        is_header=bool(c.get("is_header", False)),
                    )
                )
            except (KeyError, ValueError):
                continue
    else:
        # Synthesise flat cell list from headers and rows without span data.
        for col_idx, header in enumerate(headers):
            cells.append(
                TableCell(row=0, col=col_idx, value=header, is_header=True)
            )
        for row_idx, row in enumerate(rows, start=1):
            for col_idx, value in enumerate(row):
                cells.append(TableCell(row=row_idx, col=col_idx, value=value))

    return TableData(headers=headers, rows=rows, cells=cells, markdown=markdown)


def parse_markdown_table(markdown: str) -> tuple[list[str], list[list[str]]]:
    """Parse a GitHub-flavoured markdown table string into headers and rows.

    Lines are processed in order. The first pipe-delimited line is treated as
    the header row. A separator line immediately following the header (matching
    |---|---|) is skipped. All remaining pipe-delimited lines become data rows.
    Leading and trailing pipes on each line are stripped before splitting.

    Returns:
        tuple[list[str], list[list[str]]]: (headers, rows) where each row is
        a list of cell value strings with leading/trailing whitespace stripped.
    """
    headers: list[str] = []
    rows: list[list[str]] = []
    header_found = False
    separator_skipped = False

    for line in markdown.splitlines():
        line = line.rstrip()
        if not _MD_ROW_RE.search(line):
            continue

        # Strip leading/trailing pipes before splitting.
        stripped = line.strip().strip("|")
        cells = [cell.strip() for cell in stripped.split("|")]

        if not header_found:
            headers = cells
            header_found = True
            continue

        if not separator_skipped and _MD_SEPARATOR_RE.match(line.strip()):
            separator_skipped = True
            continue

        rows.append(cells)

    return headers, rows


def detect_table_continuation(
    prev_chunks: list[DocumentChunk],
    current_region_type: RegionType,
    page_number: int,
) -> tuple[bool, float]:
    """Detect whether the current TABLE region continues a table from the previous page.

    Heuristic: if the last chunk on the previous page is a TABLE chunk and the
    current region type is TABLE, and the last TABLE chunk's markdown or row
    data shows no terminal row containing total/sum/average keywords (which
    would indicate the table completed on the prior page), returns
    (True, confidence_score).

    Confidence is set to 0.9 when both conditions are met, 0.6 when the
    previous chunk is TABLE but a terminal keyword pattern is ambiguous.

    Returns:
        tuple[bool, float]: (is_continuation, merge_confidence).
    """
    if current_region_type is not RegionType.TABLE:
        return False, 0.0

    prev_page = page_number - 1
    prev_page_chunks = [c for c in prev_chunks if c.page_number == prev_page]
    if not prev_page_chunks:
        return False, 0.0

    last_chunk = prev_page_chunks[-1]
    if last_chunk.content_type is not RegionType.TABLE:
        return False, 0.0

    # Check whether the last table row signals completion.
    candidate_text = ""
    if last_chunk.table_data:
        if last_chunk.table_data.rows:
            last_row = last_chunk.table_data.rows[-1]
            candidate_text = " ".join(last_row)
        elif last_chunk.table_data.markdown:
            lines = [l for l in last_chunk.table_data.markdown.splitlines() if l.strip()]
            if lines:
                candidate_text = lines[-1]
    else:
        candidate_text = last_chunk.raw_text[-200:] if last_chunk.raw_text else ""

    if _TABLE_TERMINAL_KEYWORDS.search(candidate_text):
        return False, 0.0

    return True, 0.9


def reclassify_figure_as_table(response: dict) -> bool:
    """Detect whether a FIGURE VLM response actually contains tabular data.

    Inspects the raw_text and image_description fields for a markdown table
    pattern: three or more lines containing pipe characters, consistent with
    a screenshot of a table being extracted as a figure.

    Returns:
        bool: True if the figure should be reclassified as TABLE.
    """
    candidate = response.get("raw_text", "") or response.get("image_description", "")
    if not candidate:
        return False

    pipe_lines = [
        line for line in candidate.splitlines()
        if _MD_ROW_RE.search(line) and "|" in line
    ]
    return len(pipe_lines) >= 3


def _build_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Build a GitHub-flavoured markdown table from headers and row data.

    Returns:
        str: markdown table string.
    """
    if not headers:
        return ""
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    header_row = "| " + " | ".join(headers) + " |"
    data_rows = [
        "| " + " | ".join(row) + " |"
        for row in rows
    ]
    return "\n".join([header_row, sep] + data_rows)

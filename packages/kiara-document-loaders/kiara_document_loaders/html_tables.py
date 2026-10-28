"""Resolve unambiguous data-table spans before native Markdown conversion.

Data tables have a leading thead or complete leading rows of th cells. A
presentation/none role or nested table marks a layout container. Its own cells
become ordered blocks, while inner data tables retain their structure. A lone
row has no following data to label and is also retained as ordered content.
Tables made exclusively of TD cells with explicit spans also get their DOM
grid expanded. Every cell remains TD: no heading, year, unit, currency or other
meaning is inferred. Unclear column headings remain diagnosed. Incomplete but
unambiguous geometry uses explicit cell records, never a padded native grid.
Ambiguous geometry fails before native conversion. No vendor-specific rules
are used.

A navigation landmark can contain a full-width title followed by lists or
row-labelled groups. That title is not repeated as a column label. Actual
column-header rows within navigation still use the data-table path.

The grid below comes only from DOM cells, never from the converter's exported
table model. Header paths join with `` / ``; rowspan keys and values spanning
several data columns are copied into those columns. A full-width note remains
one cell to avoid multiplying footer text.
"""

from copy import deepcopy

from bs4 import BeautifulSoup, Tag

_MAX_COLUMNS = 256
_MAX_GRID_CELLS = 100_000


def _cells(row: Tag) -> list[Tag]:
    return row.find_all(["th", "td"], recursive=False)


def _header_count(rows: list[Tag]) -> int:
    count = 0
    for row in rows:
        cells = _cells(row)
        if cells and (row.parent.name == "thead" or all(cell.name == "th" for cell in cells)):
            count += 1
        else:
            break
    return count


def _span(cell: Tag, name: str, remaining_rows: int) -> int:
    value = str(cell.get(name, "1")).strip()
    if not value.isascii() or not value.isdigit():
        raise ValueError("invalid span")
    # HTML rowspan=0 means the remaining rows of this row group.
    span = int(value)
    if name == "rowspan" and span == 0:
        span = remaining_rows
    limit = remaining_rows if name == "rowspan" else _MAX_COLUMNS
    if span < 1 or span > limit:
        raise ValueError("span outside row group or size limit")
    return span


def _grid(rows: list[Tag], header_count: int) -> list[list[Tag]]:
    occupied: list[dict[int, Tag]] = [{} for _ in rows]
    total = 0
    for row_index, row in enumerate(rows):
        group_end = row_index + 1
        while group_end < len(rows) and rows[group_end].parent is row.parent:
            group_end += 1
        column = 0
        for cell in _cells(row):
            while column in occupied[row_index]:
                column += 1
            height = _span(cell, "rowspan", group_end - row_index)
            width = _span(cell, "colspan", group_end - row_index)
            if column + width > _MAX_COLUMNS or total + width * height > _MAX_GRID_CELLS:
                raise ValueError("expanded table exceeds size limit")
            if row_index < header_count < row_index + height:
                raise ValueError("header span crosses into data rows")
            for index in range(row_index, row_index + height):
                for slot in range(column, column + width):
                    if slot in occupied[index]:
                        raise ValueError("overlapping spans")
                    occupied[index][slot] = cell
            total += width * height
            column += width
    width = max((len(row) for row in occupied), default=0)
    expected = set(range(width))
    if not width or any(set(row) != expected for row in occupied):
        raise ValueError("nonrectangular table")
    return [[row[column] for column in range(width)] for row in occupied]


def _copy_cell(cell: Tag, name: str, used_ids: set[str]) -> Tag:
    copied = deepcopy(cell)
    copied.name = name
    copied.attrs.pop("rowspan", None)
    copied.attrs.pop("colspan", None)
    # The original DOM can reference an ID once. Copies retain links/content,
    # but must not introduce duplicate anchor targets.
    for node in [copied, *copied.find_all(id=True)]:
        identifier = node.get("id")
        if identifier:
            if identifier in used_ids:
                node.attrs.pop("id", None)
            else:
                used_ids.add(identifier)
    return copied


def _cell_records(
    soup: BeautifulSoup, table: Tag, rows: list[Tag], count: int, number: int
) -> None:
    """Describe original cells when a rectangular Markdown grid is impossible.

    Only existing rows receive records. A dangling rowspan retains its exact
    declared extent and row-group boundary in the label; no row or empty cell
    is invented. Occupancy over existing rows supplies explicit inheritance.
    Overlap, cross-group spans and ambiguous header boundaries still fail.
    Each record is a one-column table so the normal chunker repeats its full
    geometry, explicit TH path and row context even for a long cell value.
    """
    occupied: list[dict[int, tuple[int, int]]] = [{} for _ in rows]
    origins: dict[tuple[int, int], tuple[Tag, int, int, int, int]] = {}
    total = 0
    for index, row in enumerate(rows):
        group_start, group_end = index, index + 1
        while group_start and rows[group_start - 1].parent is row.parent:
            group_start -= 1
        while group_end < len(rows) and rows[group_end].parent is row.parent:
            group_end += 1
        column = 0
        for cell in _cells(row):
            while column in occupied[index]:
                column += 1
            height = _span(cell, "rowspan", _MAX_GRID_CELLS)
            if int(str(cell.get("rowspan", "1")).strip()) == 0:
                height = group_end - index
            width = _span(cell, "colspan", group_end - index)
            if column + width > _MAX_COLUMNS or total + width * height > _MAX_GRID_CELLS:
                raise ValueError("table geometry exceeds size limit")
            if index < count < index + height:
                raise ValueError("header span crosses into data rows")
            if index + height > group_end and group_end != len(rows):
                raise ValueError("span crosses an explicit row group")
            origin = (index, column)
            origins[origin] = (cell, height, width, group_start, group_end)
            # This bounds the records to source rows, not the declared span.
            # Its original attribute is retained below, including any overhang.
            for target in range(index, min(index + height, group_end)):
                for slot in range(column, column + width):
                    if slot in occupied[target]:
                        raise ValueError("overlapping spans")
                    occupied[target][slot] = origin
            total += width * height
            column += width

    container = soup.new_tag("div", attrs=dict(table.attrs))
    used_ids: set[str] = set()
    for caption in table.find_all("caption", recursive=False):
        container.append(deepcopy(caption))
    summary = soup.new_tag("p")
    summary.string = (
        f"Originaltabelle {number}: {len(rows)} vorhandene Zeilen; Zellen mit Original-Spannweiten."
    )
    container.append(summary)
    for index, slots in enumerate(occupied):
        first = origins[slots[0]][0] if 0 in slots else None
        context = _label_text(first) if first is not None else ""
        if not slots:
            empty_row = soup.new_tag("p")
            empty_row.string = (
                f"Originaltabelle {number}; Originalzeile {index + 1}; keine Zellen im Original."
            )
            container.append(empty_row)
        for origin in dict.fromkeys(slots[column] for column in sorted(slots)):
            source_row, column = origin
            cell, _, width, group_start, group_end = origins[origin]
            label = (
                f"Originaltabelle {number}; Originalzeile {index + 1}; "
                f"Spalten {column + 1}–{column + width}; Ursprung Z{source_row + 1}/S{column + 1}; "
                f"{cell.name}; rowspan={cell.get('rowspan', '1')}; colspan={cell.get('colspan', '1')}; "
                f"Gruppe Z{group_start + 1}–Z{group_end}"
            )
            if index >= count:
                header_origins = dict.fromkeys(
                    occupied[header][slot]
                    for header in range(count)
                    for slot in range(column, column + width)
                    if slot in occupied[header]
                )
                headers = [_label_text(origins[key][0]) for key in header_origins]
                if headers:
                    label += "; " + " / ".join(headers)
            if context:
                label += "; Spalte 1: " + context
            record = soup.new_tag("table")
            heading_row, value_row = soup.new_tag("tr"), soup.new_tag("tr")
            heading = soup.new_tag("th")
            heading.string = label
            heading_row.append(heading)
            value_row.append(_copy_cell(cell, "td", used_ids))
            record.extend([heading_row, value_row])
            container.append(record)
    table.replace_with(container)


def _label_text(cell: Tag) -> str:
    """Keep real separators without inserting spaces between inline nodes."""
    copied = deepcopy(cell)
    for separator in copied.find_all(["br", "hr"]):
        separator.replace_with("\n")
    return " ".join(copied.get_text().split())


def _apply_td_grid(rows: list[Tag], grid: list[list[Tag]]) -> None:
    """Resolve explicit occupancy; full-width cells keep their original span."""
    used_ids: set[str] = set()
    for row, cells in zip(rows, grid, strict=True):
        # Same full-width policy as explicit TH tables: one source cell keeps
        # its coverage without multiplying a title, note or other row content.
        if len(cells) > 1 and all(cell is cells[0] for cell in cells):
            copied = _copy_cell(cells[0], "td", used_ids)
            copied["colspan"] = str(len(cells))
            row.clear()
            row.append(copied)
            continue
        copies = [_copy_cell(cell, "td", used_ids) for cell in cells]
        row.clear()
        for copied in copies:
            row.append(copied)


def _apply(soup: BeautifulSoup, rows: list[Tag], grid: list[list[Tag]], count: int) -> None:
    header = soup.new_tag("tr")
    used_ids: set[str] = set()
    for column in range(len(grid[0])):
        target = soup.new_tag("th")
        seen: set[int] = set()
        for row in grid[:count]:
            source = row[column]
            if id(source) in seen:
                continue
            seen.add(id(source))
            if target.contents:
                target.append(" / ")
            copied = _copy_cell(source, "th", used_ids)
            if copied.get("id"):
                target.append(soup.new_tag("span", id=copied["id"]))
            for node in list(copied.contents):
                target.append(node.extract())
        header.append(target)
    rows[0].insert_before(header)
    for row in rows[:count]:
        row.decompose()
    for row, cells in zip(rows[count:], grid[count:], strict=True):
        # Explicit full-width notes keep a single copy, including their links.
        if len(cells) > 1 and all(cell is cells[0] for cell in cells):
            copied = _copy_cell(cells[0], "td", used_ids)
            copied["colspan"] = str(len(cells))
            row.clear()
            row.append(copied)
            continue
        copies = [_copy_cell(cell, cell.name, used_ids) for cell in cells]
        row.clear()
        for copied in copies:
            row.append(copied)


def _layout_blocks(table: Tag) -> None:
    """Keep cell order, children and IDs without inventing column headings."""
    own_elements = [
        element
        for element in table.find_all(["caption", "thead", "tbody", "tfoot", "tr", "td", "th"])
        if element.find_parent("table") is table
    ]
    for element in own_elements:
        element.name = "div"
        element.attrs.pop("rowspan", None)
        element.attrs.pop("colspan", None)
    table.name = "div"


def _navigation_layout(table: Tag, rows: list[Tag], count: int) -> bool:
    if count > 1 or rows[0].parent.name == "thead":
        return False
    if not any(
        parent.name == "nav" or str(parent.get("role", "")).lower() == "navigation"
        for parent in [table, *table.parents]
    ):
        return False
    try:
        grid = _grid(rows, count)
    except ValueError:
        return False
    if len(grid[0]) < 2:
        return False
    if count:
        cells = _cells(rows[0])
        if len(cells) != 1 or not all(cell is cells[0] for cell in grid[0]):
            return False
    for row, expanded in zip(rows[count:], grid[count:], strict=True):
        cells = _cells(row)
        if any(str(cell.get("rowspan", "1")) != "1" for cell in cells):
            return False
        if len(cells) == 1 and cells[0].name == "td" and all(cell is cells[0] for cell in expanded):
            continue
        if (
            len(cells) == 2
            and cells[0].name == "th"
            and str(cells[0].get("scope", "")).lower() == "row"
            and cells[1].name == "td"
        ):
            continue
        return False
    return True


def normalize_tables(soup: BeautifulSoup) -> tuple[str, ...]:
    """Resolve explicit geometry; never send a rejected grid to the converter."""
    diagnostics: list[str] = []
    tables = soup.find_all("table")
    numbers = {id(table): index + 1 for index, table in enumerate(tables)}
    # Classify before replacing inner layout tables. Their outer container
    # remains a layout table even when the nested table has become blocks.
    nested_containers = {id(table) for table in tables if table.find("table") is not None}
    for table in reversed(tables):
        if str(table.get("role", "")).lower() in {"presentation", "none"}:
            _layout_blocks(table)
            diagnostics.append("table_layout_role_blocks")
            continue
        if id(table) in nested_containers:
            _layout_blocks(table)
            diagnostics.append("table_nested_layout_blocks")
            continue
        rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
        if not rows:
            diagnostics.append("table_unclear_structure_preserved")
            continue
        if any(
            row.parent.name not in {"table", "thead", "tbody", "tfoot"}
            or any(cell.find(["td", "th", "tr"]) is not None for cell in _cells(row))
            for row in rows
        ):
            diagnostics.append("table_unclear_structure_preserved")
            continue
        if len(rows) == 1:
            _layout_blocks(table)
            diagnostics.append("table_single_row_blocks")
            continue
        count = _header_count(rows)
        if _navigation_layout(table, rows, count):
            _layout_blocks(table)
            diagnostics.append("table_navigation_groups_blocks")
            continue
        spans = any(
            cell.has_attr("rowspan") or cell.has_attr("colspan")
            for row in rows
            for cell in _cells(row)
        )
        if not count:
            diagnostics.append("table_unclear_headers_preserved")
            if spans and all(cell.name == "td" for row in rows for cell in _cells(row)):
                try:
                    grid = _grid(rows, 0)
                except ValueError:
                    _cell_records(soup, table, rows, 0, numbers[id(table)])
                    diagnostics.append("table_explicit_cell_records")
                else:
                    _apply_td_grid(rows, grid)
                    diagnostics.append("table_td_grid_normalized")
            continue
        if count == 1 and not spans:
            continue
        try:
            grid = _grid(rows, count)
        except ValueError:
            _cell_records(soup, table, rows, count, numbers[id(table)])
            diagnostics.append("table_explicit_cell_records")
            continue
        _apply(soup, rows, grid, count)
        diagnostics.append("table_spans_normalized")
    return tuple(dict.fromkeys(diagnostics))

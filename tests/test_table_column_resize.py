"""Spaltenbreiten per Ziehgriff in SharedUi.table (ui_core/js/table.js).

Verhaltenstest in Node mit einer kleinen DOM-Attrappe (kein jsdom im Projekt):
Ziehen, Mindestbreite, Speichern, Wiederherstellen, Zuruecksetzen und Sortierschutz.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLE_JS = ROOT / "kiara/web/static/ui_core/js/table.js"
TABLE_CSS = ROOT / "kiara/web/static/ui_core/css/table.css"

HARNESS = r"""
const fs = require('fs');
const listeners = {};
const rules = new Map();
const store = new Map();
function classList() {
  const set = new Set();
  return { add: c => set.add(c), remove: c => set.delete(c), contains: c => set.has(c),
           toggle: (c, on) => (on ? set.add(c) : set.delete(c)), _set: set };
}
global.window = global;
global.CSS = { escape: s => s };
global.document = {
  addEventListener: (type, fn) => { (listeners[type] = listeners[type] || []).push(fn); },
  getElementById: () => null,
  body: { classList: classList() },
};
global.localStorage = {
  getItem: k => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: k => store.delete(k),
};
global.SharedUi = {
  configure() {},
  runtime: { escapeHtml: s => String(s), escapeAttr: s => String(s) },
  fetch: { safeFetch: async () => ({}) },
  runtimeStyle: {
    setRule: (sel, props) => rules.set(sel, props),
    removeRule: sel => rules.delete(sel),
    classForProps: () => 'u-rt-grid',
    applyStyle() {},
  },
};
eval(fs.readFileSync(process.argv[2], 'utf8'));

const config = { id: 'ds', showHeader: true, expandable: true, gridTemplate: '30px 1fr 100px', detailFields: [],
                 columns: [{ field: 'name', label: 'Name', sortable: true }, { field: 'n', label: 'N' }] };
let sorted = 0;
const t = new SharedUi.table.ExpandableTable(config, 'c');
t.sortBy = () => { sorted++; };
const header = t.renderHeader();

// DOM-Attrappe fuer Kopfzeile mit drei Spuren (Aufklappspalte + 2 Spalten).
const handlers = {};
const handle = {
  classList: classList(),
  getAttribute: a => ({ 'data-table-resize': '1', 'data-table-id': 'ds' })[a],
  closest(sel) {
    if (sel === '[data-table-resize]') return handle;
    if (sel === '.table-header') return headerEl;
    if (sel === '[data-table-action]') return sortButton;
    return null;
  },
  setPointerCapture() {},
  addEventListener: (type, fn) => { handlers[type] = fn; },
  removeEventListener: type => { delete handlers[type]; },
};
const sortButton = { getAttribute: a => ({ 'data-table-action': 'sortBy', 'data-table-id': 'ds',
                                            'data-table-column': 'name' })[a],
                     hasAttribute: () => false, classList: classList(), tagName: 'BUTTON' };
const cell = w => ({ classList: { contains: c => c === 'header-cell' }, getBoundingClientRect: () => ({ width: w }) });
const headerEl = { children: [cell(30), cell(200), cell(100)] };
const container = { querySelectorAll: () => [] };
document.getElementById = () => container;

const ev = (type, extra) => Object.assign({ type, target: handle, button: 0, clientX: 500, pointerId: 1,
                                            preventDefault() {}, stopPropagation() {} }, extra);
listeners.pointerdown.forEach(fn => fn(ev('pointerdown')));
handlers.pointermove({ clientX: 560 });
const afterGrow = JSON.parse(JSON.stringify(rules.get('.' + t._colClass + '.' + t._colClass)));
handlers.pointermove({ clientX: 100 });           // weit nach links: Mindestbreite
handlers.pointerup({});
listeners.click.forEach(fn => fn(ev('click')));    // Klick nach dem Ziehen darf nicht sortieren

const saved = JSON.parse(localStorage.getItem('sharedui.table.colwidths.v1:ds'));
const t2 = new SharedUi.table.ExpandableTable(config, 'c2');       // stellt Breiten wieder her
const restoredRow = t2.renderRow({ id: 1, name: 'x', n: 2 }, 0);
const changed = new SharedUi.table.ExpandableTable(
  Object.assign({}, config, { id: 'ds', columns: config.columns.concat([{ field: 'z', label: 'Z' }]) }), 'c3');
listeners.dblclick.forEach(fn => fn(ev('dblclick')));
console.log(JSON.stringify({
  header, afterGrow, saved, sorted, bodyClass: document.body.classList._set.size,
  restored: t2._colWidths, restoredRowHasClass: restoredRow.includes(t2._colClass),
  changedIgnored: changed._colWidths === null,
  afterReset: localStorage.getItem('sharedui.table.colwidths.v1:ds'),
  rulesLeft: [...rules.keys()].filter(k => k.includes(changed._colClass)).length,
}));
"""


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("node nicht installiert")
    harness = tmp_path_factory.mktemp("table") / "harness.js"
    harness.write_text(HARNESS, encoding="utf-8")
    out = subprocess.run([node, str(harness), str(TABLE_JS)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_every_header_cell_has_a_resize_handle_except_expand_column(result):
    header = result["header"]
    assert header.count('class="col-resize-handle"') == 2
    assert 'data-table-resize="1"' in header and 'data-table-resize="2"' in header
    assert 'data-table-resize="0"' not in header


def test_drag_sets_pixel_tracks_with_minimum_and_persists(result):
    assert result["afterGrow"]["gridTemplateColumns"] == "30px 260px 100px minmax(0, 1fr)"
    assert result["saved"] == [30, 40, 100]
    assert result["bodyClass"] == 0


def test_click_after_drag_does_not_sort(result):
    assert result["sorted"] == 0


def test_widths_restore_and_reset(result):
    assert result["restored"] == [30, 40, 100]
    assert result["restoredRowHasClass"]
    assert result["changedIgnored"]
    assert result["afterReset"] is None


def test_css_and_csp():
    css = TABLE_CSS.read_text(encoding="utf-8")
    js = TABLE_JS.read_text(encoding="utf-8")
    assert ".col-resize-handle" in css and "cursor: col-resize" in css
    # CSP: keine Inline-Styles, Breiten nur ueber runtimeStyle-Regeln.
    assert "style=" not in js and ".style." not in js

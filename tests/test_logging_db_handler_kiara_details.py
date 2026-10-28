"""Tests fuer den `kiara_details`-Merge-Pfad in `SystemEventDBHandler.emit`.

Strukturierte Multi-Line-Felder gelangen ueber `extra={"kiara_details": {...}}`
in den `details`-JSONB der `system_events`-Tabelle. Tests pruefen:
- Werte werden gemerged.
- `None`-Werte werden gefiltert.
- Existierende Keys aus der KV-Message werden NICHT ueberschrieben.
- Nicht-Dict-Wert wird ignoriert (kein Crash).

Mock-Strategie: Die DB-Schreibung wird via `db_transaction` abgefangen, damit
wir den finalen `details`-dict beobachten koennen ohne echte DB.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from kiara.web.logging_db_handler import SystemEventDBHandler


@pytest.fixture
def captured_insert():
    """Liefert (handler, captures) — captures sammelt alle DB-INSERT-Argumente.

    `db_transaction()` wird durch einen Kontext-Manager ersetzt, dessen
    `conn.execute(sql, params)` die Argumente in `captures` ablegt.
    """
    captures: list[tuple[str, tuple]] = []

    fake_conn = MagicMock()

    def _execute(sql, params):
        captures.append((sql, params))
        return MagicMock()

    fake_conn.execute = _execute

    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_conn
    fake_ctx.__exit__.return_value = False

    handler = SystemEventDBHandler()
    # min_level-Cache vorbelegen, damit der lazy load_config-Pfad nicht greift.
    handler._min_level_cache = logging.INFO

    with patch("kiara.shared.db.db_transaction", return_value=fake_ctx):
        yield handler, captures


def _make_record(msg, *args, level=logging.ERROR, extra=None):
    """LogRecord mit optionalen extra-Attributen, wie logging.* sie produziert."""
    record = logging.LogRecord(
        name="kiara.security",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )
    if extra:
        for key, value in extra.items():
            setattr(record, key, value)
    return record


def _details_from(captures):
    """Holt den letzten gespeicherten `details`-Json-Wert aus den captures."""
    sql, params = captures[-1]
    details_arg = params[7]
    # psycopg2.extras.Json hat .adapted, sonst direkt der dict.
    return getattr(details_arg, "adapted", details_arg)


# ---------------------------------------------------------------------------
# Extra-Merge
# ---------------------------------------------------------------------------


def test_kiara_details_dict_keys_are_merged_into_details(captured_insert):
    handler, captures = captured_insert
    record = _make_record(
        "WORKER_DIED service=kiara.service pid=42 reason=unknown",
        extra={"kiara_details": {"foo": "bar", "memory_current": "12345"}},
    )
    handler.emit(record)
    details = _details_from(captures)
    assert details["foo"] == "bar"
    assert details["memory_current"] == "12345"


def test_none_values_in_kiara_details_are_filtered_out(captured_insert):
    handler, captures = captured_insert
    record = _make_record(
        "WORKER_DIED service=kiara.service pid=42 reason=unknown",
        extra={"kiara_details": {"foo": "bar", "baz": None, "qux": None}},
    )
    handler.emit(record)
    details = _details_from(captures)
    assert details["foo"] == "bar"
    assert "baz" not in details
    assert "qux" not in details


def test_empty_kiara_details_dict_does_not_break_merge(captured_insert):
    handler, captures = captured_insert
    record = _make_record(
        "EVENT_FOO bar=1",
        extra={"kiara_details": {}},
    )
    handler.emit(record)
    # Kein Crash, details enthaelt mindestens die KV-Felder ausser user/ip/reason
    details = _details_from(captures)
    assert details.get("bar") == "1"


# ---------------------------------------------------------------------------
# Setdefault-Semantik (Message-Keys gewinnen)
# ---------------------------------------------------------------------------


def test_message_kv_wins_over_kiara_details_for_same_key(captured_insert):
    """Wenn ein Key in der Message KV existiert, ueberschreibt kiara_details ihn nicht."""
    handler, captures = captured_insert
    record = _make_record(
        "EVENT_FOO existing=fromMessage another=keepMe",
        extra={"kiara_details": {"existing": "fromExtra", "added": "x"}},
    )
    handler.emit(record)
    details = _details_from(captures)
    # Message-KV gewinnt (setdefault-Semantik)
    assert details["existing"] == "fromMessage"
    # Anderer KV-Wert intakt
    assert details["another"] == "keepMe"
    # Nicht-konfligierender extra-Wert wird gemerged
    assert details["added"] == "x"


# ---------------------------------------------------------------------------
# Nicht-Dict-Wert
# ---------------------------------------------------------------------------


def test_non_dict_kiara_details_is_ignored_silently(captured_insert):
    handler, captures = captured_insert
    record = _make_record(
        "EVENT_FOO bar=1",
        extra={"kiara_details": "not-a-dict"},
    )
    # Kein Crash erwartet
    handler.emit(record)
    details = _details_from(captures)
    # Andere KV bleibt erhalten
    assert details.get("bar") == "1"
    # `not-a-dict` taucht nicht im details auf
    assert "kiara_details" not in details


def test_list_kiara_details_is_ignored_silently(captured_insert):
    handler, captures = captured_insert
    record = _make_record(
        "EVENT_FOO",
        extra={"kiara_details": ["a", "b"]},
    )
    handler.emit(record)
    # Kein Crash; details enthaelt nichts aus dem List-Object.
    details = _details_from(captures)
    assert "0" not in details
    assert "1" not in details


def test_none_kiara_details_is_ignored_silently(captured_insert):
    handler, captures = captured_insert
    record = _make_record(
        "EVENT_FOO",
        extra={"kiara_details": None},
    )
    # Kein Crash, kein Merge
    handler.emit(record)
    details = _details_from(captures)
    assert details == {} or "raw" in details  # je nach Parser-Output OK


# ---------------------------------------------------------------------------
# Integration mit Multi-Line-Werten (Realfall WORKER_DIED)
# ---------------------------------------------------------------------------


def test_multiline_journal_tail_in_kiara_details_lands_unmodified(captured_insert):
    """Multi-Line-Werte (z.B. journal_tail) duerfen ungeschnitten ins JSONB."""
    handler, captures = captured_insert
    multiline = "line1\nline2 with spaces\nline3=foo bar"
    record = _make_record(
        "WORKER_DIED service=kiara.service pid=42 reason=unknown",
        extra={
            "kiara_details": {"journal_tail": multiline, "dmesg_tail": "kernel: foo\nkernel: bar"}
        },
    )
    handler.emit(record)
    details = _details_from(captures)
    assert details["journal_tail"] == multiline
    assert details["dmesg_tail"] == "kernel: foo\nkernel: bar"

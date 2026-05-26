"""Tests fuer Multi-Select Datenquellen-Filter mit NOT-Toggle.

Testet _build_datasource_filter() mit Include-/Exclude-Logik,
QueryRequest-Felder und Permission-Sicherheit.
"""

# ===========================================================================
# Fixtures
# ===========================================================================
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from kiara.web.api_chat import QueryRequest, _build_datasource_filter


@contextmanager
def _mock_db_query(active_ids=None, public_ids=None):
    """Patcht ds_repo und perm_repo fuer _build_datasource_filter.

    Args:
        active_ids:  Alle aktiven DS-IDs (fuer Admin-Exclude-Query).
        public_ids:  IDs ohne Gruppenzuordnung (fuer Normal-User).
    """
    if active_ids is None:
        active_ids = [1, 2, 3, 4, 5]
    if public_ids is None:
        public_ids = [1, 2]

    with (
        patch("kiara.web.api_chat.ds_repo") as mock_ds,
        patch("kiara.web.api_chat.perm_repo") as mock_perm,
    ):
        mock_ds.list_filtered.return_value = (
            [{"id": i} for i in active_ids],
            len(active_ids),
        )
        mock_perm.get_public_datasource_ids.return_value = public_ids
        yield


# ===========================================================================
# QueryRequest — Neue Felder
# ===========================================================================


class TestQueryRequestMultiSelect:
    """Tests fuer die neuen Multi-Select-Felder im QueryRequest."""

    def test_default_filter_datasource_ids_none(self):
        req = QueryRequest(question="Test")
        assert req.filter_datasource_ids is None

    def test_default_filter_datasource_exclude_false(self):
        req = QueryRequest(question="Test")
        assert req.filter_datasource_exclude is False

    def test_filter_datasource_ids_list(self):
        req = QueryRequest(question="Test", filter_datasource_ids=[1, 3, 5])
        assert req.filter_datasource_ids == [1, 3, 5]

    def test_filter_datasource_exclude_true(self):
        req = QueryRequest(
            question="Test", filter_datasource_ids=[2], filter_datasource_exclude=True
        )
        assert req.filter_datasource_exclude is True
        assert req.filter_datasource_ids == [2]

    def test_filter_datasource_ids_empty_list(self):
        req = QueryRequest(question="Test", filter_datasource_ids=[])
        assert req.filter_datasource_ids == []


# ===========================================================================
# Admin — Include-Modus
# ===========================================================================


class TestBuildFilterAdminInclude:
    """Admin-User, Include-Modus (exclude=False)."""

    def test_no_selection_returns_none(self):
        """Admin ohne Auswahl → None (= alle DS)."""
        user = {"is_admin": True}
        result = _build_datasource_filter(user, None)
        assert result is None

    def test_empty_list_returns_none(self):
        """Admin mit leerer Liste → None (= alle DS)."""
        user = {"is_admin": True}
        result = _build_datasource_filter(user, [])
        assert result is None

    def test_single_selection(self):
        """Admin mit einer DS-ID → [id]."""
        user = {"is_admin": True}
        result = _build_datasource_filter(user, [3])
        assert result == [3]

    def test_multi_selection(self):
        """Admin mit mehreren DS-IDs → Liste zurueck."""
        user = {"is_admin": True}
        result = _build_datasource_filter(user, [1, 4, 5])
        assert result == [1, 4, 5]


# ===========================================================================
# Admin — Exclude-Modus
# ===========================================================================


class TestBuildFilterAdminExclude:
    """Admin-User, Exclude-Modus (exclude=True)."""

    def test_exclude_single(self):
        """Admin schliesst eine DS aus → alle aktiven minus diese."""
        user = {"is_admin": True}
        with _mock_db_query(active_ids=[1, 2, 3, 4, 5]):
            result = _build_datasource_filter(user, [3], exclude=True)
        assert sorted(result) == [1, 2, 4, 5]

    def test_exclude_multiple(self):
        """Admin schliesst mehrere DS aus."""
        user = {"is_admin": True}
        with _mock_db_query(active_ids=[1, 2, 3, 4, 5]):
            result = _build_datasource_filter(user, [1, 5], exclude=True)
        assert sorted(result) == [2, 3, 4]

    def test_exclude_all_returns_empty(self):
        """Admin schliesst ALLE aktiven DS aus → leere Liste (0 Treffer)."""
        user = {"is_admin": True}
        with _mock_db_query(active_ids=[1, 2]):
            result = _build_datasource_filter(user, [1, 2], exclude=True)
        assert result == []

    def test_exclude_none_returns_none(self):
        """Admin mit Exclude aber keine Auswahl → None (= alle)."""
        user = {"is_admin": True}
        result = _build_datasource_filter(user, None, exclude=True)
        assert result is None

    def test_exclude_empty_list_returns_none(self):
        """Admin mit Exclude + leere Liste → None (= alle)."""
        user = {"is_admin": True}
        result = _build_datasource_filter(user, [], exclude=True)
        assert result is None


# ===========================================================================
# Normal-User — Include-Modus
# ===========================================================================


class TestBuildFilterUserInclude:
    """Nicht-Admin-User, Include-Modus."""

    def test_no_selection_returns_allowed(self):
        """User ohne Auswahl → erlaubte IDs (Gruppen + oeffentliche)."""
        user = {"is_admin": False, "datasource_ids": [3, 4]}
        with _mock_db_query(public_ids=[1, 2]):
            result = _build_datasource_filter(user, None)
        assert sorted(result) == [1, 2, 3, 4]

    def test_select_allowed_id(self):
        """User waehlt erlaubte DS → nur diese."""
        user = {"is_admin": False, "datasource_ids": [3, 4]}
        with _mock_db_query(public_ids=[1, 2]):
            result = _build_datasource_filter(user, [3])
        assert result == [3]

    def test_select_multiple_allowed(self):
        """User waehlt mehrere erlaubte DS."""
        user = {"is_admin": False, "datasource_ids": [3, 4]}
        with _mock_db_query(public_ids=[1, 2]):
            result = _build_datasource_filter(user, [1, 4])
        assert sorted(result) == [1, 4]

    def test_select_forbidden_id_returns_empty(self):
        """User waehlt nicht-erlaubte DS → leere Liste (Sicherheit!)."""
        user = {"is_admin": False, "datasource_ids": [3]}
        with _mock_db_query(public_ids=[1]):
            result = _build_datasource_filter(user, [99])
        assert result == []

    def test_select_mix_allowed_forbidden(self):
        """User waehlt Mix aus erlaubten und verbotenen → nur erlaubte."""
        user = {"is_admin": False, "datasource_ids": [3]}
        with _mock_db_query(public_ids=[1]):
            result = _build_datasource_filter(user, [1, 99, 3])
        assert sorted(result) == [1, 3]

    def test_no_datasource_ids_key(self):
        """User-Dict ohne datasource_ids → nur oeffentliche."""
        user = {"is_admin": False}
        with _mock_db_query(public_ids=[1, 2]):
            result = _build_datasource_filter(user, None)
        assert sorted(result) == [1, 2]


# ===========================================================================
# Normal-User — Exclude-Modus
# ===========================================================================


class TestBuildFilterUserExclude:
    """Nicht-Admin-User, Exclude-Modus."""

    def test_exclude_one(self):
        """User schliesst eine erlaubte DS aus."""
        user = {"is_admin": False, "datasource_ids": [3, 4]}
        with _mock_db_query(public_ids=[1, 2]):
            result = _build_datasource_filter(user, [3], exclude=True)
        # Erlaubt: {1,2,3,4} minus {3} = {1,2,4}
        assert sorted(result) == [1, 2, 4]

    def test_exclude_multiple(self):
        """User schliesst mehrere erlaubte DS aus."""
        user = {"is_admin": False, "datasource_ids": [3, 4]}
        with _mock_db_query(public_ids=[1, 2]):
            result = _build_datasource_filter(user, [1, 3], exclude=True)
        assert sorted(result) == [2, 4]

    def test_exclude_all_returns_empty(self):
        """User schliesst alle erlaubten DS aus → leere Liste."""
        user = {"is_admin": False, "datasource_ids": [3]}
        with _mock_db_query(public_ids=[1]):
            result = _build_datasource_filter(user, [1, 3], exclude=True)
        assert result == []

    def test_exclude_forbidden_id_noop(self):
        """User schliesst nicht-erlaubte DS aus → kein Effekt, alle erlaubten bleiben."""
        user = {"is_admin": False, "datasource_ids": [3]}
        with _mock_db_query(public_ids=[1]):
            result = _build_datasource_filter(user, [99], exclude=True)
        assert sorted(result) == [1, 3]

    def test_exclude_no_selection_returns_allowed(self):
        """User mit Exclude aber keine Auswahl → erlaubte IDs."""
        user = {"is_admin": False, "datasource_ids": [3, 4]}
        with _mock_db_query(public_ids=[1]):
            result = _build_datasource_filter(user, None, exclude=True)
        assert sorted(result) == [1, 3, 4]


# ===========================================================================
# Sicherheits-Tests — Rechte-Eskalation verhindern
# ===========================================================================


class TestFilterSecurity:
    """Sicherstellen, dass keine Rechte-Eskalation moeglich ist."""

    def test_user_cannot_access_forbidden_via_include(self):
        """Include einer verbotenen ID liefert leere Liste, nicht die ID."""
        user = {"is_admin": False, "datasource_ids": []}
        with _mock_db_query(public_ids=[1]):
            result = _build_datasource_filter(user, [99])
        assert 99 not in (result or [])

    def test_user_cannot_access_forbidden_via_exclude(self):
        """Exclude einer verbotenen ID erweitert nicht die erlaubten IDs."""
        user = {"is_admin": False, "datasource_ids": [3]}
        with _mock_db_query(public_ids=[1]):
            result = _build_datasource_filter(user, [99], exclude=True)
        # Erlaubt: {1,3}; Exclude {99} hat keinen Effekt
        assert sorted(result) == [1, 3]
        assert 99 not in result

    def test_user_exclude_never_returns_none(self):
        """Normal-User bekommt nie None (= uneingeschraenkt) zurueck,
        auch nicht bei Exclude mit leerer Auswahl."""
        user = {"is_admin": False, "datasource_ids": [3]}
        with _mock_db_query(public_ids=[1]):
            result = _build_datasource_filter(user, None, exclude=True)
        assert result is not None
        assert isinstance(result, list)

    def test_user_include_never_returns_none(self):
        """Normal-User bekommt nie None bei Include."""
        user = {"is_admin": False, "datasource_ids": [3]}
        with _mock_db_query(public_ids=[1]):
            result = _build_datasource_filter(user, None)
        assert result is not None
        assert isinstance(result, list)


# ===========================================================================
# _build_restriction_filter — Cloud-Policy-Filter
# ===========================================================================


class TestBuildRestrictionFilter:
    """Tests fuer _build_restriction_filter() aus api_chat."""

    def test_admin_no_filter_returns_explicit_ids(self):
        """Admin ohne Filter liefert explizite aktive DS-IDs (nicht None)."""
        from kiara.web.api_chat import _build_restriction_filter

        user = {"is_admin": True}
        with patch("kiara.web.api_chat.ds_repo") as mock_ds:
            mock_ds.get_active.return_value = [{"id": 1}, {"id": 2}, {"id": 3}]
            result = _build_restriction_filter(user, None)
        assert result is not None
        assert sorted(result) == [1, 2, 3]

    def test_include_passthrough(self):
        """Include-Modus reicht IDs durch."""
        from kiara.web.api_chat import _build_restriction_filter

        user = {"is_admin": True}
        result = _build_restriction_filter(user, [3, 5])
        assert result == [3, 5]

    def test_exclude_passthrough(self):
        """Exclude-Modus berechnet Komplement."""
        from kiara.web.api_chat import _build_restriction_filter

        user = {"is_admin": True}
        with _mock_db_query(active_ids=[1, 2, 3, 4, 5]):
            result = _build_restriction_filter(user, [2], exclude=True)
        assert sorted(result) == [1, 3, 4, 5]

    def test_db_error_raises_runtime_error(self):
        """DB-Fehler bei Materialisierung wirft RuntimeError."""
        from kiara.web.api_chat import _build_restriction_filter

        user = {"is_admin": True}
        with patch("kiara.web.api_chat.ds_repo") as mock_ds:
            mock_ds.get_active.side_effect = Exception("DB kaputt")
            with pytest.raises(RuntimeError, match="Restriction-Filter"):
                _build_restriction_filter(user, None)

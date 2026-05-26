"""Tests fuer HashStore: PostgreSQL-basierte Hash-Persistenz.

Testet:
- CRUD-Operationen (load, upsert_many, remove_files, delete_datasource)
- Normalisierung (datasource_id=None → 0, model-Tag-Stripping)
- Batching bei >500 Dateien in remove_files

Alle Tests verwenden eindeutige datasource_ids (9100+) und raeumen
am Anfang ihre Daten auf, um Isolation in der geteilten PG-DB zu gewaehrleisten.
"""

import pytest

from kiara.ingestion._file_utils import HashStore

# Eindeutige DS-IDs fuer dieses Test-Modul (vermeidet Konflikte mit anderen Tests)
_DS_BASE = 9100


@pytest.fixture
def store():
    """HashStore-Instanz mit aufgeraeumten Test-IDs."""
    hs = HashStore()
    # Daten der verwendeten DS-IDs aufraeumen
    for ds_id in range(_DS_BASE, _DS_BASE + 20):
        hs.delete_datasource(ds_id)
    # Legacy DS 0 auch aufraeumen (fuer Normalisierungs-Tests)
    hs.delete_datasource(0)
    return hs


# Alias-Fixtures fuer neue Tests
@pytest.fixture
def hash_store_pg(store):
    """Alias fuer store — neue Tests verwenden diesen Namen."""
    return store


@pytest.fixture
def ds_id():
    """Feste DS-ID fuer neue Validierungs-Tests."""
    return _DS_BASE + 15


# ---------------------------------------------------------------------------
# Basis-Operationen
# ---------------------------------------------------------------------------


class TestHashStoreBasics:
    def test_load_empty(self, store):
        """Leere DB liefert leeres Dict."""
        result = store.load(_DS_BASE, "nomic-embed-text")
        assert result == {}

    def test_upsert_and_load(self, store):
        """Roundtrip: upsert_many → load gibt gleiche Daten zurueck."""
        ds_id = _DS_BASE + 1
        hashes = {"/data/file1.txt": ("abc123", 1024), "/data/file2.txt": ("def456", 2048)}
        store.upsert_many(ds_id, "nomic-embed-text", hashes)

        loaded = store.load(ds_id, "nomic-embed-text")
        assert loaded == {"/data/file1.txt": "abc123", "/data/file2.txt": "def456"}

    def test_upsert_many_overwrite(self, store):
        """Existierende Eintraege werden beim erneuten upsert aktualisiert."""
        ds_id = _DS_BASE + 2
        store.upsert_many(ds_id, "nomic", {"/data/f1.txt": ("old_hash", 100)})
        store.upsert_many(ds_id, "nomic", {"/data/f1.txt": ("new_hash", 200)})

        loaded = store.load(ds_id, "nomic")
        assert loaded["/data/f1.txt"] == "new_hash"

    def test_upsert_many_empty(self, store):
        """Leeres Dict wird ohne Fehler akzeptiert."""
        ds_id = _DS_BASE + 3
        store.upsert_many(ds_id, "nomic", {})
        assert store.load(ds_id, "nomic") == {}

    def test_remove_files(self, store):
        """Selektives Loeschen einzelner Dateipfade."""
        ds_id = _DS_BASE + 4
        hashes = {
            "/data/a.txt": ("h1", 100),
            "/data/b.txt": ("h2", 200),
            "/data/c.txt": ("h3", 300),
        }
        store.upsert_many(ds_id, "nomic", hashes)
        store.remove_files(ds_id, "nomic", ["/data/b.txt"])

        loaded = store.load(ds_id, "nomic")
        assert "/data/b.txt" not in loaded
        assert len(loaded) == 2

    def test_remove_files_empty(self, store):
        """Leere Liste wird ohne Fehler akzeptiert."""
        ds_id = _DS_BASE + 5
        store.upsert_many(ds_id, "nomic", {"/data/a.txt": ("h1", 100)})
        store.remove_files(ds_id, "nomic", [])
        assert len(store.load(ds_id, "nomic")) == 1


# ---------------------------------------------------------------------------
# delete_datasource
# ---------------------------------------------------------------------------


class TestDeleteDatasource:
    def test_delete_datasource_all_models(self, store):
        """delete_datasource ohne model loescht alle Modelle einer DS."""
        ds1 = _DS_BASE + 6
        ds2 = _DS_BASE + 7
        store.upsert_many(ds1, "nomic", {"/data/f1.txt": ("h1", 100)})
        store.upsert_many(ds1, "mxbai", {"/data/f2.txt": ("h2", 200)})
        store.upsert_many(ds2, "nomic", {"/data/f3.txt": ("h3", 300)})

        store.delete_datasource(ds1)

        assert store.load(ds1, "nomic") == {}
        assert store.load(ds1, "mxbai") == {}
        # DS 2 unberuehrt
        assert store.load(ds2, "nomic") == {"/data/f3.txt": "h3"}

    def test_delete_datasource_single_model(self, store):
        """delete_datasource mit model loescht nur dieses Modell."""
        ds_id = _DS_BASE + 8
        store.upsert_many(ds_id, "nomic", {"/data/f1.txt": ("h1", 100)})
        store.upsert_many(ds_id, "mxbai", {"/data/f2.txt": ("h2", 200)})

        store.delete_datasource(ds_id, embedding_model="nomic")

        assert store.load(ds_id, "nomic") == {}
        assert store.load(ds_id, "mxbai") == {"/data/f2.txt": "h2"}


# ---------------------------------------------------------------------------
# Normalisierung
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_datasource_id_none(self, store):
        """datasource_id=None wird auf 0 gemappt (Legacy-CLI-Pfad)."""
        store.upsert_many(None, "nomic", {"/data/f1.txt": ("h1", 100)})

        # Load mit None
        loaded = store.load(None, "nomic")
        assert loaded == {"/data/f1.txt": "h1"}

        # Load mit 0 — gleiche Daten
        loaded_zero = store.load(0, "nomic")
        assert loaded_zero == {"/data/f1.txt": "h1"}

    def test_model_passthrough(self, store):
        """HashStore reicht bereits normalisierte Modellnamen 1:1 durch."""
        ds1 = _DS_BASE + 9
        ds2 = _DS_BASE + 10
        ds3 = _DS_BASE + 11
        # Leerer String (normalisiertes None)
        store.upsert_many(ds1, "", {"/data/f1.txt": ("h1", 100)})
        assert store.load(ds1, "") == {"/data/f1.txt": "h1"}

        # Bereits normalisiert (ohne :latest)
        store.upsert_many(ds2, "nomic", {"/data/f2.txt": ("h2", 200)})
        assert store.load(ds2, "nomic") == {"/data/f2.txt": "h2"}

        # / bleibt erhalten (normalisiert, aber Sonderzeichen bleiben)
        store.upsert_many(ds3, "org/model", {"/data/f3.txt": ("h3", 300)})
        assert store.load(ds3, "org/model") == {"/data/f3.txt": "h3"}

    def test_isolation_between_datasources(self, store):
        """Verschiedene Datasources sind vollstaendig isoliert."""
        ds1 = _DS_BASE + 12
        ds2 = _DS_BASE + 13
        store.upsert_many(ds1, "nomic", {"/data/f1.txt": ("h1", 100)})
        store.upsert_many(ds2, "nomic", {"/data/f1.txt": ("h2", 200)})

        assert store.load(ds1, "nomic") == {"/data/f1.txt": "h1"}
        assert store.load(ds2, "nomic") == {"/data/f1.txt": "h2"}


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatching:
    def test_remove_files_large_batch(self, store):
        """>500 Dateien werden korrekt in Batches geloescht."""
        ds_id = _DS_BASE + 19
        # 600 Dateien anlegen
        hashes = {f"/data/file_{i:04d}.txt": (f"h{i}", 1000 + i) for i in range(600)}
        store.upsert_many(ds_id, "nomic", hashes)

        # Alle 600 loeschen
        paths = list(hashes.keys())
        store.remove_files(ds_id, "nomic", paths)

        loaded = store.load(ds_id, "nomic")
        assert loaded == {}


# ---------------------------------------------------------------------------
# Validierung: Typ-/Wert-Pruefungen in upsert_many
# ---------------------------------------------------------------------------


def test_upsert_many_legacy_string_raises(hash_store_pg, ds_id):
    """Legacy dict[str, str] Input wirft TypeError."""
    with pytest.raises(TypeError, match="erwartet.*hash.*file_size"):
        hash_store_pg.upsert_many(ds_id, "nomic", {"/a.txt": "h1"})


def test_upsert_many_none_size_raises(hash_store_pg, ds_id):
    """file_size=None wirft TypeError."""
    with pytest.raises(TypeError, match="None"):
        hash_store_pg.upsert_many(ds_id, "nomic", {"/a.txt": ("h1", None)})


def test_upsert_many_negative_size_raises(hash_store_pg, ds_id):
    """Negative file_size wirft ValueError."""
    with pytest.raises(ValueError, match="negativ"):
        hash_store_pg.upsert_many(ds_id, "nomic", {"/a.txt": ("h1", -1)})


def test_get_missing_size_paths(hash_store_pg, ds_id):
    """get_missing_size_paths liefert Pfade mit file_size IS NULL."""
    store = hash_store_pg
    store.upsert_many(ds_id, "nomic", {"/a.txt": ("h1", 100)})
    # Manuell eine Zeile ohne file_size einfuegen
    from kiara.shared.db import db_transaction

    with db_transaction() as conn:
        conn.execute(
            "INSERT INTO file_hashes (datasource_id, embedding_model, file_path, file_hash) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (datasource_id, embedding_model, file_path) DO NOTHING",
            (ds_id, "nomic", "/b.txt", "h2"),
        )
    missing = store.get_missing_size_paths(ds_id)
    assert "/b.txt" in missing
    assert "/a.txt" not in missing


def test_cross_model_file_size_sync(hash_store_pg, ds_id):
    """Write in Modell A synchronisiert file_size fuer dieselbe Datei in Modell B."""
    store = hash_store_pg
    # Modell A
    store.upsert_many(ds_id, "model_a", {"/doc.txt": ("ha", 500)})
    # Modell B — selbe Datei, anderer Hash
    store.upsert_many(ds_id, "model_b", {"/doc.txt": ("hb", 600)})
    # file_size muss fuer beide Modelle 600 sein (letzter Write gewinnt)
    from kiara.shared.db import db_query

    with db_query() as db:
        rows = db.execute(
            "SELECT embedding_model, file_size FROM file_hashes "
            "WHERE datasource_id = %s AND file_path = %s ORDER BY embedding_model",
            (ds_id, "/doc.txt"),
        ).fetchall()
    sizes = {r["embedding_model"]: r["file_size"] for r in rows}
    assert sizes["model_a"] == 600
    assert sizes["model_b"] == 600

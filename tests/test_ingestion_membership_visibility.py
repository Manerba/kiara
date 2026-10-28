"""SoT reconciliation and ES visibility with stateful backend failures."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kiara_ingestion_core.index_state import ChunkDatasources
from test_elasticsearch_datasource_lifecycle import targets

from kiara.backend_selector import BackendUnavailableError
from kiara.ingestion.runtime import _rollback_embedding_persistence, _update_external_ds_ids
from kiara.retrieval import elasticsearch_service as es_svc
from kiara.retrieval.elasticsearch_indices import document_address
from kiara.retrieval.vectorstore_base import VectorStoreBase


class MemoryES:
    def __init__(self):
        self.docs = {}
        self.visible = {}
        self.refreshes = 0
        self.updates = 0
        self.fail_update = False
        self.partial_failure = False
        self.indices = SimpleNamespace(exists=lambda **kw: True, refresh=self.refresh)

    def refresh(self, **kwargs):
        self.refreshes += 1
        self.visible = copy.deepcopy(self.docs)
        return {"_shards": {"failed": 0}}

    def count(self, **kwargs):
        return {"count": len(self.visible), "_shards": {"failed": 0}}

    def mget(self, *, ids, **kwargs):
        return {
            "docs": [
                {
                    "_id": cid,
                    "found": cid in self.docs,
                    "_source": copy.deepcopy(self.docs.get(cid, {})),
                    "_seq_no": 7,
                    "_primary_term": 1,
                }
                for cid in ids
            ]
        }

    def bulk(self, client, actions, **kwargs):
        success, errors = 0, []
        for action in actions:
            cid = action["_id"]
            if action.get("_op_type") == "update":
                if self.fail_update:
                    errors.append({"update": {"status": 409}})
                    continue
                assert action["if_seq_no"] == 7
                self.updates += 1
                self.docs[cid]["metadata"].update(copy.deepcopy(action["doc"]["metadata"]))
            elif action.get("_op_type") == "delete":
                self.docs.pop(cid, None)
            else:
                self.docs[cid] = copy.deepcopy(action["_source"])
            success += 1
            if self.partial_failure:
                errors.append({"index": {"status": 500}})
                break
        return success, errors


class MemoryStore:
    get_ds_ids = VectorStoreBase.get_ds_ids

    def __init__(self, mapping):
        self.mapping = copy.deepcopy(mapping)
        self.writes = []
        self.after_update = None

    def get_documents_by_ids(self, ids, **kwargs):
        found = [cid for cid in ids if cid in self.mapping]
        return {"ids": found, "metadatas": [{"ds_ids": self.mapping[cid]} for cid in found]}

    def update_ds_ids(self, mapping, *, wait=False):
        assert wait
        self.writes.append(copy.deepcopy(mapping))
        self.mapping.update(copy.deepcopy(mapping))
        if self.after_update:
            self.after_update()

    def delete_by_ids(self, ids):
        for cid in ids:
            self.mapping.pop(cid, None)


@pytest.fixture
def backend(monkeypatch):
    from test_elasticsearch_datasource_lifecycle import Backend

    client = Backend()
    monkeypatch.setattr(es_svc, "_get_client", lambda config: client)
    monkeypatch.setattr("elasticsearch.helpers.bulk", client.bulk)
    monkeypatch.setattr("kiara.config.get_embedding_model", lambda config: "p50")
    return client


def context(backend, memberships, actual):
    backend.docs = {
        cid: {"metadata": {"ds_ids": copy.deepcopy(ids)}} for cid, ids in actual.items()
    }
    return SimpleNamespace(
        config={"retrieval": {"hybrid_search": "elasticsearch"}},
        datasource_id=1,
        embedding_model="p50",
        stats={},
        chunk_ds=SimpleNamespace(
            get_datasource_ids_for_chunks=lambda *args: copy.deepcopy(memberships)
        ),
        store=MemoryStore(actual),
    )


def topology_context(backend, memberships, actual):
    ictx = SimpleNamespace(
        config={"retrieval": {"hybrid_search": "elasticsearch"}},
        datasource_id=1,
        embedding_model="p50",
        stats={},
        chunk_ds=SimpleNamespace(
            get_datasource_ids_for_chunks=lambda *args: copy.deepcopy(memberships)
        ),
        store=MemoryStore(actual),
    )
    for cid, owners in actual.items():
        for ds in set(owners) | set(memberships.get(cid, [])):
            name = f"kiara_ds_{ds}"
            backend.create(index=name)
            backend.docs[name][cid] = {"metadata": {"ds_ids": copy.deepcopy(owners)}}
    return ictx


def test_new_and_shared_chunks_skip_only_verified_payloads(backend):
    memberships = {"new": [1], "shared": [1, 2], "same": [2, 1]}
    ictx = topology_context(backend, memberships, {"new": [1], "shared": [1], "same": [1, 2]})
    with es_svc.ingestion_visibility(ictx.config):
        _update_external_ds_ids(ictx, list(memberships))
        assert backend.refreshes == []
        assert not any(backend.visible.values())
    assert ictx.store.writes == [{"shared": [1, 2]}]
    assert sum(a.get("_op_type") == "update" for a in backend.actions) == 2
    assert len(backend.refreshes) == 1
    assert backend.visible["kiara_ds_1"]["shared"]["metadata"]["ds_ids"] == [1, 2]


def test_sot_change_during_update_is_reconciled(backend):
    memberships = {"c": [1, 2]}
    ictx = topology_context(backend, memberships, {"c": [1]})

    def add_owner():
        if 3 not in memberships["c"]:
            memberships["c"].append(3)
            backend.create(index="kiara_ds_3")
            backend.docs["kiara_ds_3"]["c"] = {"metadata": {"ds_ids": list(memberships["c"])}}

    ictx.store.after_update = add_owner
    _update_external_ds_ids(ictx, ["c"])
    assert ictx.store.mapping["c"] == [1, 2, 3]
    assert backend.docs["kiara_ds_1"]["c"]["metadata"]["ds_ids"] == [1, 2, 3]
    assert len(ictx.store.writes) == 2


def test_unbounded_membership_churn_fails_closed(backend):
    memberships = {"c": [1, 2]}
    ictx = topology_context(backend, memberships, {"c": [1]})

    def add_owner():
        ds = max(memberships["c"]) + 1
        memberships["c"].append(ds)
        backend.create(index=f"kiara_ds_{ds}")
        backend.docs[f"kiara_ds_{ds}"]["c"] = {"metadata": {"ds_ids": list(memberships["c"])}}

    ictx.store.after_update = add_owner
    with pytest.raises(BackendUnavailableError, match="aendern sich"):
        _update_external_ds_ids(ictx, ["c"])
    assert len(ictx.store.writes) == 3


def test_es_conflict_is_not_silently_accepted(backend):
    ictx = topology_context(backend, {"c": [1, 2]}, {"c": [1]})
    backend.before_bulk = lambda: (_ for _ in ()).throw(
        es_svc.ElasticsearchPartialFailureError("unvollstaendig")
    )
    with pytest.raises(BackendUnavailableError, match="unvollstaendig"):
        with es_svc.ingestion_visibility(ictx.config):
            _update_external_ds_ids(ictx, ["c"])
    assert len(backend.refreshes) == 1


@pytest.mark.parametrize("outcome", ["success", "cancel", "error", "partial"])
def test_final_visibility_including_error_and_cancel(backend, outcome):
    config = {"retrieval": {"hybrid_search": "elasticsearch"}}
    backend.fail_after = 1 if outcome == "partial" else None

    def run():
        with es_svc.ingestion_visibility(config):
            es_svc.bulk_index_chunks(
                config, ["one", "two"], [{"ds_ids": [1]}] * 2, ["a", "b"], datasource_id=1
            )
            assert not any(backend.visible.values())
            if outcome == "error":
                raise ValueError("failed after writes")
            return outcome

    if outcome == "partial":
        with pytest.raises(es_svc.ElasticsearchPartialFailureError):
            run()
    elif outcome == "error":
        with pytest.raises(ValueError):
            run()
    else:
        assert run() == outcome
    assert len(backend.refreshes) == 1
    assert backend.visible == backend.docs
    assert "a" in backend.visible["kiara_ds_1"]


def test_guard_failure_without_writes_does_not_refresh(backend):
    with pytest.raises(ValueError):
        with es_svc.ingestion_visibility({}):
            raise ValueError("preflight blocked")
    assert backend.refreshes == []


def test_rollback_finds_unrefreshed_documents_and_preserves_other_ds(backend):
    config = {"retrieval": {"hybrid_search": "elasticsearch"}}
    with es_svc.ingestion_visibility(config):
        es_svc.bulk_index_chunks(
            config,
            ["one", "two"],
            [{"ds_ids": [1]}, {"ds_ids": [1, 2]}],
            ["a", "b"],
            datasource_id=1,
        )
        assert not any(backend.visible.values())
        es_svc.delete_by_ids(
            config, [document_address(config, 1, cid) for cid in ["a", "b", "missing"]]
        )
    assert backend.visible["kiara_ds_1"] == {}
    assert len(backend.refreshes) == 1


def test_shared_chunk_rollback_preserves_postgres_and_both_backends(test_db, backend):
    for ds_id in [1, 2]:
        test_db.execute(
            "INSERT INTO datasources(id,name,type) VALUES (%s,%s,'local')", (ds_id, f"P50-{ds_id}")
        )
    test_db.commit()
    sot = ChunkDatasources()
    sot.add_associations(["shared", "own"], 1, "p50")
    sot.add_associations(["shared"], 2, "p50")
    ictx = topology_context(backend, {}, {"shared": [1, 2], "own": [1]})
    ictx.chunk_ds = sot
    with es_svc.ingestion_visibility(ictx.config):
        assert (
            _rollback_embedding_persistence(
                ictx,
                "/file.md",
                ["shared", "own"],
                rollback_elasticsearch=True,
                rollback_db_associations=True,
                context="injected_partial_failure",
            )
            is True
        )
    assert sot.get_datasource_ids_for_chunks(["shared", "own"], "p50") == {"shared": [2]}
    assert ictx.store.mapping == {"shared": [2]}
    assert backend.visible["kiara_ds_1"] == {}
    assert backend.visible["kiara_ds_2"] == {"shared": {"metadata": {"ds_ids": [2]}}}


def test_refresh_shard_failure_is_fatal(backend):
    backend.create(index="kiara_ds_1")
    backend.indices.refresh = Mock(return_value={"_shards": {"failed": 1}})
    with pytest.raises(es_svc.ElasticsearchPartialFailureError, match="Refresh"):
        es_svc.refresh_index(
            {"retrieval": {"hybrid_search": "elasticsearch"}}, targets=targets({}, 1)
        )


def test_missing_bulk_ack_is_fatal(backend, monkeypatch):
    topology_context(backend, {"c": [1]}, {"c": [1]})
    monkeypatch.setattr("elasticsearch.helpers.bulk", lambda *a, **k: (0, []))
    with pytest.raises(es_svc.ElasticsearchPartialFailureError, match="unvollstaendig"):
        es_svc.update_ds_ids(
            {"retrieval": {"hybrid_search": "elasticsearch"}},
            {document_address({}, 1, "c"): [1]},
            refresh=False,
        )

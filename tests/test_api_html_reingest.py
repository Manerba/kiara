"""HTML re-ingestion stays a scoped mode of the existing ingestion queue."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiara.web import api_datasource_ingest, api_ingestion_jobs
from kiara.web.auth import get_admin_user


@pytest.fixture
def api(monkeypatch):
    enqueue = Mock(return_value=478)
    monkeypatch.setattr("kiara.web.ingestion_worker.enqueue_job", enqueue)
    for repo in (api_datasource_ingest.ds_repo, api_ingestion_jobs._ds_repo):
        monkeypatch.setattr(repo, "get_by_id", lambda _id: {"id": _id, "is_active": True})
    app = FastAPI()
    app.include_router(api_datasource_ingest.router)
    app.include_router(api_ingestion_jobs.router)
    app.dependency_overrides[get_admin_user] = lambda: {"username": "admin"}
    return TestClient(app), enqueue


@pytest.mark.parametrize("mode", ["outdated", "all"])
def test_datasource_html_mode_enqueues_incremental_job(api, mode):
    client, enqueue = api
    response = client.post(
        "/api/admin/datasources/12/reingest-html",
        json={"embedding_model": "nomic-embed-text", "mode": mode},
    )
    assert response.status_code == 200
    assert response.json()["job_id"] == 478
    enqueue.assert_called_once_with(
        12,
        job_type="incremental",
        triggered_by="manual",
        embedding_model="nomic-embed-text",
        html_reingest=mode,
    )


def test_datasource_html_defaults_to_outdated_and_configured_model(api, monkeypatch):
    client, enqueue = api
    monkeypatch.setattr(api_datasource_ingest, "load_config", lambda: {})
    monkeypatch.setattr(
        "kiara.config.get_embedding_model", lambda _config: "hellord/e5-mistral-7b-instruct:q4_0"
    )
    response = client.post("/api/admin/datasources/12/reingest-html")
    assert response.status_code == 200
    assert enqueue.call_args.kwargs["html_reingest"] == "outdated"
    assert enqueue.call_args.kwargs["embedding_model"] == "hellord/e5-mistral-7b-instruct:q4_0"


@pytest.mark.parametrize("mode", ["full", "", "unknown", True, None])
def test_datasource_rejects_invalid_html_mode_before_enqueue(api, mode):
    client, enqueue = api
    response = client.post("/api/admin/datasources/12/reingest-html", json={"mode": mode})
    assert response.status_code == 422
    enqueue.assert_not_called()


@pytest.mark.parametrize("mode", ["outdated", "all"])
def test_jobs_api_preserves_html_mode(api, mode):
    client, enqueue = api
    response = client.post(
        "/api/admin/ingestion-jobs",
        json={"datasource_id": 12, "embedding_model": "nomic-embed-text", "html_reingest": mode},
    )
    assert response.status_code == 200
    assert enqueue.call_args.kwargs["html_reingest"] == mode
    assert enqueue.call_args.kwargs["job_type"] == "incremental"


@pytest.mark.parametrize(
    "extra", [{"html_reingest": "full"}, {"html_reingest": "all", "job_type": "full"}]
)
def test_jobs_api_rejects_invalid_or_full_html_mode(api, extra):
    client, enqueue = api
    response = client.post("/api/admin/ingestion-jobs", json={"datasource_id": 12, **extra})
    assert response.status_code == 422
    enqueue.assert_not_called()


@pytest.mark.parametrize("endpoint", ["/datasources/12/reingest-html", "/ingestion-jobs"])
def test_backend_guard_failure_is_not_silently_queued(api, endpoint):
    client, enqueue = api
    enqueue.side_effect = ValueError("HTML-Neuaufbereitung: Modell nicht freigegeben")
    response = client.post(
        "/api/admin" + endpoint,
        json={
            "datasource_id": 12,
            "embedding_model": "text-embedding-3-small",
            "html_reingest": "outdated",
        },
    )
    assert response.status_code == 409
    assert "nicht freigegeben" in response.json()["detail"]
    enqueue.assert_called_once()


@pytest.mark.parametrize("route, job_type", [("ingest", "incremental"), ("reingest", "full")])
def test_existing_datasource_modes_keep_their_enqueue_contract(api, route, job_type):
    client, enqueue = api
    response = client.post(
        f"/api/admin/datasources/12/{route}", json={"embedding_model": "nomic-embed-text"}
    )
    assert response.status_code == 200
    enqueue.assert_called_once_with(
        12, job_type=job_type, triggered_by="manual", embedding_model="nomic-embed-text"
    )


def test_existing_jobs_mode_does_not_add_html_scope(api):
    client, enqueue = api
    response = client.post("/api/admin/ingestion-jobs", json={"datasource_id": 12})
    assert response.status_code == 200
    assert "html_reingest" not in enqueue.call_args.kwargs


@pytest.mark.parametrize("datasource, status", [(None, 404), ({"is_active": False}, 400)])
def test_html_request_requires_existing_active_datasource(api, monkeypatch, datasource, status):
    client, enqueue = api
    monkeypatch.setattr(api_datasource_ingest.ds_repo, "get_by_id", lambda _id: datasource)
    response = client.post("/api/admin/datasources/12/reingest-html")
    assert response.status_code == status
    enqueue.assert_not_called()


def test_html_retry_uses_original_job_without_overridable_scope(api, monkeypatch):
    client, enqueue = api
    retry = Mock(return_value=479)
    monkeypatch.setattr("kiara.web.ingestion_worker.retry_html_job", retry, raising=False)
    response = client.post(
        "/api/admin/ingestion-jobs/478/retry-html",
        json={"datasource_id": 15, "embedding_model": "other-model", "html_reingest": "all"},
    )
    assert response.status_code == 200
    assert response.json() == {"job_id": 479, "message": "HTML-Job erneut eingestellt"}
    retry.assert_called_once_with(478)
    enqueue.assert_not_called()


@pytest.mark.parametrize(
    "reason",
    [
        "HTML-Revision seit Jobstart geaendert; neuen HTML-Job starten",
        "HTML-Job hat bereits einen aktiven Nachfolger",
        "Nur abgebrochene oder fehlgeschlagene HTML-Jobs koennen wiederholt werden",
    ],
)
def test_html_retry_reports_guard_conflicts(api, monkeypatch, reason):
    client, enqueue = api
    retry = Mock(side_effect=ValueError(reason))
    monkeypatch.setattr("kiara.web.ingestion_worker.retry_html_job", retry, raising=False)
    response = client.post("/api/admin/ingestion-jobs/478/retry-html")
    assert response.status_code == 409
    assert response.json()["detail"] == reason
    retry.assert_called_once_with(478)
    enqueue.assert_not_called()


def test_html_retry_requires_admin(api, monkeypatch):
    from fastapi import HTTPException

    client, _enqueue = api
    retry = Mock()
    monkeypatch.setattr("kiara.web.ingestion_worker.retry_html_job", retry, raising=False)

    def deny():
        raise HTTPException(status_code=403, detail="Adminrechte erforderlich")

    client.app.dependency_overrides[get_admin_user] = deny
    response = client.post("/api/admin/ingestion-jobs/478/retry-html")
    assert response.status_code == 403
    retry.assert_not_called()

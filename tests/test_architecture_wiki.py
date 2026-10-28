"""Architecture navigation must preserve auth and the reviewed source snapshot."""

from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from kiara.web.api_wiki import _architecture_atlas, router
from kiara.web.middleware import add_security_headers_middleware

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.ids = set()
        self.duplicate_ids = set()
        self.links = []
        self.images = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.duplicate_ids.add(attrs["id"])
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "img":
            self.images.append(attrs["src"])


@pytest.fixture
def client():
    app = FastAPI()
    add_security_headers_middleware(app, csp_strict=True)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=ROOT / "kiara/web/static"))
    with patch("kiara.web.api_wiki._require_admin", return_value={"id": 1, "is_admin": True}):
        with TestClient(app, follow_redirects=False) as result:
            yield result


@pytest.mark.parametrize(
    "path",
    [
        "/wiki/guide/architecture",
        "/wiki/guide/architecture-ingestion",
        "/wiki/guide/architecture-retrieval",
        "/wiki/architecture/code/pipeline",
        "/wiki/architecture/mermaid/overview",
    ],
)
def test_all_architecture_resources_require_admin(client, path):
    with patch("kiara.web.api_wiki._require_admin", return_value=None):
        response = client.get(path)
    assert response.status_code == 302
    assert response.headers["location"] == "/login"


def test_every_diagram_step_source_and_fragment_is_reachable(client):
    atlas = _architecture_atlas()
    pages = {}
    code_pages = {}
    for page in dict.fromkeys(d["page"] for d in atlas["diagrams"]):
        path = f"/wiki/guide/{page}"
        response = client.get(path)
        assert response.status_code == 200
        assert "Content-Security-Policy" in response.headers
        pages[path] = Links(response.text)
        assert not pages[path].duplicate_ids
        for src in pages[path].images:
            response = client.get(src)
            assert response.status_code == 200
            assert "image/" in response.headers["content-type"]
    for links in pages.values():
        for href in links.links:
            target = urlsplit(href)
            if href.startswith("#"):
                assert target.fragment in links.ids, href
            elif target.path.startswith("/wiki/architecture/code/"):
                if target.path not in code_pages:
                    response = client.get(target.path)
                    assert response.status_code == 200, href
                    code_pages[target.path] = Links(response.text)
                    assert not code_pages[target.path].duplicate_ids
                assert target.fragment in code_pages[target.path].ids, href
    for diagram in atlas["diagrams"]:
        response = client.get(f"/wiki/architecture/mermaid/{diagram['id']}")
        assert response.status_code == 200
        assert response.text == diagram["mermaid"]
        assert "attachment;" in response.headers["content-disposition"]
        ids = pages[f"/wiki/guide/{diagram['page']}"].ids
        assert {n["id"] for n in diagram["nodes"]} <= ids
        assert {e["id"] for e in diagram["edges"]} <= ids


def test_source_snapshot_is_escaped_as_text(client):
    atlas = _architecture_atlas()
    reference = {**atlas["references"]["pipeline"], "code": 'value = "<script>alert(1)</script>"'}
    with patch(
        "kiara.web.api_wiki._architecture_atlas",
        return_value={
            **atlas,
            "references": {**atlas["references"], "pipeline": reference},
        },
    ):
        response = client.get("/wiki/architecture/code/pipeline")
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


@pytest.mark.parametrize(
    "path",
    [
        "/wiki/architecture/code/unknown",
        "/wiki/architecture/code/AGENTS.md",
        "/wiki/architecture/mermaid/unknown",
    ],
)
def test_unknown_snapshot_ids_are_not_filesystem_paths(client, path):
    assert client.get(path).status_code == 404



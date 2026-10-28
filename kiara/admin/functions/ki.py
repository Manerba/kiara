import json

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult


@admin_function(
    name="ki.backends.list",
    category="ki",
    description="Alle KI-Backends auflisten",
    help="Listet alle konfigurierten KI-Backends (Embedding, LLM, Cloud, VLM, DDU) "
    "mit Status, Provider und Priorität auf.",
    examples=[
        "Welche KI-Backends sind konfiguriert?",
        "Zeig mir alle Backends",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Liste aller KI-Backends mit Metadaten",
    raw_response=True,
    http=("GET", "/api/admin/ki/backends/"),
)
async def ki_backends_list(user: dict) -> AdminResult:
    from kiara.repositories.backend import KiBackendRepository
    from kiara.retrieval.embedding_contract import ProfileKind
    from kiara.retrieval.embedding_roles import (
        SEARCH_DOCUMENT,
        backend_supports_embedding_operation,
    )
    from kiara.shared.constants import PASSWORD_MASK

    backend_repo = KiBackendRepository()
    rows = backend_repo.list_ordered()

    backends = []
    for d in rows:
        if d.get("api_key"):
            d["api_key"] = PASSWORD_MASK
        try:
            d["extra_config"] = json.loads(d.get("extra_config") or "{}")
        except (json.JSONDecodeError, TypeError):
            d["extra_config"] = {}
        d["is_active"] = bool(d["is_active"])
        d["trusted"] = bool(d.get("trusted", 0))
        d["last_test_ok"] = bool(d["last_test_ok"]) if d["last_test_ok"] is not None else None
        if d.get("category") == "embedding":
            supports_late, late_reason = backend_supports_embedding_operation(
                d,
                role=SEARCH_DOCUMENT,
                kind=ProfileKind.LATE_CHUNKING,
                endpoint="/api/embed_late",
            )
            d["supports_late_chunking"] = supports_late
            d["late_chunking_contract_reason"] = late_reason
        backends.append(d)

    return AdminResult.ok("KI-Backends", data={"backends": backends})


@admin_function(
    name="ki.backends.categories",
    category="ki",
    description="Backend-Kategorien und erlaubte Provider",
    help="Zeigt die verfügbaren Kategorien (embedding, llm, cloud, vlm, ddu) "
    "und deren erlaubte Provider-Typen an.",
    examples=[
        "Welche Backend-Kategorien gibt es?",
        "Provider für LLM-Backends",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Kategorien mit Provider-Liste und Beschreibung",
    raw_response=True,
    http=("GET", "/api/admin/ki/backends/categories"),
)
async def ki_backends_categories(user: dict) -> AdminResult:
    from kiara.web.api_ki_backends import get_categories

    data = await get_categories(user=user)
    return AdminResult.ok("Kategorien", data=data)


@admin_function(
    name="ki.config",
    category="ki",
    description="KI-Konfiguration lesen",
    help="Zeigt die aktuelle KI-Backend-Konfiguration: Embedding/LLM-URLs, "
    "Modelle, Cloud-Einstellungen und Anonymisierungsparameter.",
    examples=[
        "Zeig mir die KI-Konfiguration",
        "Welches LLM-Modell ist konfiguriert?",
        "Cloud-AI Einstellungen",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="KI-Konfiguration mit allen Einstellungen",
    raw_response=True,
    http=("GET", "/api/admin/ki/config"),
)
async def ki_config(user: dict) -> AdminResult:
    from kiara.web.api_ki import get_ki_config

    data = await get_ki_config(user=user)
    return AdminResult.ok("KI-Konfiguration", data=data)


@admin_function(
    name="ki.templates.list",
    category="ki",
    description="Prompt-Templates auflisten",
    help="Listet alle verfügbaren Prompt-Templates auf, sortiert nach Reihenfolge und ID.",
    examples=[
        "Welche Templates gibt es?",
        "Prompt-Templates anzeigen",
    ],
    parameters={
        "type": "object",
        "properties": {
            "chain_type": {
                "type": "string",
                "description": "Filter nach chain_type",
                "enum": ["standard", "unchained", "agent", "rewriter", "admin_agent", "enrichment"],
            },
        },
        "required": [],
    },
    returns="Liste aller Prompt-Templates",
    raw_response=True,
    http=("GET", "/api/admin/ki/prompt-templates"),
)
async def ki_templates_list(user: dict, chain_type: str | None = None) -> AdminResult:
    if chain_type:
        from kiara.repositories.prompt import PromptTemplateRepository
        from kiara.web.api_ki import _template_row_to_dict

        prompt_repo = PromptTemplateRepository()
        rows, _total = prompt_repo.list_filtered(
            filters={"chain_type": chain_type},
            sort="sort_order ASC, id ASC",
            limit=10000,
        )
        data = {"data": [_template_row_to_dict(r) for r in rows]}
    else:
        from kiara.web.api_ki import list_prompt_templates

        data = await list_prompt_templates(user=user)
    return AdminResult.ok("Templates", data=data)


@admin_function(
    name="ki.templates.detail",
    category="ki",
    description="Details eines einzelnen Prompt-Templates",
    help="Zeigt alle Felder eines Prompt-Templates: Name, Inhalt, chain_type, "
    "Label, Beschreibung, Flags (is_cloud, show_in_chat, is_system).",
    examples=[
        "Zeig mir Template 1",
        "Details zu Template 3",
    ],
    parameters={
        "type": "object",
        "properties": {"template_id": {"type": "integer", "description": "Template-ID"}},
        "required": ["template_id"],
    },
    returns="Template-Details",
    raw_response=True,
    http=("GET", "/api/admin/ki/prompt-templates/{template_id}"),
)
async def ki_templates_detail(user: dict, template_id: int) -> AdminResult:
    from kiara.repositories.prompt import PromptTemplateRepository

    repo = PromptTemplateRepository()
    template = repo.get_by_id(template_id)
    if not template:
        return AdminResult.fail(f"Template {template_id} nicht gefunden")
    return AdminResult.ok(f"Template '{template.get('name', template_id)}'", data=dict(template))


@admin_function(
    name="ki.automaton_stats",
    category="ki",
    description="Anonymisierungs-Automaton Statistiken",
    help="Zeigt Statistiken des Aho-Corasick-Automaten für die "
    "Anonymisierung: Anzahl Einträge (ERP, LDAP), letzter Build.",
    examples=[
        "Automaton-Statistiken",
        "Wie viele Anonymisierungs-Einträge?",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Automaton-Stats mit Entry-Counts und Build-Zeitstempel",
    raw_response=True,
    http=("GET", "/api/admin/ki/automaton-stats"),
)
async def ki_automaton_stats(user: dict) -> AdminResult:
    from kiara.web.api_ki import get_automaton_stats

    data = await get_automaton_stats(user=user)
    return AdminResult.ok("Automaton-Stats", data=data)


@admin_function(
    name="ki.audit",
    category="ki",
    description="KI-Audit-Log (paginiert)",
    help="Zeigt das Anonymisierungs-Audit-Log: Welche Queries wurden "
    "anonymisiert, wie viele Entities gefunden, welcher Route.",
    examples=[
        "Audit-Log anzeigen",
        "Letzte Anonymisierungs-Events",
    ],
    parameters={
        "type": "object",
        "properties": {
            "page": {"type": "integer", "description": "Seite (Default: 1)"},
            "per_page": {
                "type": "integer",
                "description": "Einträge pro Seite (Default: 20, max: 100)",
            },
        },
        "required": [],
    },
    returns="Paginiertes Audit-Log",
    raw_response=True,
    http=("GET", "/api/admin/ki/audit"),
)
async def ki_audit(user: dict, page: int = 1, per_page: int = 20) -> AdminResult:
    from kiara.web.api_ki import get_audit_log

    data = await get_audit_log(user=user, page=page, per_page=per_page)
    return AdminResult.ok("Audit-Log", data=data)


# ---------------------------------------------------------------------------
# Sprint 06: KI-Backend CRUD (5 Funktionen)
# ---------------------------------------------------------------------------


@admin_function(
    name="ki.backends.create",
    category="ki",
    description="Neues KI-Backend erstellen",
    help="Erstellt ein neues KI-Backend. Gueltige Kategorien: embedding, llm, cloud, "
    "vlm, ddu. Provider haengt von der Kategorie ab: "
    "embedding/llm/vlm → ollama, cloud → anthropic/openai, ddu → docling. "
    "Bei Ollama-Backends werden num_ctx und embedding_dim automatisch ermittelt. "
    "DDU-Backends erhalten automatisch model='docling'.",
    examples=[
        "Erstelle ein Ollama-LLM-Backend auf http://gpu-server.local:11434 mit Modell qwen3:8b",
        "Fuege ein Anthropic-Cloud-Backend mit API-Key sk-... hinzu",
        "Neues Embedding-Backend: Ollama, URL http://localhost:11434, Modell nomic-embed-text",
    ],
    parameters={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Kategorie: embedding, llm, cloud, vlm, ddu",
            },
            "provider": {
                "type": "string",
                "description": "Provider: ollama, anthropic, openai, docling",
            },
            "name": {"type": "string", "description": "Anzeigename (auto-generiert wenn leer)"},
            "url": {"type": "string", "description": "Backend-URL (fuer Ollama/Docling)"},
            "base_url": {"type": "string", "description": "Base-URL (fuer OpenAI-kompatible APIs)"},
            "model": {"type": "string", "description": "Modell (bei DDU auto='docling')"},
            "api_key": {
                "type": "string",
                "description": "API-Key (wird verschluesselt gespeichert)",
            },
            "is_active": {"type": "boolean", "description": "Aktiv (Default: true)"},
            "priority": {"type": "integer", "description": "Prioritaet (Default: 10)"},
            "trusted": {
                "type": "boolean",
                "description": "Vertrauenswuerdiges Backend (Default: false)",
            },
            "extra_config": {"type": "object", "description": "Zusaetzliche Konfiguration"},
        },
        "required": ["category", "provider"],
    },
    returns="ID und Bestaetigungsmeldung",
    changelog_area="ki_backends",
    changelog_action="create",
    raw_response=True,
    http=("POST", "/api/admin/ki/backends/"),
)
async def ki_backends_create(
    user: dict,
    category: str,
    provider: str,
    name: str = "",
    url: str = "",
    base_url: str = "",
    model: str = "",
    api_key: str = "",
    is_active: bool = True,
    priority: int = 10,
    trusted: bool = False,
    extra_config: dict | None = None,
) -> AdminResult:
    try:
        from kiara.admin.services.ki_backend_service import create_backend

        result = await create_backend(
            category=category,
            provider=provider,
            name=name,
            url=url,
            base_url=base_url,
            model=model,
            api_key=api_key,
            is_active=is_active,
            priority=priority,
            trusted=trusted,
            extra_config=extra_config,
        )
        return AdminResult.ok("Backend erstellt", data=result)
    except ValueError as exc:
        return AdminResult.fail(str(exc))


@admin_function(
    name="ki.backends.update",
    category="ki",
    description="KI-Backend aktualisieren",
    help="Aktualisiert ein bestehendes KI-Backend. Nur uebergebene Felder werden "
    "geaendert (Partial Update). PASSWORD_MASK (********) fuer api_key "
    "behaelt den bestehenden verschluesselten Key nur bei unveraendertem "
    "Provider bei; Providerwechsel zu API-Key-Providern erfordern einen "
    "neuen Key.",
    examples=[
        "Aendere die URL von Backend 3 auf http://gpu-server.local:11434",
        "Setze Backend 5 auf Modell qwen3:8b",
    ],
    parameters={
        "type": "object",
        "properties": {
            "backend_id": {"type": "integer", "description": "Backend-ID"},
            "category": {"type": "string", "description": "Kategorie"},
            "provider": {"type": "string", "description": "Provider"},
            "name": {"type": "string", "description": "Anzeigename"},
            "url": {"type": "string", "description": "Backend-URL"},
            "base_url": {"type": "string", "description": "Base-URL"},
            "model": {"type": "string", "description": "Modell"},
            "api_key": {
                "type": "string",
                "description": "API-Key (PASSWORD_MASK behaelt bestehenden)",
            },
            "is_active": {"type": "boolean", "description": "Aktiv-Status"},
            "priority": {"type": "integer", "description": "Prioritaet"},
            "trusted": {"type": "boolean", "description": "Vertrauenswuerdiges Backend"},
            "extra_config": {"type": "object", "description": "Zusaetzliche Konfiguration"},
        },
        "required": ["backend_id"],
    },
    returns="Bestaetigungsmeldung",
    requires_confirmation=True,
    confirm_template="backend {backend_id} aktualisieren",
    pk_param="backend_id",
    changelog_area="ki_backends",
    changelog_action="update",
    raw_response=True,
    http=("PUT", "/api/admin/ki/backends/{backend_id}"),
)
async def ki_backends_update(
    user: dict,
    backend_id: int,
    category: str | None = None,
    provider: str | None = None,
    name: str | None = None,
    url: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    is_active: bool | None = None,
    priority: int | None = None,
    trusted: bool | None = None,
    extra_config: dict | None = None,
) -> AdminResult:
    try:
        from kiara.admin.services.ki_backend_service import update_backend

        result = await update_backend(
            backend_id,
            category=category,
            provider=provider,
            name=name,
            url=url,
            base_url=base_url,
            model=model,
            api_key=api_key,
            is_active=is_active,
            priority=priority,
            trusted=trusted,
            extra_config=extra_config,
        )
        return AdminResult.ok("Backend aktualisiert", data=result)
    except ValueError as exc:
        return AdminResult.fail(str(exc))


@admin_function(
    name="ki.backends.delete",
    category="ki",
    description="KI-Backend loeschen",
    help="Loescht ein KI-Backend. DDU-Backends nutzen nur noch Remote-URLs; "
    "Legacy-Eintraege ohne URL werden nicht automatisch neu angelegt.",
    examples=[
        "Loesche Backend 5",
        "Entferne das alte Anthropic-Backend",
    ],
    parameters={
        "type": "object",
        "properties": {
            "backend_id": {"type": "integer", "description": "Backend-ID"},
        },
        "required": ["backend_id"],
    },
    returns="Loeschbestaetigung",
    requires_confirmation=True,
    confirm_template="backend {backend_id} löschen",
    pk_param="backend_id",
    changelog_area="ki_backends",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/ki/backends/{backend_id}"),
)
async def ki_backends_delete(user: dict, backend_id: int) -> AdminResult:
    try:
        from kiara.admin.services.ki_backend_service import delete_backend

        result = await delete_backend(backend_id)
        return AdminResult.ok("Backend entfernt", data=result)
    except ValueError as exc:
        return AdminResult.fail(str(exc))


@admin_function(
    name="ki.backends.toggle",
    category="ki",
    description="KI-Backend aktivieren/deaktivieren",
    help="Setzt den Aktiv-Status eines KI-Backends.",
    examples=[
        "Deaktiviere Backend 3",
        "Aktiviere das Ollama-Embedding-Backend",
    ],
    parameters={
        "type": "object",
        "properties": {
            "backend_id": {"type": "integer", "description": "Backend-ID"},
            "is_active": {"type": "boolean", "description": "Neuer Aktiv-Status"},
        },
        "required": ["backend_id", "is_active"],
    },
    returns="Bestaetigungsmeldung",
    pk_param="backend_id",
    changelog_area="ki_backends",
    changelog_action="update",
    raw_response=True,
    http=("PATCH", "/api/admin/ki/backends/{backend_id}/toggle"),
)
async def ki_backends_toggle(user: dict, backend_id: int, is_active: bool) -> AdminResult:
    from kiara.backend_selector import invalidate_cache
    from kiara.repositories.backend import KiBackendRepository

    backend_repo = KiBackendRepository()
    existing = backend_repo.get_by_id(backend_id)
    if not existing:
        return AdminResult.fail("Backend nicht gefunden")

    backend_repo.set_active(backend_id, is_active)
    invalidate_cache(existing["category"])
    if existing["category"] == "ddu":
        try:
            from kiara.web.api_datasource_features import invalidate_ddu_cache

            invalidate_ddu_cache()
        except ImportError:
            pass

    state = "aktiviert" if is_active else "deaktiviert"
    return AdminResult.ok(
        f"Backend {backend_id} {state}",
        data={"message": f"Backend {state}", "name": existing.get("name", "")},
    )


@admin_function(
    name="ki.backends.test",
    category="ki",
    description="KI-Backend Verbindungstest",
    help="Testet die Verbindung zu einem konfigurierten Backend. "
    "Bei Ollama werden zusaetzlich num_ctx und embedding_dim ermittelt. "
    "Bei Cloud-Providern wird die Modell-Liste abgerufen. "
    "Das Testergebnis wird am Backend gespeichert.",
    examples=[
        "Teste Backend 3",
        "Ist das Ollama-Backend erreichbar?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "backend_id": {"type": "integer", "description": "Backend-ID"},
        },
        "required": ["backend_id"],
    },
    returns="Testergebnis mit Modell-Liste",
    raw_response=True,
    http=("POST", "/api/admin/ki/backends/{backend_id}/test"),
)
async def ki_backends_test(user: dict, backend_id: int) -> AdminResult:
    try:
        from kiara.admin.services.ki_backend_service import test_backend

        result = await test_backend(backend_id)
        return AdminResult.ok("Verbindungstest erfolgreich", data=result)
    except ValueError as exc:
        return AdminResult.fail(str(exc))


# ---------------------------------------------------------------------------
# Sprint 06: KI-Config (5 Chat-only Funktionen)
# ---------------------------------------------------------------------------


@admin_function(
    name="ki.config.update",
    category="ki",
    description="KI-Konfiguration aendern (Partial Update)",
    help="Aendert einzelne KI-Konfigurationsfelder. Nur uebergebene Felder "
    "werden geaendert — alle anderen behalten ihren aktuellen Wert. "
    "PASSWORD_MASK (********) fuer cloud_api_key behaelt den bestehenden Key. "
    "Prompt-Template muss {context} und {question} enthalten. "
    "Agent-Systemprompt muss {tools_prompt} enthalten. "
    "Aenderungen werden in kiara.yaml synchronisiert.",
    examples=[
        "Setze die Ollama-URL auf http://gpu-server.local:11434",
        "Aktiviere Cloud-Backend mit Provider anthropic",
        "Deaktiviere die Anonymisierung",
        "Setze das LLM-Modell auf qwen3:8b",
    ],
    parameters={
        "type": "object",
        "properties": {
            # --- Ollama ---
            "embedding_url": {"type": "string", "description": "Ollama Embedding URL"},
            "embedding_model": {"type": "string", "description": "Embedding-Modell"},
            "llm_url": {"type": "string", "description": "Ollama LLM URL"},
            "llm_model": {"type": "string", "description": "LLM-Modell"},
            # --- Cloud ---
            "cloud_enabled": {"type": "boolean", "description": "Cloud-Backend aktivieren"},
            "cloud_provider": {"type": "string", "description": "Provider (anthropic/openai)"},
            "cloud_api_key": {
                "type": "string",
                "description": "API-Key (PASSWORD_MASK behaelt bestehenden)",
            },
            "cloud_model": {"type": "string", "description": "Cloud-Modell"},
            "cloud_base_url": {"type": "string", "description": "Cloud Base-URL"},
            "cloud_mode": {
                "type": "string",
                "enum": ["auto", "local", "cloud"],
                "description": "Modus (auto/local/cloud)",
            },
            # --- Anonymisierung ---
            "anon_enabled": {"type": "boolean", "description": "Anonymisierung aktivieren"},
            "anon_erp_enabled": {"type": "boolean", "description": "ERP-Stammdaten aktivieren"},
            "anon_erp_path": {"type": "string", "description": "Pfad zu ERP-Stammdaten"},
            "anon_ldap_enabled": {"type": "boolean", "description": "LDAP-Stammdaten aktivieren"},
            "anon_word_boundaries": {"type": "boolean", "description": "Wortgrenzen-Matching"},
            "anon_min_term_length": {"type": "integer", "description": "Minimale Term-Laenge"},
            "anon_excluded_terms": {
                "type": "string",
                "description": "Ausgeschlossene Begriffe (JSON-Array)",
            },
            "anon_custom_patterns": {
                "type": "string",
                "description": "Eigene Regex-Muster (JSON-Array)",
            },
            "anon_audit_log": {"type": "boolean", "description": "Audit-Log aktivieren"},
            "anon_custom_entries": {
                "type": "string",
                "description": "Eigene Eintraege (JSON-Array)",
            },
            "anon_llm_enabled": {"type": "boolean", "description": "LLM-Detection aktivieren"},
            "anon_llm_model": {"type": "string", "description": "LLM-Detection Modell"},
            "anon_llm_temperature": {"type": "string", "description": "LLM-Detection Temperatur"},
            "anon_llm_confidence": {
                "type": "string",
                "description": "LLM-Detection Confidence-Schwelle",
            },
            "anon_llm_timeout": {
                "type": "integer",
                "description": "LLM-Detection Timeout (Sekunden)",
            },
            "anon_llm_cache_size": {
                "type": "integer",
                "description": "LLM-Detection Cache-Groesse",
            },
            "anon_llm_num_ctx": {"type": "integer", "description": "LLM-Detection Context-Fenster"},
        },
        "required": [],
    },
    returns="Bestaetigungsmeldung",
    requires_confirmation=True,
    confirm_template="ki konfiguration speichern",
    changelog_area="ki",
    changelog_action="update",
)
async def ki_config_update(user: dict, **kwargs) -> AdminResult:
    from kiara.repositories.config import ConfigRepository
    from kiara.shared.constants import PASSWORD_MASK
    from kiara.shared.crypto import encrypt, is_encrypted
    from kiara.shared.enums import RoutingMode

    config_repo = ConfigRepository()
    current = config_repo.get("ai_config") or {}

    # Merge: bestehende Werte als Basis, uebergebene Werte ueberschreiben
    merged = dict(current)
    for k, v in kwargs.items():
        if v is not None:
            merged[k] = v

    valid_cloud_modes = {mode.value for mode in RoutingMode}
    if "cloud_mode" in merged and merged.get("cloud_mode") not in valid_cloud_modes:
        return AdminResult.fail("Ungueltiger Cloud-Modus. Erlaubt sind: auto, local, cloud")

    # PASSWORD_MASK fuer cloud_api_key
    cloud_key = merged.get("cloud_api_key", "")
    if cloud_key == PASSWORD_MASK:
        cloud_key = current.get("cloud_api_key", "")
        if not cloud_key:
            return AdminResult.fail(
                "PASSWORD_MASK kann ohne bestehenden Cloud-API-Key nicht uebernommen werden."
            )
        if cloud_key and not is_encrypted(cloud_key):
            try:
                cloud_key = encrypt(cloud_key)
            except Exception as exc:
                return AdminResult.fail(f"API-Key-Verschluesselung fehlgeschlagen: {exc}")
    elif cloud_key and not is_encrypted(cloud_key):
        try:
            cloud_key = encrypt(cloud_key)
        except Exception as exc:
            return AdminResult.fail(f"API-Key-Verschluesselung fehlgeschlagen: {exc}")
    merged["cloud_api_key"] = cloud_key

    # YAML-Sync vor Konvertierung der beiden Integer-Schalter.
    from kiara.web.api_ki import _sync_to_yaml

    _sync_to_yaml(
        merged.get("embedding_url", ""),
        merged.get("embedding_model", ""),
        merged.get("llm_url", ""),
        merged.get("llm_model", ""),
        **{k: v for k, v in merged.items() if k.startswith(("cloud_", "anon_"))},
    )

    # Nur diese Schalter sind INTEGER; PostgreSQL-BOOLEAN braucht Python-bool.
    integer_flags = ("anon_word_boundaries", "anon_audit_log")
    db_fields = dict(merged)
    for field in integer_flags:
        if isinstance(db_fields.get(field), bool):
            db_fields[field] = int(db_fields[field])

    config_repo.set("ai_config", db_fields)

    changed_keys = [k for k in kwargs if kwargs[k] is not None]
    return AdminResult.ok(
        f"KI-Konfiguration gespeichert ({len(changed_keys)} Felder)",
        data={
            "message": "KI-Konfiguration gespeichert",
            "changed_fields": changed_keys,
            "name": "KI-Konfiguration",
        },
    )


@admin_function(
    name="ki.config.test_cloud",
    category="ki",
    description="Cloud-API-Verbindungstest",
    help="Testet die Verbindung zu einem Cloud-Provider (Anthropic oder OpenAI). "
    "Gibt bei Erfolg die verfuegbaren Modelle zurueck.",
    examples=[
        "Teste Cloud-Verbindung mit API-Key sk-ant-...",
        "Ist Anthropic erreichbar?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "provider": {"type": "string", "description": "Provider (anthropic/openai)"},
            "api_key": {"type": "string", "description": "API-Key"},
            "model": {"type": "string", "description": "Modell (optional)"},
            "base_url": {
                "type": "string",
                "description": "Base-URL (optional, fuer OpenAI-kompatible)",
            },
        },
        "required": ["provider", "api_key"],
    },
    returns="Verbindungsstatus und Modell-Liste",
)
async def ki_config_test_cloud(
    user: dict, provider: str, api_key: str, model: str = "", base_url: str = ""
) -> AdminResult:
    if not api_key:
        return AdminResult.fail("API-Key ist erforderlich")
    try:
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            result = client.models.list()
            model_ids = [m.id for m in result.data] if hasattr(result, "data") else []
            return AdminResult.ok(
                f"Verbindung OK — {len(model_ids)} Modelle",
                data={"success": True, "models": model_ids},
            )
        elif provider == "openai":
            import openai

            client = openai.OpenAI(api_key=api_key, base_url=base_url or None)
            result = client.models.list()
            model_ids = [m.id for m in result.data]
            return AdminResult.ok(
                f"Verbindung OK — {len(model_ids)} Modelle",
                data={"success": True, "models": model_ids},
            )
        else:
            return AdminResult.fail(f"Unbekannter Provider: {provider}")
    except Exception as exc:
        return AdminResult.fail(f"Cloud-Verbindung fehlgeschlagen: {exc}")


@admin_function(
    name="ki.config.test_anonymize",
    category="ki",
    description="Anonymisierung testen",
    help="Anonymisiert einen Beispieltext mit der aktuellen Konfiguration. "
    "Zeigt gefundene Treffer und verwendete Pseudonyme.",
    examples=[
        "Teste die Anonymisierung mit dem Text 'Max Mustermann aus Berlin'",
        "Anonymisiere: Herr Schmidt hat Auftrag 12345",
    ],
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Beispieltext zum Anonymisieren"},
        },
        "required": ["text"],
    },
    returns="Anonymisierter Text mit Trefferliste",
)
async def ki_config_test_anonymize(user: dict, text: str) -> AdminResult:
    from kiara.anonymization.anonymizer import Anonymizer
    from kiara.anonymization.stammdaten import build_automaton_from_sources
    from kiara.config import load_config

    config = load_config()
    automaton = build_automaton_from_sources(config)
    pseudonymizer = Anonymizer(automaton, config)
    anonymized = pseudonymizer.anonymize(text)

    matches = [
        {"original": original, "pseudonym": pseudonym}
        for pseudonym, original in pseudonymizer.reverse_map.items()
    ]

    return AdminResult.ok(
        f"{len(matches)} Treffer anonymisiert",
        data={
            "original": text,
            "anonymized": anonymized,
            "matches": matches,
            "stats": pseudonymizer.stats,
        },
    )


@admin_function(
    name="ki.config.test_connection",
    category="ki",
    description="Ollama-Verbindungstest",
    help="Testet die Verbindung zu einer Ollama-Instanz. Gibt bei Erfolg die "
    "verfuegbaren Modelle zurueck. Optional mit Kategorie-Filter.",
    examples=[
        "Teste Ollama auf http://gpu-server.local:11434",
        "Ist der Ollama-Server erreichbar?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Ollama-URL (z.B. http://localhost:11434)"},
            "category": {
                "type": "string",
                "description": "Kategorie-Filter (embedding/llm/vlm, optional)",
            },
        },
        "required": ["url"],
    },
    returns="Verbindungsstatus und Modell-Liste",
)
async def ki_config_test_connection(
    user: dict, url: str, category: str | None = None
) -> AdminResult:
    from kiara.web.api_ki import _test_ollama_connection

    result = await _test_ollama_connection(url, category=category)
    if result["success"]:
        models = result.get("models", [])
        return AdminResult.ok(
            f"Ollama erreichbar — {len(models)} Modelle", data={"success": True, "models": models}
        )
    return AdminResult.fail(result.get("error", "Verbindung fehlgeschlagen"))


@admin_function(
    name="ki.automaton.rebuild",
    category="ki",
    description="Aho-Corasick-Automaton neu bauen",
    help="Laedt Stammdaten (ERP, LDAP) neu und baut den Anonymisierungs-Automaton "
    "neu auf. Der Automaton wird im Speicher gecacht.",
    examples=[
        "Baue den Automaton neu",
        "Stammdaten neu laden",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Anzahl der Eintraege im neuen Automaton",
    requires_confirmation=True,
    confirm_template="automaton neu bauen",
    changelog_area="ki",
    changelog_action="update",
)
async def ki_automaton_rebuild(user: dict) -> AdminResult:
    from kiara.anonymization.stammdaten import build_automaton_from_sources
    from kiara.config import load_config

    config = load_config()
    try:
        automaton = build_automaton_from_sources(config)
    except Exception as exc:
        return AdminResult.fail(f"Automaton-Build fehlgeschlagen: {exc}")

    # Cache in api_ki.py aktualisieren (gleicher Speicherort wie reload_automaton)
    from datetime import datetime, timezone

    from kiara.web.api_ki import _automaton_cache

    _automaton_cache["automaton"] = automaton
    _automaton_cache["last_built"] = datetime.now(timezone.utc).isoformat()
    _sc = getattr(automaton, "source_counts", {})
    _automaton_cache["erp_count"] = _sc.get("erp", 0)
    _automaton_cache["ldap_count"] = _sc.get("ldap", 0)

    return AdminResult.ok(
        f"Automaton neu gebaut: {automaton.entry_count} Eintraege",
        data={
            "entry_count": automaton.entry_count,
            "sources": {"erp": _sc.get("erp", 0), "ldap": _sc.get("ldap", 0)},
            "name": "Automaton",
        },
    )


# ---------------------------------------------------------------------------
# Sprint 06: Template-CRUD (3 Funktionen)
# ---------------------------------------------------------------------------


@admin_function(
    name="ki.templates.create",
    category="ki",
    description="Neues Prompt-Template erstellen",
    help="Erstellt ein neues Prompt-Template. chain_type bestimmt die Pflicht-Platzhalter: "
    "standard → {context} + {question}, agent → {tools_prompt}, enrichment → {chunk_text} (Pflicht), "
    "unchained → keine. Template-Name muss eindeutig sein.",
    examples=[
        "Erstelle ein Template 'Zusammenfassung' mit chain_type standard",
        "Neues Agent-Template mit Name 'Recherche'",
    ],
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Eindeutiger Template-Name"},
            "content": {"type": "string", "description": "Template-Inhalt"},
            "label": {"type": "string", "description": "Anzeige-Label"},
            "description": {"type": "string", "description": "Beschreibung"},
            "final_prompt": {"type": "string", "description": "Finaler Prompt"},
            "cloud_system_prompt": {
                "type": "string",
                "description": "System-Prompt fuer Cloud-AI-Backends",
            },
            "chain_type": {
                "type": "string",
                "description": "Typ: standard, unchained, agent, enrichment (Default: standard)",
                "enum": ["standard", "unchained", "agent", "enrichment"],
            },
            "is_cloud": {"type": "boolean", "description": "Cloud-Template (Default: false)"},
            "show_in_chat": {"type": "boolean", "description": "Im Chat anzeigen (Default: false)"},
            "badge_color": {"type": "string", "description": "Badge-Farbe"},
            "sort_order": {"type": "integer", "description": "Sortierung (Default: 0)"},
        },
        "required": ["name", "content"],
    },
    returns="ID und Bestaetigungsmeldung",
    changelog_area="prompt_templates",
    changelog_action="create",
    raw_response=True,
    http=("POST", "/api/admin/ki/prompt-templates"),
)
async def ki_templates_create(
    user: dict,
    name: str,
    content: str,
    label: str = "",
    description: str = "",
    final_prompt: str = "",
    chain_type: str = "standard",
    is_cloud: bool = False,
    show_in_chat: bool = False,
    badge_color: str = "",
    sort_order: int = 0,
    cloud_system_prompt: str = "",
) -> AdminResult:
    from fastapi import HTTPException

    from kiara.repositories.prompt import PromptTemplateRepository
    from kiara.shared.db import db_query
    from kiara.web.api_ki import _validate_template_content

    if chain_type not in ("standard", "unchained", "agent", "enrichment"):
        return AdminResult.fail(f"Ungueltiger chain_type: {chain_type}")

    chat_visible_chain_types = ("standard", "unchained", "agent", "rewriter")
    if show_in_chat and (chain_type not in chat_visible_chain_types or is_cloud):
        return AdminResult.fail("Dieses Template darf nicht als Chat-Modus angezeigt werden")

    try:
        _validate_template_content(content, chain_type)
    except HTTPException as exc:
        return AdminResult.fail(exc.detail)

    prompt_repo = PromptTemplateRepository()
    # Uniqueness-Check (aktive + inaktive)
    if prompt_repo.get_by_name(name):
        return AdminResult.fail(f"Template-Name '{name}' existiert bereits")
    with db_query() as db:
        existing_any = db.execute(
            "SELECT id FROM prompt_templates WHERE name = %s", (name,)
        ).fetchone()
    if existing_any:
        return AdminResult.fail(f"Template-Name '{name}' existiert bereits")

    new_id = prompt_repo.create(
        name=name,
        label=label,
        description=description,
        content=content,
        final_prompt=final_prompt,
        chain_type=chain_type,
        is_cloud=int(is_cloud),
        show_in_chat=int(show_in_chat),
        badge_color=badge_color,
        sort_order=sort_order,
        is_system=0,
        cloud_system_prompt=cloud_system_prompt,
    )

    return AdminResult.ok(
        f"Template '{name}' erstellt",
        data={"id": new_id, "message": "Template erstellt", "name": label or name},
    )


@admin_function(
    name="ki.templates.update",
    category="ki",
    description="Prompt-Template aktualisieren",
    help="Aktualisiert ein bestehendes Prompt-Template. Nur uebergebene Felder "
    "werden geaendert. Bei System-Templates kann chain_type nicht geaendert "
    "werden. Content wird gegen die chain_type-Platzhalter validiert. "
    "enrichment → {chunk_text} ist Pflicht.",
    examples=[
        "Aendere den Content von Template 3",
        "Aktiviere Template 5 fuer den Chat",
    ],
    parameters={
        "type": "object",
        "properties": {
            "template_id": {"type": "integer", "description": "Template-ID"},
            "label": {"type": "string", "description": "Anzeige-Label"},
            "description": {"type": "string", "description": "Beschreibung"},
            "content": {"type": "string", "description": "Template-Inhalt"},
            "final_prompt": {"type": "string", "description": "Finaler Prompt"},
            "cloud_system_prompt": {
                "type": "string",
                "description": "System-Prompt fuer Cloud-AI-Backends",
            },
            "chain_type": {
                "type": "string",
                "description": "Typ. Fuer Custom-Templates editierbar: standard, unchained, agent, enrichment. System-Templates behalten ihren festen Typ.",
                "enum": ["standard", "unchained", "agent", "rewriter", "admin_agent", "enrichment"],
            },
            "is_cloud": {"type": "boolean", "description": "Cloud-Template"},
            "show_in_chat": {"type": "boolean", "description": "Im Chat anzeigen"},
            "badge_color": {"type": "string", "description": "Badge-Farbe"},
            "sort_order": {"type": "integer", "description": "Sortierung"},
            "is_active": {"type": "boolean", "description": "Aktiv-Status"},
        },
        "required": ["template_id"],
    },
    returns="Bestaetigungsmeldung",
    pk_param="template_id",
    changelog_area="prompt_templates",
    changelog_action="update",
    raw_response=True,
    http=("PUT", "/api/admin/ki/prompt-templates/{template_id}"),
)
async def ki_templates_update(
    user: dict,
    template_id: int,
    label: str | None = None,
    description: str | None = None,
    content: str | None = None,
    final_prompt: str | None = None,
    cloud_system_prompt: str | None = None,
    chain_type: str | None = None,
    is_cloud: bool | None = None,
    show_in_chat: bool | None = None,
    badge_color: str | None = None,
    sort_order: int | None = None,
    is_active: bool | None = None,
) -> AdminResult:
    from fastapi import HTTPException

    from kiara.repositories.base import RawSQL
    from kiara.repositories.prompt import PromptTemplateRepository
    from kiara.shared.db import db_transaction
    from kiara.web.api_ki import _validate_template_content

    prompt_repo = PromptTemplateRepository()

    all_chain_types = ("standard", "unchained", "agent", "rewriter", "admin_agent", "enrichment")
    custom_editable_chain_types = ("standard", "unchained", "agent", "enrichment")
    chat_visible_chain_types = ("standard", "unchained", "agent", "rewriter")

    with db_transaction() as tx:
        row = tx.execute(
            "SELECT * FROM prompt_templates WHERE id = %s FOR UPDATE",
            (template_id,),
        ).fetchone()
        if not row:
            return AdminResult.fail("Template nicht gefunden")
        template = dict(row)
        is_system = bool(template.get("is_system", 0))

        if chain_type is not None and chain_type not in all_chain_types:
            return AdminResult.fail(f"Ungueltiger chain_type: {chain_type}")

        if is_system and chain_type is not None and chain_type != template["chain_type"]:
            return AdminResult.fail("chain_type von System-Templates kann nicht geaendert werden")
        if (
            not is_system
            and chain_type is not None
            and chain_type not in custom_editable_chain_types
        ):
            return AdminResult.fail(
                f"chain_type '{chain_type}' ist fuer System-Templates reserviert"
            )

        effective_chain_type = chain_type if chain_type is not None else template["chain_type"]
        effective_is_cloud = is_cloud if is_cloud is not None else bool(template.get("is_cloud", 0))
        if show_in_chat is True and (
            effective_chain_type not in chat_visible_chain_types or effective_is_cloud
        ):
            return AdminResult.fail("Dieses Template darf nicht als Chat-Modus angezeigt werden")

        if content is not None or chain_type is not None:
            effective_content = content if content is not None else template["content"]
            try:
                _validate_template_content(effective_content, effective_chain_type)
            except HTTPException as exc:
                return AdminResult.fail(exc.detail)

        update_fields = {}
        for field_name, value in [
            ("label", label),
            ("description", description),
            ("content", content),
            ("final_prompt", final_prompt),
            ("cloud_system_prompt", cloud_system_prompt),
            ("chain_type", chain_type if not is_system else None),
            ("is_cloud", bool(is_cloud) if is_cloud is not None else None),
            ("show_in_chat", int(show_in_chat) if show_in_chat is not None else None),
            ("badge_color", badge_color),
            ("sort_order", sort_order),
            ("is_active", bool(is_active) if is_active is not None else None),
        ]:
            if value is not None:
                update_fields[field_name] = value

        if bool(template.get("show_in_chat", 0)) and (
            effective_chain_type not in chat_visible_chain_types or effective_is_cloud
        ):
            update_fields["show_in_chat"] = 0

        if not update_fields:
            return AdminResult.ok("Keine Aenderungen", data={"message": "Keine Aenderungen"})

        content_changed = (
            "content" in update_fields and update_fields["content"] != template["content"]
        )
        csp_changed = "cloud_system_prompt" in update_fields and update_fields[
            "cloud_system_prompt"
        ] != (template.get("cloud_system_prompt") or "")
        update_fields["updated_at"] = RawSQL("NOW()")

        if content_changed or csp_changed:
            prompt_repo.save_history(
                template_id,
                template["content"],
                cloud_system_prompt=template.get("cloud_system_prompt", ""),
                changed_by=user.get("username", ""),
                conn=tx,
            )
        prompt_repo.update(template_id, **update_fields, conn=tx)

    return AdminResult.ok(
        f"Template {template_id} aktualisiert",
        data={
            "message": "Template aktualisiert",
            "name": template.get("label") or template.get("name", ""),
        },
    )


@admin_function(
    name="ki.templates.delete",
    category="ki",
    description="Prompt-Template loeschen",
    help="Loescht ein Prompt-Template. System-Templates (is_system=1) koennen "
    "nicht geloescht werden.",
    examples=[
        "Loesche Template 5",
        "Entferne das Template 'Test-Modus'",
    ],
    parameters={
        "type": "object",
        "properties": {
            "template_id": {"type": "integer", "description": "Template-ID"},
        },
        "required": ["template_id"],
    },
    returns="Loeschbestaetigung",
    requires_confirmation=True,
    confirm_template="template {template_id} löschen",
    pk_param="template_id",
    changelog_area="prompt_templates",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/ki/prompt-templates/{template_id}"),
)
async def ki_templates_delete(user: dict, template_id: int) -> AdminResult:
    from kiara.repositories.prompt import PromptTemplateRepository
    from kiara.shared.db import db_transaction

    prompt_repo = PromptTemplateRepository()

    with db_transaction() as tx:
        row = tx.execute(
            "SELECT * FROM prompt_templates WHERE id = %s FOR UPDATE",
            (template_id,),
        ).fetchone()
        if not row:
            return AdminResult.fail("Template nicht gefunden")
        template = dict(row)

        if bool(template.get("is_system", 0)):
            return AdminResult.fail("System-Templates koennen nicht geloescht werden")

        has_history = tx.execute(
            "SELECT 1 FROM prompt_template_history WHERE template_id = %s LIMIT 1",
            (template_id,),
        ).fetchone()
        if has_history:
            return AdminResult.fail("Versionierte Templates koennen nicht geloescht werden")

        prompt_repo.delete(template_id, conn=tx)
    return AdminResult.ok(
        f"Template {template_id} geloescht",
        data={
            "message": "Template geloescht",
            "name": template.get("label") or template.get("name", ""),
        },
    )


@admin_function(
    name="ki.templates.history",
    category="ki",
    description="History eines Prompt-Templates anzeigen",
    help="Zeigt die Aenderungshistorie eines Prompt-Templates an.",
    examples=["Zeig mir die History von Template 1"],
    parameters={
        "type": "object",
        "properties": {
            "template_id": {"type": "integer", "description": "Template-ID"},
        },
        "required": ["template_id"],
    },
    returns="Liste der History-Eintraege",
    raw_response=True,
    http=("GET", "/api/admin/ki/prompt-templates/{template_id}/history"),
)
async def ki_templates_history(user: dict, template_id: int) -> AdminResult:
    """History-Eintraege fuer ein Template."""
    from kiara.repositories.prompt import PromptTemplateRepository

    prompt_repo = PromptTemplateRepository()
    template = prompt_repo.get_by_id(template_id)
    if not template:
        return AdminResult.fail("Template nicht gefunden")
    entries = prompt_repo.get_history(template_id)
    return AdminResult.ok("History", data={"data": entries})


@admin_function(
    name="ki.templates.rollback",
    category="ki",
    description="Stellt einen frueheren versionierten Prompt-Stand wieder her",
    help="Laedt einen History-Eintrag und ueberschreibt content und "
    "cloud_system_prompt des Templates. Der aktuelle Stand wird vorher "
    "als History-Eintrag archiviert.",
    examples=["Setze Template 7 auf History-Eintrag 42 zurueck"],
    parameters={
        "type": "object",
        "properties": {
            "template_id": {"type": "integer", "description": "Template-ID"},
            "history_id": {"type": "integer", "description": "History-Eintrag-ID"},
        },
        "required": ["template_id", "history_id"],
    },
    returns="Aktualisiertes Template inkl. Bestaetigung",
    requires_confirmation=True,
    confirm_template="template {template_id} auf Version {history_id} zuruecksetzen",
    pk_param="template_id",
    changelog_area="prompt_templates",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/ki/prompt-templates/{template_id}/rollback/{history_id}"),
)
async def ki_templates_rollback(user: dict, template_id: int, history_id: int) -> AdminResult:
    from fastapi import HTTPException

    from kiara.repositories.base import RawSQL
    from kiara.repositories.prompt import PromptTemplateRepository
    from kiara.shared.db import db_transaction
    from kiara.web.api_ki import _validate_template_content

    prompt_repo = PromptTemplateRepository()

    with db_transaction() as tx:
        # 1. Template laden und sperren
        row = tx.execute(
            "SELECT * FROM prompt_templates WHERE id = %s FOR UPDATE",
            (template_id,),
        ).fetchone()
        if not row:
            return AdminResult.fail("Template nicht gefunden")
        template = dict(row)

        # 2. History-Eintrag laden und Zugehoerigkeit pruefen
        hist_row = tx.execute(
            "SELECT * FROM prompt_template_history WHERE id = %s",
            (history_id,),
        ).fetchone()
        if not hist_row:
            return AdminResult.fail(f"History-Eintrag {history_id} nicht gefunden")
        history = dict(hist_row)
        if history["template_id"] != template_id:
            return AdminResult.fail(
                f"History-Eintrag {history_id} gehoert nicht zu Template {template_id}"
            )

        # 2b. History-Content gegen aktuellen chain_type validieren
        try:
            _validate_template_content(history["content"], template["chain_type"])
        except HTTPException as exc:
            return AdminResult.fail(f"Rollback abgebrochen: {exc.detail}")

        # 3. Aktuellen Stand archivieren
        prompt_repo.save_history(
            template_id,
            template["content"],
            cloud_system_prompt=template.get("cloud_system_prompt", ""),
            changed_by=user.get("username", ""),
            change_reason=f"rollback auf History {history_id}",
            conn=tx,
        )

        # 4. Template mit History-Daten ueberschreiben
        prompt_repo.update(
            template_id,
            content=history["content"],
            cloud_system_prompt=history.get("cloud_system_prompt", ""),
            updated_at=RawSQL("NOW()"),
            conn=tx,
        )

    return AdminResult.ok(
        f"Template {template_id} auf Version {history_id} zurueckgesetzt",
        data={
            "message": "Template zurueckgesetzt",
            "name": template.get("label") or template.get("name", ""),
            "history_id": history_id,
        },
    )


@admin_function(
    name="ki.enrichment.test",
    category="ki",
    description="Testet einen Enrichment-Prompt mit Beispiel-Text",
    help="Laedt ein Enrichment-Template, rendert es mit Beispieltext und fuehrt "
    "einen Test-Call gegen ein explizites oder das Standard-LLM-Backend aus.",
    examples=[
        "Teste das Enrichment-Template 7 mit einem Beispieltext",
        "Fuehre einen Prompt-Test fuer Template 12 mit Backend 4 aus",
    ],
    parameters={
        "type": "object",
        "properties": {
            "template_id": {"type": "integer", "description": "Prompt-Template-ID"},
            "sample_text": {"type": "string", "description": "Beispieltext"},
            "backend_id": {"type": "integer", "description": "Optionale Backend-ID (llm/cloud)"},
        },
        "required": ["template_id", "sample_text"],
    },
    returns="Prefix, enriched_text, gerenderter Prompt, Backend-Info und Dauer",
    raw_response=True,
    http=("POST", "/api/admin/ki/enrichment/test"),
)
async def ki_enrichment_test(
    user: dict, template_id: int, sample_text: str, backend_id: int | None = None
) -> AdminResult:
    import time

    from kiara.config import load_config
    from kiara.ingestion.enrichment import _render_prompt, enrich_chunk
    from kiara.repositories.prompt import PromptTemplateRepository

    prompt_repo = PromptTemplateRepository()
    template = prompt_repo.get_by_id(template_id)
    if not template:
        return AdminResult.fail(f"Template {template_id} nicht gefunden")
    if template.get("chain_type") != "enrichment":
        return AdminResult.fail(
            f"Template {template_id} ist kein Enrichment-Template "
            f"(chain_type={template.get('chain_type')})"
        )

    config = load_config()

    # Backend erstellen
    from kiara.generation.backends.factory import create_backend, wrap_with_anonymization

    if backend_id is not None:
        from kiara.backend_selector import get_backend_by_id

        backend_dict = get_backend_by_id(backend_id)
        if not backend_dict:
            return AdminResult.fail(f"Backend {backend_id} nicht gefunden oder inaktiv")
        cat = backend_dict.get("category", "")
        if cat not in ("llm", "cloud"):
            return AdminResult.fail(
                f"Backend {backend_id} hat Kategorie '{cat}' — nur llm/cloud erlaubt"
            )
        backend = create_backend(config, backend_id)
        # Cloud + untrusted: Anonymisierung wrappen
        if cat == "cloud" and not backend_dict.get("trusted", False):
            backend = wrap_with_anonymization(backend, config)
    else:
        backend = create_backend(config)

    backend_info = f"{backend.provider}/{backend.model}"

    # Prompt rendern (fuer Anzeige)
    prompt_used = _render_prompt(
        template_content=template["content"],
        document_context=sample_text,
        chunk_text=sample_text,
    )

    # Enrichment ausfuehren
    t0 = time.monotonic()
    try:
        enriched_text = enrich_chunk(
            chunk_text=sample_text,
            chunk_index=0,
            full_text=sample_text,
            all_chunk_texts=[sample_text],
            llm_generate_fn=backend.generate,
            prompt_template=template["content"],
        )
    except Exception as exc:
        return AdminResult.fail(f"Enrichment fehlgeschlagen: {exc}")
    duration_ms = round((time.monotonic() - t0) * 1000)

    # Prefix extrahieren
    if enriched_text != sample_text and "\n\n" in enriched_text:
        prefix = enriched_text[: enriched_text.index("\n\n" + sample_text)]
    else:
        prefix = ""

    return AdminResult.ok(
        "Enrichment-Test abgeschlossen",
        data={
            "prefix": prefix,
            "enriched_text": enriched_text,
            "prompt_used": prompt_used,
            "backend_used": backend_info,
            "duration_ms": duration_ms,
        },
    )

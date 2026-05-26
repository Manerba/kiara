from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult


@admin_function(
    name="features.ddu_available",
    category="features",
    description="DDU-Backend (Docling) verfügbar?",
    help="Prüft ob mindestens ein aktives DDU-Backend (Deep Document Understanding) "
    "erreichbar ist. Nutzt den gecachten Status (30s TTL).",
    examples=[
        "Ist DDU verfügbar?",
        "Kann ich Docling nutzen?",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="DDU-Verfügbarkeitsstatus",
)
async def features_ddu(user: dict) -> AdminResult:
    from kiara.web.api_datasource_features import ddu_available

    data = await ddu_available(user=user)
    return AdminResult.ok("DDU-Status", data=data)


@admin_function(
    name="features.vlm_available",
    category="features",
    description="VLM-Modell (Vision) verfügbar?",
    help="Prüft ob ein Vision Language Model für Bildbeschreibungen "
    "verfügbar ist. Nutzt den gecachten Status (30s TTL).",
    examples=[
        "Ist VLM verfügbar?",
        "Kann ich Bilder verarbeiten?",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="VLM-Verfügbarkeitsstatus",
)
async def features_vlm(user: dict) -> AdminResult:
    from kiara.web.api_datasource_features import vlm_available

    data = await vlm_available(user=user)
    return AdminResult.ok("VLM-Status", data=data)


@admin_function(
    name="features.embedding_models",
    category="features",
    description="Verfügbare Embedding-Modelle",
    help="Listet alle verfügbaren Embedding-Modelle auf und zeigt welches "
    "aktuell aktiv ist. Nutzt den gecachten Status (30s TTL).",
    examples=[
        "Welche Embedding-Modelle gibt es?",
        "Verfügbare Modelle für Vektorisierung",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Liste der Embedding-Modelle mit Aktivitätsstatus",
)
async def features_embedding_models(user: dict) -> AdminResult:
    from kiara.web.api_datasource_features import list_embedding_models

    data = await list_embedding_models(user=user)
    return AdminResult.ok("Embedding-Modelle", data=data)

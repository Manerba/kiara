"""Exact CAD facts for the existing chat agent and source cards."""

import asyncio
import json

from kiara.tools.base import BaseTool, ToolDefinition, ToolResult


class CadQueryTool(BaseTool):
    def __init__(self, config, allowed_datasource_ids, embedding_model=None):
        self.config = config
        self.allowed = allowed_datasource_ids
        self.embedding_model = embedding_model

    def definition(self):
        return ToolDefinition(
            name="cad_query",
            description=(
                "Exakte CAD-Abfragen: search findet Bauteile/Baugruppen nach Name oder Eigenschaften. "
                "Danach object, bom (Stückliste) oder where_used (Einbaupfade) mit model_id aus dem Treffer. "
                "object/where_used benötigen object_id. source_hash aus dem Treffer bindet die Dateiversion. "
                "bom recursive=true zählt Teile, false direkte Kinder; occurrence_id begrenzt auf ein konkretes Vorkommen. "
                "Zahlen: dichte_g_cm3 ist gespeicherte CAD-Dichte, blechdicke_mm die gespeicherte Blechdicke. "
                "minimum/maximum sind inklusive; minimum_exclusive/maximum_exclusive setzen strikte Grenzen. "
                "numeric_unit erlaubt g/cm³ oder kg/m³ für Dichte und mm/cm/m/in für Blechdicke; "
                "ohne Einheit gelten g/cm³ bzw. mm. Unbekannte Werte werden ausgelassen. "
                "property_name/property_value vergleichen exakt, führende Nullen bleiben. "
                "Für semantische Kandidaten zuerst search_documents verwenden. Keine Mengen aus Suchtexten schätzen. "
                "Bei has_more mit next_offset weiterblättern; Ergebnisse nicht über Dateiversionen summieren."
            ),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["search", "object", "bom", "where_used"],
                    },
                    **{
                        k: {"type": "string"}
                        for k in (
                            "query",
                            "model_id",
                            "object_id",
                            "occurrence_id",
                            "source_hash",
                            "property_name",
                            "property_namespace",
                            "property_value",
                        )
                    },
                    "numeric_field": {"type": "string", "enum": ["dichte_g_cm3", "blechdicke_mm"]},
                    "minimum": {"type": "number"},
                    "maximum": {"type": "number"},
                    "numeric_unit": {"type": "string"},
                    "minimum_exclusive": {"type": "boolean"},
                    "maximum_exclusive": {"type": "boolean"},
                    "recursive": {"type": "boolean"},
                    "datasource_id": {"type": "integer"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                    "offset": {"type": "integer", "minimum": 0},
                },
                "required": ["operation"],
            },
            category="builtin",
        )

    def _query(self, params):
        from kiara.cad.repository import query
        from kiara.repositories.datasource import DatasourceRepository

        allowed = self.allowed
        if allowed is None:
            allowed = [d["id"] for d in DatasourceRepository().get_active()]
        selected = params.get("datasource_id")
        if selected is not None:
            if type(selected) is not int or selected not in allowed:
                raise ValueError("Datenquelle ist nicht erlaubt")
            allowed = [selected]
        return query(
            params,
            allowed_datasource_ids=allowed,
        )

    async def execute(self, params, user_id=None):
        if error := self._validate_or_error(params):
            return error
        try:
            result = await asyncio.to_thread(self._query, params)
        except (ValueError, LookupError) as exc:
            return ToolResult(False, None, str(exc), str(exc))
        result["next_offset"] = params.get("offset", 0) + len(result["items"])
        sources = []
        for item in result["items"]:
            sources.append(
                {
                    "source": item["file_path"],
                    "score": 1.0,
                    "excerpt": "CAD "
                    + result["operation"]
                    + ": "
                    + json.dumps(
                        {
                            k: item[k]
                            for k in ("names", "path", "quantity", "source_hash", "object_id")
                            if k in item
                        },
                        ensure_ascii=False,
                    ),
                    "cad_object_id": item.get("object_id"),
                    "cad_source_hash": item["source_hash"],
                }
            )
        # The existing chat adapter applies cloud routing/anonymization to this
        # complete display before truncation; MCP uses RetrievalOnlyService.
        return ToolResult(
            True, {**result, "sources": sources}, None, json.dumps(result, ensure_ascii=False)
        )

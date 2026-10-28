"""Konfigurierbare Grenzen der ersten sequenziellen Recherche-Ausbaustufe."""

DEFAULT_RESEARCH = {
    "enabled": True,
    "num_ctx": 32768,
    "output_tokens": 2048,
    "safety_tokens": 1024,
    "bytes_per_token": 2.0,
    "max_rounds": 3,
    "reads_per_round": 10,
    "max_model_calls": 40,
    "max_seconds": 600,
    "call_timeout": 60,
    "page_chars": 12000,
    "read_chunks": 4,
    "max_results": 120,
    "store_bytes": 2000000,
    "max_evidence": 8,
    "max_parallel_inference": 1,
    "max_active_runs": 4,
    "keyword_match_operator": "and",
    "keyword_content_only": True,
    "keyword_fuzziness": "none",
    "lock_dir": "/var/lib/kiara/research-locks",
    "mcp_ttl_seconds": 600,
    "mcp_max_sessions": 16,
    "mcp_max_output_chars": 20000,
}


def research_settings(config):
    values = {**DEFAULT_RESEARCH, **config.get("chat", {}).get("research", {})}
    if values.keys() - DEFAULT_RESEARCH.keys():
        raise ValueError("Unbekannte Recherche-Einstellung")
    if type(values["enabled"]) is not bool:
        raise ValueError("chat.research.enabled muss boolesch sein")
    for key, default in DEFAULT_RESEARCH.items():
        value = values[key]
        if type(default) is int and (type(value) is not int or value < 1):
            raise ValueError(f"chat.research.{key} muss eine positive ganze Zahl sein")
    if (
        not isinstance(values["bytes_per_token"], (int, float))
        or not 1 <= values["bytes_per_token"] <= 4
    ):
        raise ValueError("chat.research.bytes_per_token muss zwischen 1 und 4 liegen")
    if values["max_parallel_inference"] != 1:
        raise ValueError("Die erste Recherche-Ausbaustufe erlaubt genau einen Inferenzaufruf")
    if values["keyword_match_operator"] not in {"and", "or"}:
        raise ValueError("chat.research.keyword_match_operator muss and oder or sein")
    if type(values["keyword_content_only"]) is not bool:
        raise ValueError("chat.research.keyword_content_only muss boolesch sein")
    if values["keyword_fuzziness"] not in {"none", "AUTO", "1", "2"}:
        raise ValueError("chat.research.keyword_fuzziness ist ungueltig")
    if not isinstance(values["lock_dir"], str) or not values["lock_dir"].startswith("/"):
        raise ValueError("chat.research.lock_dir muss ein absoluter Pfad sein")
    if values["mcp_max_output_chars"] < 2000:
        raise ValueError("chat.research.mcp_max_output_chars muss mindestens 2000 sein")
    return values

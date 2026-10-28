"""KIara - Seed-Daten beim App-Startup.

Idempotente Initialisierungen (kein Migrations-Code). Wird von ``init_db()``
nach dem Schema-Apply aufgerufen.

- ``ensure_admin_user``: Default-Admin bei leerer users-Tabelle anlegen.
- ``ensure_ai_config``: ai_config-Zeile aus kiara.yaml vorbelegen wenn leer.
- ``ensure_ddu_local_backend``: Legacy-Noop; DDU ist remote-only.
"""

import logging

from kiara.shared.constants import DEFAULT_OLLAMA_URL

logger = logging.getLogger(__name__)


def ensure_admin_user(conn) -> None:
    """Default-Admin (username=admin, password=admin) bei leerer users-Tabelle."""
    row = conn.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()
    if row["cnt"] > 0:
        return
    import bcrypt

    hashed = bcrypt.hashpw("admin".encode(), bcrypt.gensalt()).decode()
    conn.execute(
        "INSERT INTO users (username, password_hash, vorname, nachname, is_admin, must_change_password) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        ("admin", hashed, "Administrator", "", True, True),
    )
    logger.info("Default-Admin angelegt (username=admin, password=admin, must_change_password=1)")


def ensure_ai_config(conn) -> None:
    """ai_config-Zeile aus kiara.yaml vorbelegen wenn Tabelle leer."""
    row = conn.execute("SELECT COUNT(*) AS cnt FROM ai_config").fetchone()
    if row["cnt"] > 0:
        return
    from kiara.config import load_config
    from kiara.shared.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL

    cfg = load_config()
    o = cfg.get("ollama", {})
    conn.execute(
        "INSERT INTO ai_config (id, embedding_url, embedding_model, llm_url, llm_model) "
        "VALUES (1, %s, %s, %s, %s)",
        (
            o.get("embedding_url") or o.get("url", DEFAULT_OLLAMA_URL),
            o.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            o.get("llm_url") or o.get("url", DEFAULT_OLLAMA_URL),
            o.get("llm_model", DEFAULT_LLM_MODEL),
        ),
    )
    logger.info("ai_config aus kiara.yaml seeded")


def ensure_ddu_local_backend(conn) -> None:
    """Legacy-Noop: KIara nutzt DDU nur noch remote ueber docling-serve."""
    return


def run_seeds(conn) -> None:
    """Fuehrt alle Seeding-Funktionen aus. Aufruf aus ``init_db()``."""
    ensure_admin_user(conn)
    ensure_ai_config(conn)
    ensure_ddu_local_backend(conn)

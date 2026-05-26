"""Admin-Funktionen fuer Mail-Konfiguration."""

import asyncio
import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult
from kiara.shared.constants import PASSWORD_MASK

logger = logging.getLogger(__name__)


@admin_function(
    name="mail.config.read",
    category="mail",
    description="SMTP-Konfiguration anzeigen",
    help="Zeigt die aktuelle Mail/SMTP-Konfiguration an. Passwörter werden maskiert.",
    examples=[
        "Zeig mir die Mail-Konfiguration",
        "Wie ist SMTP konfiguriert?",
        "Mail-Einstellungen?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="SMTP-Konfiguration (Passwort maskiert)",
    raw_response=True,
    http=("GET", "/api/admin/mail/config"),
)
async def mail_config_read(user: dict) -> AdminResult:
    from kiara.repositories.config import ConfigRepository

    repo = ConfigRepository()
    config = repo.get("mail_config")
    if not config:
        return AdminResult.ok("Keine Mail-Konfiguration vorhanden", data={"config": None})

    # Passwort maskieren
    if config.get("password"):
        config["password"] = PASSWORD_MASK
    # Boolean-Felder konvertieren (DB speichert als int)
    for key in (
        "notify_ollama_down",
        "notify_datasource_down",
        "notify_index_error",
        "status_report_enabled",
    ):
        if key in config:
            config[key] = bool(config[key])

    return AdminResult.ok("Mail-Konfiguration", data={"config": config})


@admin_function(
    name="mail.config.update",
    category="mail",
    description="SMTP-Konfiguration speichern",
    help="Speichert die Mail/SMTP-Konfiguration. Bei maskiertem Passwort "
    "(********) wird das bestehende Passwort beibehalten.",
    examples=[
        "Konfiguriere SMTP mit Server smtp.firma.de:587",
        "Mail-Konfiguration speichern",
    ],
    parameters={
        "type": "object",
        "properties": {
            "smtp_server": {
                "type": "string",
                "description": "SMTP-Server-Adresse",
            },
            "smtp_port": {
                "type": "integer",
                "description": "SMTP-Port (Standard: 587)",
                "default": 587,
            },
            "encryption": {
                "type": "string",
                "description": "Verschlüsselung (none, starttls, ssl)",
                "enum": ["none", "starttls", "ssl"],
                "default": "starttls",
            },
            "username": {
                "type": "string",
                "description": "SMTP-Benutzername",
                "default": "",
            },
            "password": {
                "type": "string",
                "description": "SMTP-Passwort (******** = bestehendes beibehalten)",
                "default": "",
            },
            "sender_address": {
                "type": "string",
                "description": "Absender-Adresse",
            },
            "recipient_addresses": {
                "type": "string",
                "description": "Empfänger-Adressen (kommagetrennt)",
            },
            "notify_ollama_down": {
                "type": "boolean",
                "description": "Benachrichtigung bei Ollama-Ausfall",
                "default": True,
            },
            "notify_datasource_down": {
                "type": "boolean",
                "description": "Benachrichtigung bei Datenquellen-Fehler",
                "default": True,
            },
            "notify_index_error": {
                "type": "boolean",
                "description": "Benachrichtigung bei Index-Fehler",
                "default": True,
            },
            "status_report_enabled": {
                "type": "boolean",
                "description": "Regelmäßigen Status-Report aktivieren",
                "default": False,
            },
            "status_report_interval": {
                "type": "string",
                "description": "Intervall für Status-Reports",
                "enum": ["daily", "weekly", "monthly"],
                "default": "weekly",
            },
        },
        "required": ["smtp_server", "sender_address", "recipient_addresses"],
    },
    returns="Bestätigung der Speicherung",
    requires_confirmation=True,
    confirm_template="mail-konfiguration speichern",
    changelog_area="mail",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/mail/config"),
)
async def mail_config_update(
    user: dict,
    smtp_server: str,
    sender_address: str,
    recipient_addresses: str,
    smtp_port: int = 587,
    encryption: str = "starttls",
    username: str = "",
    password: str = "",
    notify_ollama_down: bool = True,
    notify_datasource_down: bool = True,
    notify_index_error: bool = True,
    status_report_enabled: bool = False,
    status_report_interval: str = "weekly",
) -> AdminResult:
    from kiara.repositories.config import ConfigRepository
    from kiara.shared.crypto import encrypt

    repo = ConfigRepository()

    # PASSWORD_MASK Drei-Wege-Logik (Spec §5)
    effective_password = password
    if password == PASSWORD_MASK or not password:
        existing = repo.get("mail_config")
        if existing and existing.get("password"):
            effective_password = existing["password"]
        elif password == PASSWORD_MASK:
            return AdminResult.fail("Kein bestehendes Passwort vorhanden")
        else:
            effective_password = ""  # Leer bei Create ohne bestehendes PW
    else:
        effective_password = encrypt(password)

    repo.set(
        "mail_config",
        {
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "encryption": encryption,
            "username": username,
            "password": effective_password,
            "sender_address": sender_address,
            "recipient_addresses": recipient_addresses,
            "notify_ollama_down": int(notify_ollama_down),
            "notify_datasource_down": int(notify_datasource_down),
            "notify_index_error": int(notify_index_error),
            "status_report_enabled": int(status_report_enabled),
            "status_report_interval": status_report_interval,
        },
    )

    return AdminResult.ok(
        "Mail-Konfiguration gespeichert",
        data={"message": "Mail-Konfiguration gespeichert"},
    )


@admin_function(
    name="mail.config.delete",
    category="mail",
    description="SMTP-Konfiguration löschen",
    help="Löscht die gesamte Mail/SMTP-Konfiguration. "
    "Danach werden keine Benachrichtigungen mehr versendet.",
    examples=[
        "Lösche die Mail-Konfiguration",
        "SMTP-Einstellungen entfernen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestätigung der Löschung",
    requires_confirmation=True,
    confirm_template="mail-konfiguration löschen",
    changelog_area="mail",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/mail/config"),
)
async def mail_config_delete(user: dict) -> AdminResult:
    from kiara.repositories.config import ConfigRepository

    repo = ConfigRepository()
    repo.delete("mail_config")
    return AdminResult.ok(
        "Mail-Konfiguration gelöscht",
        data={"message": "Mail-Konfiguration gelöscht"},
    )


@admin_function(
    name="mail.test",
    category="mail",
    description="Test-Mail versenden",
    help="Sendet eine Test-Mail an die konfigurierten Empfänger. "
    "Prüft damit die SMTP-Verbindung und -Konfiguration.",
    examples=[
        "Sende eine Test-Mail",
        "Teste die SMTP-Verbindung",
        "Mail-Test durchführen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Ergebnis des Mailversands",
    raw_response=True,
    http=("POST", "/api/admin/mail/test"),
)
async def mail_test(user: dict) -> AdminResult:
    from kiara.repositories.config import ConfigRepository

    repo = ConfigRepository()
    row = repo.get("mail_config")
    if not row:
        return AdminResult.fail("Keine Mail-Konfiguration vorhanden")

    # Passwort entschlüsseln
    from cryptography.fernet import InvalidToken

    from kiara.shared.crypto import decrypt, is_encrypted

    password = row["password"]
    if password and is_encrypted(password):
        try:
            password = decrypt(password)
        except InvalidToken:
            return AdminResult.fail(
                "Passwort-Entschluesselung fehlgeschlagen: Schluessel-Mismatch (Secret-Key gewechselt?)"
            )
        except Exception as exc:
            return AdminResult.fail(f"Passwort-Entschluesselung fehlgeschlagen: {exc}")

    config = {
        "smtp_server": row["smtp_server"],
        "smtp_port": row["smtp_port"],
        "encryption": row["encryption"],
        "username": row["username"],
        "password": password,
        "sender_address": row["sender_address"],
        "recipient_addresses": row["recipient_addresses"],
    }

    html = """
    <html><body style="font-family:Arial,sans-serif;">
        <h3 style="color:#1e40af;">KIara Test-Mail</h3>
        <p>Diese E-Mail wurde als Test von der KIara Admin-Oberflaeche gesendet.</p>
        <p>Wenn Sie diese E-Mail erhalten, ist die SMTP-Konfiguration korrekt.</p>
    </body></html>
    """

    from kiara.web.mail_service import send_mail

    success, message = await asyncio.to_thread(send_mail, config, "KIara Test-Mail", html)
    if not success:
        return AdminResult.fail(f"Mail-Versand fehlgeschlagen: {message}")

    return AdminResult.ok(
        "Test-Mail versendet",
        data={"message": message},
    )


@admin_function(
    name="mail.send_report",
    category="mail",
    description="Status-Report per Mail versenden",
    help="Generiert einen aktuellen Status-Report und versendet ihn "
    "per Mail an die konfigurierten Empfänger.",
    examples=[
        "Sende einen Status-Report",
        "Status-Mail verschicken",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Ergebnis des Reportversands",
    raw_response=True,
    http=("POST", "/api/admin/mail/status-report"),
)
async def mail_send_report(user: dict) -> AdminResult:
    from kiara.repositories.config import ConfigRepository

    repo = ConfigRepository()
    row = repo.get("mail_config")
    if not row:
        return AdminResult.fail("Keine Mail-Konfiguration vorhanden")

    # Passwort entschlüsseln
    from cryptography.fernet import InvalidToken

    from kiara.shared.crypto import decrypt, is_encrypted

    password = row["password"]
    if password and is_encrypted(password):
        try:
            password = decrypt(password)
        except InvalidToken:
            return AdminResult.fail(
                "Passwort-Entschluesselung fehlgeschlagen: Schluessel-Mismatch (Secret-Key gewechselt?)"
            )
        except Exception as exc:
            return AdminResult.fail(f"Passwort-Entschluesselung fehlgeschlagen: {exc}")

    mail_config = {
        "smtp_server": row["smtp_server"],
        "smtp_port": row["smtp_port"],
        "encryption": row["encryption"],
        "username": row["username"],
        "password": password,
        "sender_address": row["sender_address"],
        "recipient_addresses": row["recipient_addresses"],
    }

    # Stats für Report aufbauen
    from kiara.config import load_config
    from kiara.shared.db import db_query

    cfg = load_config()

    # Ollama-Status (blockierend → asyncio.to_thread)
    ollama_url = cfg.get("ollama", {}).get("url", "")
    ollama_status = "offline"
    try:
        import httpx

        resp = await asyncio.to_thread(httpx.get, f"{ollama_url}/api/version", timeout=5)
        if resp.status_code == 200:
            ollama_status = "online"
    except Exception:
        pass

    # VectorDB-Stats
    try:
        from kiara.retrieval.vectorstore import VectorStore

        vs = VectorStore(cfg)
        vs_stats = vs.get_stats()
    except Exception:
        vs_stats = {}

    # User-Stats
    with db_query() as db:
        user_count = db.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()["cnt"]

    stats = {
        "ollama": {
            "status": ollama_status,
            "url": ollama_url,
            "llm_model": cfg.get("ollama", {}).get("llm_model", ""),
            "llm_available": ollama_status == "online",
        },
        "index": {
            "total_chunks": vs_stats.get("total_chunks", 0),
            "collection": vs_stats.get("collection", ""),
        },
        "datasource": {},
        "users": {"total": user_count, "active_sessions": 0},
    }

    from kiara.web.mail_service import build_status_report, send_mail

    try:
        report_html = build_status_report(stats)
        success, msg = await asyncio.to_thread(
            send_mail, mail_config, "KIara Status-Report", report_html
        )
        if not success:
            return AdminResult.fail(f"Report-Versand fehlgeschlagen: {msg}")
    except Exception as exc:
        return AdminResult.fail(f"Report-Versand fehlgeschlagen: {exc}")

    return AdminResult.ok(
        "Status-Report versendet",
        data={"message": f"Status-Report an {row['recipient_addresses']} versendet"},
    )

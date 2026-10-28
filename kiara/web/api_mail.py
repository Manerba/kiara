"""KIara Admin - Mail API.

SMTP-Konfiguration, Test-Mail-Versand und Status-Reports.
Die Config wird als Singleton (id=1) in der Tabelle ``mail_config`` gespeichert.

Beziehungen:
    - kiara.web.mail_service   → send_mail(), build_status_report()
    - kiara.web.config_utils   → get_masked_config, resolve_masked_field
    - kiara.web.api_dashboard  → Dashboard-Daten fuer Status-Reports
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from kiara.repositories.config import ConfigRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/mail", tags=["mail"])

config_repo = ConfigRepository()


class MailConfigRequest(BaseModel):
    """Request zum Speichern der SMTP-Mail-Konfiguration."""

    smtp_server: str
    smtp_port: int = 587
    encryption: str = "starttls"
    username: str = ""
    password: str = ""
    sender_address: str
    recipient_addresses: str
    notify_ollama_down: bool = True
    notify_datasource_down: bool = True
    notify_index_error: bool = True
    status_report_enabled: bool = False
    status_report_interval: str = "weekly"


# --- Migriert zu Admin-Registry: mail.config.read (Sprint 08) ---
# @router.get("/config")
# async def get_mail_config(user=Depends(get_admin_user)):
#     """Get mail configuration (password masked)."""
#     config = get_masked_config("mail_config", mask_fields=["password"])
#     if config:
#         for bool_field in ("notify_ollama_down", "notify_datasource_down",
#                            "notify_index_error", "status_report_enabled"):
#             config[bool_field] = bool(config.get(bool_field, 0))
#     return {"config": config}

# --- Migriert zu Admin-Registry: mail.config.update (Sprint 08) ---
# @router.post("/config")
# async def save_mail_config(data: MailConfigRequest, user=Depends(get_admin_user)):
#     """Save mail configuration."""
#     with db_query() as db:
#         password = resolve_masked_field(db, "mail_config", "password", data.password) if data.password else ""
#     fields = {
#         "smtp_server": data.smtp_server,
#         "smtp_port": data.smtp_port,
#         "encryption": data.encryption,
#         "username": data.username,
#         "password": password,
#         "sender_address": data.sender_address,
#         "recipient_addresses": data.recipient_addresses,
#         "notify_ollama_down": int(data.notify_ollama_down),
#         "notify_datasource_down": int(data.notify_datasource_down),
#         "notify_index_error": int(data.notify_index_error),
#         "status_report_enabled": int(data.status_report_enabled),
#         "status_report_interval": data.status_report_interval,
#     }
#     config_repo.set("mail_config", fields)
#     logger.info("Mail config saved by %s", user["username"])
#     return {"message": "Mail-Konfiguration gespeichert"}

# --- Migriert zu Admin-Registry: mail.config.delete (Sprint 08) ---
# @router.delete("/config")
# async def delete_mail_config(user=Depends(get_admin_user)):
#     """Delete mail configuration."""
#     config_repo.delete("mail_config")
#     return {"message": "Mail-Konfiguration geloescht"}

# --- Migriert zu Admin-Registry: mail.test (Sprint 08) ---
# @router.post("/test")
# async def test_mail(user=Depends(get_admin_user)):
#     """Send a test email using the saved configuration."""
#     from kiara.web.mail_service import send_mail
#     row = config_repo.get("mail_config")
#     if not row:
#         raise HTTPException(status_code=404, detail="Mail nicht konfiguriert")
#     password = row["password"]
#     if password and is_encrypted(password):
#         password = decrypt(password)
#     config = {
#         "smtp_server": row["smtp_server"],
#         "smtp_port": row["smtp_port"],
#         "encryption": row["encryption"],
#         "username": row["username"],
#         "password": password,
#         "sender_address": row["sender_address"],
#         "recipient_addresses": row["recipient_addresses"],
#     }
#     html = """
#     <html><body style="font-family:Arial,sans-serif;">
#         <h3 style="color:#1e40af;">KIara Test-Mail</h3>
#         <p>Diese E-Mail wurde als Test von der KIara Admin-Oberflaeche gesendet.</p>
#         <p>Wenn Sie diese E-Mail erhalten, ist die SMTP-Konfiguration korrekt.</p>
#     </body></html>
#     """
#     success, message = send_mail(config, "KIara Test-Mail", html)
#     if not success:
#         raise HTTPException(status_code=500, detail=message)
#     return {"message": message}

# --- Migriert zu Admin-Registry: mail.send_report (Sprint 08) ---
# @router.post("/status-report")
# async def send_status_report(user=Depends(get_admin_user)):
#     """Send a status report email."""
#     from kiara.web.mail_service import send_mail, build_status_report
#     from kiara.web.api_dashboard import _check_ollama, _get_index_status, _get_datasource_status, _get_user_stats
#     from kiara.config import load_config
#     row = config_repo.get("mail_config")
#     if not row:
#         raise HTTPException(status_code=404, detail="Mail nicht konfiguriert")
#     password = row["password"]
#     if password and is_encrypted(password):
#         password = decrypt(password)
#     mail_config = {
#         "smtp_server": row["smtp_server"],
#         "smtp_port": row["smtp_port"],
#         "encryption": row["encryption"],
#         "username": row["username"],
#         "password": password,
#         "sender_address": row["sender_address"],
#         "recipient_addresses": row["recipient_addresses"],
#     }
#     config = load_config()
#     stats = {
#         "ollama": await _check_ollama(config),
#         "index": _get_index_status(config),
#         "datasource": _get_datasource_status(config),
#         "users": _get_user_stats(),
#     }
#     html = build_status_report(stats)
#     success, message = send_mail(mail_config, "KIara Status-Report", html)
#     if not success:
#         raise HTTPException(status_code=500, detail=message)
#     return {"message": "Status-Report gesendet"}

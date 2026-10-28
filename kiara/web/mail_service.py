"""
KIara - Mail Service
SMTP-Client und HTML-Reports.
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr

from kiara.config import DEFAULT_TIMEOUTS
from kiara.shared.constants import SMTP_PORT_STARTTLS

logger = logging.getLogger(__name__)


def send_mail(config: dict, subject: str, html_body: str) -> tuple[bool, str]:
    """Send an email using the given SMTP configuration.

    Args:
        config: Dict with smtp_server, smtp_port, encryption, username, password,
                sender_address, recipient_addresses (comma-separated string).
        subject: Email subject.
        html_body: HTML body content.

    Returns:
        Tuple of (success, message).
    """
    try:
        subject = subject.replace("\r", " ").replace("\n", " ")
        sender_address = config["sender_address"].replace("\r", " ").replace("\n", " ")
        _, sender_addr = parseaddr(sender_address)
        if not sender_addr or "@" not in sender_addr:
            return False, "Ungültige Absender-Adresse"
        recipient_addresses = config["recipient_addresses"].replace("\r", " ").replace("\n", " ")

        recipients = []
        for raw in recipient_addresses.split(","):
            raw = raw.strip()
            if not raw:
                continue
            _, addr = parseaddr(raw)
            if addr and "@" in addr:
                recipients.append(addr)
            else:
                logger.warning("Ungültige E-Mail-Adresse übersprungen: %s", raw)
        if not recipients:
            return False, "Keine gültigen Empfänger-Adressen"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender_address
        # Bcc-Leak-Schutz: bei Mehrfach-Empfaengern Sender im To, Verteilung
        # erfolgt ueber SMTP-Envelope. KEIN Bcc-Header, da msg.as_string() den
        # Header serialisieren wuerde und dieser MUAs/MTAs erreichen koennte.
        if len(recipients) > 1:
            msg["To"] = sender_address
        else:
            msg["To"] = recipient_addresses

        msg.attach(MIMEText(html_body, "html", "utf-8"))

        encryption = config.get("encryption", "starttls")
        server = config["smtp_server"]
        port = config.get("smtp_port", SMTP_PORT_STARTTLS)

        if encryption == "ssl":
            smtp = smtplib.SMTP_SSL(server, port, timeout=DEFAULT_TIMEOUTS["smtp"])
        else:
            smtp = smtplib.SMTP(server, port, timeout=DEFAULT_TIMEOUTS["smtp"])
        try:
            if encryption == "starttls":
                smtp.starttls()

            if config.get("username") and config.get("password"):
                if encryption not in ("starttls", "ssl"):
                    raise RuntimeError(
                        f"SMTP-Login ohne Verschluesselung (encryption={encryption}) verweigert: "
                        "Credentials wuerden im Klartext uebertragen. "
                        "Fuer Login 'starttls' oder 'ssl' konfigurieren."
                    )
                smtp.login(config["username"], config["password"])

            smtp.sendmail(sender_addr, recipients, msg.as_string())
        finally:
            smtp.quit()

        logger.info("Mail sent: %s -> %d recipient(s)", subject, len(recipients))
        return True, "E-Mail gesendet"
    except smtplib.SMTPAuthenticationError:
        return False, "SMTP-Authentifizierung fehlgeschlagen"
    except smtplib.SMTPConnectError:
        return False, "SMTP-Verbindung fehlgeschlagen"
    except Exception as exc:
        logger.error("Mail send failed: %s", exc)
        return False, str(exc)


def build_status_report(stats: dict) -> str:
    """Build an HTML status report."""
    ollama = stats.get("ollama", {})
    index = stats.get("index", {})
    datasource = stats.get("datasource", {})
    users = stats.get("users", {})

    ollama_color = "#22c55e" if ollama.get("status") == "online" else "#ef4444"
    ds_color = "#22c55e" if datasource.get("accessible") else "#ef4444"

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2 style="color: #1e40af;">KIara Status-Report</h2>

        <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
            <tr style="background:#f8fafc;">
                <td style="padding:8px; border:1px solid #e2e8f0;"><strong>Ollama</strong></td>
                <td style="padding:8px; border:1px solid #e2e8f0;">
                    <span style="color:{ollama_color};">{ollama.get("status", "unbekannt")}</span>
                    ({ollama.get("url", "-")})
                </td>
            </tr>
            <tr>
                <td style="padding:8px; border:1px solid #e2e8f0;"><strong>LLM-Modell</strong></td>
                <td style="padding:8px; border:1px solid #e2e8f0;">
                    {ollama.get("llm_model", "-")}
                    {"&#x2705;" if ollama.get("llm_available") else "&#x274C;"}
                </td>
            </tr>
            <tr style="background:#f8fafc;">
                <td style="padding:8px; border:1px solid #e2e8f0;"><strong>Vektor-Index</strong></td>
                <td style="padding:8px; border:1px solid #e2e8f0;">
                    {index.get("total_chunks", 0)} Chunks
                    ({index.get("collection", "-")})
                </td>
            </tr>
            <tr>
                <td style="padding:8px; border:1px solid #e2e8f0;"><strong>Datenquelle</strong></td>
                <td style="padding:8px; border:1px solid #e2e8f0;">
                    <span style="color:{ds_color};">{"Erreichbar" if datasource.get("accessible") else "Nicht erreichbar"}</span>
                    ({datasource.get("path", "-")}, {datasource.get("file_count", 0)} Dateien)
                </td>
            </tr>
            <tr style="background:#f8fafc;">
                <td style="padding:8px; border:1px solid #e2e8f0;"><strong>Benutzer</strong></td>
                <td style="padding:8px; border:1px solid #e2e8f0;">
                    {users.get("total", 0)} gesamt, {users.get("active_sessions", 0)} aktive Sessions
                </td>
            </tr>
        </table>

        <p style="font-size:12px; color:#94a3b8;">
            Automatisch generiert von KIara
        </p>
    </body>
    </html>
    """
    return html

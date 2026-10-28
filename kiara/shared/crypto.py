"""Verschluesselung fuer Credentials und Secret-Key-Verwaltung.

Stellt Fernet-basierte Ver-/Entschluesselung bereit sowie die
Verwaltung des Applikations-Secrets (Lesen/Erstellen).

Der Schluessel wird deterministisch abgeleitet:
``secret → SHA-256 → base64url → Fernet``.
Verschluesselte Werte beginnen mit ``gAAAAA`` (Fernet-Token-Prefix).

Dieses Modul ist die **Single Source of Truth** fuer alle
Verschluesselungs-Operationen und das Applikations-Secret.

Beziehungen:
    - kiara.web.crypto_utils → Re-Export (Rueckwaertskompatibilitaet)
    - kiara.web.auth          → importiert get_secret_key()
    - kiara.web.config_utils  → importiert encrypt/decrypt/is_encrypted
    - kiara.web.api_*         → importieren encrypt/decrypt/is_encrypted
    - kiara.ingestion.pipeline → importiert decrypt/is_encrypted (lazy)
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import tempfile
import time
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from kiara.config import DEFAULT_PATHS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Secret-Key-Verwaltung
# ---------------------------------------------------------------------------

SECRET_KEY_PATH: Path = Path(DEFAULT_PATHS["secret_key"])
"""Pfad zur Secret-Key-Datei (/var/lib/kiara/secret.key)."""


def get_secret_key() -> str:
    """Liest oder erstellt den Secret Key fuer Cookie-Signing und Fernet.

    Erstellt die Datei atomar (tempfile + hardlink), falls sie nicht existiert.
    Leere-Datei-Schutz: Retry-Loop fuer den Fall, dass ein anderer Prozess
    die Datei gerade erst erstellt.
    """

    def _try_read() -> str | None:
        """Liest Key aus Datei; gibt None bei fehlender/leerer Datei zurueck."""
        # Permissions VOR Lesen haerten (TOCTOU-Defense): falls die Datei
        # mit zu offenen Permissions vorliegt, vor dem ersten Read auf 0o600
        # ziehen statt erst danach.
        try:
            os.chmod(str(SECRET_KEY_PATH), 0o600)
        except FileNotFoundError:
            return None
        try:
            content = SECRET_KEY_PATH.read_text().strip()
        except FileNotFoundError:
            return None
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"Secret-Key-Datei {SECRET_KEY_PATH} ist nicht als Text lesbar"
            ) from exc
        return content or None

    # --- Bestehenden Key lesen ---
    existing = _try_read()
    if existing is not None:
        return existing

    # --- Neuen Key atomar erstellen ---
    SECRET_KEY_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = secrets.token_hex(32)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(SECRET_KEY_PATH.parent), prefix=".secret_key_")
    try:
        os.write(tmp_fd, key.encode())
        os.fchmod(tmp_fd, 0o600)
        os.fsync(tmp_fd)
        os.close(tmp_fd)
        tmp_fd = -1
        # Hardlink ist atomar: Zieldatei enthaelt sofort den vollen Key
        # oder der Call schlaegt fehl (anderer Worker war schneller).
        os.link(tmp_path, str(SECRET_KEY_PATH))
    except FileExistsError:
        # Anderer Worker hat die Datei erstellt — mit Retry lesen
        for _ in range(20):
            winner_key = _try_read()
            if winner_key is not None:
                return winner_key
            time.sleep(0.05)
        raise RuntimeError(f"Secret-Key-Datei {SECRET_KEY_PATH} existiert aber bleibt leer")
    finally:
        if tmp_fd >= 0:
            os.close(tmp_fd)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return key


# ---------------------------------------------------------------------------
# Fernet-Verschluesselung
# ---------------------------------------------------------------------------


def _get_fernet() -> Fernet:
    """Erstellt eine Fernet-Instanz aus dem Applikations-Secret.

    Leitet den Schluessel deterministisch ab: secret → SHA-256 → base64url.
    """
    secret = get_secret_key()
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def encrypt(value: str | None) -> str | None:
    """Verschluesselt einen Klartext-Wert mit Fernet.

    Leere/None-Werte werden unveraendert zurueckgegeben.
    """
    if not value:
        return value
    return _get_fernet().encrypt(value.encode()).decode()


def decrypt(value: str | None) -> str | None:
    """Entschluesselt einen Fernet-Token zurueck zu Klartext.

    Raises:
        InvalidToken: Bei ungueltigem Token (Key-Mismatch, Korruption).
            Alle Aufrufer muessen vorher ``is_encrypted()`` pruefen.
    """
    if value is None:
        return None
    if value == "":
        return ""
    try:
        return _get_fernet().decrypt(value.encode()).decode()
    except InvalidToken:
        logger.error(
            "Entschluesselung fehlgeschlagen (Key-Mismatch oder korrupter Wert). "
            "Pruefen Sie ob der Secret-Key (/var/lib/kiara/secret.key) gewechselt wurde."
        )
        raise


def is_encrypted(value: str) -> bool:
    """Prueft ob ein Wert ein valides Fernet-Token ist.

    Prueft nicht nur den Prefix, sondern validiert den Token mit dem aktuellen
    Fernet-Key. Syntaktisch plausible, aber undecryptbare Werte duerfen im
    Persistenzpfad nicht als "bereits verschluesselt" durchgereicht werden.
    """
    if not isinstance(value, str) or not value.startswith("gAAAAA"):
        return False
    try:
        # Fernet-Token haben fixe Struktur: base64url-encoded, mindestens
        # 57 Bytes nach Decodierung (Version 1 Byte + Timestamp 8 Byte +
        # IV 16 Byte + HMAC 32 Byte + Ciphertext).
        decoded = base64.urlsafe_b64decode(value.encode())
        if len(decoded) < 57 or decoded[0:1] != b"\x80":
            return False
        _get_fernet().decrypt(value.encode())
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Config-Feld Verschluesselung (DRY-Helper fuer api_datasource, api_tools, etc.)
# ---------------------------------------------------------------------------


def encrypt_secret_fields(
    config: dict,
    secret_field_names: list[str],
    existing_config: dict | None = None,
) -> dict:
    """Verschluesselt Secret-Felder in einem Config-Dict fuer DB-Persistierung.

    Behandelt drei Faelle pro Secret-Feld:
    1. Wert == PASSWORD_MASK: User hat nicht geaendert → bestehenden
       (verschluesselten) Wert aus existing_config beibehalten.
    2. Wert bereits verschluesselt (is_encrypted): Unveraendert uebernehmen.
    3. Neuer Klartext-Wert: Per Fernet verschluesseln.

    Args:
        config:             Config-Dict mit potentiellen Klartext-Secrets.
        secret_field_names: Liste der Feldnamen, die Secrets enthalten.
        existing_config:    Bestehende Config aus der DB (fuer Mask-Aufloesung
                            bei Updates). None bei neuen Eintraegen.

    Returns:
        Neues Dict mit verschluesselten Secret-Feldern.
    """
    from kiara.shared.constants import PASSWORD_MASK

    encrypted = dict(config)
    for field in secret_field_names:
        new_value = encrypted.get(field, "")
        if new_value == PASSWORD_MASK:
            if existing_config and field in existing_config:
                encrypted[field] = existing_config[field]
            else:
                raise ValueError(
                    f"PASSWORD_MASK fuer '{field}' gesendet, aber "
                    f"existing_config fehlt — Secret kann nicht aufgeloest werden"
                )
        elif new_value:
            if is_encrypted(new_value):
                encrypted[field] = new_value
            elif isinstance(new_value, str) and new_value.startswith("gAAAAA"):
                raise ValueError(
                    f"Feld '{field}' enthaelt einen Fernet-aehnlichen, "
                    "aber mit dem aktuellen Secret-Key nicht entschluesselbaren Wert"
                )
            else:
                encrypted[field] = encrypt(new_value)
        else:
            encrypted[field] = ""
    return encrypted

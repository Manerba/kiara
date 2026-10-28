"""AD objectSid Normalisierung fuer externe Identitaeten."""

from __future__ import annotations

import base64
import re

AD_SID_RE = re.compile(r"^S-1-[0-9]+(-[0-9]+)+$")


class ADSidError(ValueError):
    """Ungueltige AD-SID."""


def normalize_ad_sid(value: str, *, allow_base64: bool = True) -> str:
    """Normalisiert kanonische oder Base64-kodierte AD objectSid-Werte.

    Keycloak liefert AD ``objectSid`` bei Binary-LDAP-Mapping als Base64. Kiara
    speichert und auditiert aber immer den kanonischen SID-String.
    """
    if not isinstance(value, str):
        raise ADSidError("AD-SID muss String sein")
    raw = value.strip()
    if not raw:
        raise ADSidError("AD-SID ist leer")

    if raw[:2].lower() == "s-":
        canonical = "S-" + raw[2:]
        if not AD_SID_RE.fullmatch(canonical):
            raise ADSidError("Ungueltiges AD-SID-Format")
        return canonical

    if not allow_base64:
        raise ADSidError("AD-SID ist nicht kanonisch")

    try:
        return _binary_sid_to_string(_decode_base64_sid(raw))
    except ADSidError:
        raise
    except Exception as exc:
        raise ADSidError("AD objectSid Base64 konnte nicht dekodiert werden") from exc


def _decode_base64_sid(raw: str) -> bytes:
    padded = raw + ("=" * (-len(raw) % 4))
    try:
        decoded = base64.b64decode(padded, validate=True)
    except Exception as exc:
        raise ADSidError("AD objectSid ist kein valides Base64") from exc
    return decoded


def _binary_sid_to_string(data: bytes) -> str:
    if len(data) < 8:
        raise ADSidError("AD objectSid ist zu kurz")
    revision = data[0]
    sub_authority_count = data[1]
    expected_length = 8 + (sub_authority_count * 4)
    if revision != 1:
        raise ADSidError("AD objectSid Revision wird nicht unterstuetzt")
    if len(data) != expected_length:
        raise ADSidError("AD objectSid Laenge passt nicht zur SubAuthority-Anzahl")

    identifier_authority = int.from_bytes(data[2:8], byteorder="big", signed=False)
    parts = [f"S-{revision}-{identifier_authority}"]
    offset = 8
    for _ in range(sub_authority_count):
        parts.append(str(int.from_bytes(data[offset : offset + 4], byteorder="little")))
        offset += 4

    sid = "-".join(parts)
    if not AD_SID_RE.fullmatch(sid):
        raise ADSidError("Dekodierte AD-SID ist nicht kanonisch")
    return sid

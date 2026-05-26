"""Rekursive Secret-Redaction fuer beliebig verschachtelte Strukturen.

Wird ueberall dort genutzt, wo Tool-Parameter, Connector-Configs oder
aehnliche Strukturen die Anwendungsgrenzen verlassen (SSE-Streaming,
Persistenz, OTel-Tracing) und keine sensiblen Werte enthalten duerfen.
"""

import re

_SECRET_KEY_PATTERNS = ("password", "api_key", "secret", "dsn", "token", "bind_password")

# Sekret-Schluesselnamen, die in Freitext, Repr() oder JSON auftauchen koennen.
_SECRET_KEY_REGEX_GROUP = r"(?:password|api[_-]?key|secret|dsn|token|bind_password)"

_SECRET_TEXT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # HTTP-Authorization
    (
        re.compile(r"(authorization\s*[:=]\s*)(Bearer|Basic)?\s*([^\s\"'<>,;}\)]+)", re.IGNORECASE),
        r"\1***",
    ),
    (re.compile(r"(Bearer\s+)([^\s\"'<>,;}\)]+)", re.IGNORECASE), r"\1***"),
    (re.compile(r"(Basic\s+)([A-Za-z0-9+/=_-]+)", re.IGNORECASE), r"\1***"),
    # Dict-/JSON-Repr: 'password': 'x'  oder  "api_key": "y"
    # Schluessel und Wert sind in passenden Quotes; der Wert wird zwischen
    # den Quotes durch *** ersetzt, die Quotes bleiben erhalten.
    (
        re.compile(
            r"(?P<q1>['\"])(?P<key>" + _SECRET_KEY_REGEX_GROUP + r")(?P=q1)"
            r"(?P<sep>\s*:\s*)"
            r"(?P<q2>['\"])(?P<val>[^'\"]+)(?P=q2)",
            re.IGNORECASE,
        ),
        r"\g<q1>\g<key>\g<q1>\g<sep>\g<q2>***\g<q2>",
    ),
    # Unquoted key=value oder key: value (klassisches Logging)
    (re.compile(r"(api[_-]?key\s*[:=]\s*)([^\s\"'<>,;}\)]+)", re.IGNORECASE), r"\1***"),
    (re.compile(r"(token\s*[:=]\s*)([^\s\"'<>,;}\)]+)", re.IGNORECASE), r"\1***"),
    (re.compile(r"(password\s*[:=]\s*)([^\s\"'<>,;}\)]+)", re.IGNORECASE), r"\1***"),
    (re.compile(r"(secret\s*[:=]\s*)([^\s\"'<>,;}\)]+)", re.IGNORECASE), r"\1***"),
    (re.compile(r"(bind_password\s*[:=]\s*)([^\s\"'<>,;}\)]+)", re.IGNORECASE), r"\1***"),
    (re.compile(r"(dsn\s*[:=]\s*)([^\s\"'<>,;}\)]+)", re.IGNORECASE), r"\1***"),
    # URL-Credentials in DSN-/Endpoint-Strings
    (re.compile(r"(://)([^/@\s]+)@"), r"\1***@"),
)


def redact_secrets_recursive(obj):
    """Maskiert Secrets in beliebig verschachtelten Strukturen (dict/list)."""
    from kiara.shared.constants import PASSWORD_MASK

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key_is_secret = isinstance(k, str) and any(p in k.lower() for p in _SECRET_KEY_PATTERNS)
            if key_is_secret and v and v != PASSWORD_MASK and not isinstance(v, (dict, list)):
                out[k] = PASSWORD_MASK
            elif isinstance(v, (dict, list)):
                out[k] = redact_secrets_recursive(v)
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [redact_secrets_recursive(item) for item in obj]
    return obj


def redact_secrets_in_text(text: str) -> str:
    """Maskiert typische Secret-Patterns in einem Freitext.

    Nutzbar fuer Exception-Messages, HTTP-Response-Bodies oder andere
    Strings, die in Error-Logs landen koennten und potenziell
    Bearer-Tokens, API-Keys oder URL-eingebettete Credentials enthalten.
    """
    if not isinstance(text, str) or not text:
        return text
    for pattern, repl in _SECRET_TEXT_PATTERNS:
        text = pattern.sub(repl, text)
    return text

"""LDAP-Operationen: Verbindung, Gruppen, Mitglieder, Authentifizierung.

Reine LDAP-Logik ohne Web-Abhaengigkeiten. Nutzt ldap3 fuer alle
Active-Directory-/LDAP-Interaktionen.

Beziehungen:
    - kiara.web.ldap_utils       → Re-Export (Rueckwaertskompatibilitaet)
    - kiara.anonymization.*      → nutzt get_group_members (Stammdaten)
    - kiara.web.api_ldap         → nutzt test_connection, get_groups
    - kiara.web.api_auth         → nutzt authenticate_user
    - kiara.web.ldap_sync        → nutzt get_group_members, get_user_groups
"""

import logging
import ssl
from contextlib import contextmanager

from ldap3 import NONE, SUBTREE, Connection, Server, Tls

from kiara.config import DEFAULT_TIMEOUTS

logger = logging.getLogger(__name__)


# bugscan:by-design[ldap-cert-validation-default-off] — Interne AD/LDAP-Setups
# verwenden haeufig Self-Signed-Zertifikate; Zertifikatspruefung ist opt-in und
# darf nicht per Auto-Fix auf CERT_REQUIRED als Default umgestellt werden.
def _create_server(server_addr: str, port: int, use_ssl: bool, verify_cert: bool = False) -> Server:
    """Erstellt ein ldap3.Server-Objekt.

    SSL-Zertifikatspruefung ist bewusst optional (verify_cert=False als Default),
    da viele interne AD/LDAP-Setups Self-Signed-Zertifikate verwenden.
    """
    tls = Tls(validate=ssl.CERT_REQUIRED) if verify_cert else Tls(validate=ssl.CERT_NONE)
    return Server(
        server_addr,
        port=port,
        use_ssl=bool(use_ssl),
        tls=tls,
        get_info=NONE,
        connect_timeout=DEFAULT_TIMEOUTS["ldap_connect"],
    )


def _bind_connection(
    server: Server,
    bind_dn: str,
    bind_password: str,
    use_ssl: bool,
    receive_timeout: int | None = None,
) -> Connection:
    """Erstellt und bindet eine Connection. Bei nicht-SSL wird StartTLS verwendet."""
    conn = Connection(
        server,
        user=bind_dn,
        password=bind_password,
        auto_bind=False,
        auto_range=True,
        receive_timeout=receive_timeout
        if receive_timeout is not None
        else DEFAULT_TIMEOUTS["ldap_connect"],
        raise_exceptions=True,
    )
    conn.open()
    try:
        if not use_ssl:
            conn.start_tls()
        conn.bind()
    except Exception:
        try:
            conn.unbind()
        except Exception:
            pass
        raise
    return conn


@contextmanager
def _ldap_connection(config: dict, receive_timeout: int | None = None):
    """Kontextmanager fuer LDAP-Verbindungen.

    Erstellt Server, bindet Connection und trennt im finally-Block sauber.

    Args:
        config: LDAP-Konfiguration mit server, port, use_ssl, bind_dn, bind_password.
        receive_timeout: Optionaler Receive-Timeout (sonst Default).

    Yields:
        Gebundene ldap3.Connection.
    """
    conn = None
    try:
        srv = _create_server(config["server"], config["port"], config.get("use_ssl", 0))
        conn = _bind_connection(
            srv,
            config["bind_dn"],
            config["bind_password"],
            config.get("use_ssl", 0),
            receive_timeout=receive_timeout,
        )
        yield conn
    finally:
        if conn is not None:
            try:
                conn.unbind()
            except Exception as exc:
                logger.debug("LDAP unbind failed: %s", exc)


def test_connection(config: dict) -> tuple[bool, str]:
    """Verbindungstest. Liefert (True, None) bei Erfolg, (False, Fehlermeldung) bei Fehler."""
    try:
        with _ldap_connection(config, receive_timeout=DEFAULT_TIMEOUTS["ldap_receive"]) as conn:
            conn.search(
                config["search_base"],
                "(objectClass=*)",
                search_scope=SUBTREE,
                size_limit=1,
                attributes=[],
            )
            return True, None
    except Exception as exc:
        return False, str(exc)


def _first_ldap_value(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _format_object_sid(value) -> str:
    """Formatiert objectSid als SDDL-SID (S-1-...)."""
    value = _first_ldap_value(value)
    if value is None:
        return ""
    if isinstance(value, str):
        sid = value.strip()
        return sid if sid.startswith("S-") else ""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes) or len(value) < 8:
        return ""

    revision = value[0]
    sub_authority_count = value[1]
    expected_len = 8 + sub_authority_count * 4
    if len(value) < expected_len:
        return ""

    identifier_authority = int.from_bytes(value[2:8], "big")
    sub_authorities = [
        str(int.from_bytes(value[8 + i * 4 : 12 + i * 4], "little"))
        for i in range(sub_authority_count)
    ]
    return "S-" + "-".join([str(revision), str(identifier_authority), *sub_authorities])


def get_groups(config: dict) -> list[dict]:
    """Alle Gruppen im search_base auflisten. Liefert SID plus Anzeige-Metadaten."""
    try:
        with _ldap_connection(config) as conn:
            conn.search(
                config["search_base"],
                "(objectClass=group)",
                search_scope=SUBTREE,
                attributes=["cn", "distinguishedName", "member", "objectSid"],
            )
            groups = []
            for entry in conn.entries:
                raw = entry.entry_attributes_as_dict
                cn = str(entry.cn)
                members = entry.member.values if hasattr(entry, "member") and entry.member else []
                groups.append(
                    {
                        "sid": _format_object_sid(raw.get("objectSid")),
                        "dn": str(entry.distinguishedName),
                        "cn": cn,
                        "name": cn,
                        "member_count": len(members),
                    }
                )
            groups.sort(key=lambda g: g["cn"].lower())
            return groups
    except Exception as exc:
        from kiara.exceptions import AuthenticationError

        raise AuthenticationError(f"Gruppen konnten nicht geladen werden: {exc}") from exc


def get_group_members(config: dict) -> list[dict]:
    """Alle Benutzer der konfigurierten Gruppe abrufen.
    Liefert Liste von {dn, sAMAccountName, sn, givenName, mail, telephoneNumber, department, displayName}.
    """
    try:
        with _ldap_connection(config, receive_timeout=DEFAULT_TIMEOUTS["ldap_receive"]) as conn:
            # Mitglieder der Gruppe abrufen
            conn.search(
                config["group_dn"],
                "(objectClass=group)",
                search_scope=SUBTREE,
                attributes=["member"],
            )

            if not conn.entries:
                return []

            members_dns = conn.entries[0].member.values if conn.entries[0].member else []
            if not members_dns:
                return []

            # Batch-Search statt N einzelner Queries (N+1 → ceil(N/50)+1)
            _BATCH_SIZE = 50
            search_base = _get_domain_root(config["search_base"])
            members = []

            for i in range(0, len(members_dns), _BATCH_SIZE):
                batch = members_dns[i : i + _BATCH_SIZE]
                dn_filters = "".join(
                    f"(distinguishedName={_escape_ldap_filter(str(dn))})" for dn in batch
                )
                conn.search(
                    search_base,
                    f"(&(objectClass=user)(|{dn_filters}))",
                    search_scope=SUBTREE,
                    attributes=[
                        "sAMAccountName",
                        "sn",
                        "givenName",
                        "mail",
                        "distinguishedName",
                        "telephoneNumber",
                        "department",
                        "displayName",
                    ],
                )
                for entry in conn.entries:
                    raw = entry.entry_attributes_as_dict
                    members.append(
                        {
                            "dn": str(entry.distinguishedName),
                            "sAMAccountName": str(entry.sAMAccountName)
                            if entry.sAMAccountName
                            else "",
                            "sn": str(entry.sn) if entry.sn else "",
                            "givenName": str(entry.givenName) if entry.givenName else "",
                            "mail": str(entry.mail) if entry.mail else "",
                            "telephoneNumber": str((raw.get("telephoneNumber") or [""])[0]),
                            "department": str((raw.get("department") or [""])[0]),
                            "displayName": str((raw.get("displayName") or [""])[0]),
                        }
                    )

            return members
    except Exception as exc:
        from kiara.exceptions import AuthenticationError

        raise AuthenticationError(f"Gruppenmitglieder konnten nicht geladen werden: {exc}") from exc


def get_user_groups(config: dict, user_dn: str) -> list[str]:
    """Get group SIDs that a user is a member of.

    Raises bei LDAP-Fehlern — Aufrufer (sync_user_groups) muss Exception
    behandeln, damit leere Liste nicht faelschlicherweise als
    'keine Gruppen' interpretiert und bestehende Mitgliedschaften geloescht werden.
    """
    with _ldap_connection(config) as conn:
        conn.search(
            config["search_base"],
            f"(&(objectClass=group)(member={_escape_ldap_filter(user_dn)}))",
            search_scope=SUBTREE,
            attributes=["objectSid"],
        )
        group_sids: list[str] = []
        missing_sid_dns: list[str] = []
        for entry in conn.entries:
            sid = _format_object_sid(entry.entry_attributes_as_dict.get("objectSid"))
            if sid:
                group_sids.append(sid)
            else:
                missing_sid_dns.append(str(entry.entry_dn))
        if missing_sid_dns:
            raise ValueError("LDAP-Gruppen ohne objectSid gefunden: " + ", ".join(missing_sid_dns))
        return group_sids


def authenticate_user(config: dict, username: str, password: str) -> tuple[bool, str]:
    """LDAP-Bind mit Benutzer-DN. Sucht DN via sAMAccountName, dann Bind-Versuch."""
    if not password:
        return False, "Passwort darf nicht leer sein"
    try:
        # Service-Account: DN des Benutzers suchen
        with _ldap_connection(config) as conn:
            domain_root = _get_domain_root(config["search_base"])
            search_filter = f"(&(objectClass=user)(sAMAccountName={_escape_ldap_filter(username)}))"
            conn.search(
                domain_root,
                search_filter,
                search_scope=SUBTREE,
                attributes=["distinguishedName"],
            )

            if not conn.entries:
                return False, f"Benutzer '{username}' im LDAP nicht gefunden"

            user_dn = str(conn.entries[0].distinguishedName)

        # Bind mit Benutzer-Credentials
        try:
            srv = _create_server(config["server"], config["port"], config.get("use_ssl", 0))
            user_conn = _bind_connection(srv, user_dn, password, config.get("use_ssl", 0))
            user_conn.unbind()
            return True, ""
        except Exception as bind_err:
            return False, f"Authentifizierung fehlgeschlagen: {bind_err}"

    except Exception as exc:
        return False, f"LDAP-Authentifizierung fehlgeschlagen: {exc}"


def _get_domain_root(search_base: str) -> str:
    """Extrahiert den Domain-Root (DC-Komponenten) aus einem DN."""
    parts = [p.strip() for p in search_base.split(",")]
    dc_parts = [p for p in parts if p.upper().startswith("DC=")]
    return ",".join(dc_parts) if dc_parts else search_base


def _escape_ldap_filter(value: str) -> str:
    """Escaped Sonderzeichen fuer LDAP-Filter (RFC 4515)."""
    result = ""
    for char in value:
        if char == "\\":
            result += "\\5c"
        elif char == "*":
            result += "\\2a"
        elif char == "(":
            result += "\\28"
        elif char == ")":
            result += "\\29"
        elif char == "\x00":
            result += "\\00"
        else:
            result += char
    return result

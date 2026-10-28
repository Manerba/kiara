"""Ergaenzt den Anonymisierungs-Suchbaum mit Active-Directory-Daten.

Nutzt die bereits in der DB vorhandenen LDAP-Daten (Tabellen: users, groups)
und optional eine direkte LDAP-Abfrage fuer erweiterte Attribute.
"""

import logging

from kiara.repositories.config import ConfigRepository
from kiara.repositories.group import GroupRepository
from kiara.repositories.user import UserRepository

logger = logging.getLogger(__name__)

# Module-level Repo-Instanzen
_user_repo = UserRepository()
_group_repo = GroupRepository()
_config_repo = ConfigRepository()


class LDAPEnricher:
    """Extrahiert personenbezogene Daten aus dem bestehenden LDAP-Sync.

    Nutzt die bereits in der DB vorhandenen LDAP-Daten (Tabellen: users, groups)
    und optional eine direkte LDAP-Abfrage fuer erweiterte Attribute.
    """

    @staticmethod
    def from_database() -> list[dict]:
        """Extrahiert Benutzernamen und bekannte Attribute aus der KIara-DB.

        Liest aus den Tabellen 'users' und 'groups' alle relevanten
        Personendaten, die fuer die Anonymisierung benoetigt werden.

        Returns:
            Liste von Entitaeten im Format:
            [{"value": "Max Mustermann", "category": "employee", "id": "ldap-..."}, ...]
        """
        entities = []
        try:
            # Benutzer aus der users-Tabelle (list_safe liefert ohne password_hash)
            rows = _user_repo.list_safe()

            for row_dict in rows:
                user_id = row_dict.get("id")
                dn = row_dict.get("ldap_dn", "")
                if not dn and user_id is None:
                    continue
                entity_id = f"ldap-{dn}" if dn else f"db-user-{user_id}"

                vorname = (row_dict.get("vorname") or "").strip()
                nachname = (row_dict.get("nachname") or "").strip()
                email = (row_dict.get("email") or "").strip()

                # Vollstaendiger Name
                if vorname and nachname:
                    entities.append(
                        {
                            "value": f"{vorname} {nachname}",
                            "category": "employee",
                            "id": entity_id,
                        }
                    )
                    # Nachname, Vorname
                    entities.append(
                        {
                            "value": f"{nachname}, {vorname}",
                            "category": "employee",
                            "id": entity_id,
                        }
                    )

                # Nachname einzeln (nur wenn lang genug)
                if nachname and len(nachname) >= 3:
                    entities.append(
                        {
                            "value": nachname,
                            "category": "employee",
                            "id": entity_id,
                        }
                    )

                # E-Mail
                if email:
                    entities.append(
                        {
                            "value": email,
                            "category": "email",
                            "id": entity_id,
                        }
                    )

            # Gruppen-Namen
            group_cns = _group_repo.get_all_cns()
            for cn in group_cns:
                cn = cn.strip()
                if cn and len(cn) >= 3:
                    entities.append(
                        {
                            "value": cn,
                            "category": "employee",
                            "id": f"group-{cn}",
                        }
                    )

        except Exception as exc:
            logger.warning("DB-Benutzer konnten nicht geladen werden: %s", exc)

        return entities

    @staticmethod
    def from_ldap(config: dict) -> list[dict]:
        """Direkte LDAP-Abfrage fuer displayName, mail, telephoneNumber, department.

        Args:
            config: Gesamt-Config mit ldap-Sektion oder LDAP-Config aus der DB.

        Returns:
            Liste von Entitaeten.
        """
        # Config-Parameter auswerten: ldap-Sektion (kein Fallback auf gesamte
        # config — das wuerde Top-Level-Keys faelschlich als LDAP-Felder lesen)
        ldap_section = config.get("ldap") if config else None

        # Fallback auf DB wenn config keinen LDAP-Server enthaelt
        if not ldap_section or (
            not ldap_section.get("server") and not ldap_section.get("server_url")
        ):
            try:
                from kiara.web.config_utils import build_ldap_config

                ldap_row = _config_repo.get("ldap_config")
                ldap_section = build_ldap_config(ldap_row) if ldap_row else None
            except Exception:
                ldap_section = None

        if not ldap_section:
            return []

        ldap_cfg = {
            "server": ldap_section.get("server") or ldap_section.get("server_url", ""),
            "port": ldap_section.get("port", 389),
            "use_ssl": ldap_section.get("use_ssl", False),
            "bind_dn": ldap_section.get("bind_dn", ""),
            "bind_password": ldap_section.get("bind_password", ""),
            "search_base": ldap_section.get("search_base", ""),
            "group_dn": ldap_section.get("group_dn", ""),
        }

        if not ldap_cfg["server"] or not ldap_cfg["bind_dn"] or not ldap_cfg["search_base"]:
            logger.warning("LDAP-Config unvollstaendig — Stammdaten-Import uebersprungen")
            return []

        if not ldap_cfg["group_dn"]:
            return []

        try:
            from kiara.shared.ldap import get_group_members

            members = get_group_members(ldap_cfg)
        except Exception as exc:
            logger.warning("LDAP-Mitglieder konnten nicht geladen werden: %s", exc)
            return []

        entities = []
        for member in members:
            dn = (member.get("dn") or "").strip()
            sam = (member.get("sAMAccountName") or "").strip()
            surname = (member.get("sn") or "").strip()
            given_name = (member.get("givenName") or "").strip()
            mail = (member.get("mail") or "").strip()
            phone = (member.get("telephoneNumber") or "").strip()
            department = (member.get("department") or "").strip()
            entity_id = f"ldap-{dn}" if dn else f"ldap-{sam}"

            # Vollstaendiger Name
            if given_name and surname:
                entities.append(
                    {
                        "value": f"{given_name} {surname}",
                        "category": "employee",
                        "id": entity_id,
                    }
                )
                entities.append(
                    {
                        "value": f"{surname}, {given_name}",
                        "category": "employee",
                        "id": entity_id,
                    }
                )

            # Nachname einzeln
            if surname and len(surname) >= 3:
                entities.append(
                    {
                        "value": surname,
                        "category": "employee",
                        "id": entity_id,
                    }
                )

            # E-Mail
            if mail:
                entities.append(
                    {
                        "value": mail,
                        "category": "email",
                        "id": entity_id,
                    }
                )

            # Telefon
            if phone:
                entities.append(
                    {
                        "value": phone,
                        "category": "phone",
                        "id": entity_id,
                    }
                )

            # Abteilung
            if department and len(department) >= 3:
                entities.append(
                    {
                        "value": department,
                        "category": "location",
                        "id": f"dept-{department}",
                    }
                )

        return entities

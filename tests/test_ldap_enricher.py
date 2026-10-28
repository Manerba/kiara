from unittest.mock import patch

from kiara.anonymization.ldap_enricher import LDAPEnricher


class TestLDAPEnricherCategories:
    def test_database_email_uses_email_category(self):
        with (
            patch("kiara.anonymization.ldap_enricher._user_repo") as users,
            patch("kiara.anonymization.ldap_enricher._group_repo") as groups,
        ):
            users.list_safe.return_value = [
                {
                    "id": 1,
                    "ldap_dn": "cn=max,dc=example",
                    "vorname": "Max",
                    "nachname": "Mustermann",
                    "email": "max@example.de",
                }
            ]
            groups.get_all_cns.return_value = []

            entities = LDAPEnricher.from_database()

        email_entities = [e for e in entities if e["value"] == "max@example.de"]
        assert email_entities == [
            {
                "value": "max@example.de",
                "category": "email",
                "id": "ldap-cn=max,dc=example",
            }
        ]

    def test_ldap_mail_and_phone_use_specific_categories(self):
        cfg = {
            "ldap": {
                "server": "ldap.example.de",
                "bind_dn": "cn=admin,dc=example",
                "bind_password": "secret",
                "search_base": "dc=example",
                "group_dn": "cn=users,dc=example",
            }
        }
        with patch(
            "kiara.shared.ldap.get_group_members",
            return_value=[
                {
                    "dn": "cn=anna,dc=example",
                    "sAMAccountName": "anna",
                    "sn": "Schmidt",
                    "givenName": "Anna",
                    "mail": "anna@example.de",
                    "telephoneNumber": "+49 30 123456",
                    "department": "",
                }
            ],
        ):
            entities = LDAPEnricher.from_ldap(cfg)

        by_value = {e["value"]: e["category"] for e in entities}
        assert by_value["anna@example.de"] == "email"
        assert by_value["+49 30 123456"] == "phone"

"""Tests fuer IMAP Folder Enumeration (V10).

Testet:
- _decode_imap_utf7: Modified UTF-7 Dekodierung
- _parse_imap_list: IMAP LIST Parsing (Bytes, Tuples, Flags, Delimiter)
- _build_folder_tree: Hierarchische Baumstruktur
- Wildcard-Scan: * -> alle Ordner, Connection-Cleanup
- test_connection: Connection-Leak Fix
- _resolve_masked_config: PASSWORD_MASK Aufloesung
"""

from unittest.mock import MagicMock, patch

import pytest

from kiara.ingestion.imap_scanner import IMAPScanner

# ---------------------------------------------------------------------------
# _decode_imap_utf7
# ---------------------------------------------------------------------------


class TestDecodeImapUtf7:
    """IMAP Modified UTF-7 Dekodierung (RFC 3501 Section 5.1.3)."""

    def test_plain_ascii(self):
        assert IMAPScanner._decode_imap_utf7("INBOX") == "INBOX"

    def test_literal_ampersand(self):
        assert IMAPScanner._decode_imap_utf7("Tom &- Jerry") == "Tom & Jerry"

    def test_german_umlaut_ae(self):
        # &AOQ- = ä
        assert IMAPScanner._decode_imap_utf7("&AOQ-") == "\u00e4"

    def test_german_umlaut_ue_inline(self):
        # Entw&APw-rfe = Entwürfe
        result = IMAPScanner._decode_imap_utf7("Entw&APw-rfe")
        assert result == "Entw\u00fcrfe"

    def test_broken_no_terminator(self):
        # Fehlender '-' Terminator — kein Crash, Original beibehalten
        result = IMAPScanner._decode_imap_utf7("&AOQ")
        assert "&AOQ" in result

    def test_broken_corrupt_base64(self):
        # Korrupte Base64-Sequenz — kein Crash, Ergebnis ist egal
        result = IMAPScanner._decode_imap_utf7("&!!!-")
        assert isinstance(result, str)  # Sollte nicht crashen

    def test_empty_string(self):
        assert IMAPScanner._decode_imap_utf7("") == ""

    def test_multiple_encoded_sequences(self):
        # Zwei kodierte Sequenzen hintereinander
        result = IMAPScanner._decode_imap_utf7("&AOQ-&APw-")
        assert "\u00e4" in result
        assert "\u00fc" in result


# ---------------------------------------------------------------------------
# _parse_imap_list
# ---------------------------------------------------------------------------


class TestParseImapList:
    """IMAP LIST Response Parsing."""

    def test_slash_delimiter(self):
        raw = [
            b'(\\HasNoChildren) "/" "INBOX"',
            b'(\\HasChildren) "/" "INBOX/Sent"',
        ]
        result = IMAPScanner._parse_imap_list(raw)
        assert len(result) == 2
        assert result[0]["name"] == "INBOX"
        assert result[0]["delimiter"] == "/"
        assert result[1]["name"] == "INBOX/Sent"

    def test_dot_delimiter(self):
        raw = [b'(\\HasNoChildren) "." "INBOX.Sent"']
        result = IMAPScanner._parse_imap_list(raw)
        assert result[0]["delimiter"] == "."
        assert result[0]["name"] == "INBOX.Sent"

    def test_nil_delimiter(self):
        raw = [b'(\\Noselect) NIL "FlatFolder"']
        result = IMAPScanner._parse_imap_list(raw)
        assert result[0]["delimiter"] == "/"  # Default
        assert result[0]["name"] == "FlatFolder"

    def test_noselect_case_insensitive(self):
        """RFC 3501: Flags sind case-insensitive."""
        for flag_variant in [b"\\Noselect", b"\\NOSELECT", b"\\noselect"]:
            raw = [b"(" + flag_variant + b') "/" "Test"']
            result = IMAPScanner._parse_imap_list(raw)
            assert "\\noselect" in result[0]["flags"]

    def test_literal_tuple(self):
        """imaplib liefert Tuples fuer Literal-Antworten."""
        raw = [
            (b'(\\HasNoChildren) "/" {10}', b"Sent Items"),
        ]
        result = IMAPScanner._parse_imap_list(raw)
        assert len(result) == 1
        assert result[0]["name"] == "Sent Items"
        assert result[0]["raw_name"] == "Sent Items"
        assert result[0]["delimiter"] == "/"

    def test_literal_with_quotes_in_name(self):
        """Ordnername mit Anfuehrungszeichen im Literal."""
        raw = [
            (b'(\\HasNoChildren) "/" {13}', b'My "Special"'),
        ]
        result = IMAPScanner._parse_imap_list(raw)
        assert len(result) == 1
        assert result[0]["name"] == 'My "Special"'

    def test_short_tuple_skipped(self):
        """Tuple mit weniger als 2 Elementen wird uebersprungen."""
        raw = [(b'(\\HasNoChildren) "/" {5}',)]
        result = IMAPScanner._parse_imap_list(raw)
        assert len(result) == 0

    def test_non_bytes_skipped(self):
        """Unbekannte Typen werden uebersprungen."""
        raw = [42, None, "string"]
        result = IMAPScanner._parse_imap_list(raw)
        assert len(result) == 0

    def test_raw_name_preserved(self):
        """raw_name enthaelt den Original-IMAP-kodierten Namen."""
        raw = [b'(\\HasNoChildren) "/" "Entw&APw-rfe"']
        result = IMAPScanner._parse_imap_list(raw)
        assert result[0]["raw_name"] == "Entw&APw-rfe"
        assert result[0]["name"] == "Entw\u00fcrfe"


# ---------------------------------------------------------------------------
# _build_folder_tree
# ---------------------------------------------------------------------------


class TestBuildFolderTree:
    """Hierarchische Baumstruktur aus flacher Ordnerliste."""

    def test_flat_list(self):
        parsed = [
            {"name": "INBOX", "raw_name": "INBOX", "flags": [], "delimiter": "/"},
            {"name": "Sent", "raw_name": "Sent", "flags": [], "delimiter": "/"},
        ]
        tree = IMAPScanner._build_folder_tree(parsed)
        assert len(tree) == 2
        assert tree[0]["name"] == "INBOX"
        assert tree[0]["selectable"] is True
        assert tree[0]["children"] == []

    def test_nested(self):
        parsed = [
            {"name": "INBOX", "raw_name": "INBOX", "flags": [], "delimiter": "/"},
            {"name": "INBOX/Sub", "raw_name": "INBOX/Sub", "flags": [], "delimiter": "/"},
            {"name": "INBOX/Sub/Deep", "raw_name": "INBOX/Sub/Deep", "flags": [], "delimiter": "/"},
        ]
        tree = IMAPScanner._build_folder_tree(parsed)
        assert len(tree) == 1
        inbox = tree[0]
        assert inbox["name"] == "INBOX"
        assert len(inbox["children"]) == 1
        sub = inbox["children"][0]
        assert sub["name"] == "Sub"
        assert len(sub["children"]) == 1
        deep = sub["children"][0]
        assert deep["name"] == "Deep"

    def test_missing_multiple_parents(self):
        """Synthetische Knoten fuer alle fehlenden Vorfahren."""
        parsed = [
            {"name": "INBOX/A/B/C", "raw_name": "INBOX/A/B/C", "flags": [], "delimiter": "/"},
        ]
        tree = IMAPScanner._build_folder_tree(parsed)
        assert len(tree) == 1
        inbox = tree[0]
        assert inbox["name"] == "INBOX"
        assert inbox["selectable"] is False  # synthetisch
        a = inbox["children"][0]
        assert a["name"] == "A"
        assert a["selectable"] is False
        b = a["children"][0]
        assert b["name"] == "B"
        assert b["selectable"] is False
        c = b["children"][0]
        assert c["name"] == "C"
        assert c["selectable"] is True

    def test_synthetic_replaced_by_real(self):
        """Synthetischer Knoten wird durch echten Server-Ordner aktualisiert."""
        parsed = [
            {"name": "INBOX/Sub/Deep", "raw_name": "INBOX/Sub/Deep", "flags": [], "delimiter": "/"},
            {"name": "INBOX/Sub", "raw_name": "INBOX/Sub", "flags": [], "delimiter": "/"},
        ]
        tree = IMAPScanner._build_folder_tree(parsed)
        inbox = tree[0]
        sub = inbox["children"][0]
        assert sub["selectable"] is True  # Real, nicht mehr synthetisch
        assert sub["raw_path"] == "INBOX/Sub"

    def test_raw_path_preserved(self):
        """raw_path traegt den IMAP-kodierten Pfad."""
        parsed = [
            {"name": "Entw\u00fcrfe", "raw_name": "Entw&APw-rfe", "flags": [], "delimiter": "/"},
        ]
        tree = IMAPScanner._build_folder_tree(parsed)
        assert tree[0]["raw_path"] == "Entw&APw-rfe"
        assert tree[0]["path"] == "Entw\u00fcrfe"

    def test_input_unchanged(self):
        """sorted() statt sort() — Eingabeliste bleibt unveraendert."""
        parsed = [
            {"name": "Zzz", "raw_name": "Zzz", "flags": [], "delimiter": "/"},
            {"name": "Aaa", "raw_name": "Aaa", "flags": [], "delimiter": "/"},
        ]
        original = list(parsed)
        IMAPScanner._build_folder_tree(parsed)
        assert parsed == original

    def test_noselect_flag(self):
        """Ordner mit \\Noselect-Flag sind nicht selektierbar."""
        parsed = [
            {
                "name": "Namespace",
                "raw_name": "Namespace",
                "flags": ["\\noselect"],
                "delimiter": "/",
            },
        ]
        tree = IMAPScanner._build_folder_tree(parsed)
        assert tree[0]["selectable"] is False

    def test_empty_input(self):
        assert IMAPScanner._build_folder_tree([]) == []


# ---------------------------------------------------------------------------
# Wildcard-Scan
# ---------------------------------------------------------------------------


class TestWildcardScan:
    """Wildcard-Modus: * -> alle selektierbaren Ordner."""

    def test_wildcard_scan_uses_raw_names(self):
        """Im Wildcard-Modus werden raw_names fuer conn.select() verwendet."""
        config = {
            "imap_server": "test",
            "imap_port": 993,
            "imap_ssl": True,
            "imap_username": "u",
            "imap_password": "p",
            "imap_folders": "*",
            "imap_since_days": 1,
            "imap_download_attachments": False,
            "imap_max_attachment_mb": 1,
        }
        scanner = IMAPScanner(config)
        assert scanner._all_folders is True
        assert scanner._folders == []

        mock_conn = MagicMock()
        mock_conn.list.return_value = (
            "OK",
            [
                b'(\\HasNoChildren) "/" "INBOX"',
                b'(\\Noselect) "/" "Namespace"',
                b'(\\HasNoChildren) "/" "Entw&APw-rfe"',
            ],
        )
        mock_conn.select.return_value = ("OK", [b"0"])
        mock_conn.search.return_value = ("OK", [b""])
        mock_sock = MagicMock()
        mock_conn.sock = mock_sock

        with patch.object(scanner, "connect", return_value=mock_conn):
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                scanner.scan(tmpdir)

        # Nur selektierbare Ordner, raw_name verwendet
        select_calls = [c[0][0] for c in mock_conn.select.call_args_list]
        assert "INBOX" in select_calls
        assert "Entw&APw-rfe" in select_calls
        assert "Namespace" not in select_calls

    def test_wildcard_scan_logout_on_list_error(self):
        """Bei conn.list() Fehler wird conn.logout() aufgerufen."""
        config = {
            "imap_server": "test",
            "imap_port": 993,
            "imap_ssl": True,
            "imap_username": "u",
            "imap_password": "p",
            "imap_folders": "*",
            "imap_since_days": 1,
            "imap_download_attachments": False,
            "imap_max_attachment_mb": 1,
        }
        scanner = IMAPScanner(config)

        mock_conn = MagicMock()
        mock_conn.list.side_effect = Exception("LIST failed")
        mock_conn.sock = MagicMock()

        with patch.object(scanner, "connect", return_value=mock_conn):
            with pytest.raises(Exception, match="LIST failed"):
                import tempfile

                with tempfile.TemporaryDirectory() as tmpdir:
                    scanner.scan(tmpdir)

        # logout wird beim LIST-Fehler in _discover_folders und erneut im finally-Block
        # von scan() aufgerufen (doppelter Logout ist harmlos, da Ausnahmen abgefangen werden).
        mock_conn.logout.assert_called()


# ---------------------------------------------------------------------------
# test_connection — Connection-Leak Fix
# ---------------------------------------------------------------------------


class TestTestConnectionCleanup:
    """test_connection() schliesst die Verbindung auch bei Fehlern."""

    def test_logout_on_list_error(self):
        config = {
            "imap_server": "test",
            "imap_port": 993,
            "imap_ssl": True,
            "imap_username": "u",
            "imap_password": "p",
            "imap_folders": "INBOX",
            "imap_since_days": 1,
            "imap_download_attachments": False,
            "imap_max_attachment_mb": 1,
        }
        scanner = IMAPScanner(config)

        mock_conn = MagicMock()
        mock_conn.list.side_effect = Exception("LIST failed")

        with patch.object(scanner, "connect", return_value=mock_conn):
            result = scanner.test_connection()

        assert result["success"] is False
        assert "LIST failed" in result["message"]
        mock_conn.logout.assert_called_once()

    def test_logout_on_success(self):
        config = {
            "imap_server": "test",
            "imap_port": 993,
            "imap_ssl": True,
            "imap_username": "u",
            "imap_password": "p",
            "imap_folders": "INBOX",
            "imap_since_days": 1,
            "imap_download_attachments": False,
            "imap_max_attachment_mb": 1,
        }
        scanner = IMAPScanner(config)

        mock_conn = MagicMock()
        mock_conn.list.return_value = ("OK", [b'(\\HasNoChildren) "/" "INBOX"'])

        with patch.object(scanner, "connect", return_value=mock_conn):
            result = scanner.test_connection()

        assert result["success"] is True
        assert "INBOX" in result["folders"]
        mock_conn.logout.assert_called_once()


# ---------------------------------------------------------------------------
# _resolve_masked_config
# ---------------------------------------------------------------------------


class TestResolveMaskedConfig:
    """PASSWORD_MASK-Aufloesung in _resolve_masked_config."""

    def test_resolve_from_stored(self):
        from kiara.shared.constants import PASSWORD_MASK
        from kiara.web.api_datasource import _resolve_masked_config

        class FakeConnector:
            @staticmethod
            def secret_fields():
                return ["password"]

        config = {"server": "mail.test.de", "password": PASSWORD_MASK}

        mock_row = {
            "connector_config": '{"server": "mail.test.de", "password": "geheim"}',
            "type": "imap",
        }

        with patch("kiara.web.api_datasource.ds_repo") as mock_ds_repo:
            mock_ds_repo.get_by_id.return_value = mock_row

            result = _resolve_masked_config(config, FakeConnector, ds_id=1)

        assert result["password"] == "geheim"
        assert result["server"] == "mail.test.de"

    def test_no_mask_passthrough(self):
        from kiara.web.api_datasource import _resolve_masked_config

        class FakeConnector:
            @staticmethod
            def secret_fields():
                return ["password"]

        config = {"server": "test", "password": "klartext"}
        result = _resolve_masked_config(config, FakeConnector, ds_id=None)
        assert result["password"] == "klartext"

    def test_mask_without_ds_id_raises(self):
        """PASSWORD_MASK ohne ds_id muss HTTPException 400 werfen."""
        from fastapi import HTTPException

        from kiara.shared.constants import PASSWORD_MASK
        from kiara.web.api_datasource import _resolve_masked_config

        class FakeConnector:
            @staticmethod
            def secret_fields():
                return ["password"]

        config = {"password": PASSWORD_MASK}
        with pytest.raises(HTTPException) as exc_info:
            _resolve_masked_config(config, FakeConnector, ds_id=None)
        assert exc_info.value.status_code == 400

    def test_mask_with_empty_stored_value_raises(self):
        """PASSWORD_MASK mit ds_id aber leerem gespeichertem Wert -> HTTPException."""
        from fastapi import HTTPException

        from kiara.shared.constants import PASSWORD_MASK
        from kiara.web.api_datasource import _resolve_masked_config

        class FakeConnector:
            @staticmethod
            def secret_fields():
                return ["password"]

        config = {"password": PASSWORD_MASK}
        mock_row = {"connector_config": '{"password": ""}', "type": "imap"}

        with patch("kiara.web.api_datasource.ds_repo") as mock_ds_repo:
            mock_ds_repo.get_by_id.return_value = mock_row

            with pytest.raises(HTTPException) as exc_info:
                _resolve_masked_config(config, FakeConnector, ds_id=1)
            assert exc_info.value.status_code == 500

    def test_input_dict_not_mutated(self):
        """_resolve_masked_config darf das uebergebene config-Dict nicht mutieren."""
        from kiara.web.api_datasource import _resolve_masked_config

        class FakeConnector:
            @staticmethod
            def secret_fields():
                return ["password"]

        config = {"server": "test", "password": "klartext"}
        original_password = config["password"]
        _resolve_masked_config(config, FakeConnector, ds_id=None)
        assert config["password"] == original_password

    def test_mask_with_wrong_datasource_type_raises(self):
        """PASSWORD_MASK darf nicht aus einer DS anderen Typs uebernommen werden."""
        from fastapi import HTTPException

        from kiara.shared.constants import PASSWORD_MASK
        from kiara.web.api_datasource import _resolve_masked_config

        class FakeConnector:
            @staticmethod
            def secret_fields():
                return ["password"]

        config = {"password": PASSWORD_MASK}
        mock_row = {"connector_config": '{"password": "secret"}', "type": "imap"}

        with patch("kiara.web.api_datasource.ds_repo") as mock_ds_repo:
            mock_ds_repo.get_by_id.return_value = mock_row

            with pytest.raises(HTTPException) as exc_info:
                _resolve_masked_config(
                    config,
                    FakeConnector,
                    ds_id=1,
                    expected_type="webdav",
                )

        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# list_folders() Connector-Methode
# ---------------------------------------------------------------------------


class TestListFoldersConnector:
    """list_folders() auf IMAPConnector / ExchangeOnlineConnector."""

    def test_imap_list_folders_returns_tree(self):
        """IMAPConnector.list_folders() gibt hierarchischen Baum zurueck."""
        from kiara.connectors.imap_connector import IMAPConnector

        mock_conn = MagicMock()
        mock_conn.list.return_value = (
            "OK",
            [
                b'(\\HasChildren) "/" "INBOX"',
                b'(\\HasNoChildren) "/" "INBOX/Sent"',
                b'(\\HasNoChildren) "/" "Drafts"',
            ],
        )
        mock_conn.sock = MagicMock()

        config = {
            "server": "mail.test.de",
            "port": 993,
            "ssl": True,
            "username": "u",
            "password": "p",
            "folders": "INBOX",
        }

        # Lazy Import in list_folders() — patch auf Quell-Modul
        with patch.object(IMAPScanner, "connect", return_value=mock_conn):
            tree = IMAPConnector.list_folders(config)

        assert len(tree) >= 2  # INBOX (mit Kind) + Drafts
        inbox = next(n for n in tree if n["name"] == "INBOX")
        assert len(inbox["children"]) == 1
        assert inbox["children"][0]["name"] == "Sent"
        mock_conn.logout.assert_called_once()

    def test_imap_list_folders_logout_on_error(self):
        """list_folders() schliesst Verbindung auch bei Fehler."""
        from kiara.connectors.imap_connector import IMAPConnector

        mock_conn = MagicMock()
        mock_conn.list.side_effect = Exception("Timeout")
        mock_conn.sock = MagicMock()

        config = {
            "server": "mail.test.de",
            "port": 993,
            "ssl": True,
            "username": "u",
            "password": "p",
            "folders": "INBOX",
        }

        with patch.object(IMAPScanner, "connect", return_value=mock_conn):
            with pytest.raises(Exception, match="Timeout"):
                IMAPConnector.list_folders(config)

        mock_conn.logout.assert_called_once()

    def test_exchange_list_folders_returns_tree(self):
        """ExchangeOnlineConnector.list_folders() gibt Baum zurueck."""
        from kiara.connectors.exchange_connector import ExchangeOnlineConnector
        from kiara.ingestion.oauth2_imap_scanner import OAuth2IMAPScanner

        mock_conn = MagicMock()
        mock_conn.list.return_value = (
            "OK",
            [
                b'(\\HasNoChildren) "/" "INBOX"',
                b'(\\HasNoChildren) "/" "Sent Items"',
            ],
        )
        mock_conn.sock = MagicMock()

        config = {
            "tenant_id": "t",
            "client_id": "c",
            "client_secret": "s",
            "username": "u@test.de",
            "folders": "INBOX",
        }

        with patch.object(OAuth2IMAPScanner, "connect", return_value=mock_conn):
            tree = ExchangeOnlineConnector.list_folders(config)

        assert len(tree) == 2
        names = [n["name"] for n in tree]
        assert "INBOX" in names
        assert "Sent Items" in names
        mock_conn.logout.assert_called_once()


# ---------------------------------------------------------------------------
# list-folders API-Endpoint
# ---------------------------------------------------------------------------


class TestListFoldersEndpoint:
    """API-Endpoint /api/admin/datasources/list-folders."""

    def test_unsupported_connector_type(self):
        """Connector ohne list_folders() wirft HTTPException 400."""
        from fastapi.exceptions import HTTPException

        from kiara.web.api_datasource import ConnectorActionRequest, list_datasource_folders

        body = ConnectorActionRequest(
            type="local",
            config={"path": "/tmp"},
        )

        # Connector ohne list_folders
        class FakeLocal:
            pass

        with (
            patch("kiara.web.api_datasource.ensure_loaded"),
            patch("kiara.web.api_datasource.get_connector", return_value=FakeLocal),
        ):
            with pytest.raises(HTTPException) as exc_info:
                list_datasource_folders(body, user={"username": "admin"})

        assert exc_info.value.status_code == 400
        assert "nicht unterstuetzt" in exc_info.value.detail


# ---------------------------------------------------------------------------
# _parse_imap_list — zusaetzliche Edge Cases
# ---------------------------------------------------------------------------


class TestParseImapListExtended:
    """Erweiterte Parsing-Tests."""

    def test_multiple_flags(self):
        """Mehrere Flags in einem Eintrag, alle normalisiert."""
        raw = [b'(\\HasChildren \\Marked \\Subscribed) "/" "INBOX"']
        result = IMAPScanner._parse_imap_list(raw)
        flags = result[0]["flags"]
        assert "\\haschildren" in flags
        assert "\\marked" in flags
        assert "\\subscribed" in flags
        assert len(flags) == 3

    def test_empty_flags(self):
        """Leere Flag-Liste."""
        raw = [b'() "/" "INBOX"']
        result = IMAPScanner._parse_imap_list(raw)
        assert result[0]["flags"] == []

    def test_mixed_bytes_and_tuples(self):
        """Gemischte Liste aus normalen bytes und Literal-Tuples."""
        raw = [
            b'(\\HasNoChildren) "/" "INBOX"',
            (b'(\\HasNoChildren) "/" {11}', b"Sent Items"),
            b'(\\HasNoChildren) "/" "Drafts"',
        ]
        result = IMAPScanner._parse_imap_list(raw)
        assert len(result) == 3
        names = [r["name"] for r in result]
        assert "INBOX" in names
        assert "Sent Items" in names
        assert "Drafts" in names


# ---------------------------------------------------------------------------
# _build_folder_tree — zusaetzliche Edge Cases
# ---------------------------------------------------------------------------


class TestBuildFolderTreeExtended:
    """Erweiterte Baum-Tests."""

    def test_dot_delimiter(self):
        """Baumaufbau mit Punkt-Delimiter (z.B. Strstr-Server)."""
        parsed = [
            {"name": "INBOX", "raw_name": "INBOX", "flags": [], "delimiter": "."},
            {"name": "INBOX.Sent", "raw_name": "INBOX.Sent", "flags": [], "delimiter": "."},
            {
                "name": "INBOX.Sent.Archive",
                "raw_name": "INBOX.Sent.Archive",
                "flags": [],
                "delimiter": ".",
            },
        ]
        tree = IMAPScanner._build_folder_tree(parsed)
        assert len(tree) == 1
        inbox = tree[0]
        assert inbox["name"] == "INBOX"
        sent = inbox["children"][0]
        assert sent["name"] == "Sent"
        assert sent["children"][0]["name"] == "Archive"

    def test_siblings_sorted_alphabetically(self):
        """Geschwister-Knoten werden alphabetisch sortiert."""
        parsed = [
            {"name": "Zzz", "raw_name": "Zzz", "flags": [], "delimiter": "/"},
            {"name": "Aaa", "raw_name": "Aaa", "flags": [], "delimiter": "/"},
            {"name": "Mmm", "raw_name": "Mmm", "flags": [], "delimiter": "/"},
        ]
        tree = IMAPScanner._build_folder_tree(parsed)
        names = [n["name"] for n in tree]
        assert names == ["Aaa", "Mmm", "Zzz"]

    def test_synthetic_parent_children_preserved(self):
        """Wenn ein synthetischer Parent durch einen realen ersetzt wird,
        bleiben die Children erhalten."""
        parsed = [
            {"name": "A/B", "raw_name": "A/B", "flags": [], "delimiter": "/"},
            {"name": "A/C", "raw_name": "A/C", "flags": [], "delimiter": "/"},
            {"name": "A", "raw_name": "A", "flags": [], "delimiter": "/"},
        ]
        tree = IMAPScanner._build_folder_tree(parsed)
        a_node = tree[0]
        assert a_node["name"] == "A"
        assert a_node["selectable"] is True
        assert len(a_node["children"]) == 2


# ---------------------------------------------------------------------------
# __init__ Wildcard-Erkennung
# ---------------------------------------------------------------------------


class TestWildcardInit:
    """Wildcard-Erkennung im Scanner-Konstruktor."""

    def test_star_with_whitespace(self):
        """' * ' (mit Leerzeichen) wird als Wildcard erkannt."""
        config = {
            "imap_server": "test",
            "imap_port": 993,
            "imap_ssl": True,
            "imap_username": "u",
            "imap_password": "p",
            "imap_folders": "  *  ",
            "imap_since_days": 1,
            "imap_download_attachments": False,
            "imap_max_attachment_mb": 1,
        }
        scanner = IMAPScanner(config)
        assert scanner._all_folders is True

    def test_star_with_other_folders_not_wildcard(self):
        """'*,INBOX' ist KEIN Wildcard — wird als Ordnerliste behandelt."""
        config = {
            "imap_server": "test",
            "imap_port": 993,
            "imap_ssl": True,
            "imap_username": "u",
            "imap_password": "p",
            "imap_folders": "*,INBOX",
            "imap_since_days": 1,
            "imap_download_attachments": False,
            "imap_max_attachment_mb": 1,
        }
        scanner = IMAPScanner(config)
        assert scanner._all_folders is False
        assert "*" in scanner._folders

    def test_normal_folders_parsed(self):
        """Kommaseparierte Ordner werden korrekt aufgeteilt."""
        config = {
            "imap_server": "test",
            "imap_port": 993,
            "imap_ssl": True,
            "imap_username": "u",
            "imap_password": "p",
            "imap_folders": "INBOX, Sent, Drafts",
            "imap_since_days": 1,
            "imap_download_attachments": False,
            "imap_max_attachment_mb": 1,
        }
        scanner = IMAPScanner(config)
        assert scanner._all_folders is False
        assert scanner._folders == ["INBOX", "Sent", "Drafts"]

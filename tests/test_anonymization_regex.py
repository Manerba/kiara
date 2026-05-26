"""Tests fuer RegexPatternMatcher (Strukturmuster-Erkennung).

Testet: IBAN, Steuernummern, Telefonnummern, E-Mail-Adressen,
False-Positive-Vermeidung, already_covered-Parameter und IPv4-Validierung.
"""

import pytest

from kiara.anonymization.patterns import _BUILTIN_PATTERNS, RegexPatternMatcher

# ---------------------------------------------------------------------------
# IBAN-Erkennung
# ---------------------------------------------------------------------------


class TestIBAN:
    def test_valid_iban_compact(self):
        """DE-IBAN ohne Leerzeichen wird erkannt."""
        matcher = RegexPatternMatcher()
        text = "Konto: DE89370400440532013000 bitte ueberweisen."
        results = matcher.find_patterns(text)
        ibans = [r for r in results if r["entity_type"] == "IBAN"]
        assert len(ibans) == 1
        assert ibans[0]["value"] == "DE89370400440532013000"

    def test_valid_iban_with_spaces(self):
        """DE-IBAN mit Leerzeichen wird erkannt."""
        matcher = RegexPatternMatcher()
        text = "IBAN: DE89 3704 0044 0532 0130 00"
        results = matcher.find_patterns(text)
        ibans = [r for r in results if r["entity_type"] == "IBAN"]
        assert len(ibans) == 1
        assert "DE89" in ibans[0]["value"]

    def test_invalid_iban_too_short(self):
        """Zu kurze Nummern werden nicht als IBAN erkannt."""
        matcher = RegexPatternMatcher()
        text = "Code: DE891234 ist kein IBAN."
        results = matcher.find_patterns(text)
        ibans = [r for r in results if r["entity_type"] == "IBAN"]
        assert len(ibans) == 0

    def test_iban_non_de_prefix_also_matched(self):
        """Nicht-DE-IBANs werden mit Laenderlaenge und Checksumme erkannt."""
        matcher = RegexPatternMatcher()
        text = "IBAN: AT611904300234573201"
        results = matcher.find_patterns(text)
        ibans = [r for r in results if r["entity_type"] == "IBAN"]
        assert len(ibans) == 1

    def test_invalid_iban_checksum_rejected(self):
        """IBANs mit falscher Mod-97-Checksumme werden nicht erkannt."""
        matcher = RegexPatternMatcher()
        text = "Konto: DE89370400440532013001"
        results = matcher.find_patterns(text)
        ibans = [r for r in results if r["entity_type"] == "IBAN"]
        assert ibans == []

    def test_invalid_iban_country_length_rejected(self):
        """Landesspezifisch falsche IBAN-Laengen werden nicht erkannt."""
        matcher = RegexPatternMatcher()
        text = "Konto: DE893704004405320130001"
        results = matcher.find_patterns(text)
        ibans = [r for r in results if r["entity_type"] == "IBAN"]
        assert ibans == []

    def test_invalid_iban_unknown_country_rejected(self):
        """Unbekannte IBAN-Laendercodes werden nicht erkannt."""
        matcher = RegexPatternMatcher()
        text = "Konto: ZZ89370400440532013000"
        results = matcher.find_patterns(text)
        ibans = [r for r in results if r["entity_type"] == "IBAN"]
        assert ibans == []

    def test_multiple_ibans(self):
        """Mehrere IBANs im Text werden alle erkannt."""
        matcher = RegexPatternMatcher()
        text = "Von DE89370400440532013000 nach DE44500105175407324931."
        results = matcher.find_patterns(text)
        ibans = [r for r in results if r["entity_type"] == "IBAN"]
        assert len(ibans) == 2

    def test_iban_positions_correct(self):
        """Start/End-Positionen stimmen mit dem Textausschnitt ueberein."""
        matcher = RegexPatternMatcher()
        text = "IBAN: DE89370400440532013000."
        results = matcher.find_patterns(text)
        assert len(results) >= 1
        iban = next(r for r in results if r["entity_type"] == "IBAN")
        assert text[iban["start"] : iban["end"]] == iban["value"]


# ---------------------------------------------------------------------------
# Steuernummern
# ---------------------------------------------------------------------------


class TestTaxId:
    def test_tax_id_three_part_format(self):
        """Steuernummer im Format 123/456/78901."""
        matcher = RegexPatternMatcher()
        text = "Steuernummer: 123/456/78901."
        results = matcher.find_patterns(text)
        tax = [r for r in results if r["entity_type"] == "TAX_ID"]
        assert len(tax) == 1
        assert tax[0]["value"] == "123/456/78901"

    def test_tax_id_short_format(self):
        """Steuernummer im Format 12/345/6789."""
        matcher = RegexPatternMatcher()
        text = "StNr: 12/345/6789."
        results = matcher.find_patterns(text)
        tax = [r for r in results if r["entity_type"] == "TAX_ID"]
        assert len(tax) == 1
        assert tax[0]["value"] == "12/345/6789"

    def test_ust_id_de(self):
        """USt-ID DE123456789."""
        matcher = RegexPatternMatcher()
        text = "USt-IdNr: DE123456789."
        results = matcher.find_patterns(text)
        tax = [r for r in results if r["entity_type"] == "TAX_ID"]
        assert len(tax) == 1
        assert "DE123456789" in tax[0]["value"]

    def test_ust_id_with_space(self):
        """USt-ID mit Leerzeichen: DE 123456789."""
        matcher = RegexPatternMatcher()
        text = "USt-ID: DE 123456789."
        results = matcher.find_patterns(text)
        tax = [r for r in results if r["entity_type"] == "TAX_ID"]
        assert len(tax) == 1


# ---------------------------------------------------------------------------
# Telefonnummern
# ---------------------------------------------------------------------------


class TestPhone:
    def test_international_format(self):
        """Internationale Nummer +49 711 12345678."""
        matcher = RegexPatternMatcher()
        text = "Tel: +49 711 12345678."
        results = matcher.find_patterns(text)
        phones = [r for r in results if r["entity_type"] == "PHONE"]
        assert len(phones) == 1
        assert "+49" in phones[0]["value"]

    def test_national_format_with_space(self):
        """Nationale Nummer 0711 1234567."""
        matcher = RegexPatternMatcher()
        text = "Telefon: 0711 1234567."
        results = matcher.find_patterns(text)
        phones = [r for r in results if r["entity_type"] == "PHONE"]
        assert len(phones) == 1

    def test_national_format_with_slash(self):
        """Nationale Nummer 0711/1234567."""
        matcher = RegexPatternMatcher()
        text = "Ruf: 0711/1234567."
        results = matcher.find_patterns(text)
        phones = [r for r in results if r["entity_type"] == "PHONE"]
        assert len(phones) == 1

    def test_international_compact(self):
        """+49711-12345678 (kompakt)."""
        matcher = RegexPatternMatcher()
        text = "Fax: +49711-12345678."
        results = matcher.find_patterns(text)
        phones = [r for r in results if r["entity_type"] == "PHONE"]
        assert len(phones) == 1

    def test_mobile_number(self):
        """Mobilnummer 0172 1234567."""
        matcher = RegexPatternMatcher()
        text = "Mobil: 0172 1234567."
        results = matcher.find_patterns(text)
        phones = [r for r in results if r["entity_type"] == "PHONE"]
        assert len(phones) == 1

    def test_national_compact_number(self):
        """Kompakte nationale Nummer 07111234567."""
        matcher = RegexPatternMatcher()
        text = "Zentrale: 07111234567."
        results = matcher.find_patterns(text)
        phones = [r for r in results if r["entity_type"] == "PHONE"]
        assert len(phones) == 1
        assert phones[0]["value"] == "07111234567"

    def test_international_grouped_number(self):
        """Gruppierte internationale Nummer +49 711 123 4567."""
        matcher = RegexPatternMatcher()
        text = "Hotline: +49 711 123 4567."
        results = matcher.find_patterns(text)
        phones = [r for r in results if r["entity_type"] == "PHONE"]
        assert len(phones) == 1
        assert phones[0]["value"] == "+49 711 123 4567"

    def test_international_german_optional_trunk_prefix(self):
        """Internationales Format +49 (0)30 12345."""
        matcher = RegexPatternMatcher()
        text = "Telefon: +49 (0)30 12345."
        results = matcher.find_patterns(text)
        phones = [r for r in results if r["entity_type"] == "PHONE"]
        assert len(phones) == 1
        assert phones[0]["value"] == "+49 (0)30 12345"

    def test_international_german_optional_trunk_prefix_with_space(self):
        """Internationales Format +49 (0) 30 12345."""
        matcher = RegexPatternMatcher()
        text = "Telefon: +49 (0) 30 12345."
        results = matcher.find_patterns(text)
        phones = [r for r in results if r["entity_type"] == "PHONE"]
        assert len(phones) == 1
        assert phones[0]["value"] == "+49 (0) 30 12345"


# ---------------------------------------------------------------------------
# E-Mail
# ---------------------------------------------------------------------------


class TestEmail:
    def test_standard_email(self):
        """Standard-E-Mail user@example.com."""
        matcher = RegexPatternMatcher()
        text = "Mail an user@example.com senden."
        results = matcher.find_patterns(text)
        emails = [r for r in results if r["entity_type"] == "EMAIL"]
        assert len(emails) == 1
        assert emails[0]["value"] == "user@example.com"

    def test_email_with_dots(self):
        """E-Mail mit Punkten vorname.nachname@firma.de."""
        matcher = RegexPatternMatcher()
        text = "Kontakt: hans.mueller@firma.de"
        results = matcher.find_patterns(text)
        emails = [r for r in results if r["entity_type"] == "EMAIL"]
        assert len(emails) == 1
        assert emails[0]["value"] == "hans.mueller@firma.de"

    def test_email_with_subdomain(self):
        """E-Mail mit Subdomain user@mail.corp.example.com."""
        matcher = RegexPatternMatcher()
        text = "Erreichbar unter admin@mail.corp.example.com"
        results = matcher.find_patterns(text)
        emails = [r for r in results if r["entity_type"] == "EMAIL"]
        assert len(emails) == 1

    def test_email_with_plus(self):
        """E-Mail mit Plus-Adressierung user+tag@example.com."""
        matcher = RegexPatternMatcher()
        text = "Filter: user+newsletter@example.com"
        results = matcher.find_patterns(text)
        emails = [r for r in results if r["entity_type"] == "EMAIL"]
        assert len(emails) == 1

    def test_email_with_hyphen(self):
        """E-Mail mit Bindestrich my-name@my-domain.org."""
        matcher = RegexPatternMatcher()
        text = "Kontakt: my-name@my-domain.org"
        results = matcher.find_patterns(text)
        emails = [r for r in results if r["entity_type"] == "EMAIL"]
        assert len(emails) == 1

    def test_email_inside_url_userinfo_not_matched(self):
        """E-Mail in URL-Userinfo wird nicht als separate Mail maskiert."""
        matcher = RegexPatternMatcher()
        text = "Siehe http://user@example.com/path und mail user@example.com."
        results = matcher.find_patterns(text)
        emails = [r for r in results if r["entity_type"] == "EMAIL"]
        assert len(emails) == 1
        assert emails[0]["value"] == "user@example.com"
        assert emails[0]["start"] == text.rindex("user@example.com")


# ---------------------------------------------------------------------------
# False Positives
# ---------------------------------------------------------------------------


class TestFalsePositives:
    def test_year_not_matched_as_tax_id(self):
        """Jahreszahlen 2024, 2025 sind keine Steuernummern."""
        matcher = RegexPatternMatcher()
        text = "Im Jahr 2024 und 2025 wurde umgestellt."
        results = matcher.find_patterns(text)
        tax = [r for r in results if r["entity_type"] == "TAX_ID"]
        assert len(tax) == 0

    def test_article_number_not_phone(self):
        """Kurze Artikelnummer darf nicht als Telefon erkannt werden."""
        matcher = RegexPatternMatcher()
        text = "Artikel 42 wurde bestellt."
        results = matcher.find_patterns(text)
        phones = [r for r in results if r["entity_type"] == "PHONE"]
        assert len(phones) == 0

    def test_short_number_not_tax_id(self):
        """Kurze Zahlen wie 12345 duerfen nicht als Steuernummer erkannt werden."""
        matcher = RegexPatternMatcher()
        text = "Bestellnummer 12345 bearbeitet."
        results = matcher.find_patterns(text)
        tax = [r for r in results if r["entity_type"] == "TAX_ID"]
        assert len(tax) == 0

    def test_plain_text_no_matches(self):
        """Normaler Text ohne PII gibt leere Liste zurueck."""
        matcher = RegexPatternMatcher()
        text = "Das Wetter ist heute sonnig und warm."
        results = matcher.find_patterns(text)
        assert results == []


# ---------------------------------------------------------------------------
# already_covered Parameter
# ---------------------------------------------------------------------------


class TestAlreadyCovered:
    def test_covered_range_skipped(self):
        """Bereiche in already_covered werden uebersprungen."""
        matcher = RegexPatternMatcher()
        text = "Mail: test@example.com erreichbar."
        email_start = text.index("test@example.com")
        email_end = email_start + len("test@example.com")

        # Ohne Coverage: Email wird erkannt
        results_without = matcher.find_patterns(text)
        emails = [r for r in results_without if r["entity_type"] == "EMAIL"]
        assert len(emails) == 1

        # Mit Coverage: Email-Bereich ist abgedeckt
        results_with = matcher.find_patterns(text, already_covered=[(email_start, email_end)])
        emails_covered = [r for r in results_with if r["entity_type"] == "EMAIL"]
        assert len(emails_covered) == 0

    def test_partial_overlap_skipped(self):
        """Teilweise ueberlappende Bereiche werden ebenfalls uebersprungen."""
        matcher = RegexPatternMatcher()
        text = "IBAN: DE89370400440532013000 hier."
        # Decke die ersten Zeichen der IBAN ab
        iban_start = text.index("DE89")
        covered = [(iban_start, iban_start + 10)]

        results = matcher.find_patterns(text, already_covered=covered)
        ibans = [r for r in results if r["entity_type"] == "IBAN"]
        assert len(ibans) == 0

    def test_non_overlapping_still_found(self):
        """Nicht abgedeckte Bereiche werden weiterhin erkannt."""
        matcher = RegexPatternMatcher()
        text = "Mail: a@b.com, IBAN: DE89370400440532013000."
        email_start = text.index("a@b.com")
        email_end = email_start + len("a@b.com")

        # Nur die E-Mail abdecken
        results = matcher.find_patterns(text, already_covered=[(email_start, email_end)])
        ibans = [r for r in results if r["entity_type"] == "IBAN"]
        assert len(ibans) == 1

    def test_empty_covered_list(self):
        """Leere already_covered-Liste verhindert nichts."""
        matcher = RegexPatternMatcher()
        text = "Mail: test@example.com"
        results_empty = matcher.find_patterns(text, already_covered=[])
        results_none = matcher.find_patterns(text, already_covered=None)
        assert len(results_empty) == len(results_none)


# ---------------------------------------------------------------------------
# Custom Patterns
# ---------------------------------------------------------------------------


class TestCustomPatterns:
    def test_custom_pattern_added(self):
        """Benutzerdefiniertes Muster wird erkannt."""
        custom = [
            {"name": "projekt_id", "pattern": r"PRJ-\d{4}", "category": "project"},
        ]
        matcher = RegexPatternMatcher(custom_patterns=custom)
        text = "Projekt PRJ-1234 gestartet."
        results = matcher.find_patterns(text)
        projects = [r for r in results if r["entity_type"] == "PROJECT"]
        assert len(projects) == 1
        assert projects[0]["value"] == "PRJ-1234"

    def test_invalid_custom_pattern_ignored(self):
        """Ungueltige Regex-Muster werden stillschweigend ignoriert."""
        custom = [
            {"name": "bad", "pattern": r"[invalid", "category": "custom"},
        ]
        matcher = RegexPatternMatcher(custom_patterns=custom)
        # Kein Fehler, andere Patterns funktionieren weiterhin
        text = "Mail: test@example.com"
        results = matcher.find_patterns(text)
        assert len(results) >= 1

    def test_custom_and_builtin_coexist(self):
        """Builtin- und Custom-Patterns arbeiten zusammen."""
        custom = [
            {"name": "ticket", "pattern": r"TKT-\d{6}", "category": "ticket"},
        ]
        matcher = RegexPatternMatcher(custom_patterns=custom)
        text = "TKT-123456 betrifft test@example.com."
        results = matcher.find_patterns(text)
        categories = {r["entity_type"] for r in results}
        assert "TICKET" in categories
        assert "EMAIL" in categories

    def test_longer_later_custom_match_wins_overlap(self):
        """Laengerer spaeterer Treffer gewinnt gegen frueheres Kurzpattern."""
        custom = [
            {"name": "short_ref", "pattern": r"ABC-\d{3}", "type": "SHORT"},
            {"name": "long_ref", "pattern": r"ABC-\d{3}-XYZ", "type": "LONG"},
        ]
        matcher = RegexPatternMatcher(custom_patterns=custom)
        results = matcher.find_patterns("Referenz ABC-123-XYZ ist aktiv.")
        refs = [r for r in results if r["entity_type"] in {"SHORT", "LONG"}]
        assert len(refs) == 1
        assert refs[0]["entity_type"] == "LONG"
        assert refs[0]["value"] == "ABC-123-XYZ"

    @pytest.mark.parametrize(
        "pattern",
        [
            r"^(a|aa)+$",
            r"([A-Z]+)+",
            r"(.*)+",
        ],
    )
    def test_dangerous_custom_regex_ignored(self, pattern):
        """Offensichtlich ReDoS-gefaehrliche Custom-Regexe werden abgelehnt."""
        custom = [{"name": "bad", "pattern": pattern, "type": "BAD"}]
        matcher = RegexPatternMatcher(custom_patterns=custom)
        results = matcher.find_patterns("ABC aaaa")
        assert [r for r in results if r["entity_type"] == "BAD"] == []

    @pytest.mark.parametrize("bad_type", [1, True, None, 3.14])
    def test_custom_pattern_non_string_type_does_not_crash(self, bad_type):
        """Nicht-String type/category in YAML-Configs sprengt nicht den Loop.

        Vor dem Fix scheiterte .islower()/.upper() mit AttributeError, sobald
        type oder category aus YAML als Integer/Bool/None geladen wurden, und
        ALLE custom_patterns wurden verworfen. Der Fix coerc't via str().upper().
        """
        custom = [
            {"name": "kaputt", "pattern": r"FOO-\d+", "type": bad_type},
            {"name": "gut", "pattern": r"BAR-\d+", "type": "GOOD"},
        ]
        matcher = RegexPatternMatcher(custom_patterns=custom)
        results = matcher.find_patterns("FOO-123 und BAR-456 zusammen.")
        good = [r for r in results if r["entity_type"] == "GOOD"]
        assert len(good) == 1
        assert good[0]["value"] == "BAR-456"


# ---------------------------------------------------------------------------
# IPv4-Validierung (regex_fallback — Pseudonymizer-Pipeline)
# ---------------------------------------------------------------------------


def _get_ipv4_pattern():
    """Holt das IPv4-Pattern aus den Fallback-Patterns."""
    for p in _BUILTIN_PATTERNS:
        if p["name"] == "ipv4_private":
            return p["pattern"]
    pytest.fail("IPv4-Pattern nicht in _BUILTIN_PATTERNS gefunden")


class TestIPv4Fallback:
    def test_valid_private_10(self):
        """10.x.x.x Private-Range wird erkannt."""
        pat = _get_ipv4_pattern()
        assert pat.search("Server kiara-server.local ist erreichbar")

    def test_valid_private_172(self):
        """172.16-31.x.x Private-Range wird erkannt."""
        pat = _get_ipv4_pattern()
        assert pat.search("Netzwerk 172.16.0.1 konfiguriert")
        assert pat.search("Gateway 172.31.255.254 aktiv")

    def test_valid_private_192(self):
        """192.168.x.x Private-Range wird erkannt."""
        pat = _get_ipv4_pattern()
        assert pat.search("DHCP 192.168.1.100 vergeben")

    def test_rejects_public_ip(self):
        """Oeffentliche IPs werden NICHT erkannt (nur private Ranges)."""
        pat = _get_ipv4_pattern()
        assert not pat.search("Google DNS 8.8.8.8 erreichbar")
        assert not pat.search("Server 1.2.3.4 antwortet")

    def test_rejects_invalid_octet_over_255(self):
        """Oktette > 255 werden nicht akzeptiert."""
        pat = _get_ipv4_pattern()
        assert not pat.search("Fehler bei 10.0.256.1")
        assert not pat.search("IP 192.168.1.999 ungueltig")
        assert not pat.search("Adresse 10.300.1.1 falsch")

    def test_rejects_172_outside_private(self):
        """172.x ausserhalb 16-31 wird nicht erkannt."""
        pat = _get_ipv4_pattern()
        assert not pat.search("172.15.0.1 ist public")
        assert not pat.search("172.32.0.1 ist public")

    def test_boundary_octets(self):
        """Grenzwerte 0 und 255 werden korrekt erkannt."""
        pat = _get_ipv4_pattern()
        assert pat.search("10.0.0.0")
        assert pat.search("10.255.255.255")
        assert pat.search("192.168.0.1")
        assert pat.search("192.168.255.255")

    def test_embedded_in_text(self):
        """IP eingebettet in Fliesstext wird korrekt extrahiert."""
        pat = _get_ipv4_pattern()
        m = pat.search("Der Server unter gpu-server.local laeuft Ollama.")
        assert m
        assert m.group() == "gpu-server.local"

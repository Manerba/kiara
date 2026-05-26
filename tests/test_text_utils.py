"""Tests fuer kiara.shared.text_utils — sanitize_utf8()."""

from kiara.shared.text_utils import sanitize_utf8


class TestSanitizeUtf8:
    """Testet die sanitize_utf8() Utility-Funktion."""

    def test_empty_string(self):
        assert sanitize_utf8("") == ""

    def test_none_input(self):
        assert sanitize_utf8(None) == ""

    def test_ascii_unchanged(self):
        assert sanitize_utf8("Hello World") == "Hello World"

    def test_umlauts_unchanged(self):
        """Deutsche Umlaute (valides Unicode) duerfen NICHT veraendert werden."""
        text = "Hallo Welt äöüÄÖÜß"
        assert sanitize_utf8(text) == text

    def test_lone_surrogate_replaced(self):
        """Lone Surrogates (U+D800-U+DFFF) werden durch '?' ersetzt."""
        text = "text\ud800rest"
        result = sanitize_utf8(text)
        assert result == "text?rest"

    def test_multiple_surrogates(self):
        """Mehrere Lone Surrogates werden alle ersetzt."""
        text = "\ud800abc\udc00def\udfff"
        result = sanitize_utf8(text)
        assert "\ud800" not in result
        assert "\udc00" not in result
        assert "\udfff" not in result
        assert "abc" in result
        assert "def" in result

    def test_mixed_content_preserved(self):
        """Gemischter Text mit Umlauten + Sonderzeichen bleibt erhalten."""
        text = "Prüfbericht für Ölfilter — 100% geprüft"
        assert sanitize_utf8(text) == text

    def test_emoji_preserved(self):
        """Emoji (valides Unicode) werden nicht veraendert."""
        text = "Status: ✅ OK"
        assert sanitize_utf8(text) == text

    def test_return_type_always_str(self):
        """Rueckgabetyp ist immer str, nie None."""
        assert isinstance(sanitize_utf8(None), str)
        assert isinstance(sanitize_utf8(""), str)
        assert isinstance(sanitize_utf8("test"), str)

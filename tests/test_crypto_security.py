"""Security-Tests fuer F-01: decrypt() muss bei ungueltigem Token eine Exception
werfen statt den Klartext zurueckzugeben.

Testet encrypt/decrypt-Roundtrip, Sonderfaelle (None, leer) und das
korrekte Verhalten bei korrupten/ungueltigen Eingaben.
"""

from unittest.mock import patch

import pytest
from cryptography.fernet import InvalidToken

# ---------------------------------------------------------------------------
# Fixture: temporaerer Secret-Key
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tmp_secret_key(tmp_path):
    """Erstellt einen temporaeren Secret-Key, damit die Tests
    unabhaengig vom System-Key laufen (kein /var/lib/kiara noetig).
    """
    key_file = tmp_path / "secret.key"
    key_file.write_text("deadbeef" * 8)
    key_file.chmod(0o600)
    with patch("kiara.shared.crypto.SECRET_KEY_PATH", key_file):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

from kiara.shared.crypto import decrypt, encrypt, is_encrypted


class TestDecryptSecurity:
    """F-01: decrypt() darf bei ungueltigem Token NICHT den Input zurueckgeben."""

    def test_decrypt_invalid_token_raises_exception(self):
        """decrypt() mit ungueltigem Wert muss eine Exception werfen."""
        with pytest.raises(InvalidToken):
            decrypt("ungueltigerWert")

    def test_decrypt_empty_returns_empty(self):
        """decrypt('') gibt '' zurueck (Leerstring-Guard)."""
        assert decrypt("") == ""

    def test_decrypt_none_returns_none(self):
        """decrypt(None) gibt None zurueck wie encrypt(None)."""
        assert decrypt(None) is None

    def test_encrypt_decrypt_roundtrip(self):
        """encrypt → decrypt muss den Original-Klartext zurueckgeben."""
        original = "geheim"
        encrypted = encrypt(original)
        assert encrypted != original
        assert decrypt(encrypted) == original

    def test_decrypt_does_not_return_input_on_failure(self):
        """Bei ungueltigem Input darf der Rueckgabewert NICHT der Input selbst sein."""
        bad_input = "dies_ist_kein_fernet_token"
        with pytest.raises(InvalidToken):
            result = decrypt(bad_input)
            # Falls decrypt wider Erwarten nicht wirft, sicherstellen
            # dass der Input nicht einfach durchgereicht wird:
            assert result != bad_input  # pragma: no cover

    def test_decrypt_random_base64_raises(self):
        """Zufaelliger Base64-String (kein gueltiger Fernet-Token) wirft Exception."""
        import base64

        fake = base64.urlsafe_b64encode(b"x" * 32).decode()
        with pytest.raises(InvalidToken):
            decrypt(fake)

    def test_decrypt_gaaaaa_prefix_but_invalid_raises(self):
        """Ein String mit gAAAAA-Prefix, aber ohne gueltigen Token, wirft Exception."""
        with pytest.raises(InvalidToken):
            decrypt("gAAAAABthis_is_not_valid")

    def test_is_encrypted_rejects_undecryptable_fernet_like_value(self):
        """is_encrypted() darf Fernet-aehnliche Fremdtokens nicht akzeptieren."""
        from cryptography.fernet import Fernet

        foreign = Fernet(Fernet.generate_key()).encrypt(b"geheim").decode()

        assert foreign.startswith("gAAAAA")
        assert is_encrypted(foreign) is False

    def test_is_encrypted_accepts_current_key_token(self):
        """Mit aktuellem Secret erzeugte Tokens bleiben als verschluesselt erkannt."""
        token = encrypt("geheim")

        assert is_encrypted(token) is True

    def test_encrypt_secret_fields_rejects_undecryptable_fernet_like_value(self):
        """Persistenzpfad darf Fremd-Fernet-Tokens nicht doppelt verschluesseln."""
        from cryptography.fernet import Fernet

        from kiara.shared.crypto import encrypt_secret_fields

        foreign = Fernet(Fernet.generate_key()).encrypt(b"geheim").decode()

        with pytest.raises(ValueError, match="nicht entschluesselbaren Wert"):
            encrypt_secret_fields({"password": foreign}, ["password"])

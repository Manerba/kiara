"""Tests fuer kiara.shared.secret_redact.

Verifiziert insbesondere die in Welle 1 (Plan V15) ergaenzten Patterns
fuer Dict-/JSON-Repr-Strings, die in Exception-Texten erscheinen,
wenn Tools oder Backends ihre Parameter im Fehler-`str()` ausgeben.
"""

import pytest

from kiara.shared.secret_redact import (
    redact_secrets_in_text,
    redact_secrets_recursive,
)


class TestRedactSecretsInTextDictRepr:
    """Dict-/JSON-Repr von Toolparametern wird zuverlaessig redaktiert."""

    def test_python_dict_repr_single_quotes(self):
        text = "params={'password': 'supersecret', 'username': 'alice'}"
        out = redact_secrets_in_text(text)
        assert "supersecret" not in out
        assert "'password': '***'" in out
        assert "alice" in out

    def test_json_dict_repr_double_quotes(self):
        text = 'config={"api_key": "sk-live-123", "username": "bob"}'
        out = redact_secrets_in_text(text)
        assert "sk-live-123" not in out
        assert '"api_key": "***"' in out
        assert "bob" in out

    def test_mixed_secret_keys_in_dict(self):
        text = (
            "{'password': 'p1', 'api_key': 'a1', 'token': 't1', 'dsn': 'd1', "
            "'secret': 's1', 'bind_password': 'b1', 'public': 'visible'}"
        )
        out = redact_secrets_in_text(text)
        for leaked in ("p1", "a1", "t1", "d1", "s1", "b1"):
            assert leaked not in out, f"{leaked!r} leaked: {out!r}"
        assert "visible" in out

    def test_unknown_key_in_dict_kept(self):
        text = "{'totally_safe': 'keepme'}"
        assert redact_secrets_in_text(text) == text


class TestRedactSecretsInTextUnquoted:
    """Unquoted key=value/key: value-Logging-Patterns bleiben abgedeckt."""

    def test_unquoted_keyvalue_pairs(self):
        text = "api_key=sk-live-123 password=supersecret token=tk-987"
        out = redact_secrets_in_text(text)
        for leaked in ("sk-live-123", "supersecret", "tk-987"):
            assert leaked not in out

    def test_authorization_bearer_token(self):
        text = "Authorization: Bearer abc.def.ghi"
        out = redact_secrets_in_text(text)
        assert "abc.def.ghi" not in out

    def test_url_with_credentials(self):
        text = "postgresql://user:supersecret@db.host/dbname"
        out = redact_secrets_in_text(text)
        assert "supersecret" not in out
        assert "***@db.host/dbname" in out


class TestRedactSecretsInTextEdgeCases:
    def test_empty_text_returns_unchanged(self):
        assert redact_secrets_in_text("") == ""

    def test_non_string_returns_unchanged(self):
        # Non-strings werden unveraendert zurueckgegeben (kein Crash).
        assert redact_secrets_in_text(None) is None  # type: ignore[arg-type]

    def test_idempotent(self):
        text = "password=geheim"
        once = redact_secrets_in_text(text)
        twice = redact_secrets_in_text(once)
        assert once == twice
        assert "geheim" not in twice


class TestRedactSecretsRecursive:
    """Rekursive Variante bleibt fuer dict/list-Strukturen unveraendert."""

    def test_nested_dict_secrets_masked(self):
        from kiara.shared.constants import PASSWORD_MASK

        obj = {
            "user": "alice",
            "credentials": {"password": "geheim", "api_key": "sk-x"},
            "list": [{"token": "t1"}, {"public": "ok"}],
        }
        out = redact_secrets_recursive(obj)
        assert out["user"] == "alice"
        assert out["credentials"]["password"] == PASSWORD_MASK
        assert out["credentials"]["api_key"] == PASSWORD_MASK
        assert out["list"][0]["token"] == PASSWORD_MASK
        assert out["list"][1]["public"] == "ok"


@pytest.mark.parametrize(
    "leak_input,expected_redacted",
    [
        ("api_key=sk-live-123", "***"),
        ("password: hunter2", "***"),
        ("{'password': 'hunter2'}", "***"),
        ('{"api_key": "sk-live-123"}', "***"),
        ("Bearer abc.def", "***"),
    ],
)
def test_parametrized_redaction(leak_input, expected_redacted):
    out = redact_secrets_in_text(leak_input)
    assert expected_redacted in out

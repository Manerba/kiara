"""Tests fuer Sprint 06: User-CRUD Admin-Funktionen.

Testet users.list, .create, .update, .delete, .reset_password — 13 Unit-Tests.
"""

import asyncio
from unittest.mock import ANY, patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# users.list
# ---------------------------------------------------------------------------


class TestUsersList:
    @patch("kiara.repositories.user.UserRepository.list_safe")
    def test_list_users(self, mock_list):
        """Test 1: list_safe() Aufruf, Feld-Mapping (bool-Cast)."""
        mock_list.return_value = [
            {
                "id": 1,
                "username": "admin",
                "vorname": "A",
                "nachname": "B",
                "email": "a@b.de",
                "is_admin": True,
                "auth_source": "local",
                "ldap_dn": "",
                "is_active": True,
                "created_at": "2025-01-01",
            },
        ]
        from kiara.admin.functions.users import users_list

        result = _run(users_list(user=MOCK_USER))

        assert result.success is True
        assert len(result.data["data"]) == 1
        row = result.data["data"][0]
        assert row["is_admin"] is True
        assert row["is_active"] is True
        mock_list.assert_called_once()


# ---------------------------------------------------------------------------
# users.create
# ---------------------------------------------------------------------------


class TestUsersCreate:
    @patch("kiara.repositories.user.UserRepository.create")
    @patch("kiara.web.auth.hash_password")
    @patch("kiara.repositories.user.UserRepository.username_exists")
    def test_create_user(self, mock_exists, mock_hash, mock_create):
        """Test 2: username>=2, password>=8, hash_password, create."""
        mock_exists.return_value = False
        mock_hash.return_value = "hashed_pw"
        mock_create.return_value = 5

        from kiara.admin.functions.users import users_create

        result = _run(
            users_create(
                user=MOCK_USER,
                username="testuser",
                password="Abc12345",
            )
        )

        assert result.success is True
        assert result.data["id"] == 5
        assert result.data["name"] == "testuser"
        mock_hash.assert_called_once_with("Abc12345")
        mock_create.assert_called_once()

    @patch("kiara.repositories.user.UserRepository.username_exists")
    def test_create_user_duplicate(self, mock_exists):
        """Test 3: username_exists → fail."""
        mock_exists.return_value = True

        from kiara.admin.functions.users import users_create

        result = _run(
            users_create(
                user=MOCK_USER,
                username="existing",
                password="Abc12345",
            )
        )

        assert result.success is False
        assert "existiert bereits" in result.message

    @patch("kiara.repositories.user.UserRepository.create")
    @patch("kiara.web.auth.hash_password")
    @patch("kiara.repositories.user.UserRepository.username_exists")
    def test_create_user_short_password_allowed_in_dev(self, mock_exists, mock_hash, mock_create):
        """Test 4: Die Entwicklungsumgebung erzwingt keine Mindestlaenge."""
        mock_exists.return_value = False
        mock_hash.return_value = "hashed_pw"
        mock_create.return_value = 5

        from kiara.admin.functions.users import users_create

        result = _run(
            users_create(
                user=MOCK_USER,
                username="testuser",
                password="short",
            )
        )

        assert result.success is True
        mock_hash.assert_called_once_with("short")
        mock_create.assert_called_once()

    def test_create_user_short_username(self):
        """Test 5: <2 Zeichen → fail."""
        from kiara.admin.functions.users import users_create

        result = _run(
            users_create(
                user=MOCK_USER,
                username="x",
                password="abc12345",
            )
        )

        assert result.success is False
        assert "2 Zeichen" in result.message


# ---------------------------------------------------------------------------
# users.update
# ---------------------------------------------------------------------------


class TestUsersUpdate:
    @patch("kiara.repositories.user.UserRepository.update")
    @patch("kiara.repositories.user.UserRepository.exists")
    def test_update_user_partial(self, mock_exists, mock_update):
        """Test 6: Nur uebergebene Felder werden gesetzt."""
        mock_exists.return_value = True

        from kiara.admin.functions.users import users_update

        result = _run(
            users_update(
                user=MOCK_USER,
                user_id=5,
                email="new@test.de",
            )
        )

        assert result.success is True
        mock_update.assert_called_once_with(5, email="new@test.de")

    @patch("kiara.repositories.user.SessionRepository.delete_by_user")
    @patch("kiara.repositories.user.UserRepository.update")
    @patch("kiara.repositories.user.UserRepository.get_by_id")
    @patch("kiara.repositories.user.UserRepository.exists")
    def test_update_user_deactivate_sessions(
        self, mock_exists, mock_get, mock_update, mock_delete_sessions
    ):
        """Test 7: is_active=False → delete_by_user."""
        mock_exists.return_value = True
        mock_get.return_value = {"id": 5, "is_admin": False, "is_active": True}

        from kiara.admin.functions.users import users_update

        result = _run(
            users_update(
                user=MOCK_USER,
                user_id=5,
                is_active=False,
            )
        )

        assert result.success is True
        mock_delete_sessions.assert_called_once_with(5, conn=ANY)

    @patch("kiara.repositories.user.UserRepository.update")
    @patch("kiara.repositories.user.UserRepository.count")
    @patch("kiara.repositories.user.UserRepository.get_by_id")
    @patch("kiara.repositories.user.UserRepository.exists")
    def test_update_deactivate_last_active_admin_blocked(
        self, mock_exists, mock_get, mock_count, mock_update
    ):
        """Bug #1900: Cross-Admin-Deaktivierung des letzten aktiven Admins muss blockiert werden."""
        mock_exists.return_value = True
        mock_get.return_value = {"id": 5, "is_admin": True, "is_active": True}
        mock_count.return_value = 1

        from kiara.admin.functions.users import users_update

        result = _run(
            users_update(
                user={"id": 1, "username": "admin"},
                user_id=5,
                is_active=False,
            )
        )

        assert result.success is False
        assert "Letzter aktiver Admin" in result.message
        mock_update.assert_not_called()

    @patch("kiara.repositories.user.SessionRepository.delete_by_user")
    @patch("kiara.repositories.user.UserRepository.update")
    @patch("kiara.repositories.user.UserRepository.count")
    @patch("kiara.repositories.user.UserRepository.get_by_id")
    @patch("kiara.repositories.user.UserRepository.exists")
    def test_update_deactivate_admin_with_other_admins_allowed(
        self, mock_exists, mock_get, mock_count, mock_update, mock_delete_sessions
    ):
        """Cross-Admin-Deaktivierung erlaubt, solange weitere aktive Admins uebrig bleiben."""
        mock_exists.return_value = True
        mock_get.return_value = {"id": 5, "is_admin": True, "is_active": True}
        mock_count.return_value = 2

        from kiara.admin.functions.users import users_update

        result = _run(
            users_update(
                user={"id": 1, "username": "admin"},
                user_id=5,
                is_active=False,
            )
        )

        assert result.success is True
        mock_delete_sessions.assert_called_once_with(5, conn=ANY)

    @patch("kiara.web.auth.hash_password")
    @patch("kiara.repositories.user.UserRepository.update")
    @patch("kiara.repositories.user.UserRepository.exists")
    def test_update_user_password_hash(self, mock_exists, mock_update, mock_hash):
        """Test 8: password → hash_password aufgerufen."""
        mock_exists.return_value = True
        mock_hash.return_value = "hashed_new"

        from kiara.admin.functions.users import users_update

        result = _run(
            users_update(
                user=MOCK_USER,
                user_id=5,
                password="NewPassword1",
            )
        )

        assert result.success is True
        mock_hash.assert_called_once_with("NewPassword1")
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["password_hash"] == "hashed_new"


# ---------------------------------------------------------------------------
# users.delete
# ---------------------------------------------------------------------------


class TestUsersDelete:
    @patch("kiara.repositories.user.UserRepository.delete_with_sessions")
    @patch("kiara.repositories.user.UserRepository.get_by_id")
    @patch("kiara.repositories.user.UserRepository.exists")
    def test_delete_user(self, mock_exists, mock_get, mock_delete):
        """Test 9: delete_with_sessions aufgerufen."""
        mock_exists.return_value = True
        mock_get.return_value = {"id": 5, "is_admin": False, "is_active": True}

        from kiara.admin.functions.users import users_delete

        result = _run(users_delete(user=MOCK_USER, user_id=5))

        assert result.success is True
        mock_delete.assert_called_once_with(5)

    def test_delete_user_self_prevention(self):
        """Test 10: eigene ID → fail."""
        from kiara.admin.functions.users import users_delete

        result = _run(users_delete(user={"id": 5, "username": "admin"}, user_id=5))

        assert result.success is False
        assert "Eigenen Account" in result.message

    @patch("kiara.repositories.user.UserRepository.exists")
    def test_delete_user_nonexistent(self, mock_exists):
        """Test 11: nicht vorhanden → fail."""
        mock_exists.return_value = False

        from kiara.admin.functions.users import users_delete

        result = _run(users_delete(user=MOCK_USER, user_id=999))

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.repositories.user.UserRepository.delete_with_sessions")
    @patch("kiara.repositories.user.UserRepository.count")
    @patch("kiara.repositories.user.UserRepository.get_by_id")
    @patch("kiara.repositories.user.UserRepository.exists")
    def test_delete_last_active_admin_blocked(self, mock_exists, mock_get, mock_count, mock_delete):
        """Bug #1901: Cross-Admin-Loeschung des letzten aktiven Admins muss blockiert werden."""
        mock_exists.return_value = True
        mock_get.return_value = {"id": 5, "is_admin": True, "is_active": True}
        mock_count.return_value = 1

        from kiara.admin.functions.users import users_delete

        result = _run(users_delete(user={"id": 1, "username": "admin"}, user_id=5))

        assert result.success is False
        assert "Letzter aktiver Admin" in result.message
        mock_delete.assert_not_called()

    @patch("kiara.repositories.user.UserRepository.delete_with_sessions")
    @patch("kiara.repositories.user.UserRepository.count")
    @patch("kiara.repositories.user.UserRepository.get_by_id")
    @patch("kiara.repositories.user.UserRepository.exists")
    def test_delete_admin_with_other_admins_allowed(
        self, mock_exists, mock_get, mock_count, mock_delete
    ):
        """Cross-Admin-Loeschung erlaubt, solange weitere aktive Admins uebrig bleiben."""
        mock_exists.return_value = True
        mock_get.return_value = {"id": 5, "is_admin": True, "is_active": True}
        mock_count.return_value = 2

        from kiara.admin.functions.users import users_delete

        result = _run(users_delete(user={"id": 1, "username": "admin"}, user_id=5))

        assert result.success is True
        mock_delete.assert_called_once_with(5)


# ---------------------------------------------------------------------------
# users.reset_password
# ---------------------------------------------------------------------------


class TestUsersResetPassword:
    @patch("kiara.repositories.user.UserRepository.set_password")
    @patch("kiara.web.auth.hash_password")
    @patch("kiara.repositories.user.UserRepository.exists")
    def test_reset_password(self, mock_exists, mock_hash, mock_set):
        """Test 12: set_password mit hash aufgerufen."""
        mock_exists.return_value = True
        mock_hash.return_value = "hashed_reset"

        from kiara.admin.functions.users import users_reset_password

        result = _run(
            users_reset_password(
                user=MOCK_USER,
                user_id=5,
                new_password="Neu12345x",
            )
        )

        assert result.success is True
        mock_hash.assert_called_once_with("Neu12345x")
        mock_set.assert_called_once_with(5, "hashed_reset", conn=ANY)

    def test_reset_password_short(self):
        """Test 13: <8 Zeichen → fail."""
        from kiara.admin.functions.users import users_reset_password

        result = _run(
            users_reset_password(
                user=MOCK_USER,
                user_id=5,
                new_password="short",
            )
        )

        assert result.success is False
        assert "8 Zeichen" in result.message

    @patch("kiara.repositories.user.UserRepository.exists")
    def test_reset_password_user_not_found(self, mock_exists):
        """Benutzer nicht gefunden → fail."""
        mock_exists.return_value = False

        from kiara.admin.functions.users import users_reset_password

        result = _run(
            users_reset_password(
                user=MOCK_USER,
                user_id=999,
                new_password="Neu12345x",
            )
        )

        assert result.success is False
        assert "nicht gefunden" in result.message


# ---------------------------------------------------------------------------
# Ergaenzende Edge-Case Tests (Coverage-Erweiterung)
# ---------------------------------------------------------------------------


class TestUsersUpdateEdgeCases:
    @patch("kiara.repositories.user.UserRepository.update")
    @patch("kiara.repositories.user.UserRepository.exists")
    def test_update_user_is_admin_bool_passthrough(self, mock_exists, mock_update):
        """is_admin-Bool wird als bool an die DB weitergegeben (is_admin ist BOOLEAN in PostgreSQL)."""
        mock_exists.return_value = True

        from kiara.admin.functions.users import users_update

        result = _run(
            users_update(
                user=MOCK_USER,
                user_id=5,
                is_admin=True,
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["is_admin"] is True
        assert isinstance(call_kwargs["is_admin"], bool)

    @patch("kiara.repositories.user.UserRepository.update")
    @patch("kiara.repositories.user.UserRepository.exists")
    def test_update_user_active_true_no_session_delete(self, mock_exists, mock_update):
        """is_active=True darf KEINE Sessions loeschen."""
        mock_exists.return_value = True

        from kiara.admin.functions.users import users_update

        # SessionRepository.delete_by_user darf nicht aufgerufen werden
        with patch("kiara.repositories.user.SessionRepository.delete_by_user") as mock_del:
            result = _run(
                users_update(
                    user=MOCK_USER,
                    user_id=5,
                    is_active=True,
                )
            )

        assert result.success is True
        mock_del.assert_not_called()

    @patch("kiara.repositories.user.UserRepository.exists")
    def test_update_user_not_found(self, mock_exists):
        """Benutzer nicht gefunden → fail."""
        mock_exists.return_value = False

        from kiara.admin.functions.users import users_update

        result = _run(users_update(user=MOCK_USER, user_id=999))

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.web.auth.hash_password")
    @patch("kiara.repositories.user.UserRepository.update")
    @patch("kiara.repositories.user.UserRepository.exists")
    def test_update_user_short_password_allowed_in_dev(self, mock_exists, mock_update, mock_hash):
        """Kurze Passwoerter werden in der Entwicklungsumgebung gespeichert."""
        mock_exists.return_value = True
        mock_hash.return_value = "hashed_pw"

        from kiara.admin.functions.users import users_update

        result = _run(
            users_update(
                user=MOCK_USER,
                user_id=5,
                password="kurz",
            )
        )

        assert result.success is True
        mock_hash.assert_called_once_with("kurz")
        mock_update.assert_called_once_with(5, conn=ANY, password_hash="hashed_pw")

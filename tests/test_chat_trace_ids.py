from unittest.mock import patch

import pytest


class TestGetChatTraceIds:
    @pytest.mark.asyncio
    async def test_get_chat_stabilizes_same_timestamps_and_prefers_full_answer_match(self):
        from kiara.web.api_chat import get_chat

        shared = "X" * 200
        messages = [
            {
                "id": 3,
                "role": "assistant",
                "content": shared + "B",
                "sources": None,
                "created_at": "2026-04-04 10:00:01",
            },
            {
                "id": 2,
                "role": "assistant",
                "content": shared + "A",
                "sources": None,
                "created_at": "2026-04-04 10:00:01",
            },
        ]
        chat = {"id": 7, "title": "Test", "status": "active", "created_at": "2026-04-04 09:59:00"}
        user = {"id": 11, "is_admin": False}

        with (
            patch("kiara.web.api_chat.chat_repo") as chat_repo,
            patch("kiara.web.api_chat.msg_repo") as msg_repo,
            patch("kiara.web.api_chat.db_query") as db,
        ):
            chat_repo.get_owned.return_value = chat
            msg_repo.list_by_chat.return_value = messages
            db.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
                {"trace_id": "trace-b", "answer_excerpt": shared, "full_answer": shared + "B"},
                {"trace_id": "trace-a", "answer_excerpt": shared, "full_answer": shared + "A"},
            ]

            result = await get_chat(7, user)

        assert [(m["id"], m["trace_id"]) for m in result["messages"]] == [
            (2, "trace-a"),
            (3, "trace-b"),
        ]

    @pytest.mark.asyncio
    async def test_get_chat_context_messages_do_not_consume_trace_slots(self):
        from kiara.web.api_chat import get_chat

        messages = [
            {
                "id": 1,
                "role": "user",
                "content": "Q1",
                "sources": None,
                "created_at": "2026-04-04 10:00:00",
            },
            {
                "id": 2,
                "role": "context",
                "content": '{"text": "CTX"}',
                "sources": None,
                "created_at": "2026-04-04 10:00:01",
            },
            {
                "id": 3,
                "role": "assistant",
                "content": "A1",
                "sources": None,
                "created_at": "2026-04-04 10:00:02",
            },
        ]
        chat = {"id": 7, "title": "Test", "status": "active", "created_at": "2026-04-04 09:59:00"}
        user = {"id": 11, "is_admin": False}

        with (
            patch("kiara.web.api_chat.chat_repo") as chat_repo,
            patch("kiara.web.api_chat.msg_repo") as msg_repo,
            patch("kiara.web.api_chat.db_query") as db,
        ):
            chat_repo.get_owned.return_value = chat
            msg_repo.list_by_chat.return_value = messages
            db.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
                {"trace_id": "trace-1", "answer_excerpt": "A1", "full_answer": "A1"},
            ]

            result = await get_chat(7, user)

        assert [m["trace_id"] for m in result["messages"]] == [None, None, "trace-1"]

    @pytest.mark.asyncio
    async def test_get_chat_falls_back_to_answer_excerpt_when_full_answer_missing(self):
        from kiara.web.api_chat import get_chat

        messages = [
            {
                "id": 1,
                "role": "assistant",
                "content": "Legacy Antwort mit mehr Text",
                "sources": None,
                "created_at": "2026-04-04 10:00:01",
            },
        ]
        chat = {"id": 7, "title": "Test", "status": "active", "created_at": "2026-04-04 09:59:00"}
        user = {"id": 11, "is_admin": False}

        with (
            patch("kiara.web.api_chat.chat_repo") as chat_repo,
            patch("kiara.web.api_chat.msg_repo") as msg_repo,
            patch("kiara.web.api_chat.db_query") as db,
        ):
            chat_repo.get_owned.return_value = chat
            msg_repo.list_by_chat.return_value = messages
            db.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
                {"trace_id": "trace-legacy", "answer_excerpt": "Legacy Antwort", "full_answer": ""},
            ]

            result = await get_chat(7, user)

        assert [m["trace_id"] for m in result["messages"]] == ["trace-legacy"]

    @pytest.mark.asyncio
    async def test_get_chat_normalizes_legacy_think_blocks_before_matching(self):
        from kiara.web.api_chat import get_chat

        messages = [
            {
                "id": 1,
                "role": "assistant",
                "content": "<think>Analyse</think>Legacy Antwort mit mehr Text",
                "sources": None,
                "created_at": "2026-04-04 10:00:01",
            },
        ]
        chat = {"id": 7, "title": "Test", "status": "active", "created_at": "2026-04-04 09:59:00"}
        user = {"id": 11, "is_admin": False}

        with (
            patch("kiara.web.api_chat.chat_repo") as chat_repo,
            patch("kiara.web.api_chat.msg_repo") as msg_repo,
            patch("kiara.web.api_chat.db_query") as db,
        ):
            chat_repo.get_owned.return_value = chat
            msg_repo.list_by_chat.return_value = messages
            db.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
                {
                    "trace_id": "trace-think",
                    "answer_excerpt": "Legacy Antwort",
                    "full_answer": "Legacy Antwort mit mehr Text",
                },
            ]

            result = await get_chat(7, user)

        assert [m["trace_id"] for m in result["messages"]] == ["trace-think"]

    @pytest.mark.asyncio
    async def test_get_chat_handles_missing_created_at_without_type_error(self):
        from kiara.web.api_chat import get_chat

        messages = [
            {
                "id": 2,
                "role": "assistant",
                "content": "Ohne Timestamp",
                "sources": None,
                "created_at": None,
            },
            {
                "id": 1,
                "role": "assistant",
                "content": "Mit Timestamp",
                "sources": None,
                "created_at": "2026-04-04 10:00:01",
            },
        ]
        chat = {"id": 7, "title": "Test", "status": "active", "created_at": "2026-04-04 09:59:00"}
        user = {"id": 11, "is_admin": False}

        with (
            patch("kiara.web.api_chat.chat_repo") as chat_repo,
            patch("kiara.web.api_chat.msg_repo") as msg_repo,
            patch("kiara.web.api_chat.db_query") as db,
        ):
            chat_repo.get_owned.return_value = chat
            msg_repo.list_by_chat.return_value = messages
            db.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
                {
                    "trace_id": "trace-ts",
                    "answer_excerpt": "Mit Timestamp",
                    "full_answer": "Mit Timestamp",
                },
                {
                    "trace_id": "trace-none",
                    "answer_excerpt": "Ohne Timestamp",
                    "full_answer": "Ohne Timestamp",
                },
            ]

            result = await get_chat(7, user)

        assert [(m["id"], m["trace_id"]) for m in result["messages"]] == [
            (1, "trace-ts"),
            (2, "trace-none"),
        ]

    @pytest.mark.asyncio
    async def test_get_chat_ignores_empty_answer_rows_and_agent_error_gaps(self):
        from kiara.web.api_chat import get_chat

        messages = [
            {
                "id": 1,
                "role": "assistant",
                "content": "A1",
                "sources": None,
                "created_at": "2026-04-04 10:00:01",
            },
            {
                "id": 2,
                "role": "assistant",
                "content": "A2",
                "sources": None,
                "created_at": "2026-04-04 10:00:02",
            },
            {
                "id": 3,
                "role": "assistant",
                "content": "A3",
                "sources": None,
                "created_at": "2026-04-04 10:00:03",
            },
        ]
        chat = {"id": 7, "title": "Test", "status": "active", "created_at": "2026-04-04 09:59:00"}
        user = {"id": 11, "is_admin": False}

        with (
            patch("kiara.web.api_chat.chat_repo") as chat_repo,
            patch("kiara.web.api_chat.msg_repo") as msg_repo,
            patch("kiara.web.api_chat.db_query") as db,
        ):
            chat_repo.get_owned.return_value = chat
            msg_repo.list_by_chat.return_value = messages
            db.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
                {"trace_id": "trace-1", "answer_excerpt": "A1", "full_answer": "A1"},
                {"trace_id": "should-be-ignored", "answer_excerpt": "", "full_answer": ""},
                # Agent-Error/Unchained-Luecke: fuer A2 existiert bewusst kein trace_id-Row
                {"trace_id": "trace-3", "answer_excerpt": "A3", "full_answer": "A3"},
            ]

            result = await get_chat(7, user)

        assert [m["trace_id"] for m in result["messages"]] == ["trace-1", None, "trace-3"]

    @pytest.mark.asyncio
    async def test_get_chat_leaves_trace_id_none_when_no_trace_row_matches(self):
        from kiara.web.api_chat import get_chat

        messages = [
            {
                "id": 1,
                "role": "assistant",
                "content": "A1",
                "sources": None,
                "created_at": "2026-04-04 10:00:01",
            },
            {
                "id": 2,
                "role": "assistant",
                "content": "A2",
                "sources": None,
                "created_at": "2026-04-04 10:00:02",
            },
        ]
        chat = {"id": 7, "title": "Test", "status": "active", "created_at": "2026-04-04 09:59:00"}
        user = {"id": 11, "is_admin": False}

        with (
            patch("kiara.web.api_chat.chat_repo") as chat_repo,
            patch("kiara.web.api_chat.msg_repo") as msg_repo,
            patch("kiara.web.api_chat.db_query") as db,
        ):
            chat_repo.get_owned.return_value = chat
            msg_repo.list_by_chat.return_value = messages
            db.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
                {"trace_id": "trace-1", "answer_excerpt": "A1", "full_answer": "A1"},
                {
                    "trace_id": "trace-foreign",
                    "answer_excerpt": "komplett anderer Text",
                    "full_answer": "komplett anderer Text",
                },
            ]

            result = await get_chat(7, user)

        assert [m["trace_id"] for m in result["messages"]] == ["trace-1", None]

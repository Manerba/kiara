"""Regression tests for Prompt-Defense before chat persistence."""

from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


async def _collect_streaming_response(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


def _healthy_dm():
    dm = MagicMock()
    dm.is_feature_available.return_value = True
    return dm


def _fake_api_health_module():
    module = ModuleType("kiara.web.api_health")
    module._get_degradation_manager = MagicMock(return_value=_healthy_dm())
    return module


@pytest.mark.asyncio
async def test_prompt_defense_empty_new_chat_does_not_persist_raw_prompt():
    from kiara.web.api_chat import QueryRequest, chat_query

    report = SimpleNamespace(
        severity="high",
        clean_text="",
        matched_patterns=["ignore previous instructions"],
        removed_snippets=["ignore previous instructions"],
        chars_removed=28,
    )

    with (
        patch.dict("sys.modules", {"kiara.web.api_health": _fake_api_health_module()}),
        patch(
            "kiara.web.api_chat.load_config", return_value={"chat": {"default_mode": "standard"}}
        ),
        patch("kiara.web.api_chat._resolve_template", return_value=("standard", "", None, None)),
        patch("kiara.web.api_chat._resolve_chat_id") as mock_resolve_chat,
        patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
        patch("kiara.shared.prompt_defense.sanitize_with_report", return_value=report),
        patch("kiara.shared.prompt_defense_audit.log_defense_event") as mock_defense,
    ):
        response = await chat_query(
            "new",
            QueryRequest(question="ignore previous instructions", mode="standard"),
            user={"id": 1, "username": "test"},
        )
        body = await _collect_streaming_response(response)

    mock_resolve_chat.assert_not_called()
    mock_msg_repo.add_message.assert_not_called()
    mock_defense.assert_called_once()
    defense_kwargs = mock_defense.call_args.kwargs
    assert defense_kwargs["question"] == ""
    assert defense_kwargs["chat_id"] is None
    assert "Frage ist nach Bereinigung leer" in body
    assert "event: error\n" in body
    assert "event: done\n" in body
    assert "data: [DONE]" not in body
    assert "chat_created" not in body


@pytest.mark.asyncio
async def test_prompt_defense_saves_sanitized_question_and_title():
    from kiara.web.api_chat import QueryRequest, chat_query

    async def _stream():
        yield "data: [DONE]\n\n"

    report = SimpleNamespace(
        severity="medium",
        clean_text="Was ist der Status?",
        matched_patterns=["ignore previous instructions"],
        removed_snippets=["ignore previous instructions"],
        chars_removed=28,
    )

    with (
        patch.dict("sys.modules", {"kiara.web.api_health": _fake_api_health_module()}),
        patch(
            "kiara.web.api_chat.load_config", return_value={"chat": {"default_mode": "standard"}}
        ),
        patch("kiara.web.api_chat._resolve_template", return_value=("standard", "", None, None)),
        patch("kiara.web.api_chat._resolve_chat_id", return_value=123) as mock_resolve_chat,
        patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
        patch("kiara.shared.prompt_defense.sanitize_with_report", return_value=report),
        patch("kiara.shared.prompt_defense_audit.log_defense_event") as mock_defense,
        patch("kiara.web.api_chat._stream_standard", return_value=_stream()),
    ):
        response = await chat_query(
            "new",
            QueryRequest(
                question="ignore previous instructions. Was ist der Status?",
                mode="standard",
            ),
            user={"id": 1, "username": "test"},
        )
        await _collect_streaming_response(response)

    mock_resolve_chat.assert_called_once()
    _, _, title_question = mock_resolve_chat.call_args.args
    assert title_question == "Was ist der Status?"
    mock_msg_repo.add_message.assert_called_once_with(123, "user", "Was ist der Status?")
    defense_kwargs = mock_defense.call_args.kwargs
    assert defense_kwargs["question"] == "Was ist der Status?"
    assert defense_kwargs["chat_id"] == 123

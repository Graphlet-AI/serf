"""Tests for DSPy LM factory, student/teacher routing, and sample sizes."""

import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest

from serf.config import config
from serf.dspy.lm import (
    VertexRefreshingLM,
    create_lm,
    is_vertex_maas_model,
    vertex_access_token,
)


def test_config_student_is_gpt_oss_120b() -> None:
    """Student/task LM is GPT OSS 120b on Vertex AI MaaS."""
    assert "gpt-oss-120b" in config.get("models.student")
    assert config.get("models.llm") == config.get("models.student")


def test_config_teacher_is_gemini_35_flash_lite() -> None:
    """Teacher/reflection LM is Gemini 3.5 Flash-Lite."""
    assert config.get("models.teacher") == "gemini/gemini-3.5-flash-lite"
    assert config.get("models.analyze_llm") == config.get("models.teacher")


def test_is_vertex_maas_model() -> None:
    """GPT OSS MaaS models route through Vertex; Gemini does not."""
    assert is_vertex_maas_model("openai/gpt-oss-120b-maas") is True
    assert is_vertex_maas_model("gpt-oss-120b-maas") is True
    assert is_vertex_maas_model("gemini/gemini-3.5-flash-lite") is False


def test_vertex_access_token_accepts_raw_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-minted access token in VERTEX_AI_TOKEN is used as-is."""
    monkeypatch.setenv("VERTEX_AI_TOKEN", "ya29.fake-access-token")
    assert vertex_access_token() == "ya29.fake-access-token"


def test_vertex_access_token_decodes_unpadded_base64_service_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VERTEX_AI_TOKEN may omit base64 padding."""
    info = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "abc",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMII\n-----END PRIVATE KEY-----\n",
        "client_email": "sa@test-project.iam.gserviceaccount.com",
        "client_id": "1",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    encoded = base64.b64encode(json.dumps(info).encode()).decode().rstrip("=")
    assert "=" not in encoded
    monkeypatch.setenv("VERTEX_AI_TOKEN", encoded)

    mock_creds = MagicMock()
    mock_creds.token = "refreshed-token"

    with (
        patch("serf.dspy.lm.service_account.Credentials") as mock_cls,
        patch("serf.dspy.lm.Request"),
    ):
        mock_cls.from_service_account_info.return_value = mock_creds
        token = vertex_access_token()

    assert token == "refreshed-token"
    mock_cls.from_service_account_info.assert_called_once()


def test_vertex_access_token_decodes_base64_service_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VERTEX_AI_TOKEN may be a base64-encoded service account JSON."""
    info = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "abc",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMII\n-----END PRIVATE KEY-----\n",
        "client_email": "sa@test-project.iam.gserviceaccount.com",
        "client_id": "1",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    encoded = base64.b64encode(json.dumps(info).encode()).decode()
    monkeypatch.setenv("VERTEX_AI_TOKEN", encoded)

    mock_creds = MagicMock()
    mock_creds.token = "refreshed-token"
    mock_creds.valid = False

    with (
        patch("serf.dspy.lm.service_account.Credentials") as mock_cls,
        patch("serf.dspy.lm.Request"),
    ):
        mock_cls.from_service_account_info.return_value = mock_creds
        token = vertex_access_token()

    assert token == "refreshed-token"
    mock_cls.from_service_account_info.assert_called_once()
    kwargs: dict[str, Any] = mock_cls.from_service_account_info.call_args.kwargs
    loaded = mock_cls.from_service_account_info.call_args.args[0]
    assert loaded["project_id"] == "test-project"
    assert kwargs["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
    mock_creds.refresh.assert_called_once()


def test_vertex_access_token_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing VERTEX_AI_TOKEN raises a clear error."""
    monkeypatch.delenv("VERTEX_AI_TOKEN", raising=False)
    with pytest.raises(ValueError, match="VERTEX_AI_TOKEN"):
        vertex_access_token()


@patch("serf.dspy.lm.dspy.LM")
@patch("serf.dspy.lm.vertex_access_token", return_value="vertex-token")
def test_create_lm_student_uses_vertex(
    mock_token: MagicMock, mock_lm: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Student LM uses GPT OSS 120b via the Vertex OpenAI-compatible endpoint."""
    monkeypatch.setenv("VERTEX_AI_TOKEN", "ya29.fake-access-token")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0349392143")
    create_lm(role="student")

    mock_token.assert_called_once()
    model = mock_lm.call_args.args[0]
    kwargs = mock_lm.call_args.kwargs
    assert "gpt-oss-120b-maas" in model
    assert kwargs["api_key"] == "vertex-token"
    assert "aiplatform.googleapis.com" in kwargs["api_base"]
    assert "gen-lang-client-0349392143" in kwargs["api_base"]
    assert "/endpoints/openapi" in kwargs["api_base"]
    assert kwargs["temperature"] == 0.0


@patch("serf.dspy.lm.dspy.LM")
def test_create_lm_teacher_uses_gemini(mock_lm: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Teacher LM uses Gemini 3.5 Flash-Lite with GEMINI_API_KEY."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    create_lm(role="teacher", temperature=1.0)

    model = mock_lm.call_args.args[0]
    kwargs = mock_lm.call_args.kwargs
    assert model == "gemini/gemini-3.5-flash-lite"
    assert kwargs["api_key"] == "gemini-key"
    assert kwargs["temperature"] == 1.0
    assert "api_base" not in kwargs


@patch("serf.dspy.lm.dspy.LM")
def test_create_lm_gemini_requires_key(mock_lm: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini models fail fast without GEMINI_API_KEY."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        create_lm(model="gemini/gemini-3.5-flash-lite")
    mock_lm.assert_not_called()


def _fake_credentials(token: str, expires_in: timedelta | None) -> Any:
    """Build mock service-account credentials with a naive UTC expiry."""
    credentials = MagicMock()
    credentials.token = token
    credentials.expired = expires_in is not None and expires_in <= timedelta(0)
    if expires_in is None:
        credentials.expiry = None
    else:
        credentials.expiry = datetime.now(UTC).replace(tzinfo=None) + expires_in

    def _refresh(_request: Any) -> None:
        credentials.token = "refreshed-token"
        credentials.expired = False
        credentials.expiry = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1)

    credentials.refresh.side_effect = _refresh
    return credentials


def _refreshing_lm(credentials: Any) -> VertexRefreshingLM:
    """Build a VertexRefreshingLM around mock credentials."""
    return VertexRefreshingLM(
        "openai/openai/gpt-oss-120b-maas",
        credentials=credentials,
        api_base="https://us-central1-aiplatform.googleapis.com/v1",
        api_key=credentials.token,
        temperature=0.0,
        max_tokens=32768,
    )


def test_refreshing_lm_keeps_a_valid_token() -> None:
    """A token far from expiry is reused without a refresh call."""
    credentials = _fake_credentials("first-token", timedelta(hours=1))
    lm = _refreshing_lm(credentials)
    with patch("serf.dspy.lm.Request"):
        lm.refresh_token_if_needed()
    credentials.refresh.assert_not_called()
    assert lm.kwargs["api_key"] == "first-token"


def test_refreshing_lm_refreshes_near_expiry() -> None:
    """A token inside the safety margin is refreshed before the request."""
    credentials = _fake_credentials("first-token", timedelta(seconds=30))
    lm = _refreshing_lm(credentials)
    with patch("serf.dspy.lm.Request"):
        lm.refresh_token_if_needed()
    credentials.refresh.assert_called_once()
    assert lm.kwargs["api_key"] == "refreshed-token"


def test_refreshing_lm_refreshes_expired_token() -> None:
    """An already expired token is refreshed instead of being sent again."""
    credentials = _fake_credentials("stale-token", timedelta(seconds=-60))
    lm = _refreshing_lm(credentials)
    with patch("serf.dspy.lm.Request"):
        lm.refresh_token_if_needed()
    credentials.refresh.assert_called_once()
    assert lm.kwargs["api_key"] == "refreshed-token"


def test_refreshing_lm_refreshes_when_expiry_is_unknown() -> None:
    """Credentials without an expiry are refreshed defensively."""
    credentials = _fake_credentials("first-token", None)
    lm = _refreshing_lm(credentials)
    with patch("serf.dspy.lm.Request"):
        lm.refresh_token_if_needed()
    credentials.refresh.assert_called_once()
    assert lm.kwargs["api_key"] == "refreshed-token"


def test_refreshing_lm_accepts_timezone_aware_expiry() -> None:
    """Timezone-aware expiries are compared without raising."""
    credentials = _fake_credentials("first-token", timedelta(hours=1))
    credentials.expiry = datetime.now(UTC) + timedelta(hours=1)
    lm = _refreshing_lm(credentials)
    with patch("serf.dspy.lm.Request"):
        lm.refresh_token_if_needed()
    credentials.refresh.assert_not_called()


def test_refreshing_lm_forward_refreshes_first() -> None:
    """forward() refreshes the token before delegating to dspy.LM."""
    credentials = _fake_credentials("stale-token", timedelta(seconds=-60))
    lm = _refreshing_lm(credentials)
    with (
        patch("serf.dspy.lm.Request"),
        patch.object(dspy.LM, "forward", return_value="ok") as mock_forward,
    ):
        assert lm.forward(messages=[{"role": "user", "content": "hi"}]) == "ok"
    credentials.refresh.assert_called_once()
    assert lm.kwargs["api_key"] == "refreshed-token"
    mock_forward.assert_called_once()


def test_refreshing_lm_aforward_refreshes_first() -> None:
    """aforward() refreshes the token before delegating to dspy.LM."""
    credentials = _fake_credentials("stale-token", timedelta(seconds=-60))
    lm = _refreshing_lm(credentials)
    with (
        patch("serf.dspy.lm.Request"),
        patch.object(dspy.LM, "aforward", new=AsyncMock(return_value="ok")),
    ):
        result = asyncio.run(lm.aforward(messages=[{"role": "user", "content": "hi"}]))
    assert result == "ok"
    credentials.refresh.assert_called_once()
    assert lm.kwargs["api_key"] == "refreshed-token"


def test_refreshing_lm_copies_share_credentials() -> None:
    """A copied LM picks up a token refreshed by its sibling."""
    credentials = _fake_credentials("stale-token", timedelta(seconds=-60))
    lm = _refreshing_lm(credentials)
    clone = lm.copy(rollout_id=1)
    with patch("serf.dspy.lm.Request"):
        lm.refresh_token_if_needed()
        clone.refresh_token_if_needed()
    credentials.refresh.assert_called_once()
    assert clone.kwargs["api_key"] == "refreshed-token"


def test_create_lm_student_uses_refreshing_lm_for_service_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A service-account VERTEX_AI_TOKEN yields an LM that can refresh itself."""
    info = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "abc",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMII\n-----END PRIVATE KEY-----\n",
        "client_email": "sa@test-project.iam.gserviceaccount.com",
        "client_id": "1",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    monkeypatch.setenv("VERTEX_AI_TOKEN", json.dumps(info))
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)

    credentials = _fake_credentials("initial-token", None)
    with (
        patch("serf.dspy.lm.service_account.Credentials") as mock_cls,
        patch("serf.dspy.lm.Request"),
    ):
        mock_cls.from_service_account_info.return_value = credentials
        lm = create_lm(role="student")

    assert isinstance(lm, VertexRefreshingLM)
    assert lm.credentials is credentials
    assert lm.kwargs["api_key"] == "refreshed-token"
    assert lm.model == "openai/openai/gpt-oss-120b-maas"
    assert "test-project" in lm.kwargs["api_base"]


def test_create_lm_student_raw_token_is_not_refreshable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw ya29. bearer token is used as-is and cannot be refreshed."""
    monkeypatch.setenv("VERTEX_AI_TOKEN", "ya29.fake-access-token")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    lm = create_lm(role="student")
    assert not isinstance(lm, VertexRefreshingLM)
    assert lm.kwargs["api_key"] == "ya29.fake-access-token"

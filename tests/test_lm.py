"""Tests for DSPy LM factory, student/teacher routing, and sample sizes."""

import base64
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from serf.config import config
from serf.dspy.lm import (
    create_lm,
    get_train_sample_size,
    get_train_sample_sizes,
    is_vertex_maas_model,
    vertex_access_token,
)


def test_config_student_is_gpt_oss_120b() -> None:
    """Student/task LM is GPT OSS 120b on Vertex AI MaaS."""
    assert "gpt-oss-120b" in config.get("models.student")
    assert config.get("models.llm") == config.get("models.student")


def test_config_teacher_is_gemini_37_flash() -> None:
    """Teacher/reflection LM is Gemini 3.7 Flash."""
    assert config.get("models.teacher") == "gemini/gemini-3.7-flash"
    assert config.get("models.analyze_llm") == config.get("models.teacher")


def test_train_sample_size_configured_for_each_benchmark_dataset() -> None:
    """Each benchmark dataset has an explicit training sample size."""
    sizes = get_train_sample_sizes()
    expected = {
        "walmart-amazon": 1500,
        "abt-buy": 1500,
        "amazon-google": 1500,
        "dblp-acm": 1500,
        "dblp-scholar": 1500,
    }
    assert sizes == expected
    for name, size in expected.items():
        assert get_train_sample_size(name) == size


def test_is_vertex_maas_model() -> None:
    """GPT OSS MaaS models route through Vertex; Gemini does not."""
    assert is_vertex_maas_model("openai/gpt-oss-120b-maas") is True
    assert is_vertex_maas_model("gpt-oss-120b-maas") is True
    assert is_vertex_maas_model("gemini/gemini-3.7-flash") is False
    assert is_vertex_maas_model("gemini/gemini-3.5-flash-lite") is False


def test_vertex_access_token_accepts_raw_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-minted access token in VERTEX_AI_TOKEN is used as-is."""
    monkeypatch.setenv("VERTEX_AI_TOKEN", "ya29.fake-access-token")
    assert vertex_access_token() == "ya29.fake-access-token"


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
    """Teacher LM uses Gemini 3.7 Flash with GEMINI_API_KEY."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    create_lm(role="teacher", temperature=1.0)

    model = mock_lm.call_args.args[0]
    kwargs = mock_lm.call_args.kwargs
    assert model == "gemini/gemini-3.7-flash"
    assert kwargs["api_key"] == "gemini-key"
    assert kwargs["temperature"] == 1.0
    assert "api_base" not in kwargs


@patch("serf.dspy.lm.dspy.LM")
def test_create_lm_gemini_requires_key(mock_lm: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini models fail fast without GEMINI_API_KEY."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        create_lm(model="gemini/gemini-3.7-flash")
    mock_lm.assert_not_called()

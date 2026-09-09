"""DSPy language-model factory for student (task) and teacher (reflection) LMs."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import dspy
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from serf.config import config
from serf.logs import get_logger

logger = get_logger(__name__)

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def is_vertex_maas_model(model: str) -> bool:
    """Return whether a model identifier is a Vertex AI MaaS GPT-OSS endpoint.

    Parameters
    ----------
    model : str
        DSPy/LiteLLM model identifier

    Returns
    -------
    bool
        True when the model is served via Vertex AI MaaS (gpt-oss)
    """
    return "gpt-oss" in model.lower()


def vertex_access_token() -> str:
    """Return a Vertex AI access token from VERTEX_AI_TOKEN.

    VERTEX_AI_TOKEN may be:
    - a short-lived Google access token (``ya29.`` prefix)
    - a service-account JSON document
    - a base64-encoded service-account JSON document

    Returns
    -------
    str
        Bearer access token for the Vertex OpenAI-compatible endpoint

    Raises
    ------
    ValueError
        If VERTEX_AI_TOKEN is missing or cannot be parsed
    """
    raw = os.environ.get("VERTEX_AI_TOKEN", "").strip()
    if not raw:
        raise ValueError("VERTEX_AI_TOKEN environment variable required for Vertex AI MaaS models")
    if raw.startswith("ya29."):
        return raw

    info = _parse_service_account_info(raw)
    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=[_CLOUD_PLATFORM_SCOPE],
    )
    credentials.refresh(Request())
    if not credentials.token:
        raise ValueError("Failed to refresh Vertex AI access token from VERTEX_AI_TOKEN")
    return str(credentials.token)


def create_lm(
    model: str | None = None,
    *,
    role: str = "student",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dspy.LM:
    """Create a DSPy LM for the student (task) or teacher (reflection) role.

    Parameters
    ----------
    model : str | None
        Explicit model identifier. Defaults to ``models.student``/``models.llm``
        for role ``student`` and ``models.teacher`` for role ``teacher``.
    role : str
        ``student`` for the task LM or ``teacher`` for the GEPA reflection LM
    temperature : float | None
        Sampling temperature. Defaults to ``models.temperature``.
    max_tokens : int | None
        Max output tokens. Defaults to ``models.max_tokens``.

    Returns
    -------
    dspy.LM
        Configured language model client
    """
    if model is None:
        model = config.get("models.teacher") if role == "teacher" else config.get("models.llm")
    if temperature is None:
        temperature = float(config.get("models.temperature", 0.0))
    if max_tokens is None:
        max_tokens = int(config.get("models.max_tokens", 8192))

    logger.info(f"Creating {role} LM: {model}")
    if is_vertex_maas_model(model):
        return _create_vertex_maas_lm(model, temperature=temperature, max_tokens=max_tokens)
    return _create_gemini_lm(model, temperature=temperature, max_tokens=max_tokens)


def _parse_service_account_info(raw: str) -> dict[str, Any]:
    """Parse VERTEX_AI_TOKEN as JSON or base64-encoded JSON.

    Parameters
    ----------
    raw : str
        Raw VERTEX_AI_TOKEN value

    Returns
    -------
    dict[str, Any]
        Service account info dictionary

    Raises
    ------
    ValueError
        If the value is not valid service-account JSON
    """
    candidates = [raw]
    decoded = _decode_base64(raw)
    if decoded is not None:
        candidates.append(decoded)

    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped.startswith("{"):
            continue
        try:
            info = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(info, dict) and info.get("type") == "service_account":
            return info

    raise ValueError(
        "VERTEX_AI_TOKEN must be a Google access token, a service account JSON, "
        "or a base64-encoded service account JSON"
    )


def _decode_base64(raw: str) -> str | None:
    """Decode standard or URL-safe base64, adding padding if needed.

    Parameters
    ----------
    raw : str
        Base64-encoded text, possibly without ``=`` padding

    Returns
    -------
    str | None
        Decoded UTF-8 text, or ``None`` if decoding fails
    """
    padded = raw + "=" * ((4 - len(raw) % 4) % 4)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            return decoder(padded).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def _vertex_project_id(info: dict[str, Any] | None = None) -> str:
    """Resolve the GCP project ID for Vertex AI MaaS.

    Parameters
    ----------
    info : dict[str, Any] | None
        Optional service-account info that may contain ``project_id``

    Returns
    -------
    str
        GCP project ID

    Raises
    ------
    ValueError
        If no project ID can be resolved
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEXAI_PROJECT")
    if not project and info is not None:
        raw_project = info.get("project_id")
        if isinstance(raw_project, str):
            project = raw_project
    if not project:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT environment variable required for Vertex AI MaaS models"
        )
    return project


def _create_vertex_maas_lm(model: str, *, temperature: float, max_tokens: int) -> dspy.LM:
    """Create a DSPy LM pointed at Vertex AI's OpenAI-compatible MaaS endpoint.

    Parameters
    ----------
    model : str
        Model identifier (e.g. ``openai/gpt-oss-120b-maas``)
    temperature : float
        Sampling temperature
    max_tokens : int
        Max output tokens

    Returns
    -------
    dspy.LM
        LM configured for the Vertex OpenAI-compatible endpoint
    """
    location = config.get("models.vertex_ai.location", "us-central1")
    raw = os.environ.get("VERTEX_AI_TOKEN", "").strip()
    info: dict[str, Any] | None = None
    if (
        raw
        and not raw.startswith("ya29.")
        and not os.environ.get("GOOGLE_CLOUD_PROJECT")
        and not os.environ.get("VERTEXAI_PROJECT")
    ):
        info = _parse_service_account_info(raw)
    project = _vertex_project_id(info)
    api_base = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{location}/endpoints/openapi"
    )
    token = vertex_access_token()
    logger.info(f"Using Vertex AI MaaS endpoint in {location} for {model}")
    return dspy.LM(
        model,
        api_base=api_base,
        api_key=token,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _create_gemini_lm(model: str, *, temperature: float, max_tokens: int) -> dspy.LM:
    """Create a DSPy LM for the Gemini Developer API.

    Parameters
    ----------
    model : str
        Gemini model identifier (e.g. ``gemini/gemini-3.7-flash``)
    temperature : float
        Sampling temperature
    max_tokens : int
        Max output tokens

    Returns
    -------
    dspy.LM
        LM configured for the Gemini Developer API

    Raises
    ------
    ValueError
        If GEMINI_API_KEY is not set
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable required")
    return dspy.LM(
        model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

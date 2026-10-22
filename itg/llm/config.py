"""Choose the chat model provider from environment settings.

Both providers expose an OpenAI-compatible API, so one client works for
either:
  - foundry: an Azure AI Foundry deployment, called through the /openai/v1/
    endpoint with the resource's API key.
  - github: GitHub Models, authenticated with GITHUB_TOKEN or, failing that,
    the token from `gh auth token`.

Settings come from `.env` (see `.env.example`). Secrets are never logged.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROVIDERS = ("foundry", "github")
GITHUB_MODELS_URL = "https://models.github.ai/inference"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 30.0

    def __repr__(self) -> str:  # keep the key out of logs and tracebacks
        return f"LLMSettings(provider={self.provider!r}, base_url={self.base_url!r}, model={self.model!r})"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def _foundry_base_url(endpoint: str) -> str:
    """Accept https://<res>.openai.azure.com, .../openai or .../openai/v1 and normalise to .../openai/v1/."""
    url = endpoint.rstrip("/")
    if url.endswith("/openai/v1"):
        return url + "/"
    if url.endswith("/openai"):
        return url + "/v1/"
    return url + "/openai/v1/"


def _gh_token() -> str:
    if not shutil.which("gh"):
        return ""
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def load_settings(env: dict[str, str] | None = None) -> LLMSettings:
    """Build settings from `env` (defaults to os.environ after loading .env)."""
    if env is None:
        _load_dotenv()
        env = dict(os.environ)
    provider = env.get("ITG_LLM_PROVIDER", "foundry").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"ITG_LLM_PROVIDER must be one of {PROVIDERS}, got {provider!r}")
    timeout = float(env.get("ITG_LLM_TIMEOUT_S", "30"))

    if provider == "foundry":
        missing = [k for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT")
                   if not env.get(k)]
        if missing:
            raise ValueError(f"Foundry provider needs {', '.join(missing)} in .env")
        return LLMSettings("foundry", _foundry_base_url(env["AZURE_OPENAI_ENDPOINT"]),
                           env["AZURE_OPENAI_API_KEY"], env["AZURE_OPENAI_DEPLOYMENT"], timeout)

    token = env.get("GITHUB_TOKEN") or _gh_token()
    if not token:
        raise ValueError("GitHub provider needs GITHUB_TOKEN in .env or a logged-in `gh` CLI")
    return LLMSettings("github", env.get("GITHUB_MODELS_URL", GITHUB_MODELS_URL), token,
                       env.get("GITHUB_MODELS_MODEL", "openai/gpt-4.1-mini"), timeout)


def get_client(settings: LLMSettings | None = None):
    """Return (OpenAI client, model name) for the configured provider."""
    from openai import OpenAI

    s = settings or load_settings()
    return OpenAI(base_url=s.base_url, api_key=s.api_key, timeout=s.timeout_s, max_retries=2), s.model


def chat(messages: list[dict], settings: LLMSettings | None = None, **kwargs) -> str:
    """One chat completion; returns the reply text. Callers handle failures and fall back to templates."""
    client, model = get_client(settings)
    resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
    if not getattr(resp, "choices", None):  # e.g. a 200 with a plain-text body instead of a completion
        raise RuntimeError(f"{settings.provider if settings else 'provider'} returned no completion: {str(resp)[:80]!r}")
    return resp.choices[0].message.content or ""

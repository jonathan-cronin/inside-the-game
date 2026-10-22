"""Provider config tests: no network, no real secrets."""
import pytest

from itg.llm.config import load_settings

FOUNDRY = {"ITG_LLM_PROVIDER": "foundry", "AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_DEPLOYMENT": "mini"}


@pytest.mark.parametrize("endpoint", ["https://r.openai.azure.com", "https://r.openai.azure.com/",
                                      "https://r.openai.azure.com/openai", "https://r.openai.azure.com/openai/v1/"])
def test_foundry_endpoint_normalised(endpoint):
    s = load_settings({**FOUNDRY, "AZURE_OPENAI_ENDPOINT": endpoint})
    assert s.base_url == "https://r.openai.azure.com/openai/v1/"
    assert s.model == "mini"


def test_foundry_missing_settings_named():
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        load_settings(FOUNDRY)


def test_github_token_and_default_model():
    s = load_settings({"ITG_LLM_PROVIDER": "github", "GITHUB_TOKEN": "t"})
    assert s.base_url == "https://models.github.ai/inference"
    assert s.model == "openai/gpt-4.1-mini"


def test_key_not_in_repr():
    s = load_settings({**FOUNDRY, "AZURE_OPENAI_ENDPOINT": "https://r.openai.azure.com",
                       "AZURE_OPENAI_API_KEY": "supersecret"})
    assert "supersecret" not in repr(s)


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        load_settings({"ITG_LLM_PROVIDER": "nope"})

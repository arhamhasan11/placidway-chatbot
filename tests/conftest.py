"""Shared test setup: no test may reach a paid API, even when a real key sits in .env."""

import pytest

import config


@pytest.fixture(autouse=True)
def no_real_api_key(monkeypatch):
    # Tests that need a key set their own fake one; everything else sees none, so a forgotten
    # mock fails fast with LLMError instead of quietly calling OpenAI.
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")

"""
Tests for OpenAI provider.
"""

import pytest
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Skip tests if API key not available
if not os.environ.get("OPENAI_API_KEY"):
    pytest.skip("OpenAI API key not available", allow_module_level=True)

from just_prompt.atoms.llm_providers import openai

def test_list_models():
    """Test listing OpenAI models."""
    models = openai.list_models()
    
    # Assertions
    assert isinstance(models, list)
    assert len(models) > 0
    assert all(isinstance(model, str) for model in models)
    
    # Check for at least one expected model
    gpt_models = [model for model in models if "gpt" in model.lower()]
    assert len(gpt_models) > 0, "No GPT models found"


def test_prompt():
    """Test sending prompt to OpenAI."""
    response = openai.prompt("What is the capital of France?", "gpt-4o-mini")
    
    # Assertions
    assert isinstance(response, str)
    assert len(response) > 0
    assert "paris" in response.lower() or "Paris" in response
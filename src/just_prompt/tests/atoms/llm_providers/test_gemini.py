"""
Tests for Gemini provider.
"""

import pytest
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Skip tests if API key not available
if not os.environ.get("GEMINI_API_KEY"):
    pytest.skip("Gemini API key not available", allow_module_level=True)

from just_prompt.atoms.llm_providers import gemini

def test_list_models():
    """Test listing Gemini models."""
    models = gemini.list_models()
    
    # Assertions
    assert isinstance(models, list)
    assert len(models) > 0
    assert all(isinstance(model, str) for model in models)
    
    # Check for at least one expected model containing gemini
    gemini_models = [model for model in models if "gemini" in model.lower()]
    assert len(gemini_models) > 0, "No Gemini models found"


def test_prompt():
    """Test sending prompt to Gemini."""
    # Using gemini-1.5-flash as the model for testing
    response = gemini.prompt("What is the capital of France?", "gemini-1.5-flash")
    
    # Assertions
    assert isinstance(response, str)
    assert len(response) > 0
    assert "paris" in response.lower() or "Paris" in response
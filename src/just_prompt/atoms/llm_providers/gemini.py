"""
Google Gemini provider implementation.
"""

import os
from typing import List
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def prompt(text: str, model: str) -> str:
    """
    Send a prompt to Google Gemini and get a response.
    
    Args:
        text: The prompt text
        model: The model name
        
    Returns:
        Response string from the model
    """
    try:
        logger.info(f"Sending prompt to Gemini model: {model}")
        
        # Create generative model
        gemini_model = genai.GenerativeModel(model_name=model)

        # Generate content
        response = gemini_model.generate_content(text)
        
        return response.text
    except Exception as e:
        logger.error(f"Error sending prompt to Gemini: {e}")
        raise ValueError(f"Failed to get response from Gemini: {str(e)}")


def list_models() -> List[str]:
    """
    List available Google Gemini models.
    
    Returns:
        List of model names
    """
    try:
        logger.info("Listing Gemini models")
        
        # Get the list of models
        models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                models.append(m.name)
                
        # Format model names - strip the "models/" prefix if present
        formatted_models = [model.replace("models/", "") for model in models]
        
        return formatted_models
    except Exception as e:
        logger.error(f"Error listing Gemini models: {e}")
        # Return some known models if API fails
        logger.info("Returning hardcoded list of known Gemini models")
        return [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.0-pro",
            "gemini-2.0-flash",
            "gemini-2.5-pro-preview-03-25",
        ]
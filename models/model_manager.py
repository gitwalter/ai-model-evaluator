#!/usr/bin/env python3
"""
Model Manager for Gemini API interactions.
Handles model discovery, configuration, and content generation.
"""

import streamlit as st
import time
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import os


class ModelManager:
    """Manages Gemini model interactions and API calls."""
    
    def __init__(self):
        """Initialize the ModelManager with API configuration."""
        self._api_key = None
        self._available_models = None
        self._last_model_fetch = 0
        self._cache_duration = 300  # 5 minutes cache
        self._configure_api()
    
    def _configure_api(self) -> None:
        """Configure the Gemini API with the API key from Streamlit secrets."""
        # Try to get API key from Streamlit secrets first
        try:
            self._api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            # Fallback to environment variable for backward compatibility
            load_dotenv()
            self._api_key = os.getenv("GEMINI_API_KEY")
        
        # If no API key found, we'll handle it in the UI
        if self._api_key:
            genai.configure(api_key=self._api_key)
    
    def _check_session_api_key(self) -> bool:
        """Check if there's an API key in session state and configure it."""
        try:
            import streamlit as st
            if "api_key" in st.session_state and st.session_state.api_key:
                return self.set_api_key(st.session_state.api_key)
        except Exception:
            pass
        return False
    
    def set_api_key(self, api_key: str) -> bool:
        """Set the API key manually and configure the API."""
        if not api_key or not api_key.strip():
            return False
        
        try:
            # Test the API key by making a simple call
            genai.configure(api_key=api_key.strip())
            
            # Try to list models to verify the key works
            test_models = genai.list_models()
            if test_models:
                self._api_key = api_key.strip()
                # Clear cached models to force refresh
                self._available_models = None
                self._last_model_fetch = 0
                return True
            else:
                return False
        except Exception:
            return False
    
    def get_api_key_status(self) -> Dict[str, Any]:
        """Get the current API key status."""
        # Check session state for API key if not already configured
        if not self._api_key:
            self._check_session_api_key()
        
        return {
            'has_api_key': bool(self._api_key),
            'is_valid': self._is_api_key_valid() if self._api_key else False
        }
    
    def _is_api_key_valid(self) -> bool:
        """Check if the current API key is valid."""
        if not self._api_key:
            return False
        
        try:
            # Try to list models to verify the key works
            genai.list_models()
            return True
        except Exception:
            return False
    
    def get_available_models(self, force_refresh: bool = False) -> List[str]:
        """Get list of available Gemini models that support content generation."""
        # Check if we have a valid API key
        if not self._api_key:
            # Try to get API key from session state
            if not self._check_session_api_key():
                return []
        
        current_time = time.time()
        
        # Return cached models if still valid
        if (not force_refresh and 
            self._available_models and 
            current_time - self._last_model_fetch < self._cache_duration):
            return self._available_models
        
        try:
            models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    models.append(model.name)
            
            self._available_models = sorted(models)
            self._last_model_fetch = current_time
            return self._available_models
            
        except Exception as e:
            st.error(f"Error fetching models: {str(e)}")
            # Fallback to common models if API call fails
            fallback_models = [
                "gemini-1.5-pro-latest",
                "gemini-1.5-flash-latest", 
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite"
            ]
            self._available_models = fallback_models
            self._last_model_fetch = current_time
            return fallback_models
    
    def test_model(self, model_name: str, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """Test a single model and return results."""
        # Check if we have a valid API key
        if not self._api_key:
            # Try to get API key from session state
            if not self._check_session_api_key():
                return {
                    "model_name": model_name,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "response": "Error: No API key configured. Please enter your API key in the sidebar.",
                    "status": "error",
                    "error": "No API key configured",
                    "response_time": 0
                }
        
        start_time = time.time()
        
        try:
            # Create model instance
            model = genai.GenerativeModel(model_name)
            
            # Prepare the message
            if system_prompt:
                # For models that support system prompts
                try:
                    response = model.generate_content(
                        prompt,
                        system_instruction=system_prompt
                    )
                except:
                    # Fallback for models that don't support system prompts
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                    response = model.generate_content(full_prompt)
            else:
                response = model.generate_content(prompt)
            
            # Extract response text
            response_text = response.text if response.text else "No response text"
            
            end_time = time.time()
            
            return {
                "model_name": model_name,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response": response_text,
                "status": "success",
                "response_time": end_time - start_time
            }
            
        except Exception as e:
            end_time = time.time()
            
            return {
                "model_name": model_name,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response": f"Error: {str(e)}",
                "status": "error",
                "error": str(e),
                "response_time": end_time - start_time
            }
    
    def test_multiple_models(self, model_names: List[str], prompt: str, 
                           system_prompt: str = "") -> List[Dict[str, Any]]:
        """Test multiple models and return results."""
        results = []
        
        for model_name in model_names:
            result = self.test_model(model_name, prompt, system_prompt)
            results.append(result)
        
        return results
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        try:
            models = genai.list_models()
            for model in models:
                if model.name == model_name:
                    return {
                        "name": model.name,
                        "display_name": model.display_name,
                        "description": model.description,
                        "generation_methods": model.supported_generation_methods,
                        "input_token_limit": getattr(model, 'input_token_limit', None),
                        "output_token_limit": getattr(model, 'output_token_limit', None)
                    }
            return None
        except Exception as e:
            st.error(f"Error fetching model info: {str(e)}")
            return None

import logging
from typing import Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import settings

logger = logging.getLogger(__name__)

class GemmaError(Exception):
    """Base exception for all Gemma service errors."""
    pass

class GemmaConfigurationError(GemmaError):
    """Raised when the configuration is invalid or missing."""
    pass

class GemmaAPIError(GemmaError):
    """Raised when the Google GenAI API returns an error."""
    pass

class GemmaService:
    """
    Service class to handle communications with Google AI Studio (Gemini/Gemma API).
    Provides robust text generation with automatic retries and structured error handling.
    """
    
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        
        # Guard warning for unconfigured API keys during initialization
        if not self.api_key or self.api_key == "mock_key_for_testing":
            logger.warning(
                "GEMINI_API_KEY is not configured with a valid production key. "
                "API calls will fail until a valid key is set in .env."
            )
            
        self.client = None
        try:
            # Initialize the modern Google GenAI Client (only if key seems valid)
            if self.api_key and self.api_key != "mock_key_for_testing":
                self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize GenAI Client: {e}")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIError)
    )
    def _call_api_with_retry(
        self, 
        prompt: str, 
        system_instruction: Optional[str], 
        temperature: float, 
        max_output_tokens: Optional[int]
    ):
        """
        Internal helper method wrapped with tenacity to handle network retries 
        on transient API connection failures or rate-limiting.
        """
        # Configure generation parameters dynamically
        config_args = {"temperature": temperature}
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if max_output_tokens:
            config_args["max_output_tokens"] = max_output_tokens

        config = types.GenerateContentConfig(**config_args)
        
        logger.debug(f"Sending prompt to model {self.model_name} (temp={temperature})...")
        
        # Modern client call structure
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        return response
    def _format_gemma_chat_prompt(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Wraps prompt in instruction-tuned Gemma template format."""
        formatted = ""
        if system_instruction:
            formatted += f"<start_of_turn>user\n{system_instruction}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            formatted += f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        return formatted

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None
    ) -> str:
        """
        Generates a natural language response from the Gemma/Gemini model.
        
        Args:
            prompt: The user input prompt.
            system_instruction: Optional system rules to guide the model's behavior.
            temperature: Controls randomness (0.0 is deterministic, 1.0 is creative).
            max_output_tokens: Optional limit on response length.
            
        Returns:
            The generated text response.
            
        Raises:
            GemmaConfigurationError: If configurations are invalid or mock keys are used.
            GemmaAPIError: If the API returns an error or fails after retries.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        # Determine backend to use
        backend = settings.LLM_BACKEND.lower()
        if ":" in self.model_name:
            backend = "ollama"
        elif "/" in self.model_name:
            backend = "huggingface"

        is_gemma_model = "gemma" in self.model_name.lower()
        
        # Pre-format prompt for Gemma model structure if applicable
        active_prompt = prompt
        active_system = system_instruction
        if is_gemma_model:
            active_prompt = self._format_gemma_chat_prompt(prompt, system_instruction)
            active_system = None  # Injected directly in template prompt

        if backend == "ollama":
            try:
                import requests
                url = f"{settings.OLLAMA_HOST}/api/generate"
                payload = {
                    "model": self.model_name,
                    "prompt": active_prompt,
                    "stream": False
                }
                if active_system:
                    payload["system"] = active_system
                
                options = {}
                if temperature is not None:
                    options["temperature"] = temperature
                if max_output_tokens is not None:
                    options["num_predict"] = max_output_tokens
                if options:
                    payload["options"] = options

                logger.debug(f"Sending prompt to local Ollama {self.model_name}...")
                response = requests.post(url, json=payload, timeout=90)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                logger.error(f"Ollama execution error: {e}")
                raise GemmaAPIError(f"Ollama Error occurred: {e}") from e

        elif backend == "huggingface":
            try:
                import requests
                import time
                url = f"https://api-inference.huggingface.co/models/{self.model_name}"
                headers = {}
                if settings.HF_API_KEY:
                    headers["Authorization"] = f"Bearer {settings.HF_API_KEY}"
                
                full_prompt = active_prompt
                if active_system:
                    full_prompt = f"System: {active_system}\nUser: {active_prompt}\nAssistant:"
                
                payload = {
                    "inputs": full_prompt,
                    "parameters": {
                        "return_full_text": False
                    }
                }
                if temperature is not None:
                    payload["parameters"]["temperature"] = max(0.01, temperature)
                if max_output_tokens is not None:
                    payload["parameters"]["max_new_tokens"] = max_output_tokens

                logger.debug(f"Sending prompt to Hugging Face model {self.model_name}...")
                response = requests.post(url, headers=headers, json=payload, timeout=90)
                
                # If model is loading, wait and retry once
                if response.status_code == 503:
                    wait_time = response.json().get("estimated_time", 20)
                    logger.info(f"Hugging Face model is loading. Waiting for {wait_time}s...")
                    time.sleep(min(wait_time, 10))
                    response = requests.post(url, headers=headers, json=payload, timeout=90)

                response.raise_for_status()
                res_json = response.json()
                
                if isinstance(res_json, list) and len(res_json) > 0:
                    return res_json[0].get("generated_text", "").strip()
                elif isinstance(res_json, dict) and "generated_text" in res_json:
                    return res_json.get("generated_text", "").strip()
                else:
                    return str(res_json).strip()
            except Exception as e:
                logger.error(f"Hugging Face execution error: {e}")
                raise GemmaAPIError(f"Hugging Face API Error occurred: {e}") from e

        else:
            # Fallback to Gemini API
            if self.api_key == "mock_key_for_testing" or not self.api_key:
                raise GemmaConfigurationError(
                    "Cannot perform API operations with a mock API key. "
                    "Please configure a valid GEMINI_API_KEY in your local .env file, "
                    "or select the local Ollama backend."
                )

            try:
                response = self._call_api_with_retry(
                    prompt=active_prompt,
                    system_instruction=active_system,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens
                )
                
                if not response or not response.text:
                    raise GemmaAPIError("Received an empty response from the AI API.")
                    
                return response.text
                
            except APIError as api_err:
                logger.error(f"Google GenAI API Error: {api_err}")
                raise GemmaAPIError(f"API Error occurred: {api_err.message} (Code: {api_err.code})") from api_err
            except Exception as e:
                logger.error(f"Unexpected error in Gemma service: {e}")
                raise GemmaAPIError(f"An unexpected error occurred during generation: {e}") from e

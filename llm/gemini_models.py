# llm/gemini_models.py
import google.generativeai as genai
import numpy as np
import logging
from typing import Union, List, Dict, Any, Optional
import time
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential

# Import base classes
from .base_models import BaseTextGenerationModel, BaseEmbeddingModel

# --- Configuration ---
# It's better to configure the API key once, perhaps in main.py
# based on arguments, rather than in each class constructor.
# We'll assume genai.configure() is called elsewhere.

# Default model names (can be overridden by arguments)
DEFAULT_GEMINI_TEXT_MODEL = "gemini-1.5-flash-latest" # Or "gemini-1.0-pro", "gemini-1.5-pro-latest" etc.
DEFAULT_GEMINI_EMBEDDING_MODEL = "models/text-embedding-004" # Or "models/embedding-001"

class GeminiTextGenerationModel(BaseTextGenerationModel):
    """Gemini text generation model implementation using google-generativeai."""
    def __init__(self, api_key: str, model_name: str = DEFAULT_GEMINI_TEXT_MODEL):
        # Configure the API key upon initialization
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            logging.error(f"Failed to configure Google API key: {e}")
            raise ValueError("Invalid Google API Key provided for Gemini.") from e

        self.model_name = model_name
        # System instruction handling differs in Gemini 1.5 vs 1.0
        # For simplicity, we'll handle it via GenerationConfig or prepending for now
        self.model = genai.GenerativeModel(self.model_name)
        self.logger = logging.getLogger(__name__ + ".GeminiTextGenerationModel")
        self.logger.info(f"Initialized GeminiTextGenerationModel with model: {self.model_name}")

    # Use exponential backoff for API errors like rate limits or temporary issues
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    def generate(self, system: str, user: str, **kwargs) -> str:
        """
        Generates text using the Gemini model.

        Maps common parameters and handles potential API errors with retries.
        Constructs the 'contents' list for the API call.
        """
        try:
            # --- Parameter Mapping ---
            # Gemini uses 'candidate_count', 'stop_sequences', 'max_output_tokens', 'temperature', 'top_p', 'top_k'
            generation_config = {
                "temperature": kwargs.get('temperature', 0.8),
                "top_p": kwargs.get('top_p', 0.95), # Gemini default is often higher
                # Map max_tokens (common name) to Gemini's max_output_tokens
                "max_output_tokens": min(int(kwargs.get('max_tokens', 2048)), 8192), # Default/Max for Gemini Flash/Pro
                # Add other relevant mappings if needed
                # "top_k": kwargs.get('top_k', None),
                # "stop_sequences": kwargs.get('stop_sequences', None),
            }
            # Filter out None values
            generation_config = {k: v for k, v in generation_config.items() if v is not None}

            # --- Content Construction ---
            # Gemini expects a list of 'Content' objects (dicts with 'role' and 'parts')
            # Roles are typically 'user' and 'model' (not 'assistant')
            contents = []
            # System prompt handling (Gemini 1.5 uses system_instruction, 1.0 needs prepending)
            # Simple approach: Prepend system instructions to the first user message if model < 1.5
            # Better approach (if using 1.5+): Use system_instruction parameter of GenerativeModel
            effective_user_prompt = user
            system_instruction = None
            if system:
                 # Check if model supports system_instruction (requires google-generativeai >= 0.5.0 and 1.5 model)
                 # For broad compatibility, let's prepend for now, or use the dedicated param if available
                 # if hasattr(self.model, 'system_instruction'): # Check if attribute exists
                 #     system_instruction = genai.types.Content(role="system", parts=[genai.types.Part(text=system)])
                 # else:
                 #     effective_user_prompt = f"System Instructions:\n{system}\n\nUser Request:\n{user}"
                 # Simplest: just pass as system_instruction if model supports it
                 # Note: This requires initializing model with system_instruction if needed:
                 # self.model = genai.GenerativeModel(self.model_name, system_instruction=system)
                 # Let's stick to adding it to contents for now for simplicity across versions
                 contents.append({'role': 'user', 'content': system}) # Treat system as initial user message context
                 contents.append({'role': 'model', 'content': "Okay, I understand the instructions."}) # Acknowledge system prompt


            # Handle history if passed
            if 'history' in kwargs and isinstance(kwargs['history'], list):
                for turn in kwargs['history']:
                    # Map roles: 'assistant' -> 'model'
                    role = turn.get('role', 'user')
                    mapped_role = 'model' if role == 'assistant' else 'user'
                    contents.append({'role': mapped_role, 'content': turn.get('content', '')})

            # Add the current user prompt
            contents.append({'role': 'user', 'content': effective_user_prompt})


            self.logger.debug(f"Calling Gemini API ({self.model_name}) with config: {generation_config}")
            # self.logger.debug(f"Contents: {contents}") # Be careful logging prompts

            # --- API Call ---
            response = self.model.generate_content(
                contents=contents,
                generation_config=genai.types.GenerationConfig(**generation_config),
                # safety_settings=... # Optional: configure safety settings
            )

            # --- Response Handling ---
            # Check for blocked prompts or empty responses
            if not response.candidates:
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                     block_reason = response.prompt_feedback.block_reason
                     self.logger.warning(f"Gemini prompt blocked ({self.model_name}). Reason: {block_reason}")
                     return f"[PROMPT BLOCKED: {block_reason}]"
                 else:
                     self.logger.error(f"Gemini API Error ({self.model_name}): No candidates returned, response: {response}")
                     return "[ERROR: No Response Candidates]"

            # Extract text, handling potential errors
            try:
                generated_text = response.text
                self.logger.debug(f"Gemini ({self.model_name}) generated content successfully.")
                return generated_text
            except ValueError as e:
                # This can happen if the response was blocked for safety *after* generation started
                self.logger.warning(f"Gemini response blocked or invalid ({self.model_name}): {e}. Response: {response}")
                # Check finish_reason in candidates
                finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
                return f"[RESPONSE BLOCKED/INVALID: {finish_reason}]"
            except Exception as e:
                 self.logger.error(f"Gemini API Error ({self.model_name}): Error extracting text - {e}. Response: {response}")
                 return "[ERROR: Failed to Extract Text]"

        except Exception as e:
            # Catch potential API errors (rate limits, config errors, etc.)
            self.logger.exception(f"Gemini generation failed ({self.model_name}) after retries: {e}", exc_info=True)
            # Check for specific Google API errors if possible
            # from google.api_core import exceptions as google_exceptions
            # if isinstance(e, google_exceptions.ResourceExhausted): # Rate limit
            #     raise # Let tenacity handle retry
            return "[ERROR: Generation Failed Exception]"


class GeminiEmbeddingModel(BaseEmbeddingModel):
    """Gemini text embedding model implementation using google-generativeai."""
    def __init__(self, api_key: str, model_name: str = DEFAULT_GEMINI_EMBEDDING_MODEL):
         # Configure the API key upon initialization
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            logging.error(f"Failed to configure Google API key: {e}")
            raise ValueError("Invalid Google API Key provided for Gemini.") from e

        self.model_name = model_name
        # Determine embedding dimension based on model name
        # See: https://ai.google.dev/docs/embeddings/get-embeddings#supported_models
        if "embedding-001" in model_name:
             self._embedding_dim = 768
        elif "text-embedding-004" in model_name: # New default as of mid-2024
             self._embedding_dim = 768
        elif "text-embedding-preview" in model_name: # Example, check actual names
             self._embedding_dim = 768 # Or other dim
        else:
             self.logger.warning(f"Unknown Gemini embedding model '{model_name}'. Assuming dim 768. Verify model name.")
             self._embedding_dim = 768 # Default guess

        self.logger = logging.getLogger(__name__ + ".GeminiEmbeddingModel")
        self.logger.info(f"Initialized GeminiEmbeddingModel with model: {self.model_name}, dim: {self._embedding_dim}")
        # Gemini API handles batching internally up to a limit (e.g., 100 texts)
        # self.max_batch_size = 100 # Informational

    @property
    def embedding_dim(self) -> int:
        """Returns the embedding dimension for this Gemini model."""
        return self._embedding_dim

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encodes text using the Gemini embedding model."""
        if isinstance(text, str):
            inputs = [text]
        else:
            inputs = text

        if not inputs:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        try:
            self.logger.debug(f"Encoding batch of {len(inputs)} texts with {self.model_name}")
            # task_type="retrieval_document" is common, others exist (e.g., retrieval_query, semantic_similarity)
            # Choose based on your downstream task. 'retrieval_document' is a safe default for storage.
            result = genai.embed_content(
                model=self.model_name,
                content=inputs,
                task_type="retrieval_document",
                # title="Optional title if texts are document parts" # Optional
            )

            embeddings = result.get('embedding', None)
            if embeddings is None:
                 self.logger.error(f"Gemini Embedding API Error ({self.model_name}): 'embedding' key not found in response: {result}")
                 raise ValueError("Invalid embedding response structure from Gemini API")

            # Embeddings are returned as a list of lists
            np_embeddings = np.array(embeddings, dtype=np.float32)

            # Verify shape
            if np_embeddings.ndim != 2 or np_embeddings.shape[0] != len(inputs) or np_embeddings.shape[1] != self.embedding_dim:
                 self.logger.error(f"Gemini Embedding dim/shape mismatch! Expected ({len(inputs)}, {self.embedding_dim}), got {np_embeddings.shape}. API/Model issue?")
                 raise ValueError(f"Embedding dimension/shape mismatch from Gemini API ({self.model_name})")

            self.logger.debug(f"Gemini embedding ({self.model_name}) successful. Shape: {np_embeddings.shape}")
            return np_embeddings # Shape (N, D)

        except Exception as e:
            self.logger.exception(f"Gemini embedding failed ({self.model_name}) after retries: {e}", exc_info=True)
            # from google.api_core import exceptions as google_exceptions
            # if isinstance(e, google_exceptions.ResourceExhausted): # Rate limit
            #     raise # Let tenacity handle retry
            # Return empty array on failure
            return np.empty((0, self.embedding_dim), dtype=np.float32)

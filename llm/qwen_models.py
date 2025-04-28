# llm/qwen_models.py
from http import HTTPStatus
from typing import Union, List, Dict, Any
import dashscope
import numpy as np
import logging
import time
from tenacity import retry, stop_after_attempt, wait_fixed

# Import base classes
from .base_models import BaseTextGenerationModel, BaseEmbeddingModel

class QwenTextGenerationModel(BaseTextGenerationModel):
    """Qwen text generation model implementation using DashScope SDK."""
    def __init__(self, api_key: str, model_name: str = "qwen-turbo"):
        dashscope.api_key = api_key
        self.model_name = model_name
        self.logger = logging.getLogger(__name__ + ".QwenTextGenerationModel")
        self.logger.info(f"Initialized QwenTextGenerationModel with model: {self.model_name}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def generate(self, system: str, user: str, **kwargs) -> str:
        """
        Generates text using the Qwen model via DashScope.

        Handles potential API errors and rate limits with retries.
        Maps common parameters like temperature, top_p, max_tokens.
        Supports optional 'condition' for prepending assistant response.
        """
        try:
            # Parameter mapping and validation
            max_tokens = kwargs.get('max_tokens', 1500) # Default Qwen value
            max_tokens = min(max(int(max_tokens), 1), 8000) # Ensure integer and within Qwen limits (adjust if needed)
            temperature = kwargs.get('temperature', 0.8)
            top_p = kwargs.get('top_p', 0.8) # Default Qwen value
            condition = kwargs.get('condition', None) # Optional condition for specific prompts

            messages = [{'role': 'system', 'content': system}]
            # Handle potential history passed via kwargs (example structure)
            if 'history' in kwargs and isinstance(kwargs['history'], list):
                 messages.extend(kwargs['history']) # Append history turns
            messages.append({'role': 'user', 'content': user})

            self.logger.debug(f"Calling Qwen API ({self.model_name}) with max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
            # self.logger.debug(f"Messages: {messages}") # Be careful logging potentially sensitive prompts

            response = dashscope.Generation.call(
                model=self.model_name,
                messages=messages,
                result_format='message',
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                # Pass other specific kwargs if needed, filtering out base class ones
                **{k: v for k, v in kwargs.items() if k not in ['condition', 'history']}
            )

            if response.status_code == HTTPStatus.OK:
                if hasattr(response.output, 'choices') and response.output.choices:
                    content = response.output.choices[0].message.content
                    self.logger.debug(f"Qwen ({self.model_name}) generated content successfully.")
                    # Prepend condition if provided
                    full_response = (condition + content) if condition else content
                    return full_response
                else:
                    self.logger.error(f"Qwen API Error ({self.model_name}): Invalid response structure - {response}")
                    return "[ERROR: Invalid Response Structure]"
            # Handle specific API errors (e.g., rate limits)
            elif response.status_code == 429: # Too Many Requests
                 self.logger.warning(f"Qwen API Error ({self.model_name}) - Rate limit hit (429). Retrying...")
                 raise Exception("Rate limit hit") # Trigger tenacity retry
            else:
                self.logger.error(f"Qwen API Error ({self.model_name}) {response.code}: {response.message}")
                return f"[ERROR: API Call Failed - Code {response.code}]"

        except Exception as e:
            self.logger.exception(f"Qwen generation failed ({self.model_name}) after retries: {e}", exc_info=True)
            # Reraise the exception if retries failed, or return an error string
            # For this use case, returning an error string might be safer to avoid crashing the pipeline
            return "[ERROR: Generation Failed Exception]"

    # Note: continue_generate is not explicitly implemented as multi-turn
    # can be handled by passing 'history' in kwargs to the main 'generate' method.


class QwenEmbeddingModel(BaseEmbeddingModel):
    """Qwen text embedding model implementation using DashScope SDK."""
    def __init__(self, api_key: str, model_name: str = "text-embedding-v2"):
        dashscope.api_key = api_key
        # Ensure compatibility, v1/v2 use 1536, v3 uses 1024
        if "v1" in model_name or "v2" in model_name:
             self._embedding_dim = 1536
             self._text_type = 'query' # or 'document' depending on use case
        elif "v3" in model_name:
             self._embedding_dim = 1024
             self._text_type = 'document' # v3 uses 'document'
        else:
             # Default or raise error if model name is unknown
             self.logger.warning(f"Unknown Qwen embedding model '{model_name}'. Defaulting to dim 1536. Check compatibility.")
             self._embedding_dim = 1536
             self._text_type = 'query'

        self.model_name = model_name
        self.max_batch_size = 25  # Check DashScope docs for the specific model limit (v3 is 25)
        self.logger = logging.getLogger(__name__ + ".QwenEmbeddingModel")
        self.logger.info(f"Initialized QwenEmbeddingModel with model: {self.model_name}, dim: {self._embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        """Returns the embedding dimension for this Qwen model."""
        return self._embedding_dim

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encodes text using the Qwen embedding model via DashScope."""
        if isinstance(text, str):
            inputs = [text]
        else:
            inputs = text

        if not inputs:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        all_embeddings = []
        try:
            for i in range(0, len(inputs), self.max_batch_size):
                batch = inputs[i:i + self.max_batch_size]
                self.logger.debug(f"Encoding batch of {len(batch)} texts with {self.model_name}")

                resp = dashscope.TextEmbedding.call(
                    model=self.model_name,
                    input=batch,
                    text_type=self._text_type,
                    # dimension=self.embedding_dim, # Dimension often inferred or fixed for model
                    # encoding_format='float' # Default is float
                )

                if resp.status_code == HTTPStatus.OK:
                    if 'embeddings' in resp.output and resp.output['embeddings']:
                        batch_embeddings = [
                            np.array(item['embedding'], dtype=np.float32)
                            for item in resp.output['embeddings']
                        ]
                        # Verify dimensions
                        for j, emb in enumerate(batch_embeddings):
                             if emb.shape[0] != self.embedding_dim:
                                 self.logger.error(f"Qwen Embedding dim mismatch! Expected {self.embedding_dim}, got {emb.shape[0]} for item {i+j}. API/Model issue?")
                                 # Handle mismatch: return error, pad, or raise
                                 raise ValueError(f"Embedding dimension mismatch from Qwen API ({self.model_name})")
                        all_embeddings.extend(batch_embeddings)
                    else:
                         self.logger.error(f"Qwen Embedding API Error ({self.model_name}): Invalid response structure - {resp}")
                         raise ValueError("Invalid embedding response structure from Qwen API")
                elif resp.status_code == 429: # Rate limit
                    self.logger.warning(f"Qwen Embedding API Error ({self.model_name}) - Rate limit hit (429). Retrying...")
                    raise Exception("Rate limit hit") # Trigger retry
                else:
                    self.logger.error(f"Qwen Embedding API Error ({self.model_name}) {resp.code}: {resp.message}")
                    raise Exception(f"Qwen Embedding API call failed - Code {resp.code}")

            if not all_embeddings:
                 self.logger.error(f"Qwen embedding ({self.model_name}) returned no embeddings.")
                 return np.empty((0, self.embedding_dim), dtype=np.float32)

            return np.stack(all_embeddings) # Shape (N, D)

        except Exception as e:
            self.logger.exception(f"Qwen embedding failed ({self.model_name}) after retries: {e}", exc_info=True)
            # Return empty array on failure to avoid crashing pipeline
            return np.empty((0, self.embedding_dim), dtype=np.float32)


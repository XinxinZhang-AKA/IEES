# llm/base_models.py
import abc
import numpy as np
from typing import Union, List, Dict, Any

class BaseTextGenerationModel(abc.ABC):
    """Abstract base class for text generation models."""

    @abc.abstractmethod
    def generate(self, system: str, user: str, **kwargs) -> str:
        """
        Generates text based on system and user prompts.

        Args:
            system: The system prompt or instructions.
            user: The user's input prompt.
            **kwargs: Additional model-specific parameters (e.g., temperature, max_tokens).
                      Should handle multi-turn history if needed via kwargs.

        Returns:
            The generated text content as a string.
        """
        pass

    def continue_generate(self, system: str, user1: str, assistant1: str, user2: str, **kwargs) -> str:
        """
        Continues a conversation based on previous turns.
        This is an optional method, specific implementations might handle history
        within the `generate` method itself using kwargs.

        Args:
            system: The system prompt.
            user1: The first user message.
            assistant1: The first assistant response.
            user2: The second user message to continue from.
            **kwargs: Additional model-specific parameters.

        Returns:
            The generated text content as a string.
        """
        # Default implementation can raise NotImplementedError or be handled in generate
        raise NotImplementedError("Multi-turn continuation might be handled within 'generate' for this model.")


class BaseEmbeddingModel(abc.ABC):
    """Abstract base class for text embedding models."""

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int:
        """Returns the dimension of the embeddings produced by this model."""
        pass

    @abc.abstractmethod
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encodes text into embedding vectors.

        Args:
            text: A single string or a list of strings to encode.

        Returns:
            A numpy array of shape (N, D), where N is the number of input texts
            and D is the embedding dimension. Returns an empty array or raises
            an error if encoding fails.
        """
        pass


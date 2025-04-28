# llm/openai_models.py
import os
import logging
from typing import Union, List, Dict, Any
import numpy as np
# Make sure openai library is installed and imported
try:
    from openai import OpenAI, APIError, RateLimitError, AuthenticationError, Timeout
except ImportError:
    raise ImportError("OpenAI library not found. Please install it using 'pip install openai'")
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAITextGenerationModel:
    """
    使用 OpenAI API 进行文本生成的模型类。
    增加了对非标准 API 响应（直接返回字符串）的处理。
    """
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", base_url: str = None):
        """
        初始化 OpenAI 客户端。

        Args:
            api_key (str): OpenAI API 密钥。
            model_name (str): 要使用的 OpenAI 模型名称 (例如, "gpt-4", "gpt-3.5-turbo")。
            base_url (str, optional): OpenAI API 的基础 URL (用于代理或自定义端点)。 Defaults to None.
        """
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        try:
            # Use timeout to prevent hanging indefinitely
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=60.0) # Added timeout
            logger.info(f"OpenAI client initialized successfully for model: {self.model_name}")
            # Optional: Test connection if needed, though generate will do this
            # try:
            #     self.client.models.list()
            #     logger.info("Successfully connected to OpenAI API endpoint.")
            # except Exception as conn_err:
            #     logger.warning(f"Could not verify connection to OpenAI endpoint: {conn_err}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    @retry(
        wait=wait_exponential(min=2, max=30), # Reduced max wait time slightly
        stop=stop_after_attempt(3),          # Reduced attempts to 3
        retry=retry_if_exception_type((APIError, RateLimitError, Timeout)), # Added Timeout
        before_sleep=lambda retry_state: logger.warning(f"Retrying OpenAI API call after error: {retry_state.outcome.exception()} (Attempt {retry_state.attempt_number})")
    )
    def generate(self, system: str, user: str, condition: str = None, **kwargs) -> str:
        """
        使用 OpenAI Chat Completion API 生成文本。
        Handles both standard OpenAI responses and direct string responses from proxies.

        Args:
            system (str): 系统提示信息。
            user (str): 用户输入信息。
            condition (str, optional): 预设的助手回复开头 (如果提供)。 Defaults to None.
            **kwargs: 传递给 OpenAI API 的其他参数 (例如, temperature, max_tokens, top_p)。

        Returns:
            str: 模型生成的文本内容，如果出错则返回空字符串。
        """
        messages = [{'role': 'system', 'content': system}]
        if condition:
             messages.append({'role': 'user', 'content': user})
             messages.append({'role': 'assistant', 'content': condition})
        else:
            messages.append({'role': 'user', 'content': user})

        api_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_tokens', 1500),
            "top_p": kwargs.get('top_p', 1.0),
        }
        api_params = {k: v for k, v in api_params.items() if v is not None}

        try:
            logger.debug(f"Calling OpenAI API with params: {api_params}")
            response = self.client.chat.completions.create(**api_params)
            logger.debug(f"Raw response type from API: {type(response)}") # Log the type

            # --- 修改开始: 处理非标准响应 ---
            # 检查响应是否是预期的 OpenAI 对象类型
            # Based on openai v1+, the response should be ChatCompletion object
            if hasattr(response, 'choices') and response.choices:
                # 标准 OpenAI 响应
                content = response.choices[0].message.content
                logger.debug(f"Received standard OpenAI response: {content[:100]}...")
                return content.strip()
            # elif isinstance(response, str): # Less likely with v1+ library unless proxy modifies heavily
            #     # 如果代理直接返回字符串 (不太可能通过 client.create 发生，但作为备用检查)
            #     logger.warning("Received non-standard raw string response from API, returning directly.")
            #     return response.strip()
            else:
                # 如果响应不是预期类型，也不是字符串，记录错误
                # This might happen if the proxy returns JSON but not in the expected structure
                logger.error(f"Received unexpected response format from API. Type: {type(response)}, Content: {str(response)[:200]}...")
                # Try to extract text heuristically if possible, otherwise return empty
                # This part is highly dependent on the actual non-standard response format
                if isinstance(response, dict) and 'text' in response: # Example heuristic
                    return response['text'].strip()
                return "" # Fallback if extraction fails
            # --- 修改结束 ---

        except AuthenticationError as e:
             logger.error(f"OpenAI API Authentication Error: {e}")
             # Authentication errors usually shouldn't be retried with the same key
             return f"[AUTHENTICATION ERROR: Check API Key for {self.base_url or 'OpenAI'}]" # Return specific error message
        except RateLimitError as e:
            logger.error(f"OpenAI API rate limit exceeded: {e}")
            raise # Reraise for retry logic
        except APIError as e:
            # Handles other API-related errors (e.g., server errors, bad requests if not caught earlier)
            logger.error(f"OpenAI API error: {e}")
            # Decide if this specific APIError should be retried or not
            # For now, let retry logic handle it, but could return "" here for certain codes
            raise # Reraise for retry logic
        except Timeout as e:
             logger.error(f"OpenAI API request timed out: {e}")
             raise # Reraise for retry logic
        except Exception as e:
            # Catch unexpected errors during the API call or response processing
            logger.error(f"An unexpected error occurred during OpenAI API call: {e}", exc_info=True) # Log traceback
            # Return empty string for unexpected errors after retries fail
            return ""

    # (continue_generate method remains unchanged unless needed)


class OpenAIEmbeddingModel:
    """
    使用 OpenAI API 进行文本嵌入的模型类。
    """
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002", base_url: str = None, embedding_dim: int = 1536):
        """
        初始化 OpenAI 客户端。

        Args:
            api_key (str): OpenAI API 密钥。
            model_name (str): 要使用的 OpenAI 嵌入模型名称 (例如, "text-embedding-ada-002", "text-embedding-3-small").
            base_url (str, optional): OpenAI API 的基础 URL。 Defaults to None.
            embedding_dim (int): 期望的嵌入维度 (需要与所选模型匹配)。
        """
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_dim = embedding_dim
        logger.info(f"Initializing OpenAIEmbeddingModel with model: {self.model_name}, dimension: {self.embedding_dim}")

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30.0) # Added timeout
            logger.info("OpenAI client for embeddings initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client for embeddings: {e}")
            raise

    @retry(
        wait=wait_exponential(min=2, max=30), # Reduced max wait
        stop=stop_after_attempt(3),          # Reduced attempts
        retry=retry_if_exception_type((APIError, RateLimitError, Timeout)), # Added Timeout
        before_sleep=lambda retry_state: logger.warning(f"Retrying OpenAI Embedding API call after error: {retry_state.outcome.exception()} (Attempt {retry_state.attempt_number})")
    )
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        使用 OpenAI Embedding API 对文本进行编码。

        Args:
            text (Union[str, List[str]]): 单个字符串或字符串列表。

        Returns:
            np.ndarray: 文本的嵌入向量 (numpy array)，形状为 (N, embedding_dim)。如果出错则引发异常或返回空数组。
        """
        if isinstance(text, str):
            inputs = [text]
        elif isinstance(text, list):
            inputs = text
        else:
             # Handle cases where input might not be string or list
             logger.error(f"Invalid input type for embedding: {type(text)}. Expected str or list.")
             return np.empty((0, self.embedding_dim), dtype=np.float32)


        if not inputs:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Filter out empty strings which can cause API errors
        original_indices = [i for i, t in enumerate(inputs) if t and isinstance(t, str)]
        filtered_inputs = [inputs[i] for i in original_indices]

        if not filtered_inputs:
             logger.warning("Input to encode contained only empty strings or non-strings.")
             return np.empty((0, self.embedding_dim), dtype=np.float32)


        try:
            logger.debug(f"Calling OpenAI Embedding API for {len(filtered_inputs)} texts with model {self.model_name}")
            response = self.client.embeddings.create(
                input=filtered_inputs,
                model=self.model_name
                # dimensions=self.embedding_dim # Only for supported models like v3
            )

            if response.data and len(response.data) > 0:
                # Sort embeddings back to original order if some inputs were filtered
                embeddings_dict = {item.index: item.embedding for item in response.data}
                # Reconstruct based on original indices, filling with zeros if an embedding failed (though API usually errors out)
                final_embeddings = []
                for i in range(len(inputs)):
                     if i in original_indices and original_indices.index(i) in embeddings_dict:
                          final_embeddings.append(embeddings_dict[original_indices.index(i)])
                     else:
                          # Append zero vector for filtered or missing embeddings
                          final_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))


                embeddings_array = np.array(final_embeddings, dtype=np.float32)

                if embeddings_array.shape[1] != self.embedding_dim:
                     logger.warning(f"Warning: OpenAI returned embedding dimension {embeddings_array.shape[1]}, but expected {self.embedding_dim}. Using returned dimension.")
                     # Consider adjusting self.embedding_dim or padding/truncating if strict dimension is required downstream
                logger.debug(f"Successfully received {embeddings_array.shape[0]} embeddings with dimension {embeddings_array.shape[1]}")
                return embeddings_array
            else:
                logger.error("OpenAI Embedding API returned empty data.")
                raise ValueError("OpenAI Embedding API returned empty data.")

        except AuthenticationError as e:
             logger.error(f"OpenAI Embedding API Authentication Error: {e}")
             raise # Let the caller handle authentication issues
        except RateLimitError as e:
            logger.error(f"OpenAI Embedding API rate limit exceeded: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI Embedding API error: {e}")
            raise
        except Timeout as e:
             logger.error(f"OpenAI Embedding API request timed out: {e}")
             raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI Embedding API call: {e}", exc_info=True)
            # Decide whether to raise or return empty array for unexpected errors
            raise ValueError(f"Failed to get embeddings from OpenAI: {e}") # Raise for clarity


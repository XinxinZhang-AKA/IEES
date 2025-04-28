# llm/__init__.py
from .qwen_models import QwenTextGenerationModel, QwenEmbeddingModel
from .openai_models import OpenAITextGenerationModel, OpenAIEmbeddingModel # 添加导入

# 可以定义一个工厂函数来根据配置创建模型实例
def get_text_generation_model(provider: str, api_key: str, model_name: str, **kwargs):
    """
    根据提供者名称获取文本生成模型实例。

    Args:
        provider (str): 'qwen' 或 'openai'.
        api_key (str): 对应平台的 API 密钥。
        model_name (str): 模型名称。
        **kwargs: 其他特定于模型的参数 (例如 openai 的 base_url)。

    Returns:
        实例化的模型对象或 None。
    """
    if provider == 'qwen':
        return QwenTextGenerationModel(api_key=api_key, model_name=model_name)
    elif provider == 'openai':
        return OpenAITextGenerationModel(api_key=api_key, model_name=model_name, base_url=kwargs.get('base_url'))
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def get_embedding_model(provider: str, api_key: str, model_name: str, embedding_dim: int, **kwargs):
    """
    根据提供者名称获取嵌入模型实例。

    Args:
        provider (str): 'qwen' 或 'openai'.
        api_key (str): 对应平台的 API 密钥。
        model_name (str): 模型名称。
        embedding_dim (int): 嵌入维度。
        **kwargs: 其他特定于模型的参数 (例如 openai 的 base_url)。

    Returns:
        实例化的模型对象或 None。
    """
    if provider == 'qwen':
        # QwenEmbeddingModel 内部可能硬编码了维度或模型，需要确认
        # 假设 QwenEmbeddingModel 接受 model_name
        return QwenEmbeddingModel(api_key=api_key, model_name=model_name)
    elif provider == 'openai':
        return OpenAIEmbeddingModel(api_key=api_key, model_name=model_name, base_url=kwargs.get('base_url'), embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unsupported Embedding provider: {provider}")


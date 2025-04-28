import faiss
import numpy as np
from typing import Dict, List, Tuple
import logging # 导入 logging

class Retrieval:
    def __init__(self, text_embedding_model, logger: logging.Logger, embedding_dim: int = 1024): # 添加 embedding_dim 参数
        """
        初始化检索器。

        Args:
            text_embedding_model: 用于生成文本嵌入的模型实例 (需要有 encode 方法)。
            logger (logging.Logger): 日志记录器实例。
            embedding_dim (int): 嵌入向量的维度。默认为 1024。
        """
        self.model = text_embedding_model
        # 确保 logger 是有效的 Logger 对象
        if not isinstance(logger, logging.Logger):
             # 如果传入的不是 Logger 对象，获取一个默认的 logger
             self.logger = logging.getLogger(__name__).getChild("retrieval")
             self.logger.warning("Invalid logger passed to Retrieval, using default.")
        else:
             self.logger = logger.getChild("retrieval")

        # 使用传入的 embedding_dim
        self.embedding_dim = embedding_dim
        self.logger.info(f"Retrieval initialized with embedding dimension: {self.embedding_dim}")


    def embed(self, text: str) -> np.ndarray:
        """
        使用提供的模型生成单个文本的嵌入。
        进行维度检查和标准化。

        Args:
            text (str): 要嵌入的文本。

        Returns:
            np.ndarray: 标准化后的嵌入向量 (形状为 (1, embedding_dim))，如果出错则返回零向量。
        """
        try:
            # 调用模型的 encode 方法
            # 假设 encode 返回 numpy 数组或可以转换为 numpy 数组的列表
            raw_embedding = self.model.encode(text)

            # 确保是 numpy 数组且类型为 float32
            embedding = np.array(raw_embedding, dtype=np.float32) if isinstance(raw_embedding, list) else raw_embedding.astype(np.float32)

            # --- 维度标准化 ---
            original_shape = embedding.shape

            # 1. 确保是二维 (N, D)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1) # 变为 (1, D)
            elif embedding.ndim > 2:
                # 尝试降维，例如，如果形状是 (1, N, D)，则取第一个元素
                if embedding.shape[0] == 1:
                    embedding = embedding.reshape(embedding.shape[1], embedding.shape[2])
                else:
                    # 如果无法简单降维，记录错误并返回零向量
                    self.logger.error(f"Embedding for '{text[:50]}...' has invalid dimensions {original_shape}. Expected 1D or 2D.")
                    return np.zeros((1, self.embedding_dim), dtype=np.float32)

            # 2. 检查并调整第二维 (特征维度)
            current_dim = embedding.shape[1]
            if current_dim != self.embedding_dim:
                self.logger.warning(f"Embedding dimension mismatch for '{text[:50]}...'. Original: {current_dim}, Target: {self.embedding_dim}. Adjusting...")
                if current_dim > self.embedding_dim:
                    # 截断
                    embedding = embedding[:, :self.embedding_dim]
                else:
                    # 填充 (使用 0 填充)
                    pad_width = self.embedding_dim - current_dim
                    embedding = np.pad(embedding, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

            # 确保最终形状是 (N, embedding_dim)，对于单文本输入，N=1
            if embedding.shape[0] > 1 and isinstance(text, str):
                 self.logger.warning(f"Embedding for single text '{text[:50]}...' resulted in multiple vectors ({embedding.shape[0]}). Taking the first one.")
                 embedding = embedding[0:1, :] # 取第一个向量，保持二维

            self.logger.debug(f"Embedded '{text[:50]}...' from shape {original_shape} to {embedding.shape}")
            return embedding

        except Exception as e:
            self.logger.error(f"Embedding failed for text '{text[:50]}...': {str(e)}", exc_info=True)
            # 返回一个符合维度的零向量作为应急
            return np.zeros((1, self.embedding_dim), dtype=np.float32)

    def _process_library_embeddings(self, library: Dict) -> Tuple[np.ndarray, List[str]]:
        """
        处理策略库中的所有嵌入，进行标准化和维度对齐。

        Args:
            library (Dict): 策略库字典。

        Returns:
            Tuple[np.ndarray, List[str]]:
                - 标准化后的嵌入矩阵 (形状为 (M, embedding_dim))。
                - 反向映射列表，长度为 M，每个元素是对应的策略名称。
                如果无有效嵌入，返回空矩阵和空列表。
        """
        all_embeddings = []
        reverse_map = []

        if not library:
            self.logger.warning("Strategy library is empty.")
            return np.empty((0, self.embedding_dim), dtype=np.float32), []

        for s_name, s_info in library.items():
            # 检查 s_info 是否是字典以及是否包含 'Embeddings'
            if not isinstance(s_info, dict) or "Embeddings" not in s_info:
                self.logger.warning(f"Skipping strategy '{s_name}': Invalid format or missing 'Embeddings'.")
                continue

            raw_embeddings = s_info["Embeddings"]
            if not raw_embeddings:
                self.logger.warning(f"Strategy '{s_name}' has no embeddings.")
                continue

            # self.logger.debug(f"Processing embeddings for strategy '{s_name}'. Found {len(raw_embeddings)} raw embeddings.")

            try:
                valid_embs_for_strategy = []
                # 迭代处理每个嵌入向量
                for i, emb_item in enumerate(raw_embeddings):
                    # 确保 emb_item 是列表或 numpy 数组
                    if not isinstance(emb_item, (list, np.ndarray)):
                        self.logger.warning(f"Skipping invalid embedding item {i} for strategy '{s_name}': Not a list or ndarray.")
                        continue

                    emb_array = np.array(emb_item, dtype=np.float32)
                    original_shape = emb_array.shape

                    # --- 维度标准化 ---
                    if emb_array.ndim == 1:
                        emb_array = emb_array.reshape(1, -1)
                    elif emb_array.ndim != 2:
                        self.logger.warning(f"Skipping invalid embedding {i} for strategy '{s_name}': Invalid dimension {original_shape}.")
                        continue

                    # --- 维度对齐 ---
                    current_dim = emb_array.shape[1]
                    if current_dim != self.embedding_dim:
                        # self.logger.debug(f"Adjusting dimension for embedding {i} of '{s_name}': {current_dim} -> {self.embedding_dim}")
                        if current_dim > self.embedding_dim:
                            emb_array = emb_array[:, :self.embedding_dim]
                        else:
                            pad_width = self.embedding_dim - current_dim
                            emb_array = np.pad(emb_array, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

                    # --- 过滤全零嵌入 ---
                    if np.all(np.abs(emb_array) < 1e-6):
                        # self.logger.debug(f"Skipping zero embedding {i} for strategy '{s_name}'.")
                        continue

                    # 添加处理后的有效嵌入
                    valid_embs_for_strategy.append(emb_array)

                # 如果该策略有有效嵌入，则添加到总列表
                if valid_embs_for_strategy:
                    num_valid = len(valid_embs_for_strategy)
                    all_embeddings.extend(valid_embs_for_strategy)
                    reverse_map.extend([s_name] * num_valid)
                    # self.logger.debug(f"Added {num_valid} valid embeddings for strategy '{s_name}'.")
                else:
                    self.logger.warning(f"Strategy '{s_name}' resulted in no valid embeddings after processing.")

            except Exception as e:
                self.logger.error(f"Error processing embeddings for strategy '{s_name}': {str(e)}", exc_info=True)
                continue # 跳过这个策略

        # --- 组合最终矩阵 ---
        if not all_embeddings:
            self.logger.warning("No valid embeddings found in the entire library.")
            return np.empty((0, self.embedding_dim), dtype=np.float32), []

        try:
            # 使用 np.vstack 提高效率和鲁棒性
            embeddings_matrix = np.vstack(all_embeddings)
            self.logger.info(f"Processed library embeddings. Final matrix shape: {embeddings_matrix.shape}")
            # 最终维度校验
            if embeddings_matrix.shape[1] != self.embedding_dim:
                 self.logger.error(f"Critical Error: Final embedding matrix dimension ({embeddings_matrix.shape[1]}) does not match target ({self.embedding_dim}).")
                 # 返回空，防止后续错误
                 return np.empty((0, self.embedding_dim), dtype=np.float32), []
            return embeddings_matrix, reverse_map
        except ValueError as e:
             self.logger.error(f"Error stacking embeddings: {e}. Check if all processed embeddings have the same dimension ({self.embedding_dim}).", exc_info=True)
             return np.empty((0, self.embedding_dim), dtype=np.float32), []


    def pop(self, library: Dict, query: str, k: int = 5) -> Tuple[bool, List[Dict]]:
        """
        从策略库中检索与查询最相关的 k 个策略。

        Args:
            library (Dict): 策略库。
            query (str): 查询文本。
            k (int): 要检索的策略数量。 Defaults to 5.

        Returns:
            Tuple[bool, List[Dict]]:
                - bool: 是否成功检索到至少一个策略。
                - List[Dict]: 检索到的策略列表，每个策略包含 "Strategy", "Definition", "Example"。
        """
        self.logger.info(f"Starting retrieval for query '{query[:50]}...' with k={k}")
        try:
            # 步骤 1: 生成查询嵌入
            query_emb = self.embed(query)
            # 检查 embed 是否返回了有效的嵌入 (非全零)
            if np.all(np.abs(query_emb) < 1e-6):
                self.logger.error("Failed to generate a valid embedding for the query.")
                return False, []
            self.logger.debug(f"Query embedding generated with shape: {query_emb.shape}")


            # 步骤 2: 处理库嵌入
            embeddings_matrix, reverse_map = self._process_library_embeddings(library)

            # 防御：检查嵌入矩阵是否为空
            if embeddings_matrix.size == 0 or not reverse_map:
                self.logger.warning("No embeddings available in the library to search.")
                return False, []
            self.logger.debug(f"Library embeddings processed. Matrix shape: {embeddings_matrix.shape}, Map size: {len(reverse_map)}")


            # 步骤 3: 构建 Faiss 索引
            # 确保嵌入矩阵是 C-contiguous float32
            if not embeddings_matrix.flags['C_CONTIGUOUS']:
                embeddings_matrix = np.ascontiguousarray(embeddings_matrix, dtype=np.float32)
            elif embeddings_matrix.dtype != np.float32:
                 embeddings_matrix = embeddings_matrix.astype(np.float32)

            try:
                index = faiss.IndexFlatL2(self.embedding_dim) # 使用 L2 距离
                index.add(embeddings_matrix)
                self.logger.debug(f"Faiss index built successfully with {index.ntotal} vectors.")
            except Exception as faiss_e:
                self.logger.error(f"Failed to build Faiss index: {faiss_e}", exc_info=True)
                return False, []


            # 步骤 4: 执行检索
            # 确保查询嵌入也是 C-contiguous float32
            if not query_emb.flags['C_CONTIGUOUS']:
                 query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)
            elif query_emb.dtype != np.float32:
                  query_emb = query_emb.astype(np.float32)

            # 确定实际搜索的数量，不超过索引中的向量数
            num_to_search = min(k * 2, index.ntotal) # 搜索稍多一些以处理重复策略
            if num_to_search <= 0:
                 self.logger.warning("Number of vectors to search is zero or less.")
                 return False, []

            self.logger.debug(f"Searching for {num_to_search} nearest neighbors.")
            distances, indices = index.search(query_emb, num_to_search)
            self.logger.debug(f"Search completed. Found indices: {indices[0]}")


            # 步骤 5: 处理结果，去重并格式化
            seen_strategy_names = set()
            results = []
            if indices.size > 0:
                for i, idx in enumerate(indices[0]):
                    # 检查索引有效性
                    if idx < 0 or idx >= len(reverse_map):
                        self.logger.warning(f"Invalid index {idx} found in search results.")
                        continue

                    s_name = reverse_map[idx]

                    # 去重：确保每个策略只返回一次
                    if s_name in seen_strategy_names:
                        continue

                    # 检查策略是否存在于原始库中
                    if s_name not in library or not isinstance(library[s_name], dict):
                        self.logger.warning(f"Strategy '{s_name}' found by index but not valid in library.")
                        continue

                    strategy_info = library[s_name]
                    # 提取所需信息，提供默认值以防万一
                    result_item = {
                        "Strategy": strategy_info.get("Strategy", s_name), # 使用 s_name 作为备用
                        "Definition": strategy_info.get("Definition", "No definition available."),
                        # 取第一个示例，如果存在且不为空
                        "Example": strategy_info.get("Example", ["No example available."])[0] if strategy_info.get("Example") else "No example available."
                    }
                    results.append(result_item)
                    seen_strategy_names.add(s_name)

                    # 达到 k 个唯一策略后停止
                    if len(results) >= k:
                        break
            else:
                 self.logger.warning("Faiss search returned no indices.")


            self.logger.info(f"Retrieval finished. Found {len(results)} unique strategies.")
            return len(results) > 0, results

        except Exception as e:
            self.logger.error(f"Retrieval process failed: {str(e)}", exc_info=True)
            return False, []

    # validate_embedding 方法看起来不再直接使用，embed 方法已包含其逻辑
    # 可以考虑移除或保留作为内部辅助函数
    # def validate_embedding(self, emb: np.ndarray) -> np.ndarray:
    #     """确保嵌入为(1, embedding_dim)形状"""
    #     # ... (实现与 embed 方法中类似的标准化逻辑) ...
    #     pass


import os
import numpy as np
import json

class Library():
    def __init__(self, library, logger):
        self.library = library or {}  # 确保初始化为字典
        self.logger = logger
        # 强化初始化逻辑
        if isinstance(library, dict):
            for k, v in library.items():
                try:
                    self.library[k] = self._sanitize_strategy(v)
                except Exception as e:
                    self.logger.error(f"策略 {k} 加载失败: {str(e)}")
            self.logger.info(f"成功加载初始策略库，条目数: {len(self.library)}")
        else:
            self.logger.warning("无效的初始策略库类型，已初始化为空")

    def _sanitize_strategy(self, strategy):
        """标准化策略格式"""
        # 校验必填字段
        required_keys = ["Strategy", "Definition", "Example", "Score", "Embeddings"]
        if not all(k in strategy for k in required_keys):
            raise ValueError("策略缺少必要字段")

        # 标准化嵌入维度
        emb = np.array(strategy["Embeddings"])
        if emb.ndim == 1:
            emb = emb.reshape(-1, 1024)
        elif emb.ndim == 2 and emb.shape[1] != 1024:
            emb = emb[:, :1024] if emb.shape[1] > 1024 else np.pad(emb, [(0, 0), (0, 1024 - emb.shape[1])])

        return {
            "Strategy": str(strategy["Strategy"]),
            "Definition": str(strategy["Definition"]),
            "Example": list(strategy["Example"]),
            "Score": [float(s) for s in strategy["Score"]],
            "Embeddings": emb.tolist()
        }

    def merge(self, external_lib: dict):
        """合并外部策略库到当前实例"""
        if not isinstance(external_lib, dict):
            self.logger.error("合并失败：输入必须为字典类型")
            return

        for name, new_strat in external_lib.items():
            # 标准化策略格式
            try:
                sanitized = self._sanitize_strategy(new_strat)
            except Exception as e:
                self.logger.error(f"策略 {name} 清洗失败: {str(e)}")
                continue

            if name in self.library:
                # 合并已有策略
                existing = self.library[name]

                # 合并示例（保留最新）
                existing["Example"] = sanitized["Example"] + existing["Example"][:5]  # 保留最新+前5个旧示例

                # 合并分数（保留最高分）
                existing["Score"] = [max(sanitized["Score"][0], max(existing["Score"]))]

                # 合并嵌入向量（垂直堆叠）
                existing_emb = np.array(existing["Embeddings"])
                new_emb = np.array(sanitized["Embeddings"])
                merged_emb = np.concatenate([new_emb, existing_emb], axis=0)
                existing["Embeddings"] = merged_emb.tolist()

                self.logger.info(f"策略 {name} 合并成功，当前示例数: {len(existing['Example'])}")
            else:
                # 添加新策略
                self.library[name] = sanitized

    def add(self, new_strategy, if_notify=False):
        '''
        :param new_strategy: a dictionary containing the new strategy to be added to the library
        '''
        try:
            # 将新策略转换为标准格式
            strategy_name = new_strategy["Strategy"]
            strategy_dict = {strategy_name: new_strategy}

            # 调用合并方法（关键修改）
            self.merge(strategy_dict)  # 现在只需要传入单个字典参数

            # 通知日志
            if if_notify:
                sanitized_strategy = {
                    "Strategy": new_strategy.get("Strategy", ""),
                    "Definition": new_strategy.get("Definition", "")[:200]  # 截断长文本
                }
                self.logger.info(
                    f'New strategy added: {json.dumps(sanitized_strategy, indent=2, ensure_ascii=False)}'
                )

        except KeyError as e:
            self.logger.error(f"Invalid strategy format: missing required field {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to add strategy: {str(e)}")
            raise

    def all(self):
        return self.library

    # 在Library类中添加测试方法
    def test_embeddings_shape(self):
        for s_name, strategy in self.library.items():
            embs = np.array(strategy["Embeddings"])
            assert embs.shape[-1] == 1024, f"策略 {s_name} 维度错误: {embs.shape}"
            assert embs.ndim == 2, f"策略 {s_name} 维度数错误: {embs.ndim}"

    def check_embeddings(self):
        """验证所有策略的嵌入维度"""
        for name, strategy in self.library.items():
            emb = np.array(strategy["Embeddings"])
            assert emb.ndim == 2, f"策略 {name} 维度错误：{emb.ndim}维（应为2维）"
            assert emb.shape[1] == 1024, f"策略 {name} 维度错误：{emb.shape[1]}（应为1024）"

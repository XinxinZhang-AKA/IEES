import pickle
import math
import argparse
import os
import logging
import numpy as np
from collections import Counter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EntropyCalculator")

def calculate_shannon_entropy(probabilities):
    """
    计算香农熵 H(X) = - sum(p(x) * log2(p(x)))
    Args:
        probabilities (list or np.array): 包含每个事件概率的列表或数组。
    Returns:
        float: 计算得到的香non熵值。返回 -1.0 如果输入无效。
    """
    entropy = 0.0
    if not isinstance(probabilities, (list, np.ndarray)) or len(probabilities) == 0:
        logger.error("Invalid input: probabilities must be a non-empty list or numpy array.")
        return -1.0

    # 检查概率总和是否接近 1 (允许小的浮点误差)
    if not np.isclose(sum(probabilities), 1.0):
        logger.warning(f"Probabilities do not sum to 1 (sum={sum(probabilities)}). Normalizing...")
        prob_sum = sum(probabilities)
        if prob_sum <= 0:
             logger.error("Sum of probabilities is zero or negative. Cannot calculate entropy.")
             return -1.0
        probabilities = [p / prob_sum for p in probabilities] # 重新归一化

    for p in probabilities:
        if p < 0:
            logger.error(f"Invalid probability found: {p}. Probabilities cannot be negative.")
            return -1.0
        if p > 0:
            try:
                entropy -= p * math.log2(p)
            except ValueError:
                logger.error(f"Math domain error for log2({p}). This shouldn't happen for p > 0.")
                return -1.0
        # 如果 p == 0, p * log2(p) 约定为 0，所以不需要加任何东西
    return entropy

def analyze_strategy_library(library_path):
    """
    加载策略库并计算其多样性指标（基于分数）。

    Args:
        library_path (str): 策略库 .pkl 文件的路径。

    Returns:
        dict: 包含多样性指标的字典，或在错误时返回 None。
              Keys: 'num_strategies', 'entropy', 'max_entropy', 'normalized_entropy'
    """
    if not os.path.exists(library_path):
        logger.error(f"Strategy library file not found: {library_path}")
        return None

    try:
        with open(library_path, 'rb') as f:
            strategy_library = pickle.load(f)
        logger.info(f"Successfully loaded strategy library from: {library_path}")
    except Exception as e:
        logger.error(f"Failed to load strategy library: {e}", exc_info=True)
        return None

    if not isinstance(strategy_library, dict) or not strategy_library:
        logger.warning("Loaded library is not a valid dictionary or is empty.")
        return {'num_strategies': 0, 'entropy': 0.0, 'max_entropy': 0.0, 'normalized_entropy': 0.0}

    # --- 基于分数的熵计算 ---
    strategy_scores = []
    strategy_names = list(strategy_library.keys()) # 获取所有策略名称
    num_strategies = len(strategy_names)

    if num_strategies == 0:
         logger.info("Library contains 0 strategies.")
         return {'num_strategies': 0, 'entropy': 0.0, 'max_entropy': 0.0, 'normalized_entropy': 0.0}

    logger.info(f"Found {num_strategies} unique strategies in the library.")

    total_score = 0.0
    valid_scores = []

    for name, info in strategy_library.items():
        if isinstance(info, dict) and 'Score' in info and info['Score']:
            # 假设 Score 是一个列表，我们取第一个元素？或者求和/平均？
            # 当前代码逻辑似乎是用分数提升量，通常只有一个元素。我们取第一个。
            try:
                # 确保分数是正数，因为概率不能为负。如果分数代表提升量，应该>0。
                # 如果分数可能为0或负，需要调整处理方式（例如，加一个小的平滑值或使用绝对值）
                score_value = float(info['Score'][0])
                if score_value > 0:
                    valid_scores.append(score_value)
                    total_score += score_value
                elif score_value == 0:
                     logger.debug(f"Strategy '{name}' has a score of 0. Excluding from probability calculation.")
                else: # score_value < 0
                     logger.warning(f"Strategy '{name}' has a negative score ({score_value}). Excluding from probability calculation. Consider using absolute values if appropriate.")

            except (ValueError, IndexError, TypeError) as e:
                logger.warning(f"Could not parse score for strategy '{name}': {e}. Skipping.")
        else:
            logger.warning(f"Strategy '{name}' has missing or invalid 'Score' field. Skipping.")

    if not valid_scores or total_score <= 0:
        logger.warning("No valid positive scores found to calculate score-based entropy. Entropy will be 0.")
        probabilities_from_scores = []
        score_based_entropy = 0.0
    else:
        # 计算概率 p(x) = Score(x) / TotalScore
        probabilities_from_scores = [s / total_score for s in valid_scores]
        logger.debug(f"Probabilities based on scores: {probabilities_from_scores}")
        # 计算香农熵
        score_based_entropy = calculate_shannon_entropy(probabilities_from_scores)
        if score_based_entropy < 0: score_based_entropy = 0.0 # Handle calculation error case

    # --- 计算最大熵和归一化熵 ---
    # 最大熵 H_max = log2(N)，其中 N 是有效策略的数量
    # 注意：是参与概率计算的策略数量 (len(valid_scores)) 还是总策略数量 (num_strategies)?
    # 使用参与计算的数量更合理，表示当前分数分布下的最大可能熵
    num_valid_score_strategies = len(valid_scores)
    if num_valid_score_strategies > 0:
        max_entropy = math.log2(num_valid_score_strategies)
        # 归一化熵 (Pielou's evenness index) J = H / H_max
        # 防止 H_max 为 0 (当只有一个有效策略时)
        normalized_entropy = score_based_entropy / max_entropy if max_entropy > 0 else 1.0 if score_based_entropy == 0 else 0.0 # Handle edge cases
    else:
        max_entropy = 0.0
        normalized_entropy = 0.0 # 或者设为 None 或 NaN

    results = {
        'num_strategies': num_strategies, # 库中总策略数
        'num_strategies_with_valid_score': num_valid_score_strategies, # 用于计算熵的策略数
        'entropy': score_based_entropy, # 基于分数的熵
        'max_entropy': max_entropy, # 基于有效策略数量的最大熵
        'normalized_entropy': normalized_entropy # 归一化熵 (均匀度)
    }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the Shannon entropy of a strategy library based on strategy scores.")
    parser.add_argument("library_path", type=str, help="Path to the strategy library .pkl file (e.g., ./logs/lifelong_strategy_library.pkl)")
    args = parser.parse_args()

    entropy_results = analyze_strategy_library(args.library_path)

    if entropy_results:
        logger.info("--- Strategy Library Diversity Analysis ---")
        logger.info(f"Total unique strategies found: {entropy_results['num_strategies']}")
        logger.info(f"Strategies with valid positive scores: {entropy_results['num_strategies_with_valid_score']}")
        logger.info(f"Shannon Entropy (H) based on scores: {entropy_results['entropy']:.4f} bits")
        logger.info(f"Maximum Possible Entropy (H_max = log2(N_valid)): {entropy_results['max_entropy']:.4f} bits")
        logger.info(f"Normalized Entropy (Evenness J = H/H_max): {entropy_results['normalized_entropy']:.4f}")
        logger.info("-------------------------------------------")

        # 你可以在这里添加逻辑，例如：
        # if entropy_results['normalized_entropy'] < 0.5:
        #     logger.warning("Strategy diversity is low (normalized entropy < 0.5). Consider exploring new strategies.")

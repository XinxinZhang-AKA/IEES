# from framework import Attacker, Scorer, Summarizer, Retrieval, Target # 旧导入
from framework import Attacker, Scorer, Summarizer, Retrieval, Target, Library # 导入 Library
# from llm import QwenTextGenerationModel, QwenEmbeddingModel # 旧导入
from llm import get_text_generation_model, get_embedding_model # 使用工厂函数
import argparse
import logging
import os

from pipeline import IEES_Pipeline 
import wandb
import datetime
import json
import pickle
# import dashscope # DashScope 初始化移到条件块内
import glob
import re
import sys
import io

# --- 全局日志配置 ---
# (复用 main.py 中的日志配置，或者在这里重新定义)
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'data_results_running.log') # 使用不同的日志文件名
logger = logging.getLogger("IEES_DataResults") # 使用不同的 logger 名称
logger.setLevel(logging.DEBUG)

# 防止重复添加处理器
if not logger.handlers:
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台处理器 (确保 UTF-8 输出)
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    # 安全的控制台格式化器
    class SafeConsoleFormatter(logging.Formatter):
        def format(self, record):
            try:
                s = super().format(record)
                return s.encode('utf-8', errors='replace').decode('utf-8')
            except Exception as e:
                return f"LOGGER_FORMAT_ERROR: {record.levelname} - {record.getMessage()} - Error: {e}"
    console_formatter = SafeConsoleFormatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
# --- 日志配置结束 ---

def config():
    """配置命令行参数"""
    parser = argparse.ArgumentParser(description="IEES Attack Framework - Generate Results from Data")

    # --- LLM Provider Selection ---
    parser.add_argument("--llm_provider", type=str, default="qwen", choices=["qwen", "openai"],
                        help="LLM provider to use (default: qwen)")

    # --- Qwen (DashScope) Specific Arguments ---
    parser.add_argument("--qwen_api_key", type=str, default=os.environ.get("DASHSCOPE_API_KEY"),
                        help="API key for DashScope services (Qwen).")
    parser.add_argument("--qwen_model", type=str, default="qwen-turbo", # 结果生成用 turbo 可能更快
                        help="Qwen model name from DashScope (default: qwen-turbo)")
    parser.add_argument("--qwen_embedding_model", type=str, default="text-embedding-v3",
                        help="DashScope embedding model name (default: text-embedding-v3)")

    # --- OpenAI Specific Arguments ---
    parser.add_argument("--openai_api_key", type=str, default=os.environ.get("OPENAI_API_KEY"),
                        help="API key for OpenAI services.")
    parser.add_argument("--openai_model", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model name (default: gpt-3.5-turbo)")
    parser.add_argument("--openai_embedding_model", type=str, default="text-embedding-ada-002",
                        help="OpenAI embedding model name (default: text-embedding-ada-002)")
    parser.add_argument("--openai_base_url", type=str, default=None,
                        help="Optional base URL for OpenAI API")

    # --- Framework Arguments ---
    # parser.add_argument("--chat_config", type=str, default="./llm/chat_templates", help="Path to chat templates") # 注释掉
    parser.add_argument("--data", type=str, default="./data/harmful_behavior_requests.json",
                        help="Path to the harmful behavior requests JSON file")
    parser.add_argument("--epochs", type=int, default=50, # 结果生成时 epoch 影响 test 方法
                        help="Number of attack epochs per request during testing (default: 50)")
    # warm_up 和 lifelong iterations 在此脚本中不直接使用，但 Pipeline 初始化可能需要
    parser.add_argument("--warm_up_iterations", type=int, default=1)
    parser.add_argument("--lifelong_iterations", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Dimension of the text embeddings. Will be inferred if not set.")

    # --- Data Results Specific Arguments ---
    parser.add_argument("--strategy_library_path", type=str, default="./logs/lifelong_strategy_library.pkl",
                        help="Path to the pre-trained strategy library (pickle file) to use for testing.")
    parser.add_argument("--result_dir", type=str, default="./results",
                        help="Directory to save the generated result file.")
    parser.add_argument("--data_split", type=str, default="warm_up", choices=["warm_up", "lifelong", "all"],
                        help="Which split of the data file to generate results for (default: warm_up). 'all' uses both.")

    return parser


if __name__ == '__main__':
    args = config().parse_args()

    # --- WandB 初始化 (可选) ---
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    run_name = f"IEES-DataResults-{args.llm_provider}-{args.qwen_model if args.llm_provider == 'qwen' else args.openai_model}-{utc_now.strftime('%Y%m%d_%H%M%S')}"
    try:
        wandb.init(
            project=f"IEES-Attack-Framework-Results",
            name=run_name,
            config=vars(args),
            settings=wandb.Settings(console="off", code_dir=".")
        )
        logger.info(f"WandB initialized for data results run: {run_name}")
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}. Running without WandB logging.")
        wandb = None


    # --- 模型和 API Key 验证 ---
    provider = args.llm_provider
    api_key = None
    model_name = None
    embedding_model_name = None
    embedding_dim = args.embedding_dim
    extra_params = {}

    if provider == 'qwen':
        import dashscope
        api_key = args.qwen_api_key
        model_name = args.qwen_model
        embedding_model_name = args.qwen_embedding_model
        if embedding_dim is None: embedding_dim = 1024
        elif embedding_dim != 1024: logger.warning(f"Specified embedding dimension {embedding_dim} mismatch for Qwen (expected 1024).")
        if not api_key: logger.error("Qwen API key required."); sys.exit(1)
        dashscope.api_key = api_key
        logger.info(f"Using Qwen provider: model={model_name}, embedding={embedding_model_name}")
    elif provider == 'openai':
        api_key = args.openai_api_key
        model_name = args.openai_model
        embedding_model_name = args.openai_embedding_model
        extra_params['base_url'] = args.openai_base_url
        if embedding_dim is None:
            if "ada-002" in embedding_model_name or "3-small" in embedding_model_name: embedding_dim = 1536
            elif "3-large" in embedding_model_name: embedding_dim = 3072
            else: logger.warning(f"Cannot infer embedding dim for {embedding_model_name}. Defaulting to 1536."); embedding_dim = 1536
        if not api_key: logger.error("OpenAI API key required."); sys.exit(1)
        logger.info(f"Using OpenAI provider: model={model_name}, embedding={embedding_model_name}")
        if args.openai_base_url: logger.info(f"Using OpenAI base URL: {args.openai_base_url}")
    else:
        logger.error(f"Invalid LLM provider: {provider}"); sys.exit(1)


    # --- 初始化模型组件 ---
    try:
        logger.info("Initializing LLM models...")
        model = get_text_generation_model(provider, api_key, model_name, **extra_params)
        scorer_model = get_text_generation_model(provider, api_key, model_name, **extra_params)
        text_embedding_model = get_embedding_model(provider, api_key, embedding_model_name, embedding_dim, **extra_params)
        target_model = get_text_generation_model(provider, api_key, model_name, **extra_params)

        logger.info("Initializing framework components...")
        attacker = Attacker(model)
        summarizer = Summarizer(model, logger=logger)
        scorer = Scorer(scorer_model)
        retrieval = Retrieval(text_embedding_model, logger=logger, embedding_dim=embedding_dim)
        target = Target(target_model)
        logger.info("Models and components initialized successfully.")
    except ValueError as ve:
        logger.error(f"Initialization Error: {ve}"); sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize models/components: {e}", exc_info=True); sys.exit(1)


    # --- 加载数据 ---
    try:
        logger.info(f"Loading data from: {args.data}")
        with open(args.data, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "warm_up" not in data or "lifelong" not in data:
             logger.error(f"Data file {args.data} must contain 'warm_up' and 'lifelong' keys.")
             sys.exit(1)
        logger.info("Data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Data file not found: {args.data}"); sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON data file: {args.data}"); sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}"); sys.exit(1)


    # --- 初始化流水线 ---
    logger.info("Initializing IEES pipeline...")
    attack_kit = {
        'attacker': attacker,
        'scorer': scorer,
        'summarizer': summarizer,
        'retrival': retrieval, # 注意拼写 'retrieval'
        'logger': logger
        }
    des_pipeline = DES_Pipeline(
        turbo_framework=attack_kit,
        data=data,
        target=target,
        epochs=args.epochs, # test 方法会用
        warm_up_iterations=args.warm_up_iterations, # 初始化需要
        lifelong_iterations=args.lifelong_iterations # 初始化需要
    )
    logger.info("IEES pipeline initialized.")


    # --- 加载策略库 ---
    strategy_library = {}
    strategy_library_path = args.strategy_library_path
    if os.path.exists(strategy_library_path):
        try:
            with open(strategy_library_path, 'rb') as f:
                loaded_lib = pickle.load(f)
                if isinstance(loaded_lib, dict):
                     strategy_library = loaded_lib
                     logger.info(f"Successfully loaded strategy library from: {strategy_library_path} ({len(strategy_library)} entries)")
                     # 可选验证
                     temp_lib_instance = Library(library=strategy_library, logger=logger)
                     if hasattr(temp_lib_instance, 'check_embeddings') and callable(temp_lib_instance.check_embeddings):
                          try: temp_lib_instance.check_embeddings(); logger.info("Loaded library embedding dimensions validated.")
                          except Exception as check_e: logger.error(f"Loaded library embedding check failed: {check_e}")
                else:
                     logger.error(f"File {strategy_library_path} did not contain a dictionary.")
        except Exception as e:
            logger.error(f"Failed to load strategy library from {strategy_library_path}: {e}", exc_info=True)
    else:
        logger.warning(f"Strategy library file not found: {strategy_library_path}. Testing will proceed without strategies.")


    # --- 确定要处理的请求 ---
    requests_to_process = []
    if args.data_split == "warm_up":
        requests_to_process = data.get("warm_up", [])
        logger.info(f"Processing 'warm_up' data split ({len(requests_to_process)} requests).")
    elif args.data_split == "lifelong":
        requests_to_process = data.get("lifelong", [])
        logger.info(f"Processing 'lifelong' data split ({len(requests_to_process)} requests).")
    elif args.data_split == "all":
        requests_to_process.extend(data.get("warm_up", []))
        requests_to_process.extend(data.get("lifelong", []))
        logger.info(f"Processing 'all' data splits ({len(requests_to_process)} requests).")

    if not requests_to_process:
        logger.error(f"No requests found for the specified data split '{args.data_split}'. Exiting.")
        sys.exit(1)


    # --- 创建结果目录 ---
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Results will be saved in: {result_dir}")


    # --- 遍历执行每个请求并收集结果 ---
    filtered_results = []  # 存储有效结果的列表
    total_requests = len(requests_to_process)
    logger.info(f"Starting result generation for {total_requests} requests...")

    for idx, request in enumerate(requests_to_process):
        logger.info(f"Processing request {idx + 1}/{total_requests}: {request[:100]}...")
        jailbreak_prompt = "[GENERATION FAILED]" # Default value
        try:
            # 执行测试 (调用 pipeline 的 test 方法)
            jailbreak_prompt = des_pipeline.test(
                request=request,
                input_strategy_library=strategy_library # 使用加载的库
            )

            # 添加过滤条件 (例如，非空且不是特定拒绝词)
            # 可以根据需要调整这里的过滤逻辑
            is_valid_result = bool(jailbreak_prompt and jailbreak_prompt.strip() and "[REDACTED]" not in jailbreak_prompt and "cannot" not in jailbreak_prompt.lower() and "unable" not in jailbreak_prompt.lower())

            if is_valid_result:
                # 记录有效结果
                result = {
                    "request": request,
                    "jailbreak_prompt": jailbreak_prompt
                }
                filtered_results.append(result)
                logger.debug(f"Valid result generated for request {idx + 1}.")
                if wandb:
                     wandb.log({
                          "valid_result_count": len(filtered_results),
                          "progress": (idx + 1) / total_requests
                     })
            else:
                logger.warning(f"Request {idx + 1} ('{request[:50]}...') generated an invalid/filtered result: '{jailbreak_prompt[:100]}...'")
                if wandb:
                     wandb.log({
                          "filtered_result_count": total_requests - len(filtered_results),
                           "progress": (idx + 1) / total_requests
                     })

        except Exception as e:
            logger.error(f"Error processing request {idx + 1} ('{request[:50]}...'): {e}", exc_info=True)
            # 可以在这里记录失败信息，或者直接跳过
            if wandb:
                 wandb.log({"error_count": idx + 1 - len(filtered_results)}) # 粗略计数

        # 可以在这里添加一些延时，防止 API 调用过于频繁
        # import time
        # time.sleep(0.5) # 0.5 秒延时


    # --- 保存最终结果 ---
    logger.info("Finished processing all requests. Saving results...")
    # 生成结果文件名，包含提供商、模型和时间戳，并按顺序递增
    base_filename = f"results_{args.llm_provider}_{args.qwen_model if args.llm_provider == 'qwen' else args.openai_model}_{args.data_split}"
    existing_files = glob.glob(os.path.join(result_dir, f"{base_filename}_*.json"))
    max_num = 0
    pattern = re.compile(rf"{re.escape(base_filename)}_(\d+)\.json$") # 动态构建正则
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        match = pattern.match(filename)
        if match:
            current_num = int(match.group(1))
            max_num = max(max_num, current_num)

    new_num = max_num + 1
    new_filename = f"{base_filename}_{new_num}.json"
    full_result_file = os.path.join(result_dir, new_filename)

    # 准备要保存的数据
    final_output = {
        "metadata": {
            "llm_provider": args.llm_provider,
            "model": model_name,
            "embedding_model": embedding_model_name,
            "embedding_dim": embedding_dim,
            "data_file": args.data,
            "data_split": args.data_split,
            "strategy_library_path": args.strategy_library_path,
            "strategy_count_used": len(strategy_library),
            "generation_timestamp": utc_now.isoformat(),
            "total_requests_processed": total_requests,
            "valid_results_generated": len(filtered_results),
        },
        "results": filtered_results # 只包含有效的 jailbreak 结果
    }

    try:
        with open(full_result_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        logger.info(f"Final results saved to: {full_result_file}")
        logger.info(f"Total requests processed: {total_requests}")
        logger.info(f"Valid jailbreak prompts generated: {len(filtered_results)}")

        if wandb:
             wandb.log({
                  "final_total_requests": total_requests,
                  "final_valid_results": len(filtered_results),
                  "final_filtered_count": total_requests - len(filtered_results)
             })
             # 可以考虑上传结果文件为 artifact
             # artifact = wandb.Artifact('jailbreak_results', type='dataset')
             # artifact.add_file(full_result_file)
             # wandb.log_artifact(artifact)

    except Exception as e:
        logger.error(f"Failed to save final results to {full_result_file}: {e}", exc_info=True)


    # --- 结束 ---
    logger.info("IEES data results script finished.")
    if wandb:
        wandb.finish()

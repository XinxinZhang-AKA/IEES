# run_gemini_test.py
import argparse
import logging
import os
import json
import time

# --- Import necessary framework components ---
# Import BaseTargetLLM, Target, Attacker, Scorer etc. as needed for your test
from framework.target import BaseTargetLLM, Target
from framework.scorer import Scorer
from framework.log import Log # Assuming Logger is used
# Import other components you want to test with Gemini

# --- Import ONLY the Gemini LLM implementation ---
from llm.gemini_models import GeminiTargetLLM

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gemini_test_running.log")
    ]
)
logger = logging.getLogger(__name__)

# --- Simplified Factory Function (Only needs Gemini) ---
# Or you could copy/import the full setup_target_llm function if preferred
def setup_gemini_llm(model_name: str, api_key: str = None, **kwargs) -> BaseTargetLLM:
    """Creates and returns an instance of GeminiTargetLLM."""
    logger.info(f"Attempting to set up Gemini LLM: name='{model_name}'")
    try:
        llm_client = GeminiTargetLLM(model_name=model_name, api_key=api_key, **kwargs)
        logger.info(f"Successfully initialized LLM client: {llm_client}")
        return llm_client
    except Exception as e:
        logger.error(f"Failed to initialize GeminiTargetLLM: {e}", exc_info=True)
        raise ValueError(f"Initialization failed for Gemini model {model_name}") from e

# --- Main Test Logic ---
def main():
    parser = argparse.ArgumentParser(description="Run Specific Tests with Gemini LLM")

    # --- Arguments specific to Gemini test ---
    parser.add_argument("--model-name", type=str, required=True, default="gemini-pro",
                        help="Specific Gemini model name/identifier (e.g., 'gemini-pro', 'gemini-1.5-flash')")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Google API Key. If not provided, attempts to use GOOGLE_API_KEY environment variable.")
    parser.add_argument("--prompts-file", type=str, default="data/harmful_behavior_requests.json",
                        help="Path to the JSON file containing prompts for testing.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level.")
    parser.add_argument("--output-log-path", type=str, default="./logs/gemini_test",
                        help="Directory to save Gemini test specific logs.")
    # Add any other arguments needed for the specific test scenario

    args = parser.parse_args()

    # --- Set Log Level ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level)
    logger.info(f"Logging level set to: {args.log_level.upper()}")
    logger.info(f"Testing with Gemini model: {args.model_name}")

    # --- Create Log Directory ---
    if not os.path.exists(args.output_log_path):
        os.makedirs(args.output_log_path)
        logger.info(f"Created Gemini test log directory: {args.output_log_path}")

    # --- Initialize Logger (Framework's Logger) ---
    framework_logger = Log(args.output_log_path) # Use the specific path
    logger.info(f"Framework Logger initialized at path: {args.output_log_path}")


    # --- Load Prompts ---
    try:
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts_to_test = json.load(f)
        if not isinstance(prompts_to_test, list):
             raise TypeError("Prompts file should contain a JSON list.")
        # Extract text if needed, assuming list of strings for now
        prompts_text = prompts_to_test
        logger.info(f"Loaded {len(prompts_text)} prompts from {args.prompts_file}")
    except FileNotFoundError:
        logger.error(f"Prompts file not found: {args.prompts_file}")
        return
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error reading or parsing prompts file {args.prompts_file}: {e}")
        return

    # --- Setup Gemini LLM ---
    try:
        gemini_client = setup_gemini_llm(
            model_name=args.model_name,
            api_key=args.api_key
        )
        # Wrap in Target for consistent interface if needed by other components
        target_system = Target(llm_client=gemini_client)
    except ValueError as e:
        logger.error(f"Failed to setup Gemini LLM: {e}")
        return

    # --- Initialize necessary components for the test ---
    # Example: Initialize Scorer if you want to test scoring Gemini responses
    # scorer = Scorer(target_system, framework_logger)
    # logger.info("Scorer initialized for Gemini test.")

    # --- Run Test Scenario ---
    logger.info("Starting Gemini test scenario...")
    results = []

    # Example: Simple query loop to test basic interaction and blocking
    print("\n--- Running Gemini Test Queries ---")
    for i, prompt in enumerate(prompts_text[:10]): # Test first 10 prompts
        print(f"\nTesting prompt {i+1}/{len(prompts_text)}:")
        print(f"Prompt: {prompt}")
        try:
            response = target_system.query(prompt) # Use the Target wrapper
            print(f"Response: {response}")
            results.append({"prompt": prompt, "response": response})
            # You could add scoring here:
            # score = scorer.score(response)
            # print(f"Score: {score}")
            # results[-1]["score"] = score
        except Exception as e:
            error_msg = f"[ERROR during query in test loop: {e}]"
            print(f"Response: {error_msg}")
            results.append({"prompt": prompt, "response": error_msg})
            logger.error(f"Error during query in test loop for prompt {i+1}", exc_info=True)
        time.sleep(1) # Optional delay

    # --- Save Test Results ---
    results_path = os.path.join(args.output_log_path, "gemini_test_results.json")
    try:
        with open(results_path, 'w', encoding='utf-8') as f_res:
            json.dump(results, f_res, indent=4, ensure_ascii=False)
        logger.info(f"Saved Gemini test results to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save Gemini test results: {e}")

    logger.info("Gemini test finished.")


if __name__ == "__main__":
    logger.info("Starting Gemini test script. Ensure GOOGLE_API_KEY is set if not using --api-key.")
    main()

# ```
# **如何使用 `run_gemini_test.py`:**
#
# 1.  保存代码到项目根目录（或你喜欢的位置）。
# 2.  确保 `GOOGLE_API_KEY` 环境变量已设置。
# 3.  运行脚本，指定 Gemini 模型名称：
#     ```bash
#     python run_gemini_test.py --model-name gemini-1.5-flash # Or gemini-pro
#     # Optional: Specify API key directly (less recommended)
#     # python run_gemini_test.py --model-name gemini-pro --api-key "AI..."
#     # Optional: Specify prompts file or log path
#     # python run_gemini_test.py --model-name gemini-pro --prompts-file data/specific_test_prompts.json --output-log-path ./logs/gemini_specific_run
#     ```
#
# 这样你就可以在不影响主 `main.py` 的情况下，独立运行和调试针对 Gemini 的测
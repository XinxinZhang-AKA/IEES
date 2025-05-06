# IEES: A Framework for Automated Discovery and Learning of LLM Jailbreak Strategies

## Overview

This repository contains the source code implementation for the IEES (Information Entropy Evolution Strategies) framework. This framework is designed to facilitate the automated discovery, learning, and evaluation of strategies employed in jailbreaking attacks against Large Language Models (LLMs).

## Getting Started

### Prerequisites and Installation

Prior to execution, the repository must be cloned, and necessary dependencies installed. Execute the following commands in a terminal environment:

```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
```

*Please replace `<repository_url>` and `<repository_directory>` with the actual URL and directory name.*

### API Key Configuration

The framework requires API credentials for the selected LLM platforms. Support is currently provided for Alibaba Cloud Qwen (via DashScope) and OpenAI. Configuration can be achieved through one of the following methods:

1.  **Environment Variables (Recommended):** Define the following environment variables:
    ```bash
    export DASHSCOPE_API_KEY="sk-your_qwen_api_key"
    export OPENAI_API_KEY="sk-your_openai_api_key"
    ```

2.  **Command-Line Arguments:** Provide the API keys directly via the `--qwen_api_key` or `--openai_api_key` arguments when executing the relevant scripts.

## Framework Execution

### Primary Workflow (Training and Learning)

The `main.py` script orchestrates the complete Warm-up and Lifelong Red Teaming phases, enabling the system to learn and optimize a library of jailbreak strategies.

**Execution with Qwen (Default Provider):**

```bash
python main.py --qwen_api_key <your_qwen_api_key> \
               --qwen_model <qwen_model_name> \
               # Optional parameters: --epochs <num>, --lifelong_iterations <num>, etc.
```

*Specify the desired Qwen model (e.g., `qwen-max`).*

**Execution with OpenAI:**

```bash
python main.py --llm_provider openai \
               --openai_api_key <your_openai_api_key> \
               --openai_model <openai_model_name> \
               --openai_embedding_model <openai_embedding_model> \
               --embedding_dim <dimension> \
               # Optional parameters: --openai_base_url <your_proxy_url>, --epochs <num>, etc.
```

*Specify the relevant OpenAI models (e.g., `gpt-4`, `text-embedding-3-small`) and the corresponding embedding dimension (e.g., `1536`).*

Generated strategy libraries and operational logs are stored within the `./logs/` directory.

### Testing Phase (Single Request Evaluation)

The `test.py` script facilitates the evaluation of a pre-trained strategy library against a specific user-provided harmful request. It loads the specified library and generates a potential jailbreak prompt for the input request.

**Execution with Qwen:**

```bash
python test.py --qwen_api_key <your_qwen_api_key> \
               --request "Your specific harmful request for evaluation" \
               --strategy_library_path ./logs/lifelong_strategy_library.pkl
```

**Execution with OpenAI:**

```bash
python test.py --llm_provider openai \
               --openai_api_key <your_openai_api_key> \
               --request "Your specific harmful request for evaluation" \
               --strategy_library_path ./logs/lifelong_strategy_library.pkl \
               # Optional parameters: --openai_model <model>, --embedding_dim <dim>, etc.
```

The resulting jailbreak prompt is outputted to the console.

### Dataset Result Generation

The `data_results.py` script utilizes a trained strategy library to generate jailbreak prompts for a specified subset (`warm_up`, `lifelong`, or `all`) of requests within the designated data file (`./data/harmful_behavior_requests.json`). The outcomes are compiled and saved in JSON format.

**Execution with Qwen (testing warm_up split):**

```bash
python data_results.py --qwen_api_key <your_qwen_api_key> \
                       --data_split warm_up \
                       --strategy_library_path ./logs/lifelong_strategy_library.pkl
```

**Execution with OpenAI (testing lifelong split):**

```bash
python data_results.py --llm_provider openai \
                       --openai_api_key <your_openai_api_key> \
                       --data_split lifelong \
                       --strategy_library_path ./logs/lifelong_strategy_library.pkl \
                       # Optional parameters: --openai_model <model>, --embedding_dim <dim>, etc.
```

Generated result files are stored in the `./results/` directory.



## Usage Notes

* Ensure the validity of API keys and confirm sufficient usage quotas for the selected LLM platform.
* Execution parameters, such as `epochs`, `lifelong_iterations`, and `embedding_dim`, may require adjustment based on experimental requirements and model specifications.
* The `--openai_base_url` argument should be utilized when interfacing with custom or proxy OpenAI API endpoints.


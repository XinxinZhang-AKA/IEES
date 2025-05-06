IEES: A Framework for Automated Discovery and Learning of LLM Jailbreak StrategiesOverviewThis repository contains the source code implementation for the IEES (Placeholder: Insert Full Name for IEES if available, e.g., Information Entropy Evolution Strategies) framework. This framework is designed to facilitate the automated discovery, learning, and evaluation of strategies employed in jailbreaking attacks against Large Language Models (LLMs).Getting StartedPrerequisites and InstallationPrior to execution, the repository must be cloned, and necessary dependencies installed. Execute the following commands in a terminal environment:git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
Replace <repository_url> and <repository_directory> with the appropriate values.API Key ConfigurationThe framework requires API credentials for the selected LLM platforms. Support is currently provided for Alibaba Cloud Qwen (via DashScope) and OpenAI. Configuration can be achieved through one of the following methods:Environment Variables (Recommended): Define the following environment variables:export DASHSCOPE_API_KEY="sk-your_qwen_api_key"
export OPENAI_API_KEY="sk-your_openai_api_key"
Command-Line Arguments: Provide the API keys directly via the --qwen_api_key or --openai_api_key arguments when executing the relevant scripts.Framework ExecutionPrimary Workflow (Training and Learning)The main.py script orchestrates the complete Warm-up and Lifelong Red Teaming phases, enabling the system to learn and optimize a library of jailbreak strategies.Execution with Qwen (Default Provider):python main.py --qwen_api_key <your_qwen_api_key> \
               --qwen_model <qwen_model_name> \
               # Optional parameters: --epochs <num>, --lifelong_iterations <num>, etc.
Specify the desired Qwen model (e.g., qwen-max).Execution with OpenAI:python main.py --llm_provider openai \
               --openai_api_key <your_openai_api_key> \
               --openai_model <openai_model_name> \
               --openai_embedding_model <openai_embedding_model> \
               --embedding_dim <dimension> \
               # Optional parameters: --openai_base_url <your_proxy_url>, --epochs <num>, etc.
Specify the relevant OpenAI models (e.g., gpt-4, text-embedding-3-small) and the corresponding embedding dimension (e.g., 1536).Generated strategy libraries and operational logs are stored within the ./logs/ directory.Testing Phase (Single Request Evaluation)The test.py script facilitates the evaluation of a pre-trained strategy library against a specific user-provided harmful request. It loads the specified library and generates a potential jailbreak prompt for the input request.Execution with Qwen:python test.py --qwen_api_key <your_qwen_api_key> \
               --request "Your specific harmful request for evaluation" \
               --strategy_library_path ./logs/lifelong_strategy_library.pkl
Execution with OpenAI:python test.py --llm_provider openai \
               --openai_api_key <your_openai_api_key> \
               --request "Your specific harmful request for evaluation" \
               --strategy_library_path ./logs/lifelong_strategy_library.pkl \
               # Optional parameters: --openai_model <model>, --embedding_dim <dim>, etc.
The resulting jailbreak prompt is outputted to the console.Dataset Result GenerationThe data_results.py script utilizes a trained strategy library to generate jailbreak prompts for a specified subset (warm_up, lifelong, or all) of requests within the designated data file (./data/harmful_behavior_requests.json). The outcomes are compiled and saved in JSON format.Execution with Qwen (testing warm_up split):python data_results.py --qwen_api_key <your_qwen_api_key> \
                       --data_split warm_up \
                       --strategy_library_path ./logs/lifelong_strategy_library.pkl
Execution with OpenAI (testing lifelong split):python data_results.py --llm_provider openai \
                       --openai_api_key <your_openai_api_key> \
                       --data_split lifelong \
                       --strategy_library_path ./logs/lifelong_strategy_library.pkl \
                       # Optional parameters: --openai_model <model>, --embedding_dim <dim>, etc.
Generated result files are stored in the ./results/ directory.Repository StructureThe repository is organized as follows:.
├── framework/          # Core framework components (Attacker, Scorer, Summarizer, etc.)
├── llm/                # LLM API wrappers (Qwen, OpenAI)
├── data/               # Dataset files (e.g., harmful_behavior_requests.json)
├── logs/               # Directory for runtime logs and strategy libraries
├── results/            # Directory for generated results from data_results.py
├── main.py             # Main execution script (Training & Learning)
├── test.py             # Single request testing script
├── data_results.py     # Dataset result generation script
├── pipeline.py         # Defines Warm-up and Lifelong processes
├── calculate_entropy.py # (Optional) Script for strategy library diversity analysis
├── requirements.txt    # Python dependencies
└── README.md           # This document
Usage NotesEnsure the validity of API keys and confirm sufficient usage quotas for the selected LLM platform.Execution parameters, such as epochs, lifelong_iterations, and embedding_dim, may require adjustment based on experimental requirements and model specifications.The --openai_base_url argument should be utilized when interfacing with custom or proxy OpenAI API endpoints.

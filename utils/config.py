# Configuration module for legal RAG orchestration survey

import os
from typing import Dict, Any, Optional

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data and index directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INDEXES_DIR = os.path.join(PROJECT_ROOT, "indexes")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Dataset paths
DATASET_PATHS = {
    "CUAD": os.path.join(DATA_DIR, "CUAD", "cuad_eval.json"),
    "ECHR": os.path.join(DATA_DIR, "ECHR", "echr_eval.json"),
    "LEDGAR": os.path.join(DATA_DIR, "LEDGAR", "ledgar_eval.json")
}

# Index paths
INDEX_PATHS = {
    "CUAD": os.path.join(INDEXES_DIR, "cuad"),
    "ECHR": os.path.join(INDEXES_DIR, "echr"),
    "LEDGAR": os.path.join(INDEXES_DIR, "ledgar")
}

# Framework configurations
FRAMEWORK_CONFIGS = {
    "baseline": {
        "model_name": "llama-3.3-70b-versatile",
        "k": 3,
        "temperature": 0.0
    },
    "langchain": {
        "model_name": "llama-3.3-70b-versatile",
        "k": 3,
        "temperature": 0.0
    },
    "langgraph": {
        "model_name": "llama-3.3-70b-versatile",
        "k": 3,
        "temperature": 0.0
    },
    "llamaindex": {
        "model_name": "llama-3.3-70b-versatile",
        "k": 3,
        "temperature": 0.0
    },
    "dspy": {
        "model_name": "llama-3.3-70b-versatile",
        "k": 3,
        "temperature": 0.0
    },
    "instructor": {
        "model_name": "llama-3.3-70b-versatile",
        "k": 3,
        "temperature": 0.0
    },
    "autogen": {
        "model_name": "llama-3.3-70b-versatile",
        "k": 3,
        "temperature": 0.0
    },
    "crewai": {
        "model_name": "llama-3.3-70b-versatile",
        "k": 3,
        "temperature": 0.0
    }
}

# Task-specific hyperparameters
TASK_CONFIGS = {
    "CUAD": {
        "k": 5,  # More documents for extraction
        "max_retrieval_iterations": 2
    },
    "ECHR": {
        "k": 5,  # More documents for reasoning
        "max_iterations": 3,  # More iterations for long-context
        "chunk_size": 5000
    },
    "LEDGAR": {
        "k": 3,
        "batch_size": 10  # For batch processing
    }
}

# Evaluation settings
EVAL_CONFIG = {
    "sample_size": None,  # None = use all examples
    "batch_size": 1,  # Process examples one at a time by default
    "save_predictions": True,
    "save_per_example_metrics": True
}

# Model settings
MODEL_CONFIG = {
    "provider": "groq",  # "groq", "openai", "anthropic", etc.
    "default_model": "llama-3.3-70b-versatile",
    "max_tokens": None,  # None = no limit
    "timeout": 60  # Timeout in seconds
}

# Environment variables (for API keys)
def get_api_key(provider: str) -> Optional[str]:
    """Get API key from environment variable."""
    env_vars = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }
    env_var = env_vars.get(provider.lower())
    if env_var:
        return os.getenv(env_var)
    return None


def get_config(framework: str, task: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration for a framework and optionally a task.
    
    Args:
        framework: Framework name
        task: Optional task name
        
    Returns:
        Merged configuration dictionary
    """
    config = FRAMEWORK_CONFIGS.get(framework, {}).copy()
    
    if task and task in TASK_CONFIGS:
        # Merge task-specific config (task config overrides framework config)
        task_config = TASK_CONFIGS[task].copy()
        config.update(task_config)
    
    return config


def ensure_directories():
    """Ensure all necessary directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(INDEXES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create task-specific result directories
    for task in ["CUAD", "ECHR", "LEDGAR"]:
        os.makedirs(os.path.join(RESULTS_DIR, task), exist_ok=True)

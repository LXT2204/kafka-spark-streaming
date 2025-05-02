import os
from pathlib import Path
from typing import Dict, Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

class KafkaConfig(BaseModel):
    bootstrap_servers: str
    topics: Dict[str, str]
    group_id: str

class VideoConfig(BaseModel):
    path: str
    fps: int
    resolution: Dict[str, int]
    loop: bool

class ProcessingConfig(BaseModel):
    batch_size: int
    model: Dict[str, str]

class LoggingConfig(BaseModel):
    level: str
    file: str

class Config(BaseModel):
    kafka: KafkaConfig
    video: VideoConfig
    processing: ProcessingConfig
    logging: LoggingConfig

def load_config() -> Config:
    """Load configuration from YAML file and environment variables."""
    # Load environment variables
    load_dotenv()
    
    # Get the config file path
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    
    # Load YAML configuration
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Replace environment variables
    def replace_env_vars(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        elif isinstance(value, dict):
            return {k: replace_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace_env_vars(item) for item in value]
        return value
    
    config_dict = replace_env_vars(config_dict)
    
    # Create Pydantic model
    return Config(**config_dict) 
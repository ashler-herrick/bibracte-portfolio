import logging
import logging.config
from pathlib import Path
import yaml


def setup_logging():
    # Determine the base directory (project root)
    base_dir = Path(__file__).resolve().parent.parent  # Adjust based on your project structure

    # Construct the path to the logging configuration file
    config_path = base_dir / "logging_config.yaml"

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
        logging.info(f"Loaded logging configuration from {config_path}")
    else:
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Logging configuration file '{config_path}' not found. Using basic configuration.")

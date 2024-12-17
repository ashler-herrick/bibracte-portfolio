import logging
import logging.config
import os
from pathlib import Path
import yaml
from importlib.resources import files


def setup_logging(
    default_relpath: str = "logging_config.yaml", default_level: int = logging.INFO, env_key: str = "LOG_CFG"
):
    """
    Setup logging configuration from a YAML file.

    Args:
        default_relpath (str): Relative path to the logging configuration file within the package.
        default_level (int): Default logging level if the config file is not found.
        env_key (str): Environment variable that can specify an alternative config file path.
    """
    # Locate the logging configuration file relative to the package
    try:
        package_root = files("bet_edge")  # Replace with your package's root module name
        config_path = package_root / default_relpath
    except ImportError:
        logging.warning("Package 'bet_edge' not found. Falling back to current directory.")
        config_path = Path(__file__).parent / default_relpath

    # Check for environment variable override
    env_path = os.getenv(env_key, None)
    if env_path:
        config_path = Path(env_path)

    # Resolve the absolute path
    config_path = config_path.resolve()  # type: ignore

    if config_path.is_file():
        with open(config_path, "rt") as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                logging.debug(f"Loaded logging configuration from {config_path}")
            except Exception as e:
                logging.basicConfig(level=default_level)
                logging.error(f"Error in logging configuration file: {e}. Using basic configuration.")
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging configuration file '{config_path}' not found. Using basic configuration.")

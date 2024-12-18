# scripts/script2.py
import logging

from bet_edge.setup_logging import setup_logging


def main():
    logger = logging.getLogger(__name__)
    logger.info("Script2 started.")

    numbers = [10, 20, 30, 40, 50]
    mean = sum(numbers) / len(numbers)
    logger.info(f"Calculated mean: {mean}")
    logger.debug("This is a debug statement.")

    logger.info("Script2 finished.")

    logger.error("This is a error statement.")


if __name__ == "__main__":
    setup_logging(default_relpath="../logging_config.yaml")  # Adjust path as needed
    main()

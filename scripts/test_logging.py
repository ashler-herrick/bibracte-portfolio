# scripts/script2.py
import logging

from bet_edge.logging_setup import setup_logging


def main():
    logger = logging.getLogger(__name__)
    logger.info("Script2 started.")

    numbers = [10, 20, 30, 40, 50]
    mean = sum(numbers) / len(numbers)
    logger.info(f"Calculated mean: {mean}")

    logger.info("Script2 finished.")


if __name__ == "__main__":
    setup_logging(default_relpath="../logging_config.yaml")  # Adjust path as needed
    main()
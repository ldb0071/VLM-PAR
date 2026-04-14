import sys
import logging
import os
from loguru import logger

# Singleton pattern to ensure logger is initialized only once
_logger_initialized = False

class InterceptHandler(logging.Handler):
    """
    Intercepts Python's standard logging messages and routes them to loguru.
    """

    def emit(self, record):
        try:
            # Match standard logging levels to loguru levels
            loguru_level = logger.level(record.levelname).name
        except KeyError:
            loguru_level = record.levelno

        # Log the message with the correct depth for source file and line number
        logger.opt(depth=6, exception=record.exc_info).log(loguru_level, record.getMessage())


def create_logger():
    """
    Configures the root logger to use Loguru for all standard logging calls.
    """
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(logging.DEBUG)  # Catch all log levels


def init_logger(logging_dir="./logs", log_level="INFO"):
    """
    Initializes the logger with file and console output.
    Ensures this is only run once globally.
    """
    global _logger_initialized
    if _logger_initialized:
        return

    # Ensure the logging directory exists
    os.makedirs(logging_dir, exist_ok=True)

    # Clear previous handlers to avoid duplicate logs
    logger.remove()

    # Add handlers with custom formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
        level=log_level,
        colorize=True,
    )
    logger.add(
        f"{logging_dir}/log.txt",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level=log_level,
        rotation="10 MB",  # Rotate the file when it reaches 10 MB
        retention="7 days",  # Keep logs for 7 days
        backtrace=True,
        diagnose=True,
    )

    # Route standard logging through loguru
    create_logger()
    _logger_initialized = True


# Expose logger and init_logger globally
__all__ = [
    'logger',
    'init_logger'
]

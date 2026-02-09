
import logging

# Create a logger for the package
logger = logging.getLogger("video-judge")

# Add NullHandler to prevent "No handlers could be found" warnings
# Applications using this package should configure their own handlers
logger.addHandler(logging.NullHandler())


def get_logger(name: str = "video-judge") -> logging.Logger:
    """
    Get a logger instance for the video-judge app.

    Args:
        name: Logger name, defaults to package name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_default_logging(level: int = logging.INFO):
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    # make the others shush
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

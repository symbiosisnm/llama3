import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger that writes to stdout with timestamps.

    Parameters:
        name: Name of the logger.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        An instance of logging.Logger configured with a StreamHandler and formatter.
        If the logger already has handlers, this function does not add new ones.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Only add handler if no handlers are present to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

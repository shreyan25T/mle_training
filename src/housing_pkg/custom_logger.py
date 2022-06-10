import logging
from pathlib import Path
import sys


def configure_logger(log_level, log_file, no_console):
    """
    This is to configure the logger for any script.

    Args:
    -----
        log_level (str): Logging level from list
                        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        log_file (str): Desired name of the log file, without any path.
        no_console (bool): If you don't want to write logs on console.

    Returns:
    --------
       logging.Logger: logger for the script.
    """
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Initializing the logger
    # NOTE: StreamHandler added by default to write logs on console
    logging.basicConfig(
        level=levels[log_level],
        format="%(asctime)s|%(levelname)s|%(filename)s|%(funcName)s|%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if log_file or not no_console:
        if log_file:
            # Creating a logs folder to store log files.
            log_path = Path("../..")/"logs"
            log_path.mkdir(parents=True, exist_ok=True)

            # Adding a file handler to write in files.
            file_hndlr = logging.FileHandler(str(log_path) + f"/{log_file}")
            file_hndlr.setLevel(levels[log_level])
            file_hndlr.setFormatter(
                logging.Formatter(
                    "%(asctime)s|%(levelname)s|%(filename)s|%(funcName)s|%(message)s"
                )
            )

            logging.root.addHandler(file_hndlr)

        if not no_console:
            # Removing the StreamHandler,
            # to disable writing logs on console
            logging.root.removeHandler(logging.root.handlers[0])

    return logging.root

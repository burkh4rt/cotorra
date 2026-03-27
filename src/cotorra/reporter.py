#!/usr/bin/env python3

"""
logging
"""

import logging

import polars as pl
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

log = logging.getLogger("rich")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(500)


class Logger(logging.Logger):
    def __init__(self, name: str = __package__):
        super().__init__(name=name)
        self.setLevel(logging.WARNING)
        self.handlers.clear()

        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(message)s", datefmt="%H:%M:%S%Z"
        )
        ch = RichHandler(
            show_path=False, show_time=False, console=Console(width=200, soft_wrap=True)
        )
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        self.addHandler(ch)
        self.propagate = False


if __name__ == "__main__":
    logger = Logger()
    logger.info("Testing...")

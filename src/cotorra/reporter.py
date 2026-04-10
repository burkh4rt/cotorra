#!/usr/bin/env python3

"""
logging
"""

import logging

import numpy as np
import polars as pl
import torch as t
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
        self.setLevel(logging.INFO)
        self.handlers.clear()

        formatter = logging.Formatter(
            fmt="🦜 [%(asctime)s] %(message)s", datefmt="%H:%M:%S%Z"
        )
        ch = RichHandler(
            show_path=False, show_time=False, console=Console(width=200, soft_wrap=True)
        )
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.addHandler(ch)
        self.propagate = False

    def summarize_trained_model(
        self, model, bos_token_id, reverse, n_samp=3, max_len=2048
    ):
        for i in range(n_samp):
            sample = model.generate(
                t.tensor([[bos_token_id]], dtype=t.int32).to(model.device),
                max_length=max_len,
                do_sample=True,
                top_k=len(reverse),
            )
            self.info(
                "Sample {}: {}".format(
                    i + 1, np.vectorize(reverse.get)(sample.cpu().numpy()).tolist()
                )
            )


if __name__ == "__main__":
    self = Logger()
    self.info("Testing...")

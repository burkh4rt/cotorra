#!/usr/bin/env python3

"""
load data and prepare for training / evaluation
"""

import collections
import pathlib

import datasets as ds
import numpy as np
import polars as pl
from omegaconf import OmegaConf


def batched_iter(dset: ds.Dataset, seq_len: int):
    dq = collections.deque()
    for eg in iter(dset):
        dq.extend(eg["input_ids"])
        while len(dq) >= seq_len:
            yield {"input_ids": [dq.popleft() for _ in range(seq_len)]}


class Loader:
    """the meds format dumps training (train), validation (tuning), and test (held_out)
    data into the same file;
    we need to start by fishing out training and validation data"""

    def __init__(self, **kwargs):
        main_cfg = OmegaConf.load(
            pathlib.Path("./config/main.yaml").expanduser().resolve()
        )
        self.cfg = OmegaConf.merge(main_cfg, kwargs)
        self.rng = np.random.default_rng(42)
        self.processed_data_home = (
            pathlib.Path(self.cfg.processed_data_home).expanduser().resolve()
        )
        self.tokenizer_info = OmegaConf.load(
            self.processed_data_home / "tokenizer.yaml"
        )

        if not (
            (self.processed_data_home / "training_tokens_times.parquet").is_file()
            and (self.processed_data_home / "tuning_tokens_times.parquet").is_file()
        ):
            self.subject_splits = pl.scan_parquet(
                self.processed_data_home / "subject_splits.parquet"
            )
            self.tokens_times = pl.scan_parquet(
                self.processed_data_home / "tokens_times.parquet"
            )
            (tt := self.tokens_times.join(self.subject_splits, on="subject_id")).filter(
                pl.col("split") == "train"
            ).drop("split").sink_parquet(
                self.processed_data_home / "training_tokens_times.parquet"
            )
            tt.filter(pl.col("split") == "tuning").drop("split").sink_parquet(
                self.processed_data_home / "tuning_tokens_times.parquet"
            )

        self.dataset = (
            ds.load_dataset(
                "parquet",
                data_files={
                    "training": str(
                        self.processed_data_home / "training_tokens_times.parquet"
                    ),
                    "tuning": str(
                        self.processed_data_home / "tuning_tokens_times.parquet"
                    ),
                },
            )
            .rename_column("tokens", "input_ids")
            .remove_columns(["subject_id", "times"])
        )

    def get_training_data(self):
        return ds.Dataset.from_generator(
            batched_iter,
            gen_kwargs={
                "dset": self.dataset["training"]
                .repeat(self.cfg.n_epochs)
                .shuffle(generator=self.rng),
                "seq_len": self.cfg.max_seq_len,
            },
        ).with_format("torch")

    def get_tuning_data(self):
        return ds.Dataset.from_generator(
            batched_iter,
            gen_kwargs={
                "dset": self.dataset["tuning"],
                "seq_len": self.cfg.max_seq_len,
            },
        ).with_format("torch")


if __name__ == "__main__":
    self = Loader()

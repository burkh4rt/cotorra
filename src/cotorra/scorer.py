#!/usr/bin/env python3

"""
use @lukesolo-ml's implementation of SCORE and REACH to make generative predictions
"""

import asyncio
import collections
import itertools
import pathlib

import numpy as np
import polars as pl
import tqdm
from omegaconf import OmegaConf
from quick_sco_re import GenerationConfig, create_engine, generate_and_score

from cotorra.logger import Logger


def batched(iterable, n):
    # `itertools.batched` introduced in Python 3.12
    # cf. https://docs.python.org/3/library/itertools.html#itertools.batched
    # batched('ABCDEFG', 3) → ABC DEF G
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


class Scorer:
    def __init__(
        self,
        main_cfg: pathlib.Path | str = None,
        processed_data_home: pathlib.Path | str = None,
        output_home: pathlib.Path | str = None,
        model_home: pathlib.Path | str = None,
        **kwargs,
    ):
        main_cfg = OmegaConf.load(
            pathlib.Path(main_cfg if main_cfg is not None else "./config/main.yaml")
            .expanduser()
            .resolve()
        )
        self.cfg = OmegaConf.merge(main_cfg, OmegaConf.create(kwargs))
        self.processed_data_home = (
            pathlib.Path(
                processed_data_home
                if processed_data_home is not None
                else self.cfg.processed_data_home
            )
            .expanduser()
            .resolve()
        )
        self.output_home = (
            pathlib.Path(
                output_home
                if output_home is not None
                else self.cfg.get("output_home", self.cfg.get("output_dir"))
            )
            .expanduser()
            .resolve()
        )
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")
        self.logger = Logger()

        self.engine = create_engine(
            model_path=str(pathlib.Path(model_home).expanduser().resolve())
            if model_home is not None
            else str(self.output_home / f"mdl-{self.cfg.run_name}"),
            max_len=self.cfg.score.max_len,
            use_time_horizon="max_time" in self.cfg.score,  # use if max_time configured
        )

        self.ds = pl.scan_parquet(
            self.processed_data_home / "held_out_for_inference.parquet"
        )
        self.tokens_past = self.ds.select("tokens_past").collect().to_series().to_list()

    async def sco_re(self, target_token: str, to_score_tokens: list[int]):
        tid = self.tkzr_cfg.lookup[target_token]
        sco_re_config = GenerationConfig(
            max_len=self.cfg.score.max_len,
            n_samp=self.cfg.score.n_samp,
            target_event_id=tid,
            end_token_ids=list(
                map(self.tkzr_cfg.lookup.get, self.cfg.score.end_tokens)
            ),
            suppressed_ids=list(
                map(self.tkzr_cfg.lookup.get, self.cfg.score.suppressed_tokens)
            ),
            trunc_id=self.tkzr_cfg.lookup.get(self.cfg.score.trunc_id, -1),
            max_time=self.cfg.score.get("max_time", None),
        )
        trajectories, results = await generate_and_score(
            self.engine, sco_re_config, to_score_tokens, target_token_id=tid
        )
        return trajectories, results

    async def score(self):
        res = collections.defaultdict(lambda: np.nan * np.ones(len(self.tokens_past)))

        for tt in tqdm.tqdm(self.cfg.score.target_tokens, position=0):
            to_score = (
                self.ds.select(~pl.col(f"{tt}_past")).collect().to_series().to_numpy()
            )
            to_score_tokens = [x for x, flag in zip(self.tokens_past, to_score) if flag]
            for idx_tks in tqdm.tqdm(
                batched(enumerate(to_score_tokens), self.cfg.score.batch_size),
                position=1,
                leave=False,
                total=np.ceil(len(to_score_tokens) / self.cfg.score.batch_size),
            ):
                idx, tks = zip(*idx_tks)
                _, results = await self.sco_re(tt, tks)
                res[f"{tt}_mc_estimate"][to_score][np.array(idx).ravel()] = np.array(
                    [np.mean(r.m0_samples) for r in results]
                )
                res[f"{tt}_scope_score"][to_score][np.array(idx).ravel()] = np.array(
                    [np.mean(r.m1_samples) for r in results]
                )
                res[f"{tt}_reach_score"][to_score][np.array(idx).ravel()] = np.array(
                    [np.mean(r.m2_samples) for r in results]
                )
        return res

    def save_all(self):
        res = asyncio.run(self.score())
        self.ds.with_columns(pl.from_dict(res)).sink_parquet(
            self.output_home / f"scores-{self.cfg.wandb.run_name}.parquet"
        )


if __name__ == "__main__":
    self = Scorer()
    breakpoint()

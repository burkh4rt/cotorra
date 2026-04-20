#!/usr/bin/env python3

"""
extract representations
"""

import pathlib

import torch as t
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM

from cotorra.loader import Loader
from cotorra.reporter import Logger


class Extractor:
    """load a model and extract representations from it"""

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
        self.cfg = OmegaConf.merge(main_cfg, kwargs)
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
        self.loader = Loader(self.cfg, self.processed_data_home)
        self.logger = Logger()
        self.device = (
            "cuda"
            if t.cuda.is_available()
            else "mps"
            if t.backends.mps.is_available()
            else "cpu"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pathlib.Path(model_home).expanduser().resolve()
            if model_home is not None
            else self.output_home / f"mdl-{self.cfg.run_name}"
        )
        self.model.to(self.device).eval()
        if not isinstance(self.model.config.pad_token_id, int):
            self.model.config.pad_token_id = self.model.config.eos_token_id
        self.ds = None

    def collate_fn(self, batch, time_limit_s: int = None):
        ml = t.tensor(self.cfg.get("extract", {}).get("max_len", 4096))
        break_pt = (
            [t.minimum(t.searchsorted(x, time_limit_s), ml) for x in batch["s_elapsed"]]
            if time_limit_s is not None
            else [ml] * len(batch["input_ids"])
        )
        input_ids = pad_sequence(
            [x[:bk] for bk, x in zip(break_pt, batch["input_ids"])],
            batch_first=True,
            padding_value=self.model.config.pad_token_id,
        ).to(self.model.device)
        if "time_based_rope" in self.cfg:
            p_ids = (
                pad_sequence(
                    [x[:bk] for bk, x in zip(break_pt, batch["s_elapsed"])],
                    batch_first=True,
                    padding_value=self.model.config.pad_token_id,
                ).to(self.model.device)
                / self.cfg.time_based_rope.sec_per_pos_id
            )
            p_ids += t.arange(p_ids.shape[-1], device=p_ids.device, dtype=p_ids.dtype)
        else:
            p_ids = None
        return {"input_ids": input_ids, "position_ids": p_ids}

    def extract_final(self, batch):
        collated = self.collate_fn(
            batch, time_limit_s=self.cfg.get("extract", {}).get("time_limit_s", None)
        )
        first_eos = t.where(
            (hits := (collated["input_ids"] == self.model.config.eos_token_id)).any(
                dim=-1
            ),
            hits.long().argmax(dim=-1)
            - 1,  # -1 to get the last token before break point
            collated["input_ids"].shape[-1] - 1,
        )
        with t.inference_mode():
            features = self.model(**collated, output_hidden_states=True).hidden_states[
                -1
            ]  # last hidden layer
        batch["features"] = (
            features[t.arange(len(first_eos)), first_eos].float().cpu().numpy()
        )
        return batch

    def extract(self):
        self.ds = self.loader.dataset.with_format("torch").map(
            self.extract_final,
            batched=True,
            batch_size=self.cfg.get("extract", {}).get("batch_size", 8),
        )
        for split, dset in self.ds.items():
            dset.to_parquet(self.processed_data_home / f"features-{split}.parquet")


if __name__ == "__main__":
    self = Extractor()
    # self.extract()

    # batch_eg = self.loader.dataset.with_format("torch")["training"].batch(8)[0]
    # collated_eg = self.collate_fn(batch_eg, time_limit_s = 86400)
    # fin_rep = self.extract_final(batch_eg)

#!/usr/bin/env python3

"""
train a model
"""

import os
import pathlib

import torch as t
from omegaconf import OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainingArguments,
)
from transformers import Trainer as t_Trainer

from cotorra.loader import Loader
from cotorra.loss import Loss
from cotorra.reporter import Logger


class Trainer:
    """the meds format dumps training (train), validation (tuning), and test (held_out)
    data into the same file;
    we need to start by fishing out training and validation data"""

    def __init__(self, **kwargs):
        main_cfg = OmegaConf.load(
            pathlib.Path("./config/main.yaml").expanduser().resolve()
        )
        mdl_cfg = OmegaConf.load(
            pathlib.Path(main_cfg.model_config).expanduser().resolve()
        )
        self.cfg = OmegaConf.merge(main_cfg, mdl_cfg, OmegaConf.create(kwargs))
        self.processed_data_home = (
            pathlib.Path(self.cfg.processed_data_home).expanduser().resolve()
        )
        self.output_dir = pathlib.Path(self.cfg.output_dir).expanduser().resolve()
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")
        self.loader = Loader(self.cfg)
        self.logger = Logger()

        self.trainer = t_Trainer(
            model_init=self.model_init,
            data_collator=self.collate_fn,
            compute_loss_func=Loss(self.cfg, self.tkzr_cfg).custom_loss
            if self.cfg.custom_loss
            else None,
            train_dataset=self.loader.get_train_data(),
            eval_dataset=self.loader.get_tuning_data(),
            args=TrainingArguments(
                output_dir=str(self.output_dir), **self.cfg.training_args
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        os.environ["WANDB_PROJECT"] = self.cfg.get("wandb", {}).get(
            "project", "cotorra"
        )
        os.environ["WANDB_RUN_NAME"] = self.cfg.get("wandb", {}).get(
            "run_name", "cotorra"
        )

    def model_init(self):
        conf_param = dict(
            vocab_size=len(self.tkzr_cfg.lookup),
            bos_token_id=self.tkzr_cfg.lookup.BOS,
            eos_token_id=self.tkzr_cfg.lookup.EOS,
        )
        config = AutoConfig.from_pretrained(
            self.cfg.model_name, **conf_param, **self.cfg.model_args
        )
        mdl = AutoModelForCausalLM.from_config(config)
        self.logger.info(
            "Loaded model {name} with {num} params.".format(
                name=self.cfg.model_name, num=sum(p.numel() for p in mdl.parameters())
            )
        )
        return mdl

    def collate_fn(self, batch):
        input_ids = t.stack([x["input_ids"] for x in batch])
        if "time_based_rope" not in self.cfg:
            return {"input_ids": input_ids, "labels": input_ids}
        else:
            p_ids = (
                t.stack([x["s_elapsed"] for x in batch])
                / self.cfg.time_based_rope.sec_per_pos_id
            )
            p_ids += t.arange(p_ids.shape[-1], device=p_ids.device, dtype=p_ids.dtype)
            return {"input_ids": input_ids, "labels": input_ids, "position_ids": p_ids}

    def train(self, verbose=False):
        self.trainer.train()
        self.trainer.model.save_pretrained(
            self.output_dir / f"mdl-{self.cfg.wandb.run_name}"
        )

        if verbose:
            self.logger.summarize_trained_model(
                model=self.trainer.model,
                bos_token_id=self.tkzr_cfg.lookup["BOS"],
                reverse={v: k for k, v in self.tkzr_cfg.lookup.items()},
            )


if __name__ == "__main__":
    self = Trainer()
    self.train(verbose=True)

#!/usr/bin/env python3

"""
load data
"""

import os
import pathlib

import numpy as np
import torch as t
from omegaconf import OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
)
from transformers import Trainer as t_Trainer

from cotorra.loader import Loader
from cotorra.reporter import Logger


class NanStoppingCallback(TrainerCallback):
    """stop training on encountering a nan objective"""

    def __init__(self):
        super().__init__()
        self.logger = Logger()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            for k, v in metrics.items():
                if not np.isfinite(v):
                    if state.is_world_process_zero:
                        self.logger.warning(f"Encountered non-finite metric {k} ({v}).")
                    control.should_training_stop = True


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
        self.cfg = OmegaConf.merge(main_cfg, mdl_cfg, kwargs)
        self.processed_data_home = (
            pathlib.Path(self.cfg.processed_data_home).expanduser().resolve()
        )
        self.output_dir = pathlib.Path(self.cfg.output_dir).expanduser().resolve()
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")
        self.loader = Loader(**self.cfg)
        self.logger = Logger()

        self.vocab = np.array(
            sorted(self.tkzr_cfg.lookup, key=self.tkzr_cfg.lookup.get)
        )
        self.toi_flag = np.isin(self.vocab, self.cfg.tokens_of_interest).astype(int)
        self.weights = t.Tensor((self.cfg.toi_weight - 1) * self.toi_flag + 1)

        os.environ["WANDB_PROJECT"] = self.cfg.wandb.project
        os.environ["WANDB_RUN_NAME"] = self.cfg.wandb.run_name

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

    def train(self):

        def collate_fn(batch):
            input_ids = t.stack([x["input_ids"] for x in batch])
            labels = input_ids.clone()
            return {"input_ids": input_ids, "labels": labels}

        training_args = TrainingArguments(
            report_to="wandb",
            run_name=self.cfg.wandb.run_name,
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.per_device_eval_batch_size,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            greater_is_better=False,
            eval_strategy=self.cfg.eval_strategy,
            save_strategy=self.cfg.save_strategy,
            ddp_find_unused_parameters=False,
        )

        def custom_loss(outputs, labels, **kwargs):
            logits = outputs.logits  # (batch, seq_len, vocab_size)
            return t.nn.CrossEntropyLoss(
                weight=self.weights.to(logits.device, dtype=logits.dtype)
            )(logits.view(-1, logits.size(-1)), labels.view(-1))

        trainer = t_Trainer(
            model_init=self.model_init,
            data_collator=collate_fn,
            compute_loss_func=custom_loss if self.cfg.toi_weight != 1.0 else None,
            train_dataset=self.loader.get_training_data(),
            eval_dataset=self.loader.get_tuning_data(),
            args=training_args,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                NanStoppingCallback(),
            ],
        )

        trainer.train()
        best_ckpt = trainer.state.best_model_checkpoint
        AutoModelForCausalLM.from_pretrained(best_ckpt).save_pretrained(
            self.output_dir / f"mdl-{self.cfg.run_name}"
        )


if __name__ == "__main__":
    self = Trainer()

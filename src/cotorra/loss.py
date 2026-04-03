#!/usr/bin/env python3

"""
configurable loss functions for training;
note this code only runs when configured with `custom_loss: !!bool true`
"""

import numpy as np
import torch as t


class Loss:
    def __init__(self, cfg, tkzr_cfg):

        self.cfg = cfg
        self.tkzr_cfg = tkzr_cfg
        self.vocab = np.array(
            sorted(self.tkzr_cfg.lookup, key=self.tkzr_cfg.lookup.get)
        )

        if "label_weighted_loss" in self.cfg:
            self.toi_flag = np.isin(
                self.vocab, self.cfg.label_weighted_loss.tokens_of_interest
            )
            self.weights = t.tensor(
                (self.cfg.label_weighted_loss.toi_weight - 1) * self.toi_flag + 1
            )

        if "quantile_token_loss" in self.cfg:
            assert not self.tkzr_cfg.cfg.fused, NotImplementedError(
                "label_weighted_loss is not formulated for fused tokens"
            )
            self.qi_flag = np.isin(
                self.vocab, [f"Q{i}" for i in range(self.tkzr_cfg.cfg.n_bins)]
            )
            self.qi_labels = t.nonzero(t.tensor(self.qi_flag), as_tuple=True)[0]
            self.qi_num = t.tensor(
                (np.char.replace(self.vocab[self.qi_flag], "Q", "").astype(int) + 0.5)
                / self.tkzr_cfg.cfg.n_bins
            ).to(dtype=t.float32)
            self.label_to_q = t.full((len(self.vocab),), float("nan"))
            self.label_to_q[self.qi_labels] = self.qi_num

    def quantile_token_loss(self, outputs, labels, **kwargs):
        q_logits = outputs.logits[
            :, :, self.qi_labels.to(outputs.logits.device)
        ]  # (batch, seq_len, vocab_size)
        q_probs = t.softmax(q_logits, dim=-1)
        e_num = q_probs @ self.qi_num.to(q_logits.device, dtype=q_logits.dtype)
        shift_e_num = e_num[:, :-1]
        shift_labels_num = self.label_to_q.to(labels.device, dtype=q_logits.dtype)[
            labels[:, 1:]
        ]
        mask = ~t.isnan(shift_labels_num)
        return (
            t.nn.MSELoss()(shift_e_num[mask], shift_labels_num[mask])
            if mask.any()
            else 0.0
        )

    def label_weighted_loss(self, outputs, labels, **kwargs):
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return t.nn.CrossEntropyLoss(
            weight=self.weights.to(logits.device, dtype=logits.dtype)
        )(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    def custom_loss(self, outputs, labels, **kwargs):
        loss = 0.0
        if "label_weighted_loss" in self.cfg:
            loss += self.label_weighted_loss(outputs, labels)
        if "quantile_token_loss" in self.cfg:
            loss += self.cfg.quantile_token_loss.qt_weight * self.quantile_token_loss(
                outputs, labels
            )
        return loss

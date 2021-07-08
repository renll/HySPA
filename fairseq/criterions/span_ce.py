# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch

@dataclass
class SpanCrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("params.optimization.sentence_avg")


@register_criterion("span_ce", dataclass=SpanCrossEntropyCriterionConfig)
class SpanCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        st_target, en_target=model.decoder.ind2se(sample["target"])
        net_output_st = (net_output[1][0],net_output[1][2])
        net_output_en = (net_output[1][1],net_output[1][2])
        loss_st, _ = self.compute_loss(model, net_output_st, {"target": st_target}, reduce=reduce)
        loss_en, _ = self.compute_loss(model, net_output_en, {"target": en_target}, reduce=reduce)
        loss=(loss_en + loss_st)/2
        if loss_st>100000:

            torch.set_printoptions(threshold=10000)
            print("T:",st_target)
            print("net_output_st: ", net_output_st[0].gather(-1,st_target.unsqueeze(-1)))
            raise ValueError("") 
        elif loss_en>100000:
            torch.set_printoptions(threshold=10000)
            print("TT:",en_target)
            print("net_output_en: ", net_output_en[0].gather(-1,en_target.unsqueeze(-1)))
            raise ValueError("") 
        sample_size = (
            sample["ntokens"]
        )
        #sample["target"].size(0) if self.sentence_avg else 
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        #print(lprobs.shape,target.shape)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

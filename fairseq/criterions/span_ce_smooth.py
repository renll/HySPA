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
class SmoothSpanCrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("params.optimization.sentence_avg")

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    mask = (-lprobs<1e5).float()#B V
    tgt_vocab_size = mask.sum(-1, keepdim=True) 
    smoothing_value = epsilon / (tgt_vocab_size - 1)
    one_hot = smoothing_value.repeat(1,lprobs.shape[-1])
    one_hot = mask*one_hot
    one_hot[:,ignore_index] = 0
    confidence = 1 - epsilon 
    model_prob = one_hot
    model_prob.scatter_(1, target, confidence)
    #model_prob.masked_fill_((target == ignore_index), 0)
    #print(model_prob)

    #torch.set_printoptions(threshold=10000)
    non_pad_mask = target.squeeze(-1).ne(ignore_index)
    #print(lprobs[5])
    #print(model_prob[5])
    loss = -(lprobs*model_prob).sum(dim=-1)
    #print(loss[5])
    #print(loss.shape)
    loss = loss.masked_select(non_pad_mask)
    #print(loss)
    #raise ValueError()
    if reduce:
        loss=loss.sum()
    #print(lprobs[-1])
    #print(model_prob[-1])

    
    
   # nll_loss = -lprobs.gather(dim=-1, index=target)
   # mask = (-lprobs<1e8).float()
   # lprobs= lprobs*mask
   # smooth_loss = (-lprobs.sum(dim=-1, keepdim=True)-nll_loss)/(mask.sum(-1, keepdim=True) - 1)
   # if ignore_index is not None:
   #     pad_mask = target.eq(ignore_index)
   #     nll_loss.masked_fill_(pad_mask, 0.0)
   #     smooth_loss.masked_fill_(pad_mask, 0.0)
   #     #print(smooth_loss)
   # else:
   #     nll_loss = nll_loss.squeeze(-1)
   #     smooth_loss = smooth_loss.squeeze(-1)
   # if reduce:
   #     nll_loss = nll_loss.sum()
   #     smooth_loss = smooth_loss.sum()
   # eps_i = epsilon #/ lprobs.size(-1)
   # loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss



@register_criterion("span_ce_smooth", dataclass=SmoothSpanCrossEntropyCriterionConfig)
class SmoothSpanCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,

    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        #self.alpha = 0.3


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on


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
        
        #last_hid = net_output[1][2]["last_hid"]
        loss_st, c_st,t_st = self.compute_loss(model, net_output_st, {"target": st_target}, reduce=reduce)
        loss_en, c_en,t_en = self.compute_loss(model, net_output_en, {"target": en_target}, reduce=reduce)
        loss=(loss_en + loss_st)/2
        #loss = loss + self.alpha * last_hid.pow(2).mean()

        nll_loss=loss
        n_correct = torch.sum(c_st *c_en)
        total = (t_st + t_en)/2.0
        if loss_st>1e10:

            torch.set_printoptions(threshold=10000)
            print("T:",st_target)
            print("net_output_st: ", net_output_st[0].gather(-1,st_target.unsqueeze(-1)))
            raise ValueError("") 
        elif loss_en>1e10:
            torch.set_printoptions(threshold=10000)
            print("TT:",en_target)
            print("net_output_en: ", net_output_en[0].gather(-1,en_target.unsqueeze(-1)))
            raise ValueError("") 
        sample_size = (
            sample["ntokens"]
        )
        #sample["target"].size(0) if self.sentence_avg else 
        #print(sample_size)
        #raise ValueError()
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        #print(lprobs.shape,target.shape)
        loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        
        if self.report_accuracy:
            mask = target.ne(self.padding_idx)
            n_correct = lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
            
            total = torch.sum(mask)
        else:
            n_correct = torch.zeros(1)
            total = 0
        return loss, n_correct, total

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        
        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )



        
        
       # if sample_size != ntokens:
       #     metrics.log_scalar(
       #         "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
       #     )
       #     metrics.log_derived(
       #         "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
       #     )
       # else:
       #     metrics.log_derived(
       #         "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
       #     )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

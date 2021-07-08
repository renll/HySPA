# Copyright (c) Liliang Ren.
#               Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from argparse import Namespace

import numpy as np
from fairseq import metrics, options, utils
from fairseq.data import (
    Dictionary,
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
    BERTDGLREDataset,
    BERTACEDataset,
    Indexing,
    CharIndexing,
)
from fairseq.tasks import FairseqTask, register_task
from transformers import AutoTokenizer

import torch
EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)



class HSDictionary(Dictionary):
    def __init__(self,
        pad_index=97,
        eos_index=98,
        bos_index=99,
        sep_index=100,
        unk_index=-1,
        dict_len=101+48*512,
        extra_special_symbols=None
    ):
        super().__init__()
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = str(bos_index),str(unk_index),str(pad_index),str(eos_index)
        self.symbols = []
        self.count = []
        self.indices = {}

        for i in range(dict_len):
            self.add_symbol(str(i))

        self.pad_index = pad_index
        self.eos_index = eos_index
        self.bos_index = bos_index
        self.sep_index = sep_index
        self.unk_index = unk_index
        print("self.bos_index", self.bos_index)
        print("self.pad_index", self.pad_index)
        print("self.eos_index", self.eos_index)
        print("self.unk_index", self.unk_index)
        print("len(self.symbols)", len(self.symbols))
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)
        

@register_task("seqie")
class SeqieTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--data-dir', type=str, default='data-bin/ace05/', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument('--max-source-positions', default=512, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=128, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')
        parser.add_argument('--bert-path', type=str, default="roberta-base",help='bert model name')

        parser.add_argument('--type-len', default=-1, type=int, metavar='N',help='97+4')
        parser.add_argument('--rel-len', default=-1, type=int, metavar='N',help='7+4')
        parser.add_argument('--max-span-len', default=16, type=int, metavar='N',help='null')
        parser.add_argument('--vocab-size', default=17000, type=int, metavar='N',help='null')
        parser.add_argument('--token_emb_dim', default=100, type=int, metavar='N',help='null')
        parser.add_argument('--char_emb_dim', default=30, type=int, metavar='N',help='null')

        parser.add_argument('--wv_path', type=str, default="wv/glove.6B.100d.ace05.txt",help='bert model name')
        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # fmt: on

    def __init__(self, args, pre_model,src_dict,tgt_dict,rel_freq,rel2id,ent2id,char,glove):
        super().__init__(args)
        self.pre_model = pre_model
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.padding_idx = tgt_dict.pad()
        self.rel_freq=rel_freq
        self.rel2id=rel2id
        self.ent2id=ent2id
        self.char = char
        self.glove = glove

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        train_file = args.data_dir+'train.json'
        ###
        ori_data=[]
        with open(file=train_file, mode='r', encoding='utf-8') as fr:
            for line in fr:
                ori_data.append(json.loads(line))
        print('loading..')
        
        rel_freq={}
        ent_freq={}
        
        label_len=[]
        for i, doc in enumerate(ori_data):
            rel_labels= doc['relations']
            ent_labels= doc['ner']
            for label in rel_labels:
                for l in label:
                    d=l[-1]
                    rel_freq[d] = rel_freq.get(d,0)+1
            
            for label in ent_labels:
                for l in label:
                    d=l[-1]
                    ent_freq[d] = ent_freq.get(d,0)+1
                

            #label_len.append(len(labels))
        rel_freq["TYPE"] = max(rel_freq.values())+1
        ent_freq["NULL"] = max(ent_freq.values())+1
        #print("average label len: ",sum(label_len)*4/len(label_len))
        print(rel_freq)
        print(ent_freq)
        rel2id={r:i for i,r in enumerate(rel_freq.keys())} 
        ent2id={r:i for i,r in enumerate(ent_freq.keys())} 
        print(rel2id)
        print(ent2id)
        rel_freq={ rel2id[r]:p for (r,p) in rel_freq.items()}
        print(rel_freq)
        ####
        
        args.rel_len = len(rel_freq)+4
        args.type_len = args.rel_len + len(ent_freq)

        pre_model=args.bert_path
        tokenizer= AutoTokenizer.from_pretrained(pre_model)
        src_dict =  HSDictionary(pad_index=tokenizer.pad_token_id,
                                eos_index=tokenizer.eos_token_id,
                                bos_index=tokenizer.bos_token_id,
                                unk_index=tokenizer.unk_token_id,
                                dict_len=len(tokenizer)
                    )
        tgt_dict = HSDictionary(pad_index=args.rel_len-4,
                                eos_index=args.rel_len-3,
                                bos_index=args.rel_len-2,
                                sep_index=args.rel_len-1,
                                unk_index=-1,
                                dict_len=args.type_len+args.max_span_len*args.max_source_positions
                    )


        
        char_path = args.data_dir + "char_vocab.txt"
        gl_path = args.data_dir + "glove_vocab.txt"


        char = CharIndexing(cased=True,vocab_file=char_path)
        glove = Indexing(maxlen = None,cased=False,vocab_file=gl_path)

        return cls(args,pre_model, src_dict, tgt_dict,rel_freq,rel2id,ent2id,char,glove, **kwargs)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
    
        data_dir = self.args.data_dir

        char = self.char
        glove = self.glove
        if split=="train": 
            dfile = data_dir+'train.json'
            file_save =data_dir+'train_BERT.pkl'
            data_set = BERTACEDataset(dfile, file_save, self.ent2id, self.rel2id, char=char, glove= glove, dataset_type='train',opt=self.args,rel_freq=self.rel_freq)
        elif split=="valid":
            dfile = data_dir+'dev.json'
            file_save =data_dir+'dev_BERT.pkl'
            data_set = BERTACEDataset(dfile, file_save, self.ent2id, self.rel2id, char=char, glove= glove, dataset_type='dev',opt=self.args,rel_freq=self.rel_freq)
        else:
            dfile = data_dir+'test.json'
            file_save =data_dir+'test_BERT.pkl'
            data_set = BERTACEDataset(dfile, file_save, self.ent2id, self.rel2id, char=char, glove= glove, dataset_type='test',opt=self.args,rel_freq=self.rel_freq)


        self.datasets[split]=data_set

   
    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.args.max_source_positions
        #(self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints, bos_token=self.tgt_dict.bos_index
            )
    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        print("len self.tgt_dict", len(self.tgt_dict))
        return self.tgt_dict
    
    @property
    def pad_id(self):
        return self.padding_idx

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

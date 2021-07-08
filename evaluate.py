# from fairseq.data.dglred_dataset import BERTDGLREDataset, DGLREDataloader
# from fairseq.data.ace_dataset import BERTACEDataset
from fairseq import options, tasks#, utils, checkpoint_utils, scoring
import argparse
import os
import json
# import numpy as np
# import sklearn
from datetime import datetime
# from collections import defaultdict
# import torch
from IPython import embed

import glob
def logging(s):
    print(datetime.now(), s)



# TL = 101
# MAX_SPANL = 48



def ind2se(ind, TL, MAX_SPANL):
    start_ind = -max(0, -ind + TL) + max(0, ind - TL) // MAX_SPANL + TL
    end_ind = start_ind + max(0, ind - TL) % MAX_SPANL
    return start_ind, end_ind


def get_metrics(sent_list, preds_list, labels_list):
    n_correct, n_pred, n_label = 0, 0, 0
    i_count = 0

    for sent, preds, labels in zip(sent_list, preds_list, labels_list):
        # print(preds)
        # print(labels)
        preds = set(preds)
        labels = {tuple(x) for x in labels}

        n_pred += len(preds)
        n_label += len(labels)
        n_correct += len(preds & labels)

        i_count += 1

    precision = n_correct / (n_pred + 1e-8)
    recall = n_correct / (n_label + 1e-8)
    f1 = 2 / (1 / (precision + 1e-8) + 1 / (recall + 1e-8) + 1e-8)

    return precision, recall, f1


def jaccard_score(seta, setb):
    return len(seta.intersection(setb)) / len(seta.union(setb))


def ind2original(ind, TL ,SL):
    start, end = ind2se(ind, TL, SL)
    start, end=start - TL,end - TL + 1
    # start=max(1, start)
    # end=max(1, end)
    return start, end


def seq2pred(task, seq_labels, probs, id2rel, id2ent, bert_starts_rev_dict):
    cur_rel_id = None
    cur_rel_prob = None
    cur_head = None
    cur_head_prob = None
    relation_list = []
    entity_list = []
    SEP = task.tgt_dict.sep_index
    rel_mlen = task.args.rel_len
    type_mlen = task.args.type_len
    #print("TYPE", type_mlen)
    #print("rel", rel_mlen)
    #raise ValueError()
    for jj, span_id in enumerate(seq_labels):
        if span_id == SEP:  # if id2rel(span_id) == "TYPE":
            cur_head = None
            cur_head_prob = None
            cur_rel_id = None
            cur_rel_prob = None
            continue
        if cur_head is None:
            cur_head = span_id
            # cur_head_prob=probs[jj]
        elif cur_rel_id is None:
            cur_rel_id = span_id
            # cur_rel_prob = probs[jj]
        else:
            # tail
            if cur_head >= type_mlen and cur_rel_id < len(id2rel):
                head_start, head_end = ind2original(cur_head, type_mlen, task.args.max_span_len)
                # head_start, head_end = max(1, head_start), max(1, head_end)
                #head_start, head_end=bert_starts_rev_dict[head_start], bert_starts_rev_dict[head_end]
                if id2rel[cur_rel_id] != "TYPE" and span_id >= type_mlen:

                    tail_start, tail_end = ind2original(span_id, type_mlen, task.args.max_span_len)
                    #tail_start, tail_end=bert_starts_rev_dict[tail_start], bert_starts_rev_dict[tail_end]
                    #relation_list.append([head_start, head_end, tail_start, tail_end, id2rel[cur_rel_id]])
                    relation_list.append([tail_start, tail_end, head_start, head_end, id2rel[cur_rel_id]])
                elif id2rel[cur_rel_id] == "TYPE" and (type_mlen > span_id >= rel_mlen):
                    entity_list.append([head_start, head_end, id2ent[span_id-rel_mlen]])
            cur_rel_id = None
            cur_rel_prob = None
    return entity_list, relation_list

def triple2phrase(tokens, triples):
    if not triples: return []
    if len(triples[0])==5:
        return [(tokens[triple[0]:triple[1]], tokens[triple[2]:triple[3]], triple[-1]) for triple in triples]
    return [(tokens[triple[0]:triple[1]], triple[-1]) for triple in triples]

def evaluate(dataset, task,res_file=None):
    # with open("valid.ACE05.json", "r") as f:
    #     ori_data = json.load(f)
    if res_file is not None:
        with open(res_file, "r") as f:
            generated_data = json.load(f)
    pred_entities, pred_relations=[],[]
    label_entities, label_relations=[],[]
    pred_relations_wNER,label_relations_wNER = [],[]
    sents=[]
    type_mlen= task.args.type_len
    
    class meter():
        ree_co = 0
        all_co = 0
        sect=0
        all_sect=0
        node=0
        def mess(self,out):
            vs=[]
            for i in range(len(out)):
                if i%2==0:
                    vs.append(out[i])
            vert = set(vs)
            len_v = len(vert)
            for vi in vert:
                hi,ti=ind2se(vi,type_mlen,task.args.max_span_len)
                if hi==ti and vi >= type_mlen:
                    self.node+=1
                if vi >= type_mlen:
                    self.all_sect += 1
                for vj in vert:
                    if vi != vj :
                        hi,ti=ind2se(vi,type_mlen,task.args.max_span_len)
                        hj,tj=ind2se(vj,type_mlen,task.args.max_span_len)
                        if vi >=type_mlen and vj>=type_mlen: #and \
                        #    (ti!=hi!=hj!=tj):
                        #if True:
                            if (hj<=ti<=tj and hi<=hj) or (hj<=hi<=tj and ti>=tj) :
                                #print(hi,ti)
                                #print(hj,tj)
                                self.sect+=1
                                #raise ValueError()
            self.ree_co+=len(out)//2-len_v
            self.all_co+=len(out)//2
        def print(self):
            print("FRACTION: ", self.ree_co/self.all_co)
            print("SECT FRACTION: ", self.sect/self.all_sect)
            print("SINGLE FRACTION: ", self.node/self.all_sect)
 
    tgt_meter = meter()
    pred_meter = meter()

    for i, sample in enumerate(dataset.get_data()):
        #
        # if i !=2423:
        #     continue
        # embed()
        print("\n----sample", i, "----")
        tokens=sample['tokens']#[type_mlen:]
        id2ent = sample['id2ent']
        id2rel = sample['id2rel']
        #print("\nid2ent", id2ent)
        #print("\nid2rel", id2rel)
        #raise ValueError()

        seq_labels = sample['seq_labels']
        out = [int(item.strip()) for item in generated_data[str(i)]["hypo_str"].split()]
        #out = seq_labels
        print("\nseq_labels", seq_labels)
        print("out", out)
       
        tgt_meter.mess(seq_labels[:-1])
        pred_meter.mess(out)
        # print("\nseq_labels", seq_labels)
        # print("out", out)

        bert_starts_rev_dict = sample['bert_starts_rev_dict']


        # pred_entities_sample, pred_relations_sample = seq2pred(seq_labels, [], id2rel, id2ent, bert_starts_rev_dict)
        a=generated_data[str(i)]["score"]
        print("\nsentence:", " ".join(tokens))
        pred_entities_sample, pred_relations_sample = seq2pred(task,out, [], id2rel, id2ent, bert_starts_rev_dict)

        def is_same(seta, setb):
            seta = set([tuple(item) for item in seta])
            setb = set([tuple(item) for item in setb])
            difa, difb = seta.difference(setb), setb.difference(seta)
            if difa or difb:
                return False
                # print("\ndifa", difa)
                # print("difb", difb)
                # embed()
            return True
        pred_entities_sample.sort(key=lambda x: x)
        sample['entities'].sort(key=lambda x: x)

        if not is_same(pred_entities_sample, sample['entities']):
            print("\npred_entities_sample", triple2phrase(tokens, pred_entities_sample))
            print("sample['entities']  ", triple2phrase(tokens, sample['entities']))
        #is_same(pred_entities_sample, sample['entities'])

        pred_relations_sample.sort(key=lambda x: x)
        sample['relations'].sort(key=lambda x: x)

        if not is_same(pred_relations_sample, sample['relations']):
            print("\npred_relations_sample", triple2phrase(tokens, pred_relations_sample))
            print("sample['relations']  ", triple2phrase(tokens, sample['relations']))

        pred_span_to_etype = [{(ib, ie): etype for ib, ie, etype in pred_entities_sample}]
        label_span_to_etype = [{(ib, ie): etype for ib, ie, etype in sample['entities']}]
        if not is_same(pred_span_to_etype, label_span_to_etype):
            print('\npred_span_to_etype ', pred_span_to_etype)
            print('label_span_to_etype', label_span_to_etype)

        print("\ntokens", tokens)
        print("\nscores", a)
        pred_entities.append(pred_entities_sample)
        label_entities.append(sample['entities'])

        pred_relations.append(pred_relations_sample)
        label_relations.append(sample['relations'])

        # embed()
        # pred_relations_wNER += [
        #     [
        #         (ib, ie, m[(ib, ie)], jb, je, m[(jb, je)], rtype) for ib, ie, jb, je, rtype in x
        #     ] for x, m in zip([pred_relations_sample], pred_span_to_etype)
        # ]
        # label_relations_wNER += [
        #     [
        #         (ib, ie, m[(ib, ie)], jb, je, m[(jb, je)], rtype) for ib, ie, jb, je, rtype in x
        #     ] for x, m in zip([sample['relations']], label_span_to_etype)
        # ]
        sents.append(sample['tokens'])

    tgt_meter.print()
    pred_meter.print()
    pred_entities = [[tuple(item) for item in example] for example in pred_entities]
    pred_relations = [[tuple(item) for item in example] for example in pred_relations]
        # word_id = sample['word_id']



    rets = {}
    rets['entity_p'], rets['entity_r'], rets['entity_f1'] = get_metrics(
        sents, pred_entities, label_entities)
    rets['relation_p'], rets['relation_r'], rets['relation_f1'] = get_metrics(
        sents, pred_relations, label_relations)
    # rets['relation_p_wNER'], rets['relation_r_wNER'], rets['relation_f1_wNER'] = get_metrics(
    #     sents, pred_relations_wNER, label_relations_wNER)
    print(rets)


    if not os.path.isdir("results/"):
        os.mkdir("results/")
    fname=os.path.splitext(res_file)[0].split("__")
    with open("results/relf1_{:.5f}__entf1_{:.5f}__{}__{}".format(rets['relation_f1'], rets['entity_f1'], fname[-2], fname[-1]), "w+") as _:
        pass



if __name__ == "__main__":

    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    
    split = args.gen_subset
    task = tasks.setup_task(args)
    task.load_dataset(split)
    test_set = task.datasets[split]

    docid2outseqs = glob.glob("gen_output/docid2outs*")

    for docid2outseq in docid2outseqs:
        evaluate(test_set, task,res_file=docid2outseq)
        #raise ValueError()

# Copyright (c) Liliang Ren.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import pickle
import random
from collections import defaultdict,OrderedDict

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer,AutoModel

from fairseq.data import FairseqDataset, data_utils
import queue
from fairseq.data.indexings import Indexing, CharIndexing

IGNORE_INDEX = -100

def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

class Roberta():
    def __init__(self,model_path):
        super().__init__()
        print(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_len = 512
        self.MASK = self.tokenizer.mask_token
        self.CLS = self.tokenizer.cls_token
        self.SEP = self.tokenizer.sep_token
        self.pad_token_id =self.tokenizer.pad_token_id 

    def tokenize(self, text, masked_idxs=None):
        tokenized_text = self.tokenizer.tokenize(text)
        if masked_idxs is not None:
            for idx in masked_idxs:
                tokenized_text[idx] = self.MASK
        # prepend [CLS] and append [SEP]
        # see https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py#L195  # NOQA
        tokenized = [self.CLS] + tokenized_text + [self.SEP]
        return tokenized

    def tokenize_to_ids(self, text, masked_idxs=None, pad=False):
        tokens = self.tokenize(text, masked_idxs)
        return tokens, self.convert_tokens_to_ids(tokens, pad=pad)

    def convert_tokens_to_ids(self, tokens, pad=False):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor([token_ids])
        assert ids.size(1) < self.max_len
        if pad:
            
            #padded_ids = torch.zeros(1, self.max_len).to(ids)
            padded_ids = torch.full((1, self.max_len),self.tokenizer.pad_token_id,dtype=ids.dtype).to(ids)
            padded_ids[0, :ids.size(1)] = ids
            mask = torch.zeros(1, self.max_len).to(ids)
            mask[0, :ids.size(1)] = 1
            return padded_ids, mask
        else:
            return ids, None

    def flatten(self, list_of_lists):
        for list in list_of_lists:
            for item in list:
                yield item

    def subword_tokenize(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the special symbols required
                by Bert (CLS and SEP).
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
        """
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        subwords = [self.CLS] + list(self.flatten(subwords))[:(self.max_len-3)] + [self.SEP]
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_start_idxs[token_start_idxs > (self.max_len-3)] = self.max_len
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subword IDs, including IDs of the special
                symbols (CLS and SEP) required by Bert.
            - A mask indicating padding tokens.
            - An array of indices into the list of subwords. See
                doc of subword_tokenize.
        """
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        return subword_ids.numpy(), token_start_idxs, subwords

    def segment_ids(self, segment1_len, segment2_len):
        ids = [0] * segment1_len + [1] * segment2_len
        return torch.tensor([ids])


from operator import itemgetter, attrgetter

def se2ind(start,end,MAX_SPANL,TL):
    return start*MAX_SPANL+end-start-1+TL # python end index convention

def ind2se(ind, TL, MAX_SPANL):
    start_ind = -max(0, -ind + TL) + max(0, ind - TL) // MAX_SPANL + TL
    end_ind = start_ind + max(0, ind - TL) % MAX_SPANL
    return start_ind, end_ind


def get_metrics(sent_list, preds_list, labels_list):
    n_correct, n_pred, n_label = 0, 0, 0
    i_count = 0
    for sent, preds, labels in zip(sent_list, preds_list, labels_list):
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


def ind2original(ind, TL):
    start, end = ind2se(ind, TL, 32)
    return start - TL, end - TL + 1


def seq2pred(seq_labels, probs, id2rel, id2ent, bert_starts_rev_dict):
    cur_rel_id = None
    cur_rel_prob = None
    cur_head = None
    cur_head_prob = None
    relation_list = []
    entity_list = []
    SEP = len(id2rel) + 3
    rel_mlen = len(id2rel) + 4
    type_mlen = len(id2rel) + len(id2ent) + 4
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
                head_start, head_end = ind2original(cur_head, type_mlen)
                head_start, head_end=bert_starts_rev_dict[head_start], bert_starts_rev_dict[head_end]
                if id2rel[cur_rel_id] != "TYPE" and span_id >= type_mlen:

                    tail_start, tail_end = ind2original(span_id, type_mlen)
                    tail_start, tail_end=bert_starts_rev_dict[tail_start], bert_starts_rev_dict[tail_end]
                    relation_list.append([head_start, head_end, tail_start, tail_end, id2rel[cur_rel_id]])
                elif id2rel[cur_rel_id] == "TYPE" and (type_mlen > span_id >= rel_mlen):
                    entity_list.append([head_start, head_end, id2ent[span_id-rel_mlen]])
            cur_rel_id = None
            cur_rel_prob = None
    return entity_list, relation_list

class BERTACEDataset(FairseqDataset):

    def __init__(self, src_file, save_file, ent2id, rel2id,rel_freq,char,glove,
                 dataset_type='train', instance_in_train=None, opt=None,dump=False):

        super().__init__()

        # record training set mention triples
        self.rel_freq=rel_freq
        self.data = None
        self.document_max_length = opt.max_source_positions
        self.type_mlen=len(rel2id)+len(ent2id)+4
        self.rel_mlen=len(rel2id)+4
        self.SEP=len(rel2id)+3
        self.SOS=len(rel2id)+2
        self.EOS=len(rel2id)+1   
        self.tgt_pad_id=len(rel2id)
        self.shuffle=True
        self.buckets=None
        self.span_mlen=opt.max_span_len
        self.char = char 
        self.glove = glove 
        
        bert = Roberta(opt.bert_path)
        self.pad_id = bert.pad_token_id#source

        print('Reading data from {}.'.format(src_file))
        if not dump:
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                data = info['data']
            print('load preprocessed data from {}.'.format(save_file))

        else:

            ree_co=0
            all_co=0
            if "ace05" in opt.data_dir:
                fullname ={'PHYS': 'Physical', 'ART': 'Agent-artifact', 'GEN-AFF': 'Gen-affiliation', 'PART-WHOLE': 'Part-whole', 'ORG-AFF': 'Organization-affiliation', 'PER-SOC': 'Personal-social', 'TYPE': 'Type',
                    'GPE': 'Geopolitics', 'PER': 'Person', 'VEH': 'Vehicle', 'ORG': 'Organization', 'FAC': 'Facility', 'LOC': 'Location', 'WEA': 'Weapon', 'NULL': 'Null'}

            bert_model=AutoModel.from_pretrained(opt.bert_path,output_hidden_states=True)
            bert_tokenizer = AutoTokenizer.from_pretrained(opt.bert_path)

            id2sp= {v:fullname[k] for (k,v) in rel2id.items()}
            id2sp[len(id2sp)] = "[PAD]" 
            id2sp[len(id2sp)] = "[EOS]" 
            id2sp[len(id2sp)] = "[CLS]" 
            id2sp[len(id2sp)] = "[SEP]" 
            tmpl=len(id2sp)
            for (k,v) in ent2id.items():
                id2sp[tmpl+v] = fullname[k] 
            print(id2sp)
            prefix = [id2sp[i] for i in range(len(id2sp))]
            print(prefix)
            prefix_emb = []
            for p in prefix:
                inputs= bert_tokenizer(p,return_tensors="pt")
                outputs=bert_model(**inputs)
                x=outputs[2][-4:]
                x=sum(x)/len(x)
                x=x[:,1:-1].mean(dim=1)
                prefix_emb.append(x)
            
            prefix_emb=torch.cat(prefix_emb, dim=0)
            torch.save(prefix_emb,opt.data_dir+'prefix_emb.pt')
            
            ori_data=[]
            with open(file=src_file, mode='r', encoding='utf-8') as fr:
                for line in fr:
                    ori_data.append(json.loads(line))
            print('loading..')
 
            
            data = []
            global max_span_len
            max_span_len=0

            pred_entities, pred_relations = [], []
            label_entities, label_relations = [], []
            pred_relations_wNER, label_relations_wNER = [], []
            sents = []
           

            for i, doc in enumerate(ori_data):
                # print("doc", i)
               
                sentl = 0
                
                for j,words in enumerate(doc["sentences"]):
                    # generate document ids
                    
                    pwords= prefix + words    

                    chi=self.char([pwords]) 
                    gl=self.glove([pwords])
                   
                    

                    bert_token, bert_starts, bert_subwords = bert.subword_tokenize_to_ids(words)
                    end = len(bert_token[0])-1
                    bsn=bert_starts.tolist()+[end]
                    bword_list=[list(range(bsn[i],bsn[i+1])) for i in range(len(bsn)-1)]

                    bword_len = [ len(b) for b in bword_list]
                    max_blen = max(bword_len)
                    bwords=[w+(max_blen-len(w))*[0] for w in bword_list]

                    pbert_token, pbert_starts, pbert_subwords = bert.subword_tokenize_to_ids(prefix)
                    assert len(pbert_starts)==len(prefix)

                    bert_starts_rev_dict={}

                    def p2ind(hpos0,hpos1):
                        global max_span_len 

                        assert hpos0<len(bert_starts) and hpos1<=len(bert_starts)
                        
                        hpos0p = bert_starts[hpos0]
                        if hpos1<len(bert_starts): 
                            hpos1p = bert_starts[hpos1] 
                        else:
                            hpos1p = len(bert_subwords)-1 #tok before eos 

                        max_span_len=max(hpos1-hpos0,max_span_len)
                        
                        if hpos0p<self.document_max_length  and \
                                    hpos1p<self.document_max_length : 

                            seqh=se2ind(hpos0,hpos1,self.span_mlen,self.type_mlen)
                            return seqh 


               
                    seq_label=OrderedDict()#graph
                    
                    relations = [ [r[0]-sentl,r[1]-sentl+1,r[2]-sentl,r[3]-sentl+1,r[4]] for r in doc["relations"][j]]
                    rels = defaultdict(list)
                    for r in relations:
                        relation= r[-1]
                        h=p2ind(r[0],r[1])
                        t=p2ind(r[2],r[3])
                        assert (relation in rel2id), 'no such relation {} in rel2id'.format(relation)
                        rels[t].append((rel2id[relation],h)) 
                        #rels[h].append((rel2id[relation],t)) 

                    ners=[ [e[0]-sentl,e[1]-sentl+1,e[2]] for e in doc["ner"][j]]
                    ent=sorted(ners, key=lambda x: x[0]) 

                    for e in ent:

                        espan=p2ind(e[0],e[1])
                        
                        seq_label.setdefault(espan,[]).append((rel2id["TYPE"],self.rel_mlen+ent2id[e[-1]])) 
                        if espan in rels.keys():
                            seq_label[espan].extend(rels[espan])

                       
                    
                    for s in seq_label.keys():
                        seq_label[s]=list(set(seq_label[s]))
                        seq_label[s].sort(key=lambda x: x[1])
                        seq_label[s].sort(key=lambda x: self.rel_freq[x[0]],reverse=True)
                    
                    BFS=True
                    if BFS:
                        #bfs
                        seq_labels=[]
                        V=seq_label.keys()
                        visited={v:False for v in V}
                        for u in V:
                            if visited[u] == False:
                                q=queue.Queue()
                                visited[u]=True
                                q.put(u)
                                while (not q.empty()):
                                    u=q.queue[0]
                                    q.get()
                                    if u in V:
                                        sl=[x for tp in seq_label[u] for x in tp]
                                        seq_labels.extend([u,]+sl+[self.SEP])
                                        for p in seq_label[u]:
                                            if p[1] in V and not visited[p[1]]:
                                                visited[p[1]]=True
                                                q.put(p[1])
                    else:
                        #dfs
                        def dfs(u, visited):
                            visited[u]=True
                            if u in seq_label.keys():
                                for k in seq_label[u]:
                                    seq_labels.append(u)
                                    seq_labels.append(k[0])
                                        
                                    if not visited[k[1]]:
                                        dfs(k[1],visited)
                                    else:
                                        seq_labels.append(k[1])
                                        seq_labels.append(self.SEP)
                            else:
                                seq_labels.append(u)
                                seq_labels.append(self.SEP)
                        
                        seq_labels=[]
                        vertices = list(set(list(seq_label.keys())+[pp[1] for p in seq_label.values() for pp in p]))
                        V=sorted(vertices)
                        visited={v:False for v in V}
                        for u in V:
                            if visited[u] == False and u in seq_label.keys():
                                dfs(u,visited)

                  

                    if len(seq_labels)==0:
                        seq_labels=[self.rel_mlen+ent2id['NULL'],self.SEP]

                    seq_labels=seq_labels[:-1]
                    seq_labels.append(self.EOS)
                    
                    vertices = list(set(list(seq_label.keys())+[pp[1] for p in seq_label.values() for pp in p]))
                    vs=[]
                    for i in range(len(seq_labels)):
                        if i%2==0:
                            vs.append(seq_labels[i])
                    len_v = len(set(vs))
                    g_lenv= len(vertices)
                    if g_lenv==0:
                        g_lenv+=1
                    if len_v != g_lenv:
                        print(len_v)
                        print(g_lenv)
                        print(seq_labels)
                        print(vertices)
                        print(set(vs))
                        raise ValueError()
                    ree_co+=len(seq_labels)//2-1-len_v
                    all_co+=len(seq_labels)//2-1
                    
                    sentl = sentl+len(words)
                    
                   
                    sample_length = len(bert_token[0])
                    word_id = np.array(bert_token[0])
                    pref_id = np.array(pbert_token[0])
                    char_id = np.array(chi[0])
                    bword_pos = np.array(bwords)
                    gl_id = np.array(gl[0])
                    #assert len(gl_id)==len(bert_starts)
                    starts = np.zeros((sample_length,), dtype=np.int32)
                    starts[:len(bert_starts)]=bert_starts
                    
                    pstarts = np.zeros((len(pbert_token[0]),), dtype=np.int32)
                    pstarts[:len(pbert_starts)]=pbert_starts
 

                    assert len(list({v: k for k, v in ent2id.items()}))==len(ent2id), "ent2id"
                    assert len(list({v: k for k, v in rel2id.items()}))==len(rel2id), "rel2id"
                    data.append({
                        'seq_labels':seq_labels,
                        'starts': starts,
                        'pstarts': pstarts,
                        'word_id': word_id,
                        'bword_pos': bword_pos,
                        'pref_id': pref_id,
                        'char_id': char_id,
                        'gl_id': gl_id,
                        'tokens': words,
                        'relations': relations,
                        'entities': ners,
                        'id2ent':{v: k for k, v in ent2id.items()},
                        'id2rel':{v: k for k, v in rel2id.items()},
                        'span_mlen':self.span_mlen,
                        'bert_starts_rev_dict':bert_starts_rev_dict
                    })

            print("FRACTION: ", ree_co/all_co)
            print("MAXSPAN: ",max_span_len)
            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': data}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

        
        #print(data[5])
        self.src_data=[torch.LongTensor(x['word_id']) for x in data]
        self.src_pdata=[torch.LongTensor(x['pref_id']) for x in data]
        self.char_data=[torch.LongTensor(x['char_id']) for x in data]
        self.bword_data=[torch.LongTensor(x['bword_pos']) for x in data]
        self.glove_data=[torch.LongTensor(x['gl_id']) for x in data]
        self.src_starts=[torch.LongTensor(x['starts']) for x in data]
        self.src_pstarts=[torch.LongTensor(x['pstarts']) for x in data]
        self.tgt_data=[torch.LongTensor(x['seq_labels']) for x in data]
        self.tgt_sizes=np.array([len(x['seq_labels']) for x in data])
        print("max_target_len: ", max(self.tgt_sizes))
        print("mean_target_len: ", sum(self.tgt_sizes)/len(self.tgt_sizes))
        self.src_sizes= torch.LongTensor(
            [s.ne(self.pad_id).long().sum() for s in self.src_data]
        ).numpy()
        #self.src_sizes = np.array([len(x['word_id']) for x in data])
        print("max_src_len: ", max(self.src_sizes))
        print("REL_LEN: ", self.rel_mlen)
        print("TYPE_LEN: ", self.type_mlen)
        self.data=data

    def get_data(self):
        return self.data
    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        
        src_item=self.src_data[idx]
        src_starts=self.src_starts[idx]
        tgt_item=self.tgt_data[idx]

        example={
            "id": idx,
            "source": src_item,
            "sourcep": self.src_pdata[idx],
            "source_starts": src_starts,
            "source_pstarts": self.src_pstarts[idx],
            "bwords": self.bword_data[idx],
            "char":self.char_data[idx],
            "glove":self.glove_data[idx],
            "target": tgt_item,
        }
        
        return example
    
    def collater(self, samples, pad_to_length=None):
        if len(samples) == 0:
            return {}

        pad_to_length=None#self.document_max_length
        pad_idx=self.pad_id
        tgt_pad_idx=self.tgt_pad_id
        id = torch.LongTensor([s["id"] for s in samples])
        src_tokens =  data_utils.collate_tokens(
                [s["source"] for s in samples],
                pad_idx,
                pad_to_length=pad_to_length,
            ) 
        src_ptokens =  data_utils.collate_tokens(
                [s["sourcep"] for s in samples],
                pad_idx,
                pad_to_length=pad_to_length,
            ) 
        
        src_starts =  data_utils.collate_tokens(
                [s["source_starts"] for s in samples],
                0,
                pad_to_length=pad_to_length,
            ) 
        
        src_pstarts =  data_utils.collate_tokens(
                [s["source_pstarts"] for s in samples],
                0,
                pad_to_length=pad_to_length,
            ) 
        
        max_slen = max(s["source"].size(0) for s in samples)
        gloves = data_utils.collate_tokens(
                [s["glove"] for s in samples],
                0,
                pad_to_length=max_slen,
            ) 
        chars = data_utils.collate_chars(
                [s["char"] for s in samples],
                0,
                pad_to_length=max_slen,
            ) 
        bwords = data_utils.collate_chars(
                [s["bwords"] for s in samples],
                0,
                pad_to_length=max_slen,
            ) 
       
        
        # sort by descending source length
        src_lengths = torch.LongTensor(
            [s["source"].ne(pad_idx).long().sum() for s in samples]
        )
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)
        src_ptokens = src_ptokens.index_select(0, sort_order)
        src_starts = src_starts.index_select(0, sort_order)
        src_pstarts = src_pstarts.index_select(0, sort_order)
        gloves = gloves.index_select(0, sort_order)
        chars = chars.index_select(0, sort_order)
        bwords = bwords.index_select(0, sort_order)
        

        prev_output_tokens = None
        target = None
        if samples[0].get("target",None) is not None:
            target  =  data_utils.collate_tokens(
                    [s["target"] for s in samples],
                    tgt_pad_idx,
                    pad_to_length=None,
                ) 
            target = target.index_select(0, sort_order) 
            tgt_lengths = torch.LongTensor(
                [s["target"].ne(tgt_pad_idx).long().sum() for s in samples]
            ).index_select(0, sort_order) 
            ntokens = tgt_lengths.sum().item()
            
            if samples[0].get("prev_output_tokens", None) is not None:
                prev_output_tokens = data_utils.collate_tokens(
                    [s["prev_output_tokens"] for s in samples],
                    tgt_pad_idx,
                    self.SOS,
                    move_eos_to_beginning=False,
                    pad_to_length=None,
                )
                #prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
            else:
                prev_output_tokens = data_utils.collate_tokens(
                    [s["target"] for s in samples],
                    tgt_pad_idx,
                    self.SOS,
                    move_eos_to_beginning=True,
                    pad_to_length=None,
                )
            
        else:
            
            ntokens = src_lengths.sum().item()
        
        batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "src_starts": src_starts,
            "chars": chars,
            "bwords": bwords,
            "gloves": gloves,
            "src_ptokens": src_ptokens,
            "src_pstarts": src_pstarts,
        },
        "target": target,
        }
        if prev_output_tokens is not None: 
            batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
                0, sort_order
            ) 
        
        return batch

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        #return self.tgt_sizes[index]  
        #return self.document_max_length
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )


    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return 0

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        return indices
       # if self.buckets is None:
       #     # sort by target length, then source length
       #     if self.tgt_sizes is not None:
       #         indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
       #     return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
       # else:
       #     # sort by bucketed_num_tokens, which is:
       #     #   max(padded_src_len, padded_tgt_len)
       #     return indices[
       #         np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
       #     ]

def save_vocab(vb,path):
    with open(path, 'w', encoding = 'utf-8' ) as f:
        vb={ i:p for (p,i) in vb.items()}
        for i in range(len(vb)):
            f.write(vb[i]+"\n")
 

if __name__=="__main__":
    
    


    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--bert-path', type=str, default="roberta-large")#
    parser.add_argument('--bert-path', type=str, default="albert-xxlarge-v1")#
    parser.add_argument('--data-dir', type=str, default="../../data-bin/ace05/")#
    parser.add_argument('--max-source-positions', default=512, type=int, metavar='N',
                        help='max number of tokens in the source sequence')
    parser.add_argument('--max-span-len', default=16, type=int, metavar='N',help='null')


    opt=parser.parse_args()

    data_dir = opt.data_dir

    char = CharIndexing(cased=True)
    glove = Indexing(maxlen = None,cased=False) 
    glove.vocab = {
                '[pad]': 0,
                '[eos]': 1,
                '[cls]': 2,
                '[sep]': 3,
                '[unk]': 4,
            }

    train_file = data_dir+'train.json'
    dev_file = data_dir+'dev.json'
    test_file = data_dir+'test.json'
    train_file_save =data_dir+'train_BERT.pkl'
    dev_file_save =data_dir+'dev_BERT.pkl'
    test_file_save =data_dir+'test_BERT.pkl'

    ####
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
    #####
    train_set = BERTACEDataset(train_file, train_file_save, ent2id, rel2id, dataset_type='train', char=char, glove= glove, opt=opt,rel_freq=rel_freq,dump=True)
    dev_set = BERTACEDataset(dev_file, dev_file_save, ent2id, rel2id, dataset_type='dev',char=char, glove= glove,opt=opt,rel_freq=rel_freq,dump=True)
    test_set = BERTACEDataset(test_file, test_file_save, ent2id, rel2id, dataset_type='test',char=char, glove= glove,opt=opt,rel_freq=rel_freq,dump=True)
    
    char_path = data_dir + "char_vocab.txt"
    gl_path = data_dir + "glove_vocab.txt"
    save_vocab(glove.vocab, gl_path)
    save_vocab(char.vocab, char_path)












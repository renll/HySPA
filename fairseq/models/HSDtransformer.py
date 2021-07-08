# Copyright (c) Liliang Ren.
#               Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    FairseqDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    MultiheadAttention,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
import torch.nn.functional as F
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

from transformers import AutoTokenizer, AutoModel
from fairseq.modules.unfold import unfold1d
from .modeling_roberta import *
from .modeling_bert import *
from tqdm import tqdm
from .seqs import *

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("HSDtransformer")
class HSDTransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """


    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')

        parser.add_argument('--decoder-forget-rate', type=float, metavar='D',
                            help='dropout probability for attention weights')


        parser.add_argument('--lm-embed-dim', type=int, metavar='N')
        parser.add_argument('--dropout_o', type=float, metavar='D')
        parser.add_argument('--lm_dropout', type=float, metavar='D')
        parser.add_argument('--dropout_i', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--dropout_e', type=float, metavar='D',
                            help='dropout probability')

        #
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        glove_ind= task.glove
        tgt_dict = task.target_dictionary
        decoder_embed_tokens = tgt_dict

        st_emb = Embedding(2, args.decoder_embed_dim, None)
        encoder = cls.build_encoder(args, st_emb, glove_ind)
        decoder = cls.build_decoder(args, decoder_embed_tokens,st_emb)
        return cls(args, encoder, decoder )

    @classmethod
    def build_encoder(cls, args,st_emb, glove_ind):
        return BERTEncoder(args,st_emb, glove_ind)

    @classmethod
    def build_decoder(cls, args, embed_tokens,st_emb):
        return HSDecoder(
            args,
            None,
            embed_tokens,
            st_emb=st_emb,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        src_starts,
        chars,
        bwords,
        gloves,
        src_ptokens,
        src_pstarts,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
       # torch.autograd.set_detect_anomaly(True)
        encoder_out = self.encoder(
            src_tokens, src_lengths,src_starts, chars,bwords,gloves, 
            src_ptokens,
            src_pstarts,
            return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class BERTEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, st_emb, glove_ind):
        super().__init__(None)
        self.args = args

        self.register_buffer("version", torch.Tensor([3]))

        self.encoder_layerdrop = args.encoder_layerdrop

        self.glove_ind = glove_ind
        self.bert=AutoModel.from_pretrained(args.bert_path,hidden_dropout_prob=args.lm_dropout)
        self.tokenizer=AutoTokenizer.from_pretrained(args.bert_path)
        self.padding_idx=self.tokenizer.pad_token_id
        self.embed_positions=None
      
        self.max_source_positions = args.max_source_positions
        self.dropout_module = FairseqDropout(
            args.dropout_e, module_name=self.__class__.__name__
        )

        self.pro_layer=Linear(args.lm_embed_dim + args.char_emb_dim + args.token_emb_dim, args.decoder_embed_dim,bias=False)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.token_embedding = Embedding(args.vocab_size, args.token_emb_dim, 0, dropout=True)

        self.dropoute = 0.
        self.char_embedding = Embedding(200, args.char_emb_dim, 0)
        
        prefix_mat = torch.load(args.data_dir+"prefix_emb.pt")
        self.prefix_emb = nn.Embedding.from_pretrained(prefix_mat,freeze=True)
        self.type_embedding = Embedding(5, args.decoder_embed_dim, 0)
        
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.token_emb_dim)
        self.embed_scale_tp = 1.0 if args.no_scale_embedding else math.sqrt(args.decoder_embed_dim)
        
        self.char_encoding = CharLSTMEncoding(args.char_emb_dim)

        #self.char_emb_dropout = nn.Dropout(0.5)

        self.masking = Masking()
        self.load_pretrained(args.wv_path,True)


    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def load_pretrained(self, path, freeze=False):
        embedding_matrix = self.token_embedding.cpu().weight.data
        idx_list=[]
        with open(path, 'r') as f:
            for line in tqdm(f):
                line = line.strip().split(' ')
                token = line[0]
                vector = np.array([float(x) for x in line[1:]], dtype=np.float32)
                vector = torch.from_numpy(vector)
                idx = self.glove_ind.token2idx(token)
                if idx >= self.args.rel_len:
                    idx_list.append(idx)
                else:
                    print(idx)
                embedding_matrix[idx] = vector
        self.token_embedding.weight.data = embedding_matrix.to(self.token_embedding.weight.data)
       
        if freeze:
            def _freeze_word_embs(self, grad_in, grad_out):
                embs_grad = grad_in[0]
                embs_grad[idx_list] = 0.
                return (embs_grad,)
            self.token_embedding.register_backward_hook(_freeze_word_embs)

    def forward(
        self,
        src_tokens,
        src_lengths,
        src_starts,
        chars,
        bwords,
        gloves,
        src_ptokens,
        src_pstarts,
        return_all_hiddens: bool = True,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        # compute padding mask
        mention_tokens=None
        ner_tokens=None
        span_tokens=None
        encoder_padding_mask = src_tokens.ne(self.padding_idx)
        encoder_outputs= self.bert(input_ids=src_tokens, attention_mask=encoder_padding_mask,output_hidden_states=True,return_dict=True)
        eo=encoder_outputs.last_hidden_state

        eo=self.dropout_module(eo,locked=False)
        cls=eo[:,0,:].unsqueeze(1)

        ss=src_starts.unsqueeze(-1).repeat(1,1,eo.shape[-1])
        eo1=torch.gather(eo, 1, ss)
        
      
      
        prefix_token=torch.arange(self.args.type_len).unsqueeze(0).repeat(src_tokens.shape[0],1).to(src_tokens)
        eop1 = self.prefix_emb(prefix_token) 

        encoder_padding_mask = torch.cat([src_starts.new_zeros(src_starts.shape[0], 1).bool(),gloves.eq(0)],dim=1)
        eo = torch.cat([cls, eop1[:,:self.args.type_len], eo1[:,:gloves.shape[1]-self.args.type_len]], dim=1)

        type_tokens = src_starts.new_zeros([src_starts.shape[0],1 + gloves.shape[1]]).long()
        split1 = self.args.rel_len-4+1
        split2 = self.args.rel_len+1
        split3 = self.args.type_len+1
        type_tokens[:, 0]=4
        type_tokens[:, 1:split1]=1
        type_tokens[:, split1:split2]=2
        type_tokens[:, split2:split3]=3
        type_tokens[:, split3:]=4


        tmp = self.token_embedding(gloves.transpose(0,1),
                            dropout=self.dropoute if self.training else 0)
        tmp = tmp.transpose(0,1) * self.embed_scale
        
        masks = self.masking(gloves, mask_val=0)

        c_masks = self.masking(chars, mask_val=0)
        c_lens = c_masks.sum(dim=-1) + (1-masks.long())
        c_embeddings = self.char_embedding(chars)
        #c_embeddings = self.char_emb_dropout(c_embeddings)
        c_embeddings = self.char_encoding(c_embeddings, lens=c_lens, mask=c_masks)        
        
        tmp=torch.cat([tmp.new_zeros([tmp.shape[0],1,tmp.shape[-1]]),tmp],dim=1)
        c_embeddings=torch.cat([c_embeddings.new_zeros([c_embeddings.shape[0],1,c_embeddings.shape[-1]]),c_embeddings],dim=1)
        eo = torch.cat([eo,tmp,c_embeddings],dim=-1) 
        eo=self.pro_layer(eo)
        
        type_emb=self.type_embedding(type_tokens)*self.embed_scale_tp
        
        eo=eo+type_emb
        eo = eo * (1-encoder_padding_mask.float().unsqueeze(-1))
        x=eo.transpose(0,1)
        enc = torch.cat([x[0].unsqueeze(0),x[self.args.type_len+1:]],dim=0)

        states=[]
        if return_all_hiddens:
            states=[x.transpose(0,1) for x in encoder_outputs.hidden_states]


        return EncoderOut(
            encoder_out=enc,  # T x B x C
            encoder_padding_mask= encoder_padding_mask,  # B x T
            span_mask=span_tokens,
            encoder_embedding=x[1:],  # T X Bx C
            encoder_states=states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(1, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        new_span_mask = encoder_out.span_mask
        if new_span_mask is not None:
            new_span_mask = new_span_mask.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)


        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            span_mask=new_span_mask,
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        #for i in range(self.num_layers):
        #    # update layer norms
        #    self.layers[i].upgrade_state_dict_named(
        #        state_dict, "{}.layers.{}".format(name, i)
        #    )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict



class TreeEmbed(nn.Module):

    def __init__(self,d_model, max_nb):
        super().__init__()
        depth = 2 # only support depth = 2 for now
        embedding_dim = d_model //(max_nb*depth)
        self.d = nn.Parameter(torch.Tensor(embedding_dim))
        limit=(3/embedding_dim )** 0.5
        nn.init.uniform_(self.d, -limit,limit )
        
       # fan_in =(max_nb*depth)*embedding_dim 
       # self.tree_proj = nn.Linear(fan_in ,d_model)
       # limit = (6/(fan_in +d_model) )** 0.5
       # nn.init.uniform_(self.tree_proj.weight, -limit,limit )
       # nn.init.zeros_(self.tree_proj.bias)
        
        self.d_model = d_model
        self.max_nb = max_nb

    def forward(self,tree_token):
        d=torch.tanh(self.d)
        d1=torch.cat([torch.pow(d,0).unsqueeze(0).repeat(self.max_nb,1),d.unsqueeze(0).repeat(self.max_nb,1)],dim=0)
        norm=torch.sqrt(( 1-torch.square(d))*(self.d_model/2))
        d1=d1*norm
        d1=d1.unsqueeze(0).unsqueeze(1)
        tree_emb =d1*tree_token.unsqueeze(-1)
        tree_emb = tree_emb.view(tree_emb.shape[0],tree_emb.shape[1],-1)
        #tree_emb = self.tree_proj(tree_emb)
        return tree_emb


class HSDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens,st_emb=None, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.st_emb = st_emb

        self.dropout_module = FairseqDropout(
            args.dropout_i, module_name=self.__class__.__name__
        )
        self.dropout_o = FairseqDropout(
            args.dropout_o, module_name=self.__class__.__name__
        )
        self.dropout_oe = FairseqDropout(
            args.dropout_o, module_name=self.__class__.__name__
        )

        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        #input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.pad()
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = None
        self.embed_pc = Embedding(3, embed_dim, 0)
        self.embed_level = PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                0,
                learned=False,
            )

        self.max_nb = 16
        self.embed_tree =TreeEmbed(embed_dim,self.max_nb)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.start_layer=Linear(embed_dim, embed_dim)
        self.end_layer=Linear(embed_dim, embed_dim)


        self.span_attn=MultiheadAttention(
            embed_dim,
            1,
            kdim=embed_dim,
            vdim=embed_dim,
            dropout=0,
            encoder_decoder_attention=True,
            in_attn = True,
        )
       

        self.adaptive_softmax = None
        self.output_projection = None

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn)

    def ind2se(self, ind):
        MAX_SPANL=self.args.max_span_len
        TL=self.args.type_len
        start_ind=-torch.relu(-ind+TL)+torch.relu(ind-TL)//MAX_SPANL+TL
        end_ind=start_ind+torch.relu(ind-TL)%MAX_SPANL
        return start_ind, end_ind

    def get_traversal_token(self, strs):
        pad = self.embed_tokens.pad_index
        sep = self.embed_tokens.sep_index
        sos = self.embed_tokens.bos_index
        mask = (strs!=pad).long()
        sep_pos = (strs==sep).long()+(strs==sos).long()
        
        parent_pos = sep_pos.roll(1,dims=1)
        parent_pos[:,0] = 0

        level_token = torch.cumsum(sep_pos,dim=1)*mask
        #print(level_token)
        pc_token = (torch.arange(strs.shape[1],device = strs.device)%2+parent_pos)*mask
        #print(pc_token)
        connect_token=(torch.arange(strs.shape[1],device = strs.device)//2).unsqueeze(0).repeat(strs.shape[0],1)
        stages = connect_token*sep_pos
        
        connect_token=(connect_token-stages.cummax(dim=1).values)*mask
        

        depth = 2  # only support depth = 2 for now
        tree=F.one_hot(connect_token,num_classes= self.max_nb*depth)*\
                (1-parent_pos).unsqueeze(2)*(1-sep_pos).unsqueeze(2)*mask.unsqueeze(2)
        tree = tree.roll(-1,dims=2)
       
        c_mask = (pc_token == 1).long().unsqueeze(-1).repeat(1,1,self.max_nb*depth)
        child=c_mask*tree
        child = child.roll(self.max_nb,dims=2)
        child[:,:,0]=(pc_token == 1).long()
        tree = tree*(1-c_mask)+child
        
        return level_token,pc_token,tree,sep_pos,parent_pos


    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        level_token,pc_token,tree_token,sep_pos,parent_pos = self.get_traversal_token(prev_output_tokens)
        

        levels = self.embed_level(level_token,incremental_state=incremental_state)
        pcs = self.embed_pc(pc_token)
        trees = self.embed_tree(tree_token)
        
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
            levels,trees=levels[:, -1:],trees[:, -1:]
            pcs = pcs[:,-1:]
        Hp = encoder_out.encoder_embedding#T0 B C
       
        H = Hp[self.args.type_len:]
        S = Hp[:self.args.type_len]
        Hmask=(encoder_out.encoder_padding_mask[:,1:]).float()*(-1e8)
        Hmask=Hmask.transpose(0,1).unsqueeze(-1)#T B 1

        hmask = Hmask[self.args.type_len:]
        smask = Hmask[:self.args.type_len]
       
        start_ind, end_ind = self.ind2se(prev_output_tokens)#B P 

        end_ind+=1 # python index convention
        next_range=torch.arange(Hp.shape[0]).to(H.device).unsqueeze(0).repeat(start_ind.shape[0],1).unsqueeze(1)#B 1 T
        start_range=(next_range<start_ind.unsqueeze(2).repeat(1,1,Hp.shape[0])).float()
        end_range=(next_range<end_ind.unsqueeze(2).repeat(1,1,Hp.shape[0])).float()

        span=end_range-start_range #B P T

        assert (span>=0).all()
        
        span_mask = 1-span 
        span_mask = span_mask.masked_fill(span_mask.to(torch.bool), -1e8)
        
        q = encoder_out.encoder_out[0].unsqueeze(0).repeat(span.shape[1],1,1)#P B C
        x, _ = self.span_attn(
            query=q,
            key=Hp,
            value=Hp,
            key_padding_mask=None,
            attn_mask=None,
            span_mask= span_mask,
            incremental_state=incremental_state,
            static_kv=True,
        ) #P B C
        
        x = x.transpose(0,1)
       
        osc_mask=(prev_output_tokens<self.args.rel_len).float().unsqueeze(-1)#BP1
        
        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

       
        x+= levels+trees+self.embed_scale*pcs
       

        tgt_emb=self.st_emb(prev_output_tokens.new_ones(prev_output_tokens.shape).long())*self.embed_scale
        x+= tgt_emb
       
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)


        _self_attn_input_buffer = self.layers[0].self_attn._get_input_buffer(incremental_state)
        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        
       
        if not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            enp=encoder_out.encoder_out
            src_emb=self.st_emb(enp.new_zeros(enp.shape[:2]).long())*self.embed_scale
            enp=enp+src_emb
            x= torch.cat([enp,x], dim = 0)
            
            epm = torch.cat([encoder_out.encoder_padding_mask[:,0].unsqueeze(1),encoder_out.encoder_padding_mask[:,self.args.type_len+1:]],dim=1)

            if self_attn_padding_mask is not None: 
                self_attn_padding_mask=torch.cat([epm,self_attn_padding_mask],dim=1)
            else:
                self_attn_padding_mask= torch.cat([epm, prev_output_tokens.new_zeros(prev_output_tokens.shape).bool()],dim=1)
            
            self_attn_mask = self.buffered_future_mask(x,encoder_out)
        else:
            self_attn_mask = None
        
        
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            x= x[encoder_out.encoder_out.shape[0]:]
        
        #if self.project_out_dim is not None:
        #    x = self.project_out_dim(x)
        last_hid = x
        x=self.dropout_o(x)
        xs=self.start_layer(x)
        xe=self.end_layer(x)


        # P x B x C -> B x P x C
        xs = xs.transpose(0, 1)#.contiguous()
        xe = xe.transpose(0, 1)#.contiguous()
        osc_tp=torch.cat([osc_mask.repeat(1,1,self.args.rel_len)*(-1e8),
                            (1-osc_mask).repeat(1,1,self.args.type_len-self.args.rel_len)*(-1e8)],dim=-1)
        
        ss= S.transpose(0,1)#.contiguous()
        def calc_dist(a,b):
            #return -torch.cdist(a,b)
            #return torch.bmm(F.normalize(a, dim=-1),
            #                F.normalize(b, dim=-1).transpose(1,2))*self.temp
            return torch.einsum("bpc,btc->bpt",a,b)

       
        ty_mask = smask.permute(1,2,0)+osc_tp
        ty_st=calc_dist(xs,ss)+ ty_mask
        ty_en=calc_dist(xe,ss)+ ty_mask
        
        He=H
        h = He.transpose(0,1)#.contiguous() 
        st_seq=calc_dist(xs,h).permute(2,0,1)+hmask+(1-osc_mask).permute(2,0,1)*(-1e8)
        en_seq=calc_dist(xe,h).permute(2,0,1)+hmask+(1-osc_mask).permute(2,0,1)*(-1e8)

        ty_log= ty_st+ty_en
        en_log=unfold1d(en_seq,self.args.max_span_len,0,pad_value=-1e8)#t,b,p,L

        span_log=en_log+st_seq.unsqueeze(-1)

        span_log=span_log.permute(1,2,0,3)#b p t l

        flat_span=span_log.reshape(span_log.shape[0],span_log.shape[1],-1)
        x=torch.cat([ty_log,flat_span],dim=-1)

        train_st=torch.cat([ty_st,st_seq.permute(1,2,0)],dim=-1)
        train_en=torch.cat([ty_en,en_seq.permute(1,2,0)],dim=-1)

        return x,(train_st,train_en, {"attn": [attn],"last_hid": last_hid, "inner_states": inner_states})

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        #if self.adaptive_softmax is None:
            # project back to size of vocabulary
        #    return self.output_projection(features)
        #else:
        return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor, encoder_out):
        edim = encoder_out.encoder_out.shape[0]
        dim = tensor.size(0)
        #assert edim < dim
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
       # if (
       #     self._future_mask.size(0) == 0
       #     or (not self._future_mask.device == tensor.device)
       #     or self._future_mask.size(0) < dim
       # ):
        _future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        _future_mask[:edim, :edim]=0
        _future_mask = _future_mask.to(tensor)
        return _future_mask#[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx, dropout=False):
    if dropout:
        m= utils.DropoutEmbedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    else:
        m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    #nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    utils.truncated_normal_(m.weight,std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    #nn.init.xavier_uniform_(m.weight)
    utils.truncated_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("HSDtransformer", "HSDtransformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.lm_embed_dim = getattr(args, "lm_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2*256)
    args.encoder_layers = getattr(args, "encoder_layers", 0)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 6)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.decoder_forget_rate = getattr(args, "decoder_forget_rate", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.lm_dropout = getattr(args, "lm_dropout", 0.2)
    args.dropout_o = getattr(args, "dropout_o", 0.0)
    args.dropout_i = getattr(args, "dropout_i", 0.0)
    args.dropout_e = getattr(args, "dropout_e", 0.0)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", True
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", True)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)


@register_model_architecture("HSDtransformer", "HSDtransformer1")
def transformer_iwslt_de_en(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4 * 256)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    base_architecture(args)

@register_model_architecture("HSDtransformer", "HSDtransformer_sc")
def transformer_iwslt_de_en(args):
    args.lm_embed_dim = getattr(args, "lm_embed_dim", 768)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4 * 256)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    base_architecture(args)

@register_model_architecture("HSDtransformer", "HSDtransformer_alb")
def transformer_iwslt_de_en(args):
    args.lm_dropout = getattr(args, "lm_dropout", 0.1)
    args.lm_embed_dim = getattr(args, "lm_embed_dim", 4096)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4 * 256)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    base_architecture(args)





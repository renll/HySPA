
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import numpy as np

#from utils import *
#from functions import *

#from layers.indexings import *
#from layers.encodings import *
#from layers.encodings.lm_embeddings import PreEmbeddedLM

class Masking:
    def __call__(self, idxs, mask_val=0, dtype=None):
        if dtype is None or dtype == torch.BoolTensor:
            masks = (idxs!=mask_val).to(idxs.device)
        else:
            masks = (idxs!=mask_val).type(dtype).to(idxs.device)
        return masks

class CharLSTMEncoding(nn.Module):
    
    def __init__(self, input_dim=None, num_layers=1):
        super().__init__()
        self.lstm = LSTMEncoding(input_dim, num_layers)
        
    def forward(self, inputs, lens=None, mask=None):
        bs, t_maxlen, c_maxlen, _ = inputs.shape
        inputs = inputs.view(bs*t_maxlen, c_maxlen, -1)
        lens = lens.view(bs*t_maxlen)
        
        _, hids = self.lstm(inputs, return_cls=True, lens=lens, mask=mask) # (bs*t_maxlen, hidden_dim)
        hids = hids.view(bs, t_maxlen, -1)
        return hids

class LSTMEncoding(nn.Module):
    
    def __init__(self, input_dim=None, num_layers=1):
        super().__init__()
        
        self.hidden_dim = input_dim
        self.num_layers = num_layers
        self.bias = True #config.bias
        self.batch_first = True #config.batch_first
        self.dropout = 0.2
        self.bidirectional = True #config.bidirectional
        self.input_dim = self.hidden_dim if input_dim is None else input_dim
        
        k_bidirectional = 2 if self.bidirectional else 1
        
        self.rnn = nn.LSTM(
            self.input_dim, self.hidden_dim//k_bidirectional, 
            self.num_layers, self.bias, self.batch_first, self.dropout, self.bidirectional)
        init_lstm(self.rnn)
        
#         shape = [k_bidirectional*self.num_layers, 1, self.hidden_dim//k_bidirectional]
    
    def forward(self, inputs, return_cls=False, mask=None, lens=None):
        batch_size = inputs.shape[0] if self.batch_first else input.shape[1]
        hidden = None
        
        if mask is not None or lens is not None:
            if lens is not None:
                word_seq_lens = lens
            else:
                word_seq_lens = mask.sum(dim=-1)
            word_seq_lens = word_seq_lens + (word_seq_lens == 0).long() # avoid length == 0
            word_rep = inputs
            sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
            _, recover_idx = permIdx.sort(0, descending=False)
            sorted_seq_tensor = word_rep[permIdx]
            
            packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
            lstm_out, (h, _) = self.rnn(packed_words, None)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            outputs = lstm_out[recover_idx]
            hidden = torch.cat([h[-2,:,:], h[-1,:,:]], dim=-1)
            hidden = hidden[recover_idx]
        else:
            outputs, (h, c) = self.rnn(inputs, hidden)
            hidden = torch.cat([h[-2,:,:], h[-1,:,:]], dim=-1)
        
        if return_cls:
            return (outputs, hidden)
        else:
            return outputs
        
class GRUEncoding(nn.Module):
    
    def __init__(self, config, input_dim=None, num_layers=1):
        super().__init__()
        
        self.config = config
        
        self.hidden_dim = config.hidden_dim
        self.num_layers = num_layers
        self.bias = True #config.bias
        self.batch_first = True #config.batch_first
        self.dropout = config.dropout
        self.bidirectional = True #config.bidirectional
        self.input_dim = self.hidden_dim if input_dim is None else input_dim
        
        k_bidirectional = 2 if self.bidirectional else 1
        
        self.rnn = nn.GRU(
            self.input_dim, self.hidden_dim//k_bidirectional, 
            self.num_layers, self.bias, self.batch_first, self.dropout, self.bidirectional)

    def forward(self, inputs, return_cls=False, mask=None, lens=None):
        batch_size = inputs.shape[0] if self.batch_first else input.shape[1]
        hidden = None
        
        if mask is not None or lens is not None:
            if lens is not None:
                word_seq_lens = lens
            else:
                word_seq_lens = mask.sum(dim=-1)
            word_seq_lens = word_seq_lens + (word_seq_lens == 0).long() # avoid length == 0
            word_rep = inputs
            sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
            _, recover_idx = permIdx.sort(0, descending=False)
            sorted_seq_tensor = word_rep[permIdx]
            
            packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
            lstm_out, h = self.rnn(packed_words, None)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            outputs = lstm_out[recover_idx]
            hidden = torch.cat([h[-2,:,:], h[-1,:,:]], dim=-1)
            hidden = hidden[recover_idx]
        else:
            outputs, h = self.rnn(inputs, hidden)
            hidden = torch.cat([h[-2,:,:], h[-1,:,:]], dim=-1)
        
        if return_cls:
            return (outputs, hidden)
        else:
            return outputs
        
class CNNEncoding(nn.Module):
    ''' n to 1 '''
    def __init__(self, config, input_dim=None, Ks=[3,4,5]):
        super().__init__()
        self.config = config
        
        if input_dim is None:
            input_dim = config.hidden_dim
        
        D = input_dim      # input dim
        C = config.hidden_dim # output dim
        Ci = 1
        Co = config.hidden_dim # kernel numbers i.e. hid dim
        Ks = Ks # kernels
        
        self.convs1 = nn.ModuleList([
            nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), padding=((K-1)//2,0)) for K in Ks])

    def forward(self, x, mask=None, lens=None):
        # x (N, W, D)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [conv(x).squeeze(3) for conv in self.convs1]
        if mask is not None:
            mask = mask.view(x[0].shape[0], 1, -1).float() # (N, 1, W) c.f. x:(N, D, W)
            x = [i - (1.-mask)*999 for i in x]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x
    
    
class NGramEncoding(nn.Module):
    
    def __init__(self, config, input_dim=None, ngram=2, padding=0):
        super().__init__()
        self.config = config
        
        if input_dim is None:
            input_dim = config.hidden_dim
            
        self.conv = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=config.hidden_dim, 
            kernel_size=ngram,
            padding=padding,
        )
            
    def forward(self, x):
        # x (B, T, in)
        return self.conv(x.transpose(1,2)).transpose(1,2)
    
    
class RouterEncoding(nn.Module):
    
    def __init__(self, config, n_iter=2, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
            
        self.config = config
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            SelfRouting(self.config, config.hidden_dim, n_iter=n_iter) for i in range(num_layers)
        ])
        
        self.fcs = nn.ModuleList([
             nn.Sequential(
                 nn.Linear(config.hidden_dim, dim_feedforward),
                 nn.ReLU(),
                 nn.Dropout(dropout),
                 nn.Linear(dim_feedforward, config.hidden_dim),
             ) for i in range(num_layers)
        ])
        
        self.norms0 = nn.ModuleList([
            nn.LayerNorm(self.config.hidden_dim) for _ in range(num_layers)
        ])
        
        self.norms1 = nn.ModuleList([
            nn.LayerNorm(self.config.hidden_dim) for _ in range(num_layers)
        ])
        
        self.scales = nn.Parameter(torch.ones(num_layers).float())
        
    def forward(self, inputs, mask=None):
        outputs = inputs
        for i in range(self.num_layers):
            outputs = outputs + self.dropout(self.scales[i] * self.layers[i](outputs, mask)[0])
            outputs = self.norms0[i](outputs)
            
            outputs = outputs + self.dropout(self.fcs[i](outputs))
            outputs = self.norms1[i](outputs)
            
        return outputs
    
class ReConv(nn.Module):
    
    def __init__(self, config, kernel_size=2):
        super().__init__()
        self.config = config
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            kernel_size=[kernel_size, kernel_size], 
            in_channels=config.hidden_dim, 
            out_channels=config.hidden_dim, 
            padding=(kernel_size//2))
        
    def forward(self, embeddings_list):
        '''
        list of (B, T, H), (B, T-1, H), ... (B, T-(L-1), H)
        '''
        B, T, H = embeddings_list[0].shape
        L = len(embeddings_list)
        
        embeddings_T = pad_arbitrary(embeddings_list, padding_value=0., padding_dim=1, length_dim=1) # (B, L, T, H)
        embeddings_T = embeddings_T.permute(0, 3, 1, 2)
        if self.kernel_size % 2 == 0:
            embeddings_T = self.conv(embeddings_T)[:, :, 1:, 1:] + embeddings_T # (B, L, T, H)
        else:
            embeddings_T = self.conv(embeddings_T) + embeddings_T # (B, L, T, H)
        embeddings_T = embeddings_T.permute(0, 2, 3, 1)
        
        return embeddings_T

def init_lstm(input_lstm):
    """
    Initialize lstm
    
    PyTorch weights parameters:
    
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size * hidden_size)`
            
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size * hidden_size)`            
    """
    # Weights init for forward layer
    for ind in range(0, input_lstm.num_layers):
        
        ## Gets the weights Tensor from our model, for the input-hidden weights in our current layer
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        
        # Initialize the sampling range
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        
        # Randomly sample from our samping range using uniform distribution and apply it to our current layer
        nn.init.uniform_(weight, -sampling_range, sampling_range)
        
        # Similar to above but for the hidden-hidden weights of the current layer
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -sampling_range, sampling_range)
        
        
    # We do the above again, for the backward layer if we are using a bi-directional LSTM (our final model uses this)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)

    # Bias initialization steps
    
    # We initialize them to zero except for the forget gate bias, which is initialized to 1
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind))
            
            # Initializing to zero
            bias.data.zero_()
            
            # This is the range of indices for our forget gates for each LSTM cell
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            
            #Similar for the hidden-hidden layer
            bias = eval('input_lstm.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            
        # Similar to above, we do for backward layer if we are using a bi-directional LSTM 
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
    

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..TF_utils import (Decoder, DecoderLayer, Encoder, EncoderDecoder,
                        EncoderLayer, GeneratorWithParallelHeads626,
                        LinearEmbedding, MultiHeadAttention,
                        PointerwiseFeedforward, PositionalEncoding,
                        SublayerConnection)


class STF(nn.Module):
    def __init__(self, cfg):
        super(STF, self).__init__()
        "Helper: Construct a model from hyperparameters."

        # Hyperparameters from cfg
        hist_inp_size = cfg['in_channels']
        lane_inp_size = cfg['enc_dim']
        num_queries = cfg['queries']
        dec_inp_size = cfg['queries_dim']
        dec_out_size = cfg['out_channels']
        # Hyperparameters predefined
        N = 2
        N_lane = 2
        N_social = 2
        d_model = 128
        d_ff = 256
        pos_dim = 64
        dist_dim = 128
        h = 2
        dropout = 0
        #

        self.aux_loss = cfg['aux_task']
        c = copy.deepcopy
        dropout_atten = dropout
        #dropout_atten = 0.1
        attn = MultiHeadAttention(h, d_model, dropout=dropout_atten)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.hist_tf = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(LinearEmbedding(hist_inp_size, d_model), c(position))
        )
        self.lane_enc = Encoder(EncoderLayer(
            d_model, c(attn), c(ff), dropout), N_lane)
        self.lane_dec = Decoder(DecoderLayer(
            d_model, c(attn), c(attn), c(ff), dropout), N_lane)
        self.lane_emb = LinearEmbedding(lane_inp_size, d_model)

        self.pos_emb = nn.Sequential(
            nn.Linear(2, pos_dim, bias=True),
            nn.LayerNorm(pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim, bias=True))
        self.dist_emb = nn.Sequential(
            nn.Linear(num_queries*d_model, dist_dim, bias=True),
            nn.LayerNorm(dist_dim),
            nn.ReLU(),
            nn.Linear(dist_dim, dist_dim, bias=True))

        self.fusion1 = nn.Sequential(
            nn.Linear(d_model+pos_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))
        self.fusion2 = nn.Sequential(
            nn.Linear(dist_dim+pos_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))
        self.social_enc = Encoder(EncoderLayer(
            d_model, c(attn), c(ff), dropout), N_social)
        self.social_dec = Decoder(DecoderLayer(
            d_model, c(attn), c(attn), c(ff), dropout), N_social)

        # self.g = Generator(d_model*2, dec_out_size)
        self.prediction_header = GeneratorWithParallelHeads626(
            d_model*2, dec_out_size, dropout)
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for name, param in self.named_parameters():
            # print(name)
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        self.query_embed = nn.Embedding(self.num_queries, d_model)
        self.query_embed.weight.requires_grad == False
        nn.init.orthogonal_(self.query_embed.weight)

    # input: [inp, dec_inp, src_att, trg_att]

    def forward(self, traj, pos, social_num, social_mask, lane_enc, lane_mask):
        '''
            Args:
                traj: [batch size, max_agent_num, 19, 4]
                pos: [batch size, max_agent_num, 2]
                social_num: float = max_agent_num
                social_mask: [batch size, 1, max_agent_num]
                lane_enc: [batch size, max_lane_num, 64]
                lane_mask: [batch size, 1, max_lane_num]

            Returns:
                outputs_coord: [batch size, max_agent_num, num_query, 30, 2]
                outputs_class: [batch size, max_agent_num, num_query]
        '''

        self.query_batches = self.query_embed.weight.view(
            1, 1, *self.query_embed.weight.shape).repeat(*traj.shape[:2], 1, 1)

        # Trajectory transfomer
        hist_out = self.hist_tf(traj, self.query_batches, None, None)
        pos = self.pos_emb(pos)
        hist_out = torch.cat([pos.unsqueeze(dim=2).repeat(
            1, 1, self.num_queries, 1), hist_out], dim=-1)
        hist_out = self.fusion1(hist_out)
        
        # Lane encoder
        lane_mem = self.lane_enc(self.lane_emb(lane_enc), lane_mask)
        lane_mem = lane_mem.unsqueeze(1).repeat(1, social_num, 1, 1)
        lane_mask = lane_mask.unsqueeze(1).repeat(1, social_num, 1, 1)
        
        # Lane decoder
        lane_out = self.lane_dec(hist_out, lane_mem, lane_mask, None)
        
        # Fuse position information
        dist = lane_out.view(*traj.shape[0:2], -1)
        dist = self.dist_emb(dist)
        
        # Social layer
        social_inp = self.fusion2(torch.cat([pos, dist], -1))
        social_mem = self.social_enc(social_inp, social_mask)
        social_out = social_mem.unsqueeze(
            dim=2).repeat(1, 1, self.num_queries, 1)
        out = torch.cat([social_out, lane_out], -1)

        # Prediction head
        outputs_coord, outputs_class = self.prediction_header(out)
  
        return outputs_coord, outputs_class
 

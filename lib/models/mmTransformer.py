import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit*2

    def forward(self, lane):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            lane: [batch size, max_lane_num, 9, 7] (vectorized representation)

        Returns:
            x_max: [batch size, max_lane_num, 64]
        '''
        x = lane
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class mmTrans(nn.Module):

    def __init__(self, stacked_transformer, cfg):
        super(mmTrans, self).__init__()
        # stacked transformer class
        self.stacked_transformer = stacked_transformer(cfg)

        lane_channels = cfg['lane_channels']
        self.hist_feature_size = cfg['in_channels']

        self.polyline_vec_shape = 2*cfg['subgraph_width']
        self.subgraph = LaneNet(
            lane_channels, cfg['subgraph_width'], cfg['num_subgraph_layres'])

        self.FUTURE_LEN = cfg['future_num_frames']
        self.OBS_LEN = cfg['history_num_frames'] - 1
        self.lane_length = cfg['lane_length']

    def preprocess_traj(self, traj):
        '''
            Generate the trajectory mask for all agents (including target agent)

            Args:
                traj: [batch, max_agent_num, obs_len, 4]

            Returns:
                social mask: [batch, 1, max_agent_num]

        '''
        # social mask
        social_valid_len = self.traj_valid_len
        social_mask = torch.zeros(
            (self.B, 1, int(self.max_agent_num))).to(traj.device)
        for i in range(self.B):
            social_mask[i, 0, :social_valid_len[i]] = 1

        return social_mask

    def preprocess_lane(self, lane):
        '''
            preprocess lane segments using LaneNet

        Args:
            lane: [batch size, max_lane_num, 10, 5]

        Returns:
            lane_feature: [batch size, max_lane_num, 64 (feature_dim)]
            lane_mask: [batch size, 1, max_lane_num]

        '''

        # transform lane to vector
        lane_v = torch.cat(
            [lane[:, :, :-1, :2],
             lane[:, :, 1:, :2],
             lane[:, :, 1:, 2:]], dim=-1)  # bxnlinex9x7

        # lane mask
        lane_valid_len = self.lane_valid_len
        lane_mask = torch.zeros(
            (self.B, 1, int(self.max_lane_num))).to(lane_v.device)
        for i in range(lane_valid_len.shape[0]):
            lane_mask[i, 0, :lane_valid_len[i]] = 1

        # use vector like structure process lane
        lane_feature = self.subgraph(lane_v)  # [batch size, max_lane_num, 64]

        return lane_feature, lane_mask

    def forward(self, data: dict):
        """
        Args:
            data (Data): 
                HIST: [batch size, max_agent_num, 19, 4]
                POS: [batch size, max_agent_num, 2]
                LANE: [batch size, max_lane_num, 10, 5]
                VALID_LEN: [batch size, 2] (number of valid agents & valid lanes)

        Note:
            max_lane_num/max_agent_num indicates maximum number of agents/lanes after padding in a single batch 
        """
        # initialized
        self.B = data['HISTORY'].shape[0]

        self.traj_valid_len = data['VALID_LEN'][:, 0]
        self.max_agent_num = torch.max(self.traj_valid_len)

        self.lane_valid_len = data['VALID_LEN'][:, 1]
        self.max_lane_num = torch.max(self.lane_valid_len)

        # preprocess
        pos = data['POS']
        trajs = data['HISTORY']
        social_mask = self.preprocess_traj(data['HISTORY'])
        lane_enc, lane_mask = self.preprocess_lane(data['LANE'])

        out = self.stacked_transformer(trajs, pos, self.max_agent_num,
                                       social_mask, lane_enc, lane_mask)

        return out

from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import numpy as np
import torch

from .utils import get_heading_angle, transform_coord


class VectorizedCase(object):

    def __init__(self, cfg):

        self.striaghten = True
        self.max_agent_num = 68
        self.pad = False

    def get_straighten_angle(self, features):
        '''
            agent_features: [20,5]
            -------------Calculate-Angle-------------------
            trajs which feed into func must satisfy following condition:
            1. long enough (l > 2m)
            2. have same direction
        '''
        agent_features = features[0]

        ct = 19 - 6
        coord1, coord2 = agent_features[ct, :2], agent_features[19, :2]
        traj_dir = agent_features[-1, :2] - agent_features[0, :2]
        current_dir = coord2 - coord1
        while (np.linalg.norm(coord1-coord2, ord=2) < 2 or (current_dir*traj_dir).sum() < 0) and ct > 0:
            ct -= 1
            coord1 = agent_features[ct, :2]
            current_dir = coord2 - coord1

        theta = get_heading_angle(agent_features[ct:, :2])

        return theta

    def get_history_traj(self, features, theta):
        '''
            features: trajectory features with size (number of agents, history frame num, 5)
            Notes: index 0 of axis 0 is the target agent.  
        '''

        num_agent = features.shape[0]
        features = features[..., :4]

        if self.striaghten:
            features = features.reshape(-1, 4)
            features[:, :2] = transform_coord(features[:, :2], theta)
            features = features.reshape(num_agent, 20, 4)

        v = features[:, 1:, :2] - features[:, :-1, :2]  # na, 19, 2
        ts = (features[:, 1:, 2] + features[:, :-1, 2])/2  # na, 19
        mask = features[:, 1:, 3]*features[:, :-1, 3]  # 1,1 =>1; 1,0 =>0; 0,0=>0

        hist_traj = np.concatenate(
            [v, ts.reshape(-1, 19, 1), mask.reshape(-1, 19, 1)], -1)
        pos = features[:, -1, :2]
        assert hist_traj.shape == (num_agent, 19, 4)
        assert pos.shape == (num_agent, 2)

        if self.pad:
            # padding data
            hist_traj = np.pad(
                hist_traj, ((0, self.max_agent_num - num_agent), (0, 0), (0, 0)), "constant")
            pos = np.pad(
                pos, ((0, self.max_agent_num - num_agent), (0, 0)), "constant")

        return dict(
            HISTORY=hist_traj,
            POS=pos,
        )

    def get_future_traj(self, features, pos, theta):
        '''
            pos: nbr2target_translate (n_agent, 2)
            features: trajectory features with size (number of agents, history frame num, 3)
            Notes: index 0 of axis 0 is the target agent.  
        '''

        n_agents = features.shape[0]

        if self.striaghten:
            features = features.reshape(-1, 3)
            features[:, :2] = transform_coord(features[:, :2], theta)
            features = features.reshape(-1, 30, 3)

        v = np.concatenate([(features[:, 0, :2] - pos).reshape(-1, 1, 2),
                           features[:, 1:, :2]-features[:, :-1, :2]], 1)
        mask = features[:, :, 2].reshape(-1, 30, 1)
        future_traj = np.concatenate([v, mask], axis=-1)

        assert future_traj.shape == (n_agents, 30, 3)

        if self.pad:
            future_traj = np.pad(
                future_traj, ((0, self.max_agent_num-n_agents), (0, 0), (0, 0)), "constant")

        return dict(FUTURE=future_traj,)

    def process_case(self, data):
        '''
            vectorized each cases
            data: [
                FEATURE
                GT
                LANE
            ]

            out:
            ["HIST", "FUTURE", "POS", "VALID_AGENT","VALID_AGENT","LANE","MAX_LEN","THETA", "NAME"]

        '''

        theta = self.get_straighten_angle(data['HISTORY'])
        data['THETA'] = theta

        # --------------Trajectory------------------------------------------------------

        # Histroy traj
        hist_dict = self.get_history_traj(data['HISTORY'], data['THETA'])
        data.update(hist_dict)

        # Future traj
        future_dict = self.get_future_traj(data['FUTURE'], data['POS'], data['THETA'])
        data.update(future_dict)

        return data

    def transform_coord(self, coords, angle):
        x = coords[:, 0]
        y = coords[:, 1]
        x_transform = math.cos(angle)*x-math.sin(angle)*y
        y_transform = math.cos(angle)*y+math.sin(angle)*x
        output_coords = np.stack((x_transform, y_transform), axis=-1)
        return output_coords

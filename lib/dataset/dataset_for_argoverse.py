import math
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

# Tensorizerization & Vectorization
# from .argoverse_convertor import ArgoverseConvertor
# Utilities
from .utils import transform_coord, transform_coord_flip
from .collate import collate_single_cpu


class STFDataset(Dataset):
    """
        dataset object similar to `torchvision` 
    """

    def __init__(self, cfg: dict):
        super(STFDataset, self).__init__()
        self.cfg = cfg

        self.processed_data_path = cfg['processed_data_path']
        self.processed_maps_path = cfg['processed_maps_path']

        # self.traj_processor = ArgoverseConvertor(cfg['traj_processor_cfg'])

        # Load lane data
        with open(self.processed_maps_path, 'rb') as f:
            self.map, self.lane_id2idx = pickle.load(f)

        # Load processed trajs, land id and Misc.
        with open(self.processed_data_path, 'rb') as f:
            self.data = pickle.load(f)

        # get data list
        self.data_list = sorted(self.data.keys())

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        '''
            Returns:
                the shape in here you can refer the format sheet at [here](./README.md)
        '''

        name = self.data_list[idx]
        data_dict = {'NAME': name, 'MAX_LEN': [68, 248], }
        data_dict.update(self.get_data(name))

        return data_dict

    @classmethod
    def get_data_path_ls(cls, dir_):
        return [os.path.join(dir_, data_path) for data_path in os.listdir(dir_)]

    def get_data(self, name):
        '''
            the file name of the case

            Since we have processed the case.

            this function only needs to retrieve the lanes.
        '''
        out_dict = {}

        # load from pkl
        datadict = self.data[name]
        out_dict.update(datadict)

        # ----- LANE --------------
        lane = self.get_lane(
            datadict['LANE_ID'], datadict['THETA'], datadict['NORM_CENTER'], datadict['CITY_NAME'])

        out_dict.update(dict(LANE=lane))

        for k, v in out_dict.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            
                if v.dtype == torch.double:
                    v = v.type(torch.float32)
            
                out_dict[k] = v

        return out_dict

    def get_lane(self, lane_id, theta, center, city):
        '''
            Args:
                lane_id: [lane_num]
                center: [2]
                theta: float
                city: str

                self.map the preprocess map data
                    : Dict[city name, List[]]

            Returns:
                lane_feature: num_lane, 10, 5
        '''

        # Get lane
        # lane_feature: num_lane, 10, 5
        lane_id2idx = self.lane_id2idx[city]
        idx = list(map(lambda x: lane_id2idx[x], lane_id))
        lane_feature = self.map[city][idx].copy()  # (nline, 10, 5)
        lane = lane_feature[:, :, :2]

        # Location normalization
        lane = lane - center
        lane = transform_coord(lane, theta)
        lane_feature[:, :, :2] = lane

        return lane_feature


if __name__ == '__main__':

    import argparse
    from config.Config import Config
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(
        description='Preprocess argoverse dataset')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    validation_cfg = cfg.get('val_dataset')
    val_dataset = STFDataset(validation_cfg)
    val_dataloader = DataLoader(val_dataset,
                                shuffle=validation_cfg["shuffle"],
                                batch_size=validation_cfg["batch_size"],
                                num_workers=validation_cfg["workers_per_gpu"],
                                collate_fn=collate_single_cpu)

    val_dataloader = iter(val_dataloader)
    DATA = next(val_dataloader)
    import ipdb
    ipdb.set_trace()

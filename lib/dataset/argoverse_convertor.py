import os
import pickle
import re
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from argoverse.data_loading.argoverse_forecasting_loader import \
    ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from tqdm import tqdm

from .preprocess_utils.feature_utils import compute_feature_for_one_seq, save_features
from .preprocess_utils.map_utils_vec import save_map

# vectorization
from .vectorization import VectorizedCase


class ArgoverseConvertor(object):

    def __init__(self, cfg):

        self.data_dir = cfg['DATA_DIR']
        self.obs_len = cfg['OBS_LEN']
        self.lane_radius = cfg['LANE_RADIUS']
        self.object_radius = cfg['OBJ_RADIUS']
        self.raw_dataformat = cfg['RAW_DATA_FORMAT']
        self.am = ArgoverseMap()
        self.Afl = ArgoverseForecastingLoader
        self.out_dir = cfg['INTERMEDIATE_DATA_DIR']
        self.save_dir_pretext = cfg['info_prefix']
        self.specific_data_fold_list = cfg['specific_data_fold_list']

        # vectorization
        self.vec_processor = VectorizedCase(cfg['vectorization_cfg'])

    def preprocess_map(self):
        
        os.makedirs(self.out_dir, exist_ok=True)
        if not os.path.exists(os.path.join(self.out_dir, 'map.pkl')):
            print("Processing maps ...")
            save_map(self.out_dir)
            print('Map is save at '+ os.path.join(self.out_dir, 'map.pkl'))

    def process(self,):

        # preprocess the map
        self.preprocess_map()

        # storage the case infomation
        data_info = {}

        for folder in os.listdir(self.data_dir):

            if folder not in self.specific_data_fold_list:
                continue

            afl = self.Afl(os.path.join(self.data_dir, folder, 'data'))
            info_dict = {}
            data_info[folder] = {}

            for path_name_ext in tqdm(afl.seq_list):

                afl_ = afl.get(path_name_ext)
                path, name_ext = os.path.split(path_name_ext)
                name, ext = os.path.splitext(name_ext)

                info_dict[name] = self.process_case(afl_.seq_df)

            out_path = os.path.join(
                self.out_dir,  self.save_dir_pretext + f'{folder}.pkl')
            with open(out_path, 'wb') as f:
                pickle.dump(info_dict, f, pickle.HIGHEST_PROTOCOL)

            data_info[folder]['sample_num'] = len(afl.seq_list)
            print('Data is save at ' + out_path)

        # print info
        print("Finish Preprocessing.")
        for k in data_info.keys():
            print('dataset name: ' + k +
                  '\n sample num: {}'.format(data_info[k]['sample_num']))

    def preprocess_case(self, seq_df):
        '''
            Args:
                seq_df: 

        '''
        # retrieve info from csv
        agent_feature, obj_feature_ls, nearby_lane_ids, norm_center, city_name =\
            compute_feature_for_one_seq(
                seq_df,
                self.am,
                self.obs_len,
                self.lane_radius,
                self.object_radius,
                self.raw_dataformat,
                viz=False,
                mode='nearby'
            )

        # pack as the output
        dic = save_features(
            agent_feature, obj_feature_ls, nearby_lane_ids, norm_center, city_name
        )

        return dic

    def process_case(self, seq_df):

        # tensorized
        data = self.preprocess_case(seq_df)
        # vectorized
        vec_dic = self.vec_processor.process_case(data)

        return vec_dic


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description='Preprocess argoverse dataset')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    from config.Config import Config
    cfg = Config.fromfile(args.config)
    preprocess_cfg = cfg.get('preprocess_dataset')
    processor = ArgoverseConvertor(preprocess_cfg)
    processor.process()

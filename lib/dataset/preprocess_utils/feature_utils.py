import os
from typing import List

import numpy as np
import pandas as pd
from argoverse.data_loading.argoverse_forecasting_loader import \
    ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap

from .agent_utils import get_agent_feature_ls
from .lane_utils import get_nearby_lane_feature_ls
from .object_utils import get_nearby_moving_obj_feature_ls


def compute_feature_for_one_seq(
        traj_df: pd.DataFrame,
        am: ArgoverseMap,
        obs_len: int = 20,
        lane_radius: int = 5,
        obj_radius: int = 10,
        raw_dataformat: dict = None,
        viz: bool = False,
        mode='nearby',
        query_bbox=[-65, 65, -65, 65]) -> List[List]:
    """
    return lane & track features
    args:
        mode: 'rect' or 'nearby'
    returns:
        agent_feature_ls:
            list of target agent
        obj_feature_ls:
            list of (list of nearby agent feature)
        lane_feature_ls:
            list of (list of lane segment feature)

        norm_center np.ndarray: (2, )
    """
    # normalize timestamps
    traj_df['TIMESTAMP'] -= np.min(traj_df['TIMESTAMP'].values)
    seq_ts = np.unique(traj_df['TIMESTAMP'].values)

    city_name = traj_df['CITY_NAME'].iloc[0]
    agent_df = None
    agent_x_end, agent_y_end, start_x, start_y, query_x, query_y, norm_center = [
        None] * 7

    # agent traj & its start/end point
    for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):
        # sorted already according to timestamp
        if obj_type == 'AGENT':
            agent_df = remain_df
            start_x, start_y = agent_df[['X', 'Y']].values[0]
            agent_x_end, agent_y_end = agent_df[['X', 'Y']].values[-1]
            query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
            norm_center = np.array([query_x, query_y])
            break
        else:
            raise ValueError(f"cannot find 'agent' object type")

    # get agent features
    agent_feature = get_agent_feature_ls(agent_df, obs_len, norm_center)
    hist_xy = agent_feature[0]
    hist_len = np.sum(np.sqrt(
        (hist_xy[1:, 0]-hist_xy[:-1, 0])**2 + (hist_xy[1:, 1]-hist_xy[:-1, 1])**2))

    # search lanes from the last observed point of agent
    nearby_lane_ids = get_nearby_lane_feature_ls(
        am, agent_df, obs_len, city_name, lane_radius, norm_center, mode=mode, query_bbox=query_bbox)

    # search nearby moving objects from the last observed point of agent
    obj_feature_ls = get_nearby_moving_obj_feature_ls(
        agent_df, traj_df, obs_len, seq_ts, obj_radius, norm_center, raw_dataformat)

    return [agent_feature, obj_feature_ls, nearby_lane_ids, norm_center, city_name]


def save_features(agent_feature, obj_feature_ls, nearby_lane_ids, norm_center, city_name):
    """
    args:
        agent_feature_ls:
            list of (xys, ts, agent_df['TRACK_ID'].iloc[0], gt_xys)
        obj_feature_ls:
            list of list of (xys, ts, mask, track_id, gt_xys, gt_mask)
        lane_feature_ls:
            list of list of lane a segment feature, centerline, lane_info1, lane_info2, lane_id
    returns:
        Dict[]
    """
    nbrs_nd = np.empty((0, 4))
    nbrs_gt = np.empty((0, 3))
    lane_nd = np.empty((0, 7))

    # agent features
    # input: xy,ts,mask
    agent_len = agent_feature[0].shape[0]
    agent_nd = np.hstack(
        (agent_feature[0], agent_feature[1].reshape((-1, 1)), np.ones((agent_len, 1))))
    assert agent_nd.shape[1] == 4, "agent_traj feature dim 1 is not correct"
    # gt: xy, mask
    gt_len = agent_feature[-1].shape[0]
    agent_gt = np.hstack((agent_feature[-1], np.ones((gt_len, 1))))
    assert agent_gt.shape[1] == 3

    # obj features
    # input: xy,ts,mask
    # gt: xy, mask
    if(len(obj_feature_ls) > 0):
        for obj_feature in obj_feature_ls:
            obj_len = obj_feature[0].shape[0]
            obj_nd = np.hstack((obj_feature[0], obj_feature[1].reshape(
                (-1, 1)), obj_feature[2].reshape((-1, 1))))
            assert obj_nd.shape[1] == 4, "obj_traj feature dim 1 is not correct"
            nbrs_nd = np.vstack([nbrs_nd, obj_nd])

            gt_len = obj_feature[4].shape[0]
            obj_gt = np.hstack(
                (obj_feature[4], obj_feature[5].reshape((-1, 1))))
            assert obj_gt.shape[1] == 3, "obj_gt feature dim 1 is not correct"
            nbrs_gt = np.vstack([nbrs_gt, obj_gt])
        # nbrs_nd [nbrs_num,20,4]
        nbrs_nd = nbrs_nd.reshape([-1, 20, 4])
        # nbrs_gt [nbrs_num,30,3]
        nbrs_gt = nbrs_gt.reshape([-1, 30, 3])

    # matrix of all agents
    if(len(obj_feature_ls)>0):
        all_agents_nd = np.concatenate([agent_nd.reshape(1,-1,4),nbrs_nd])
        all_agents_gt = np.concatenate([agent_gt.reshape(1,-1,3),nbrs_gt])
    else:
        all_agents_nd = agent_nd.reshape(1,-1,4)
        all_agents_gt = agent_gt.reshape(1,-1,3)

    # lane ids: (large integer)
    lane_id = np.array(nearby_lane_ids)

    # saving
    dic = {
        "HISTORY": all_agents_nd.astype(np.float32),
        "FUTURE": all_agents_gt.astype(np.float32),
        "LANE_ID": lane_id.astype(np.int32),
        "NORM_CENTER": norm_center.astype(np.float32),
        "VALID_LEN": np.array((len(all_agents_nd), len(lane_id))),
        "CITY_NAME": city_name
    }

    return dic

import os

import numpy as np


# Only support nearby in our implementation
def get_nearby_lane_feature_ls(am, agent_df, obs_len, city_name, lane_radius, norm_center, has_attr=False, mode='nearby', query_bbox=None):
    '''
    compute lane features
    args:
        norm_center: np.ndarray
        mode: 'nearby' return nearby lanes within the radius; 'rect' return lanes within the query bbox
        **kwargs: query_bbox= List[int, int, int, int]
    returns:
        list of list of lane a segment feature, formatted in [centerline, is_intersection, turn_direction, is_traffic_control, lane_id,
         predecessor_lanes, successor_lanes, adjacent_lanes]
    '''
    query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
    nearby_lane_ids = am.get_lane_ids_in_xy_bbox(
        query_x, query_y, city_name, lane_radius)

    return nearby_lane_ids

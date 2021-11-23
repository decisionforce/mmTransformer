import os
import pickle

import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap


def save_map(dir_, name = f"map.pkl"):
    am = ArgoverseMap()
    lane_dict = am.build_centerline_index()

    # go through each lane segment
    dic = {"PIT": [], "MIA": []}
    lane_id2idx =  {"PIT": {}, "MIA": {}}
    for city_name in ["PIT", "MIA"]:
        
        for i, lane_id in enumerate(lane_dict[city_name].keys()):
            # extract from API
            lane_cl = am.get_lane_segment_centerline(lane_id, city_name)
            centerline = lane_cl[:, :2]
            is_intersection = am.lane_is_in_intersection(lane_id, city_name)
            turn_direction = am.get_lane_turn_direction(lane_id, city_name)
            traffic_control = am.lane_has_traffic_control_measure(
                lane_id, city_name)
            lane_info1 = 1
            if(is_intersection):
                lane_info1 = 2
            lane_info2 = 1
            if(turn_direction == "LEFT"):
                lane_info2 = 2
            elif(turn_direction == "RIGHT"):
                lane_info2 = 3
            lane_info3 = 1
            if(traffic_control):
                lane_info3 = 2
            
            lane_len = lane_cl.shape[0]

            # there 61 lane is not enough for size 10
            if lane_len < 10:
                lane_cl = np.pad(
                  lane_cl, ((0, 10-lane_len), (0, 0)), "edge")
                lane_len = 10

            lane_nd = np.concatenate(
                [lane_cl[:, :2],
                    np.ones((lane_len, 1)) * lane_info1,
                    np.ones((lane_len, 1)) * lane_info2,
                    np.ones((lane_len, 1)) * lane_info3], axis=-1)
            dic[city_name].append(lane_nd)
            lane_id2idx[city_name][lane_id] = i

        dic[city_name] = np.stack(dic[city_name], axis=0)

    # saving
    with open(os.path.join(dir_, name), 'wb') as f:
        pickle.dump([dic, lane_id2idx], f, pickle.HIGHEST_PROTOCOL)

import numpy as np


def get_agent_feature_ls(agent_df, obs_len, norm_center):
    """
    args:
    returns: 
        list of (track, timetamp, track_id, gt_track)
    """
    xys, gt_xys = agent_df[["X", "Y"]].values[:obs_len], agent_df[[
        "X", "Y"]].values[obs_len:]
    xys -= norm_center  # normalize to last observed timestamp point of agent
    gt_xys -= norm_center  # normalize to last observed timestamp point of agent
    ts = agent_df['TIMESTAMP'].values[:obs_len]

    return [xys, ts, agent_df['TRACK_ID'].iloc[0], gt_xys]

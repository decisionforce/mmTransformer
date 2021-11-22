import math
import numpy as np
from sklearn.linear_model import LinearRegression


def get_heading_angle(traj: np.ndarray):
    """
        get the heading angle 
        traj: [N,2] N>=6
    """
    # length == 6
    # sort position
    _traj = traj.copy()
    traj = traj.copy()

    traj = traj[traj[:, 0].argsort()]
    traj = traj[traj[:, 1].argsort()]

    if traj.T[0].max()-traj.T[0].min() > traj.T[1].max()-traj.T[1].min():  # * dominated by x
        reg = LinearRegression().fit(traj[:, 0].reshape(-1, 1), traj[:, 1])
        traj_dir = _traj[-2:].mean(0) - _traj[:2].mean(0)
        reg_dir = np.array([1, reg.coef_[0]])
        angle = np.arctan(reg.coef_[0])
    else:
        # using y as sample and x as the target to fit a line
        reg = LinearRegression().fit(traj[:, 1].reshape(-1, 1), traj[:, 0])
        traj_dir = _traj[-2:].mean(0) - _traj[:2].mean(0)
        reg_dir = np.array([reg.coef_[0], 1])*np.sign(reg.coef_[0])
        if reg.coef_[0] == 0:
            import pdb
            pdb.set_trace()
        angle = np.arctan(1/reg.coef_[0])

    if angle < 0:
        angle = 2*np.pi + angle
    if (reg_dir*traj_dir).sum() < 0:  # not same direction
        angle = (angle+np.pi) % (2*np.pi)
    # angle from y
    angle_to_y = angle-np.pi/2
    angle_to_y = -angle_to_y
    return angle_to_y


def transform_coord(coords, angle):
    x = coords[..., 0]
    y = coords[..., 1]
    x_transform = np.cos(angle)*x-np.sin(angle)*y
    y_transform = np.cos(angle)*y+np.sin(angle)*x
    output_coords = np.stack((x_transform, y_transform), axis=-1)
    
    return output_coords


def transform_coord_flip(coords, angle):
    x = coords[:, 0]
    y = coords[:, 1]
    x_transform = math.cos(angle)*x-math.sin(angle)*y
    y_transform = math.cos(angle)*y+math.sin(angle)*x
    x_transform = -1*x_transform  # flip
    # y_transform = -1*y_transform # flip
    output_coords = np.stack((x_transform, y_transform), axis=-1)
    return output_coords

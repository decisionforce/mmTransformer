import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

padding_keys = ['HISTORY', 'LANE', 'POS']
stacking_keys = ['VALID_LEN']
listing_keys = ['CITY_NAME', 'NAME', 'FUTURE',
                'LANE_ID', 'THETA', 'NORM_CENTER']


def collate_single_cpu(batch):
    """
        We only pad the HISTORY and LANE data.
        For other data, we append data with same key into a list.
    """

    keys = batch[0].keys()

    out = {k: [] for k in keys}

    for data in batch:
        for k, v in data.items():
            out[k].append(v)

    # stacking
    for k in stacking_keys:
        out[k] = torch.stack(out[k], dim=0)

    # padding
    for k in padding_keys:
        out[k] = pad_sequence(out[k], batch_first=True)

    return out

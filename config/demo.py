# cfg for preprocess
preprocess_dataset = dict(
    RAW_DATA_FORMAT={
        "TIMESTAMP": 0,
        "TRACK_ID": 1,
        "OBJECT_TYPE": 2,
        "X": 3,
        "Y": 4,
        "CITY_NAME": 5,
    },
    LANE_WIDTH={'MIA': 3.84, 'PIT': 3.97},
    # to be considered as static
    VELOCITY_THRESHOLD=0.0,
    # number of timesteps the track should exist to be considered in social context
    EXIST_THRESHOLD=(5),
    # index of the sorted velocity to look at, to call it as stationary
    STATIONARY_THRESHOLD=(13),
    LANE_RADIUS=65,  # nearby lanes
    OBJ_RADIUS=56,  # nearby objects
    OBS_LEN=20,
    DATA_DIR='./data',
    INTERMEDIATE_DATA_DIR='./interm_data',
    info_prefix='argoverse_info_',
    VIS=False,
    # sepecify which fold in data dir will be processed
    specific_data_fold_list = ['train','val','test','sample'],
    vectorization_cfg = dict(
        starighten = True,
    )
)


model = dict(
    type='stacked_transformer',
    history_num_frames= 20,
    future_num_frames= 30,
    # mode setting
    in_channels= 4,
    lane_channels= 7,
    out_channels= 60, #future_frame*2 !!!!!!!!!!!!! should change with num frame
    K= 6,
    increasetime= 3,
    queries= 6,
    num_guesses= 6,
    queries_dim= 64,
    enc_dim= 64,
    aux_task= False,


    #mmTrans main cfg
    subgraph_width = 32,
    num_subgraph_layres =2,
    lane_length = 10,
)


dataset = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    traj_processor_cfg=preprocess_dataset,
)

from copy import deepcopy

train_dataset= deepcopy(dataset)
train_dataset.update(dict(
        type= "STFDataset",
        batch_size= 128,
        shuffle= True,
        num_workers= 4,
        Providing_GT= True,
        lane_length= 10,
        dataset_path= './data/train',
        processed_data_path= './interm_data/argoverse_info_train.pkl',
        processed_maps_path='./interm_data/map.pkl',
    ))

val_dataset= deepcopy(dataset)
val_dataset.update(dict(
        type= "STFDataset",
        batch_size= 32,
        shuffle= False,
        Providing_GT= True,
        lane_length= 10,
        dataset_path= './data/val',
        processed_data_path= './interm_data/argoverse_info_val.pkl',
        processed_maps_path='./interm_data/map.pkl'
        )
    )

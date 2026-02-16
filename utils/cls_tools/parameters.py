# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    # print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,     # To do quick test. Trains/test on small subset of dataset, and # of epochs
    
        finetune_mode = False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights = '...', 

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir = 'C:/data/2024DCASE_data/',

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir = './data/feature_labels_2023',

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='mic',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        # max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite = 50,
        fmax_doa_salsalite = 2000,
        fmax_spectra_salsalite = 9000,
        label_sequence_length = 100, # Number of label frames in a training sample. Each label frame corresponds to label_hop_len_s seconds.

        # MODEL TYPE
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # SPATIAL MAP
        spatial_map=False,

        # METRIC
        average = 'macro',        # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20,
        evaluate_distance = True,
        segment_based_metrics = False,
        lad_dist_thresh=float('inf'),    # Absolute distance error threshold for computing the detection metrics
        lad_reldist_thresh=float('1'),  # Relative distance error threshold for computing the detection metrics
    )

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['unique_classes'] = 13


    # Raw Audio Chunks
    params['label_sequence_length'] = 1 # use only one time frame for tdoa training
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['raw_chunks'] = True
    params['quick_test'] = False
    params['dataset'] = 'mic'
    params['use_salsalite'] = False
    params['multi_accdoa'] = False
    params['n_mics'] = 4
    params['ngcc_channels'] = 32
    params['ngcc_out_channels'] = 16 
    params['saved_chunks'] = True
    params['use_mel'] = False
    params['nb_epochs'] = 1
    params['lambda'] = 1.0 # set to 1.0 to only train tdoa, and 0.0 to only train SELD
    params['tracks'] = 3

    return params

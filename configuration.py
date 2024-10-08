# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from data_utils import InputTypes, DataTypes, FeatureSpec
import datetime

import attention as ATTN

class ElectricityConfig:
    def __init__(self):

        self.features = [
                         FeatureSpec('id', InputTypes.ID, DataTypes.CATEGORICAL),
                         FeatureSpec('hours_from_start', InputTypes.TIME, DataTypes.CONTINUOUS),
                         FeatureSpec('power_usage', InputTypes.TARGET, DataTypes.CONTINUOUS),
                         FeatureSpec('hour', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('day_of_week', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('hours_from_start', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('categorical_id', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        ]
        # Dataset split boundaries
        self.time_ids = 'days_from_start' # This column contains time indices across which we split the data
        self.train_range = (1096, 1315)
        self.valid_range = (1308, 1339)
        self.test_range = (1332, 1346)
        self.dataset_stride = 1 #how many timesteps between examples
        self.scale_per_id = True
        self.missing_id_strategy = None
        self.missing_cat_data_strategy='encode_all'

        # Feature sizes
        self.static_categorical_inp_lens = [369]
        self.temporal_known_categorical_inp_lens = []
        self.temporal_observed_categorical_inp_lens = []
        self.quantiles = [0.1, 0.5, 0.9]

        self.example_length = 8 * 24
        self.encoder_length = 7 * 24

        self.n_head = 4
        self.hidden_size = 160
        self.dropout = 0.1
        self.attn_dropout = 0.1

        #### Derived variables ####
        self.temporal_known_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                      self.temporal_observed_continuous_inp_size,
                                      self.temporal_target_size,
                                      len(self.temporal_observed_categorical_inp_lens),
                                      ])


class TrafficConfig:
    def __init__(self):

        self.features = [
                         FeatureSpec('id', InputTypes.ID, DataTypes.CATEGORICAL),
                         FeatureSpec('hours_from_start', InputTypes.TIME, DataTypes.CONTINUOUS),
                         FeatureSpec('values', InputTypes.TARGET, DataTypes.CONTINUOUS),
                         FeatureSpec('time_on_day', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('day_of_week', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('hours_from_start', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('categorical_id', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        ]
        # Dataset split boundaries
        self.time_ids = 'sensor_day' # This column contains time indices across which we split the data
        self.train_range = (0, 151)
        self.valid_range = (144, 166)
        self.test_range = (159, float('inf'))
        self.dataset_stride = 1 #how many timesteps between examples
        self.scale_per_id = True
        self.missing_id_strategy = None
        self.missing_cat_data_strategy='encode_all'

        # Feature sizes
        self.static_categorical_inp_lens = [963]
        self.temporal_known_categorical_inp_lens = []
        self.temporal_observed_categorical_inp_lens = []
        self.quantiles = [0.1, 0.5, 0.9]

        self.example_length = 8 * 24
        self.encoder_length = 7 * 24

        self.n_head = 4
        self.hidden_size = 320
        self.dropout = 0.3
        self.attn_dropout = 0.3

        #### Derived variables ####
        self.temporal_known_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                      self.temporal_observed_continuous_inp_size,
                                      self.temporal_target_size,
                                      len(self.temporal_observed_categorical_inp_lens),
                                      ])


class VolatilityConfig:
    def __init__(self):

        self.features = [
                         FeatureSpec('Symbol', InputTypes.ID, DataTypes.CATEGORICAL),
                         FeatureSpec('date', InputTypes.TIME, DataTypes.DATE),
                         FeatureSpec('log_vol', InputTypes.TARGET, DataTypes.CONTINUOUS),
                         FeatureSpec('open_to_close', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
                         FeatureSpec('days_from_start', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('day_of_week', InputTypes.KNOWN, DataTypes.CATEGORICAL),
                         FeatureSpec('day_of_month', InputTypes.KNOWN, DataTypes.CATEGORICAL),
                         FeatureSpec('week_of_year', InputTypes.KNOWN, DataTypes.CATEGORICAL),
                         FeatureSpec('month', InputTypes.KNOWN, DataTypes.CATEGORICAL),
                         FeatureSpec('Region', InputTypes.STATIC, DataTypes.CATEGORICAL)
                        ]
        # Dataset split boundaries
        self.time_ids = 'year' # This column contains time indices across which we split the data
        self.train_range = (2000, 2016)
        self.valid_range = (2016, 2018)
        self.test_range = (2018, float('inf'))
        self.dataset_stride = 1 #how many timesteps between examples
        self.scale_per_id = True
        self.missing_id_strategy = None
        self.missing_cat_data_strategy='encode_all'

        # Feature sizes
        self.static_categorical_inp_lens = [4]
        self.temporal_known_categorical_inp_lens = [7, 31, 53, 12]
        self.temporal_observed_categorical_inp_lens = []
        self.quantiles = [0.1, 0.5, 0.9]

        self.example_length = 257
        self.encoder_length = 252

        self.n_head = 1
        self.hidden_size = 160
        self.dropout = 0.3
        self.attn_dropout = 0.3

        #### Derived variables ####
        self.temporal_known_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                      self.temporal_observed_continuous_inp_size,
                                      self.temporal_target_size,
                                      len(self.temporal_observed_categorical_inp_lens),
                                      ])

class FavoritaConfig:
    def __init__(self):

        self.features = [
                        FeatureSpec('traj_id', InputTypes.ID, DataTypes.CATEGORICAL),
                        FeatureSpec('date', InputTypes.TIME, DataTypes.DATE),
                        FeatureSpec('log_sales', InputTypes.TARGET, DataTypes.CONTINUOUS),
                        FeatureSpec('onpromotion', InputTypes.KNOWN, DataTypes.CATEGORICAL),
                        FeatureSpec('transactions', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
                        FeatureSpec('oil', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
                        FeatureSpec('day_of_week', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                        FeatureSpec('day_of_month', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                        FeatureSpec('month', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                        FeatureSpec('national_hol', InputTypes.KNOWN, DataTypes.CATEGORICAL),
                        FeatureSpec('regional_hol', InputTypes.KNOWN, DataTypes.CATEGORICAL),
                        FeatureSpec('local_hol', InputTypes.KNOWN, DataTypes.CATEGORICAL),
                        FeatureSpec('open', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                        FeatureSpec('item_nbr', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        FeatureSpec('store_nbr', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        FeatureSpec('city', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        FeatureSpec('state', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        FeatureSpec('type', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        FeatureSpec('cluster', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        FeatureSpec('family', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        FeatureSpec('class', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        FeatureSpec('perishable', InputTypes.STATIC, DataTypes.CATEGORICAL)
                        ]
        # Dataset split boundaries
        self.time_ids = 'date' # This column contains time indices across which we split the data
        self.train_range = (datetime.datetime(2015, 1, 1),
                            datetime.datetime(2015, 12, 1))
        self.valid_range = (datetime.datetime(2015, 12, 1),
                            datetime.datetime(2015, 12, 31))
        self.test_range = (datetime.datetime(2015, 12, 31),
                           datetime.datetime(2016, 1, 31))
        self.dataset_stride = 1 #how many timesteps between examples
        self.scale_per_id = True
        self.missing_id_strategy = None
        self.missing_cat_data_strategy='encode_all'

        # Feature sizes
        self.static_categorical_inp_lens = [6, 25, 220, 2, 2, 7, 55, 4, 10]
        self.temporal_known_categorical_inp_lens = [9, 6, 5, 2]
        self.temporal_observed_categorical_inp_lens = []
        self.quantiles = [0.1, 0.5, 0.9]

        self.example_length = 120
        self.encoder_length = 90

        self.n_head = 4
        self.hidden_size = 240
        self.dropout = 0.1
        self.attn_dropout = 0.1

        #### Derived variables ####
        self.temporal_known_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                      self.temporal_observed_continuous_inp_size,
                                      self.temporal_target_size,
                                      len(self.temporal_observed_categorical_inp_lens),
                                      ])

def make_attn_module_class(attn_name):
    """Gets an attention module class for arbitrary experiment.

    Returns:
      Attention module class derived from nn.Module.
    """

    attention_class = {
        'sdp': ATTN.DotProductAttention,
        'lin': ATTN.LinearKernelAttention,
        'exp': ATTN.ExpKernelAttention,
        'per': ATTN.PeriodicKernelAttention,
        'lp': ATTN.LocallyPeriodicKernelAttention,
        'rq': ATTN.RationalQuadraticKernelAttention,
        'imp': ATTN.ImplicitKernelAttention,
        'cp': ATTN.ChangePointKernelAttention
    }

    return attention_class[attn_name]

def get_attention_names():
    return ['sdp', 'lin', 'exp', 'per', 'lp', 'rq', 'imp', 'cp']

def get_attention_hparam_grid(attn_name):
    attn_names = get_attention_names()
    if attn_name not in attn_names:
        raise ValueError('Unknown attention type {}. must be one of {}'.format(attn_name, attn_names))

    hranges = {
        'sdp': [{}],
        'lin': [{}],
        'exp': [
            # with magnitude
            {"p_norm_sim": 1, "p_norm_mag": 1, "include_magnitude": True},
            {"p_norm_sim": 2, "p_norm_mag": 1, "include_magnitude": True},
            {"p_norm_sim": 1, "p_norm_mag": 2, "include_magnitude": True},
            # without magnitude
            {"p_norm_sim": 1, "p_norm_mag": 0, "include_magnitude": False},
            {"p_norm_sim": 2, "p_norm_mag": 0, "include_magnitude": False},
        ],
        'per': [
            # p_norm=0 (without magnitude)
            {"period": 0.01, "p_norm": 0, "include_magnitude": False},
            {"period": 0.1, "p_norm": 0, "include_magnitude": False},
            {"period": 1, "p_norm": 0, "include_magnitude": False},
            {"period": 10, "p_norm": 0, "include_magnitude": False},
            {"period": 100, "p_norm": 0, "include_magnitude": False},
            # p_norm=1
            {"period": 0.01, "p_norm": 1, "include_magnitude": True},
            {"period": 0.1, "p_norm": 1, "include_magnitude": True},
            {"period": 1, "p_norm": 1, "include_magnitude": True},
            {"period": 10, "p_norm": 1, "include_magnitude": True},
            {"period": 100, "p_norm": 1, "include_magnitude": True},
            # p_norm=2
            {"period": 0.01, "p_norm": 2, "include_magnitude": True},
            {"period": 0.1, "p_norm": 2, "include_magnitude": True},
            {"period": 1, "p_norm": 2, "include_magnitude": True},
            {"period": 10, "p_norm": 2, "include_magnitude": True},
            {"period": 100, "p_norm": 2, "include_magnitude": True},
        ],
        'lp': [
            # p_norm=0 (without magnitude)
            {"period": 0.01, "p_norm": 0, "include_magnitude": False},
            {"period": 0.1, "p_norm": 0, "include_magnitude": False},
            {"period": 1, "p_norm": 0, "include_magnitude": False},
            {"period": 10, "p_norm": 0, "include_magnitude": False},
            {"period": 100, "p_norm": 0, "include_magnitude": False},
            # p_norm=1
            {"period": 0.01, "p_norm": 1, "include_magnitude": True},
            {"period": 0.1, "p_norm": 1, "include_magnitude": True},
            {"period": 1, "p_norm": 1, "include_magnitude": True},
            {"period": 10, "p_norm": 1, "include_magnitude": True},
            {"period": 100, "p_norm": 1, "include_magnitude": True},
            # p_norm=2
            {"period": 0.01, "p_norm": 2, "include_magnitude": True},
            {"period": 0.1, "p_norm": 2, "include_magnitude": True},
            {"period": 1, "p_norm": 2, "include_magnitude": True},
            {"period": 10, "p_norm": 2, "include_magnitude": True},
            {"period": 100, "p_norm": 2, "include_magnitude": True},
        ],
        'rq': [
            # p_norm=0 (without magnitude)
            {"alpha": 0.01, "p_norm": 0, "include_magnitude": False},
            {"alpha": 0.1, "p_norm": 0, "include_magnitude": False},
            {"alpha": 1, "p_norm": 0, "include_magnitude": False},
            {"alpha": 10, "p_norm": 0, "include_magnitude": False},
            {"alpha": 100, "p_norm": 0, "include_magnitude": False},
            # p_norm=1
            {"alpha": 0.01, "p_norm": 1, "include_magnitude": True},
            {"alpha": 0.1, "p_norm": 1, "include_magnitude": True},
            {"alpha": 1, "p_norm": 1, "include_magnitude": True},
            {"alpha": 10, "p_norm": 1, "include_magnitude": True},
            {"alpha": 100, "p_norm": 1, "include_magnitude": True},
            # p_norm=2
            {"alpha": 0.01, "p_norm": 2, "include_magnitude": True},
            {"alpha": 0.1, "p_norm": 2, "include_magnitude": True},
            {"alpha": 1, "p_norm": 2, "include_magnitude": True},
            {"alpha": 10, "p_norm": 2, "include_magnitude": True},
            {"alpha": 100, "p_norm": 2, "include_magnitude": True},
        ],
        'imp': [
            # p_norm=0 (without magnitude)
            {"R_features": 32, "p_norm": 0, "include_magnitude": False},
            {"R_features": 64, "p_norm": 0, "include_magnitude": False},
            {"R_features": 128, "p_norm": 0, "include_magnitude": False},
            {"R_features": 256, "p_norm": 0, "include_magnitude": False},
            # p_norm=1
            {"R_features": 32, "p_norm": 1, "include_magnitude": True},
            {"R_features": 64, "p_norm": 1, "include_magnitude": True},
            {"R_features": 128, "p_norm": 1, "include_magnitude": True},
            {"R_features": 256, "p_norm": 1, "include_magnitude": True},
            # p_norm=2
            {"R_features": 32, "p_norm": 2, "include_magnitude": True},
            {"R_features": 64, "p_norm": 2, "include_magnitude": True},
            {"R_features": 128, "p_norm": 2, "include_magnitude": True},
            {"R_features": 256, "p_norm": 2, "include_magnitude": True},
        ],
        'cp': [
            ## p_norm_sim=2
            # p_norm=0 (without magnitude)
            {"period": 0.01, "p_norm_sim": 2, "p_norm_mag": 0, "include_magnitude": False},
            {"period": 0.1, "p_norm_sim": 2, "p_norm_mag": 0, "include_magnitude": False},
            {"period": 1, "p_norm_sim": 2, "p_norm_mag": 0, "include_magnitude": False},
            {"period": 10, "p_norm_sim": 2, "p_norm_mag": 0, "include_magnitude": False},
            {"period": 100, "p_norm_sim": 2, "p_norm_mag": 0, "include_magnitude": False},
            # p_norm=1
            {"period": 0.01, "p_norm_sim": 2, "p_norm_mag": 1, "include_magnitude": True},
            {"period": 0.1, "p_norm_sim": 2, "p_norm_mag": 1, "include_magnitude": True},
            {"period": 1, "p_norm_sim": 2, "p_norm_mag": 1, "include_magnitude": True},
            {"period": 10, "p_norm_sim": 2, "p_norm_mag": 1, "include_magnitude": True},
            {"period": 100, "p_norm_sim": 2, "p_norm_mag": 1, "include_magnitude": True},
            # p_norm=2
            {"period": 0.01, "p_norm_sim": 2, "p_norm_mag": 2, "include_magnitude": True},
            {"period": 0.1, "p_norm_sim": 2, "p_norm_mag": 2, "include_magnitude": True},
            {"period": 1, "p_norm_sim": 2, "p_norm_mag": 2, "include_magnitude": True},
            {"period": 10, "p_norm_sim": 2, "p_norm_mag": 2, "include_magnitude": True},
            {"period": 100, "p_norm_sim": 2, "p_norm_mag": 2, "include_magnitude": True},

            ## p_norm_sim=1
            # p_norm=0 (without magnitude)
            {"period": 0.01, "p_norm_sim": 1, "p_norm_mag": 0, "include_magnitude": False},
            {"period": 0.1, "p_norm_sim": 1, "p_norm_mag": 0, "include_magnitude": False},
            {"period": 1, "p_norm_sim": 1, "p_norm_mag": 0, "include_magnitude": False},
            {"period": 10, "p_norm_sim": 1, "p_norm_mag": 0, "include_magnitude": False},
            {"period": 100, "p_norm_sim": 1, "p_norm_mag": 0, "include_magnitude": False},
            # p_norm=1
            {"period": 0.01, "p_norm_sim": 1, "p_norm_mag": 1, "include_magnitude": True},
            {"period": 0.1, "p_norm_sim": 1, "p_norm_mag": 1, "include_magnitude": True},
            {"period": 1, "p_norm_sim": 1, "p_norm_mag": 1, "include_magnitude": True},
            {"period": 10, "p_norm_sim": 1, "p_norm_mag": 1, "include_magnitude": True},
            {"period": 100, "p_norm_sim": 1, "p_norm_mag": 1, "include_magnitude": True},
            # p_norm=2
            {"period": 0.01, "p_norm_sim": 1, "p_norm_mag": 2, "include_magnitude": True},
            {"period": 0.1, "p_norm_sim": 1, "p_norm_mag": 2, "include_magnitude": True},
            {"period": 1, "p_norm_sim": 1, "p_norm_mag": 2, "include_magnitude": True},
            {"period": 10, "p_norm_sim": 1, "p_norm_mag": 2, "include_magnitude": True},
            {"period": 100, "p_norm_sim": 1, "p_norm_mag": 2, "include_magnitude": True},
        ],
    }

    return hranges[attn_name]

CONFIGS = {'electricity':  ElectricityConfig,
           'traffic':      TrafficConfig,
           'volatility':   VolatilityConfig,
           }

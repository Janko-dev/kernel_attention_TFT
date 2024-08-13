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

: ${SEED:=1}
: ${LR:=1e-3}
: ${NGPU:=1}
: ${BATCH_SIZE:=128}
: ${EPOCHS:=100}
: ${MAX_GRAD_NORM:=100}
: ${PATIENCE:=5}

: ${ATTN_NAME:=sdp}
: ${EXP_NAME:=traffic}
: ${EXP_DATA_PATH:=/storage/data/processed/${EXP_NAME}_bin}
: ${EXP_RESULTS_PATH:=/storage/results/${EXP_NAME}_individual_exp/grid_search_${EXP_NAME}_${ATTN_NAME}}

python -m torch.distributed.run --nproc_per_node=${NGPU} grid_search.py \
        --dataset ${EXP_NAME} \
        --data_path ${EXP_DATA_PATH} \
        --attn_name=${ATTN_NAME} \
        --batch_size=${BATCH_SIZE} \
        --sample 450000 50000 \
        --lr ${LR} \
        --epochs ${EPOCHS} \
        --seed ${SEED} \
        --use_amp \
        --clip_grad ${MAX_GRAD_NORM} \
        --early_stopping ${PATIENCE} \
        --results ${EXP_RESULTS_PATH}

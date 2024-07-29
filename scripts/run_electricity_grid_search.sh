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
: ${BATCH_SIZE:=64}
: ${EPOCHS:=100}
: ${ATTN_NAME:=sdp}

python -m torch.distributed.run --nproc_per_node=${NGPU} grid_search.py \
        --dataset electricity \
        --data_path /data/processed/electricity_bin \
        --attn_name=${ATTN_NAME} \
        --batch_size=${BATCH_SIZE} \
        --sample 450000 50000 \
        --lr ${LR} \
        --epochs ${EPOCHS} \
        --seed ${SEED} \
        --use_amp \
        --clip_grad 0.01 \
        --early_stopping 5 \
        --results /results/TFT_electricity_grid${ATTN_NAME}_bs${BATCH_SIZE}_lr${LR}_seed_${SEED}

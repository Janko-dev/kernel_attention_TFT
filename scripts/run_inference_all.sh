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

ATTN_NAMES=(sdp lin exp per lp rq imp cp)

: ${EXP_NAME:=electricity}
: ${EXP_DATA_PATH:=/storage/data/processed/${EXP_NAME}_bin}


for ATTN_NAME in ${ATTN_NAMES[@]}
do

  : ${EXP:=gridsearch_${EXP_NAME}_${ATTN_NAME}_bs64_lr1e-3_seed1/}
  : ${CKECKPOINT_PATH:=/storage/results/${EXP}/best_model_checkpoint.pt}
  : ${EXP_RESULTS_PATH:=/storage/inference/${EXP_NAME}/inference_results_${ATTN_NAME}}

  python inference.py \
      --checkpoint ${CKECKPOINT_PATH} \
      --data ${EXP_DATA_PATH}/test.csv \
      --tgt_scalers ${EXP_DATA_PATH}/tgt_scalers.bin \
      --cat_encodings ${EXP_DATA_PATH}/cat_encodings.bin \
      --batch_size 64 \
      --visualize 2 \
      --results ${EXP_RESULTS_PATH}

done
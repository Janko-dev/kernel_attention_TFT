
: ${SEED:=1}
: ${LR:=1e-3}
: ${NGPU:=1}
: ${BATCH_SIZE:=64}
: ${EPOCHS:=100}
: ${MAX_GRAD_NORM:=0.01}
: ${PATIENCE:=5}

ATTN_NAMES=(sdp lin exp per lp rq imp cp)

: ${EXP_NAME:=electricity}
: ${EXP_DATA_PATH:=/storage/data/processed/${EXP_NAME}_bin}

for ATTN_NAME in ${ATTN_NAMES[@]}
do

  : ${EXP_RESULTS_PATH:=/storage/results/${EXP_NAME}/grid_search_${EXP_NAME}_${ATTN_NAME}}

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
done

for P in `ls /storage/results/${EXP_NAME}`;
do
    echo ${P}
    tail -n 1 /storage/results/${P}/dllogger.json
done
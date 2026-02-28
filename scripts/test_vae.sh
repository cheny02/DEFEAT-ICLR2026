#!/bin/bash
cd ..
# custom config
DATA=/data/dataset/
TRAINER=DEFEAT
DATASET=$1
CFG=$2
LOADEP=$3
CTP=end  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
EPS=$6
ALPHA=$7
STEPS=$8
SEED=0
W1=$9
W2=${10}
W3=${11}
W4=${12}
A=${13}
ROB_TEST=True
# ATTACK=aa
# FP=fp32

MODEL_DIR=output_${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}/W1_${W1}_W2_${W2}_W3_${W3}_W4_${W4}_A_${A}/${DATASET}/seed${SEED}
DIR=output_${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}/W1_${W1}_W2_${W2}_W3_${W3}_W4_${W4}_A_${A}/${DATASET}/seed${SEED}/test
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eps ${EPS} \
    --alpha ${ALPHA} \
    --steps ${STEPS} \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.DEFEAT.ATK_TEST ${ROB_TEST} \
    TRAINER.COOP.W1 ${W1}\
    TRAINER.COOP.W2 ${W2}\
    TRAINER.COOP.W3 ${W3}\
    TRAINER.COOP.W4 ${W4}\
    TRAINER.COOP.ALPHA ${A} \
    # TRAINER.DEFEAT.ATK ${ATTACK} \
    # TRAINER.COOP.PREC ${FP} 
fi

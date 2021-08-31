#!/usr/bin/env bash
set -x
DATAPATH="/home/yanziwei/data/project-test/Stereo_exp/HITnet/datafile/sceneflow/"
LOGDIR="./worklog"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./datasets/list/sceneflow_train.list --testlist ./datasets/list/sceneflow_train.list \
    --logdir $LOGDIR \
    --ckpt_start_epoch 4713 --summary_freq 1000 \
    --epochs 300 --lrepochs "2,5:4,2.5" \
    --batch_size 16
    --maxdisp 192

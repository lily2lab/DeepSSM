#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
cd ../..
savepath='results/3dpw/DeepSSM_short_term/v0'
modelpath='checkpoints/3dpw/DeepSSM_short_term/v0'
pretrain_modelpath='checkpoints/3dpw/DeepSSM_short_term/v0/model.ckpt-800'
logname='logs/3dpw/DeepSSM_short_term.log'
nohup python -u Train_DeepSSM_3dpw.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths data/3dpw_ske/train_3dpw0_25.npy \
    --valid_data_paths data/3dpw_ske/test_3dpw0_25.npy \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --input_length 10 \
    --seq_length 25 \
    --stacklength1 2 \
    --stacklength2 1 \
    --filter_size 3 \
    --lr 0.0001  \
    --batch_size 16 \
    --sampling_stop_iter 0 \
    --max_iterations 3000000 \
    --display_interval 10 \
    --test_interval 100 \
    --n_gpu 1 \
    --snapshot_interval 100 >>${logname}  2>&1 &

tail -f ${logname}

# --pretrained_model ${pretrain_modelpath}  \




date ; hostname ; pwd

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=19800
export NODE_RANK=0

EXP_LR_ARRAY=(1e-5 2e-5 1e-5 2e-5)
EXP_BS_ARRAY=(16 16 32 32)
EXP_TN_ARRAY1=(cola mrpc sst2 qqp)
EXP_TN_ARRAY2=(stsb rte qnli mnli)
EXP_CUDA_INDEX1=(0 1 2 3)
EXP_CUDA_INDEX2=(4 5 6 7)
EXP_MASTER_PORT1=(19800 19801 19802 19803)
EXP_MASTER_PORT2=(19804 19805 19806 19807)

for task in {0..3}
do
    task1=${EXP_TN_ARRAY1[$task]}
    task2=${EXP_TN_ARRAY2[$task]}
    for i in {0..3}
    do 
        EXP_LR=${EXP_LR_ARRAY[$i]}
        EXP_BS=${EXP_BS_ARRAY[$i]}
        EXP_PGB=$EXP_BS
        echo ${EXP_CUDA_INDEX1[$i]}, ${EXP_MASTER_PORT1[$i]}, $NODE_RANK, $EXP_BS, $EXP_LR, $task1
        echo ${EXP_CUDA_INDEX2[$i]}, ${EXP_MASTER_PORT2[$i]}, $NODE_RANK, $EXP_BS, $EXP_LR, $task2
        RUN_NAME=""$EXP_BS"_"$EXP_LR""
        CUDA_VISIBLE_DEVICES=${EXP_CUDA_INDEX1[$i]} MASTER_PORT=${EXP_MASTER_PORT1[$i]} python run_glue.py with run_name=$RUN_NAME learning_rate=$EXP_LR batch_size=$EXP_BS per_gpu_batchsize=$EXP_PGB group_name=$task1 exp_name=METER-Uni-Modal load_path=~/BT/METER_checkpoints/meter_clip16_288_roberta_pretrain.ckpt load_flag=True >/dev/null &
        CUDA_VISIBLE_DEVICES=${EXP_CUDA_INDEX2[$i]} MASTER_PORT=${EXP_MASTER_PORT2[$i]} python run_glue.py with run_name=$RUN_NAME learning_rate=$EXP_LR batch_size=$EXP_BS per_gpu_batchsize=$EXP_PGB group_name=$task2 exp_name=METER-Uni-Modal load_path=~/BT/METER_checkpoints/meter_clip16_288_roberta_pretrain.ckpt load_flag=True >/dev/null &
    done
    wait
    date
done


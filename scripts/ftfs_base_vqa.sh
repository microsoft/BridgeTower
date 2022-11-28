date ; hostname ; pwd

EXP_NODES=1
EXP_IS=576
EXP_PGB=16
EXP_PGEB=64
EXP_LR=2e-5
EXP_BS=512
EXP_ME=10
EXP_WS=0.1
EXP_WD=0.01
EXP_LMH=50
EXP_LMC=5
EXP_THL=2
EXP_HHS=1
EXP_RGM=clip_randaug # all the VQAv2 experiments without VLP for METER and BridgeTower

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=19800
export NODE_RANK=0

PREFIX_NAME="ftfs"

echo $MASTER_ADDR, $MASTER_PORT, $NODE_RANK, $EXP_NODES, $EXP_IS, $EXP_PGB, $EXP_PGEB, $EXP_LR, $EXP_BS, $EXP_ME, $EXP_WS, $EXP_WD, $EXP_LMH, $EXP_LMC, $EXP_THL, $EXP_HHS, $EXP_RGM

TIME=$(date "+%Y%m%d%H%M")
RUN_NAME=""$PREFIX_NAME"_"$EXP_IS"_"$EXP_PGB"_"$EXP_PGEB"_"$EXP_LR"_"$EXP_BS"_"$EXP_ME"_"$EXP_WS"_"$EXP_WD"_"$EXP_LMH"_"$EXP_LMC"_"$EXP_THL"_"$EXP_HHS"_"$EXP_RGM"_"$TIME""
echo $RUN_NAME
python run.py with run_name=$RUN_NAME task_finetune_vqa_clip_bert bt clip16 text_roberta $EXP_RGM num_gpus=8 num_nodes=$EXP_NODES image_size=$EXP_IS per_gpu_batchsize=$EXP_PGB per_gpu_eval_batchsize=$EXP_PGEB learning_rate=$EXP_LR batch_size=$EXP_BS max_epoch=$EXP_ME warmup_steps=$EXP_WS weight_decay=$EXP_WD lr_mult_head=$EXP_LMH lr_mult_cross_modal=$EXP_LMC task_head_layers=$EXP_THL head_hidden_scale=$EXP_HHS

date
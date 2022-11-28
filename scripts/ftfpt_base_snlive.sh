date ; hostname ; pwd

EXP_NODES=1
EXP_IS=384
EXP_PGB=8
EXP_PGEB=64
EXP_LR=3e-6
EXP_BS=64
EXP_ME=5
EXP_WS=0.06
EXP_WD=0.01
EXP_LMH=10
EXP_LMC=5
EXP_THL=2
EXP_HHS=2
EXP_LP=BridgeTower_pt_base.ckpt
EXP_RGM=blip_randaug_wc

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=19800
export NODE_RANK=0

PREFIX_NAME="ftfpt"


echo $MASTER_ADDR, $MASTER_PORT, $NODE_RANK, $EXP_NODES, $EXP_IS, $EXP_PGB, $EXP_PGEB, $EXP_LR, $EXP_BS, $EXP_ME, $EXP_WS, $EXP_WD, $EXP_LMH, $EXP_LMC, $EXP_THL, $EXP_HHS, $EXP_RGM

TIME=$(date "+%Y%m%d%H%M")
RUN_NAME=""$PREFIX_NAME"_"$EXP_IS"_"$EXP_PGB"_"$EXP_PGEB"_"$EXP_LR"_"$EXP_BS"_"$EXP_ME"_"$EXP_WS"_"$EXP_WD"_"$EXP_LMH"_"$EXP_LMC"_"$EXP_THL"_"$EXP_HHS"_"$EXP_RGM"_"$TIME""
echo $RUN_NAME
python run.py with run_name=$RUN_NAME task_finetune_snli_clip_bert bt clip16 text_roberta $EXP_RGM num_gpus=8 num_nodes=$EXP_NODES load_path=~/BT/best_checkpoints/$EXP_LP image_size=$EXP_IS per_gpu_batchsize=$EXP_PGB per_gpu_eval_batchsize=$EXP_PGEB learning_rate=$EXP_LR batch_size=$EXP_BS max_epoch=$EXP_ME warmup_steps=$EXP_WS weight_decay=$EXP_WD lr_mult_head=$EXP_LMH lr_mult_cross_modal=$EXP_LMC task_head_layers=$EXP_THL head_hidden_scale=$EXP_HHS

date
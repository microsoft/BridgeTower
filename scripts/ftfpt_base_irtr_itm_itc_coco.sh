date ; hostname ; pwd

EXP_NODES=1
EXP_IS=384
EXP_PGB=16
EXP_PGEB=16
EXP_LR=4.5e-6
EXP_BS=256
EXP_ME=30
EXP_WS=0.1
EXP_WD=0.01
EXP_LMH=5
EXP_LMC=5
EXP_LP=BridgeTower_pt_base.ckpt
EXP_RGM=blip_randaug_wc
EXP_PGEBT=256
EXP_PGEBI=128
EXP_GWG=True
EXP_GAII=False
EXP_IC=1

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=19800
export NODE_RANK=0

PREFIX_NAME="ftfpt"

echo $MASTER_ADDR, $MASTER_PORT, $NODE_RANK, $EXP_NODES, $EXP_IS, $EXP_PGB, $EXP_PGEB, $EXP_LR, $EXP_BS, $EXP_ME, $EXP_WS, $EXP_WD, $EXP_LMH, $EXP_LMC, $EXP_RGM


TIME=$(date "+%Y%m%d%H%M")

RUN_NAME=""$PREFIX_NAME"_"$EXP_IS"_"$EXP_PGB"_"$EXP_PGEB"_"$EXP_LR"_"$EXP_BS"_"$EXP_ME"_"$EXP_WS"_"$EXP_WD"_"$EXP_LMH"_"$EXP_LMC"_"$EXP_RGM"_"$TIME""

echo $RUN_NAME

python run.py with run_name=$RUN_NAME task_finetune_irtr_itm_itc_coco_clip_bert bt clip16 text_roberta $EXP_RGM num_gpus=8 num_nodes=$EXP_NODES load_path=~/BT/best_checkpoints/$EXP_LP image_size=$EXP_IS per_gpu_batchsize=$EXP_PGB per_gpu_eval_batchsize=$EXP_PGEB learning_rate=$EXP_LR batch_size=$EXP_BS max_epoch=$EXP_ME warmup_steps=$EXP_WS weight_decay=$EXP_WD lr_mult_head=$EXP_LMH lr_mult_cross_modal=$EXP_LMC per_gpu_eval_batchsize_text=$EXP_PGEBT per_gpu_eval_batchsize_image=$EXP_PGEBI gather_with_grads=$EXP_GWG gather_all_image_inputs=$EXP_GAII image_chunks=$EXP_IC

date
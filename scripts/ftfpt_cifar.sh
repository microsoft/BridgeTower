date ; hostname ; pwd

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=19800
export NODE_RANK=0

EXP_LR_ARRAY=(1e-5 1e-5 2e-5 2e-5 1e-5 1e-5 2e-5 2e-5)
EXP_LF_ARRAY=(True False True False True False True False)
EXP_GN_ARRAY=(cifar10 cifar10 cifar10 cifar10 cifar100 cifar100 cifar100 cifar100)
EXP_RB_ARRAY=(288 224 288 224 288 224 288 224)

for i in {0..7}
do
    EXP_LR=${EXP_LR_ARRAY[$i]}
    EXP_LF=${EXP_LF_ARRAY[$i]}
    EXP_GN=${EXP_GN_ARRAY[$i]}
    EXP_RB=${EXP_RB_ARRAY[$i]}
    echo $MASTER_ADDR, $MASTER_PORT, $NODE_RANK, $EXP_NODES, $EXP_LR, $EXP_LF, $EXP_GN, $EXP_RB
    RUN_NAME=""$EXP_LR"_"$EXP_LF"_10"
    python run_cifar.py with run_name=$RUN_NAME learning_rate=$EXP_LR load_flag=$EXP_LF group_name=$EXP_GN resolution_before=$EXP_RB
    RUN_NAME=""$EXP_LR"_"$EXP_LF"_100"
    python run_cifar.py with run_name=$RUN_NAME learning_rate=$EXP_LR load_flag=$EXP_LF group_name=$EXP_GN resolution_before=$EXP_RB max_epoch=100
done
date
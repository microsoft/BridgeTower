from sacred import Experiment

ex = Experiment("VL")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "itc": 0,
        "itm_itc": 0,
        "irtr_itm_itc": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "snli": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    # below params varies with the environment
    root_dir = "~/BT"
    data_root = f"{root_dir}/dataset/fine-tune"
    log_dir = f"{root_dir}/logs"
    output_dir = f"{root_dir}/checkpoints"
    load_path = ""
    num_gpus = 8
    num_nodes = 1
    num_workers = 8
    precision = 32
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    per_gpu_eval_batchsize = 0

    # Wandb Logger Setting
    exp_name = "BT"
    group_name = "exp/task"
    run_name = "finetune"
    
    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    log_every_n_steps = 50

    # Experiment Setting
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 224
    patch_size = 32
    draw_false_image = 0
    image_only = False
    resolution_before = 224

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    input_image_embed_size = 768
    input_text_embed_size = 768
    vit = 'CLIP-ViT-B/32'
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 10
    max_steps = -1
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False

    # Debug
    debug_num = 0

    # METER Setting
    meter_fusion = False
    vit_remove_last = False

    # BT Setting
    model_type = "BT" # "METER", "BT"
    vit_layernorm_shared = True
    vit_layernorm_init_from_vit = False
    task_head_layers = 2 # 1, 2
    head_hidden_scale = 1 # 1, 2, 3, 4
    per_gpu_eval_batchsize_text = 256
    per_gpu_eval_batchsize_image = 128
    per_gpu_eval_batchsize_fusion_text = 500
    k_test = 128 # 128, 256
    amp_flag = True
    task_threshold = 0 # the task will be executed if it > task_threshold
    nlvr2_drop_rate = 0.1

    ## contrastive setting
    temperature = 0.07
    contrastive_hidden_size = 256
    gather_with_grads = True
    gather_global_negative = False
    gather_all_image_inputs = False # if all image features cannot be gathered in one GPU, then gather all image inputs
    image_chunks = 1 # if k_test x image need too many memory, then split them into chunks to calculate rerank scores
    text_chunks = 1 # if k_test x text need too many memory, then split them into chunks to calculate rerank scores
    save_memory = False

# model type
@ex.named_config
def meter():
    model_type = "METER"

@ex.named_config
def bt():
    model_type = "BT"

@ex.named_config
def bt_large():
    hidden_size = 1024
    num_heads = 16
    num_layers = 6

# pre-train task setting
@ex.named_config
def task_mlm_itm_clip_bert():
    group_name = "mlm_itm"
    run_name = "pre-train"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True

    vocab_size = 30522
    max_text_len = 50
    image_size = 224
    tokenizer = "bert-base-uncased"
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    learning_rate = 1e-5
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    draw_false_image = 1

@ex.named_config
def task_mlm_itm_itc():
    group_name = "mlm_itm_itc"
    loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
    contrastive_hidden_size = 256 # 256, 512, 768

@ex.named_config
def task_mlm_itm_itc_hard():
    group_name = "mlm_itm_itc_hard"
    loss_names = _loss_names({"itm_itc": 1, "mlm": 1})
    draw_false_image = 0
    contrastive_hidden_size = 256 # 256, 512, 768

# fine-tune task setting
@ex.named_config
def task_finetune_vqa_clip_bert():
    group_name = "vqa"
    run_name = "finetune"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = -1
    warmup_steps = 0.1
    learning_rate = 1e-5
    lr_mult_head = 50
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 50
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 576

@ex.named_config
def task_finetune_snli_clip_bert():
    group_name = "snli"
    run_name = "finetune"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    max_epoch = 5
    max_steps = -1
    warmup_steps = 0.1
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 50
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 384

@ex.named_config
def task_finetune_irtr_f30k_clip_bert():
    group_name = "irtr_f30k"
    run_name = "finetune"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = -1
    warmup_steps = 0.1
    draw_false_image = 1
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 40
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 384

@ex.named_config
def task_finetune_irtr_itm_itc_f30k_clip_bert():
    group_name = "irtr_itm_itc_f30k"
    run_name = "finetune"
    datasets = ["f30k"]
    loss_names = _loss_names({"irtr_itm_itc": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = -1
    warmup_steps = 0.1
    draw_false_image = 0
    draw_false_text = 0
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 40
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 384
    k_test = 128
    get_recall_metric = True

@ex.named_config
def task_finetune_nlvr2_clip_bert():
    group_name = "nlvr2"
    run_name = "finetune"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = -1
    warmup_steps = 0.1
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 50
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 384

@ex.named_config
def task_finetune_irtr_coco_clip_bert():
    group_name = "irtr_coco"
    run_name = "finetune"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = -1
    warmup_steps = 0.1
    draw_false_image = 1
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 40
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 384

@ex.named_config
def task_finetune_irtr_itm_itc_coco_clip_bert():
    group_name = "irtr_itm_itc_coco"
    run_name = "finetune"
    datasets = ["coco"]
    loss_names = _loss_names({"irtr_itm_itc": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = -1
    warmup_steps = 0.1
    draw_false_image = 0
    draw_false_text = 0
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 40
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 384
    k_test = 256
    get_recall_metric = True

# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end

# vision encoder
@ex.named_config
def vit16_224():
    vit = 'vit_base_patch16_224'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def vit16_384():
    # used by METER
    vit = 'vit_base_patch16_384'
    image_size = 224
    patch_size = 32
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def vit32_384():
    # used by ViLT
    vit = 'vit_base_patch32_384'
    image_size = 384
    patch_size = 32
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def vit16_224_in21k():
    vit = 'vit_base_patch16_224_in21k'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def vit32_224_in21k():
    vit = 'vit_base_patch32_224_in21k'
    image_size = 224
    patch_size = 32
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def deit16_224():
    # used by ALBEF
    vit = 'vit_deit_base_patch16_224'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def deit16_384():
    # used by METER
    vit = 'vit_deit_base_patch16_384'
    image_size = 384
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

## bt don't support swin transformer
@ex.named_config
def swin32_base224():
    vit = "swin_base_patch4_window7_224_in22k"
    patch_size = 32
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024
    resolution_before = 224

@ex.named_config
def swin32_base384():
    vit = "swin_base_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024
    resolution_before = 384

@ex.named_config
def swin32_large384():
    vit = "swin_large_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1536
    resolution_before = 384

@ex.named_config
def clip32():
    vit = 'CLIP-ViT-B/32'
    image_size = 224
    patch_size = 32
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def clip16():
    vit = 'CLIP-ViT-B/16'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def clip14_large():
    vit = 'CLIP-ViT-L/14'
    image_size = 224
    patch_size = 14
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 1024

# text encoder
@ex.named_config
def text_roberta():
    tokenizer = "roberta-base"
    vocab_size = 50265
    input_text_embed_size = 768
    whole_word_masking = False

@ex.named_config
def text_roberta_large():
    tokenizer = "roberta-large"
    vocab_size = 50265
    input_text_embed_size = 1024
    whole_word_masking = False

# random augmentation
@ex.named_config
def imagenet_randaug():
    train_transform_keys = ["imagenet_randaug"]

@ex.named_config
def clip_randaug():
    train_transform_keys = ["clip_randaug"]

@ex.named_config
def clip_pure():
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]

@ex.named_config
def blip_pure():
    train_transform_keys = ["blip"]
    val_transform_keys = ["blip"]

@ex.named_config
def blip_randaug():
    train_transform_keys = ["blip_randaug"]
    val_transform_keys = ["blip"]

@ex.named_config
def blip_randaug_wc():
    train_transform_keys = ["blip_randaug_wc"]
    val_transform_keys = ["blip"]

@ex.named_config
def blip_randaug_wohf():
    train_transform_keys = ["blip_randaug_wohf"]
    val_transform_keys = ["blip"]

@ex.named_config
def blip_randaug_pretrain():
    train_transform_keys = ["blip_randaug_pretrain"]
    val_transform_keys = ["blip"]

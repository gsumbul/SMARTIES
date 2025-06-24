accelerate launch \
    --dynamo_backend='no' \
    --num_machines=1 \
    --num_processes=1 \
    main_downstream.py \
    --eval_batch_size 64 \
    --blr 6.25e-5 \
    --epochs 200 \
    --eval_type ft \
    --eval_dataset RESISC45 \
    --model smarties_vit_large_patch16 \
    --label_smoothing 0 \
    --eval_freq 1 \
    --wandb_tags ft \
    --warmup_epochs 5 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --mixup 0 \
    --cutmix 0 \
    --spectrum_specs_path config/electromagnetic_spectrum.yaml \
    --eval_specs_path config/eval_datasets.yaml \
    --sensors_specs_path config/pretraining_sensors.yaml \
    --nb_workers_per_gpu 8 \
    --pretrained_model_path weights/smarties-v1-vitl.safetensors \
    # --pretrained_model_path weights/smarties-v1-vitl-resisc45-finetune.safetensors \
    # --eval_only \

accelerate launch \
    --dynamo_backend='no' \
    --num_machines=1 \
    --num_processes=1 \
    main_downstream.py \
    --eval_batch_size 256 \
    --blr 2e-4 \
    --eval_type ft \
    --epochs 150 \
    --drop_path 0.1 \
    --model smarties_vit_large_patch16 \
    --eval_dataset EuroSAT \
    --label_smoothing 0 \
    --mixup 0.8 --cutmix 1.0 \
    --eval_freq 20 \
    --wandb_tags ft \
    --warmup_epochs 5 \
    --weight_decay 0.05 \
    --spectrum_specs_path config/electromagnetic_spectrum.yaml \
    --eval_specs_path config/eval_datasets.yaml \
    --sensors_specs_path config/pretraining_sensors.yaml \
    --nb_workers_per_gpu 8 \
    --pretrained_model_path weights/smarties-v1-vitl.safetensors \
    # --eval_only \
    # --pretrained_model_path weights/smarties-v1-vitl-eurosat-finetune.safetensors \

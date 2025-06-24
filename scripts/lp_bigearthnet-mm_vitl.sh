accelerate launch \
    --dynamo_backend='no' \
    --num_machines=1 \
    --num_processes=1 \
    main_downstream.py \
    --eval_batch_size 1024 \
    --blr 1e-3 \
    --eval_type lp \
    --epochs 100 \
    --drop_path 0.2 \
    --model smarties_vit_large_patch16 \
    --multi_modal \
    --eval_dataset BigEarthNetMM \
    --label_smoothing 0 \
    --eval_freq 30 \
    --wandb_tags lp \
    --weight_decay 0 \
    --spectrum_specs_path config/electromagnetic_spectrum.yaml \
    --eval_specs_path config/eval_datasets.yaml \
    --sensors_specs_path config/pretraining_sensors.yaml \
    --nb_workers_per_gpu 8 \
    --pretrained_model_path weights/smarties-v1-vitl.safetensors \
    # --pretrained_model_path weights/smarties-v1-vitl-bigearthnetmm-linprobe.safetensors \
    # --eval_only \

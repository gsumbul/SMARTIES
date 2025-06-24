accelerate launch \
    --dynamo_backend='no' \
    --multi_gpu \
    --num_machines=1 \
    --num_processes=8 \
    main_pretrain.py \
    --model smarties_vit_base_patch16 \
    --blr 1.5e-4 \
    --warmup_epochs 20 \
    --weight_decay 0.05 \
    --batch_size 256 \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --epochs 300 \
    --eval_freq 1 \
    --eval_batch_size 512 \
    --eval_dataset EuroSAT \
    --sensors_specs_path config/pretraining_sensors.yaml \
    --spectrum_specs_path config/electromagnetic_spectrum.yaml \
    --eval_specs_path config/eval_datasets.yaml \

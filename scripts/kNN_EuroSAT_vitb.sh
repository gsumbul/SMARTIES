accelerate launch \
    --dynamo_backend='no' \
    --num_machines=1 \
    --num_processes=1 \
    main_downstream.py \
    --eval_batch_size 512 \
    --eval_dataset EuroSAT \
    --model smarties_vit_base_patch16 \
    --eval_scale 1 \
    --eval_type kNN \
    --pretrained_model_path weights/smarties-v1-vitb.safetensors \
    --sensors_specs_path config/pretraining_sensors.yaml \
    --spectrum_specs_path config/electromagnetic_spectrum.yaml \
    --eval_specs_path config/eval_datasets.yaml \
    --nb_workers_per_gpu 8

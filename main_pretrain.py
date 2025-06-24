import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import os
import sys
from pathlib import Path
import torch
import timm.optim.optim_factory as optim_factory

from pretraining_dataset import build_pretrain_loader
from eval_datasets import build_eval_loader
from utils.utils import set_specs, load_training_state, save_training_state, save_model, kNN
from utils.arguments import get_args_parser
import models_smarties
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.9"

def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        device_placement=True,
        split_batches=False, 
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        cpu=args.cpu,
        log_with = "wandb",
        dispatch_batches = False, 
        even_batches=False, 
        use_seedable_sampler=False,
        step_scheduler_with_optimizer=True, 
        kwargs_handlers=[ddp_kwargs]
    )
    args.num_gpus = accelerator.num_processes
    set_seed(args.seed)
    set_specs(args)

    data_loader_train = build_pretrain_loader(args)
    data_loader_eval_train, data_loader_eval_test = build_eval_loader(args)

    model = models_smarties.__dict__[args.model](**vars(args))
    
    global_batch_size = args.batch_size * accelerator.num_processes * args.grad_accum_steps
    batch_ratio = global_batch_size / args.lr_base_batch_size
    args.lr = args.blr * batch_ratio
    
    metric_name = f'kNN@{args.knn}_acc_{args.eval_dataset}_train'

    param_groups = optim_factory.param_groups_layer_decay(
        model, args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    # Instantiate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(data_loader_train) * args.warmup_epochs,
        num_training_steps=len(data_loader_train) * args.epochs
    )
    
    model, optimizer, data_loader_train, data_loader_eval_train, data_loader_eval_test, lr_scheduler = accelerator.prepare(
        model, optimizer, data_loader_train, data_loader_eval_train, data_loader_eval_test, lr_scheduler
    )

    accelerator.register_for_checkpointing(lr_scheduler)

    starting_epoch, global_step, nb_skip_batches = load_training_state(args, accelerator, data_loader_train)
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project, 
            config=args.__dict__,
            init_kwargs={'wandb': {
                'entity': args.wandb_entity,
                'resume': "allow",
                'name': args.wandb_name,
                'notes':args.wandb_notes,
                'tags': args.wandb_tags
                }
            }
        )
        wandb_tracker = accelerator.get_tracker("wandb")
        wandb_tracker.run.define_metric(metric_name, summary="max")

        run_dir = os.path.join(args.out_dir, '_'.join([wandb_tracker.run.name, wandb_tracker.run.id]))
        if global_step == 0:
            os.makedirs(run_dir, exist_ok=True)

    accelerator.print(f"Start training for {args.epochs} epochs with {accelerator.num_processes} GPU(s) and {accelerator.mixed_precision} mixed-precision")
    accelerator.print("base lr: %.2e" % args.blr)
    accelerator.print("actual lr: %.2e" % args.lr)
    accelerator.print("effective batch size: %d" % global_batch_size)

    best_knn_acc = 0.
    for epoch in tqdm(range(starting_epoch, args.epochs), disable=not accelerator.is_local_main_process):
        model.train()
        if (epoch == starting_epoch):
            # We need to skip steps until we reach the resumed step
            active_dataloader = accelerator.skip_first_batches(data_loader_train, nb_skip_batches)
        else:
            # After the first iteration, we need to go back to the original dataloader
            active_dataloader = data_loader_train
        for batch in tqdm(active_dataloader, disable=not accelerator.is_local_main_process):
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                loss, _, _ = model(batch)
            accelerator.backward(loss)
            optimizer.step()
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step()
            accelerator.log({"train_loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step=global_step)
            if accelerator.is_main_process:
                save_training_state(accelerator, run_dir, epoch, global_step)
            global_step += 1
            
        if (((epoch+1) % args.eval_freq == 0) and not args.skip_eval) or ((epoch + 1) == args.epochs):
            res = kNN(
                args,
                accelerator,
                model,
                trainloader=data_loader_eval_train,
                testloader=data_loader_eval_test,
                feat_dim=1024 if "large" in args.model else 768
            )
            accelerator.log({metric_name: res}, step=global_step)
            if res >= best_knn_acc:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_model(model, accelerator, run_dir, dir='model_weights_best_epoch_{}'.format(epoch+1))
                best_knn_acc = res
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_model(model, accelerator, run_dir, dir='model_weights_last_epoch_{}'.format(epoch+1))
    accelerator.end_training()

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    main(args)

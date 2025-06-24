import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import os
import sys
from pathlib import Path
import torch
import timm.optim.optim_factory as optim_factory
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
import models_smarties_vit

from eval_datasets import build_eval_loader
from utils.model_utils import interpolate_pos_embed, apply_label_mixup_fn
from utils.loss_utils import calc_downstream_loss
from utils.utils import get_logits_reducer, get_multistep_lr_schedule, get_nb_classes, get_task_head, get_task_type, get_metric_dict, set_specs, load_training_state, save_training_state, save_model, evaluate, kNN
from utils.arguments import get_args_parser
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from safetensors.torch import load_file

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.9"

def main(args):
    vit_backbone = args.model.split('smarties_')[-1]

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        device_placement=True,
        split_batches=False, #If True the actual batch size used will be the same on any kind of distributed processes, but it must be a round multiple of the num_processes you are using. If False, actual batch size used will be the one set in your script multiplied by the number of processes.
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        cpu=args.cpu,
        log_with = "wandb",
        dispatch_batches = False, #If set to True, the dataloader prepared by the Accelerator is only iterated through on the main process and then the batches are split and broadcast to each process. Will default to True for DataLoader whose underlying dataset is an IterableDataset, False otherwise.
        even_batches=False, #If set to True, in cases where the total batch size across all processes does not exactly divide the dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among all workers.
        use_seedable_sampler=False,
        step_scheduler_with_optimizer=True, #Set Trueif the learning rate scheduler is stepped at the same time as the optimizer
        kwargs_handlers=[ddp_kwargs]
    )
    args.num_gpus = accelerator.num_processes
    set_seed(args.seed)
    set_specs(args)
    args.nb_classes = get_nb_classes(args.eval_dataset)
    args.task_type = get_task_type(args.eval_dataset)

    data_loader_eval_train, data_loader_eval_test = build_eval_loader(args)

    model = models_smarties_vit.__dict__[vit_backbone](
        num_classes=args.nb_classes,
        drop_path_rate=0.0,
        global_pool=args.global_pool,
        spectrum_specs=args.spectrum_specs,
        img_size=args.input_size,
        mixed_precision=args.mixed_precision,
        multi_modal=args.multi_modal,
        num_sources=len(args.eval_specs[args.eval_dataset]['sources']) if args.multi_modal else 1,
        all_tokens=True if args.task_type == 'SS' else False
    )
    state_dict = model.state_dict()
    pretrained_state_dict = load_file(args.pretrained_model_path)
    for k in ["head.weight", "head.bias"]:
        if (
            k in pretrained_state_dict and pretrained_state_dict[k].shape != state_dict[k].shape
        ):
            accelerator.print(f"Removing key {k} from pretrained checkpoint")
            del pretrained_state_dict[k]

    if args.eval_only:
        model.head = get_task_head(model, args.task_type, args.eval_type, args.nb_classes)
    
    if args.eval_type == 'ft' and not args.eval_only:
        # interpolate position embedding
        interpolate_pos_embed(model, pretrained_state_dict)

    msg = model.load_state_dict(pretrained_state_dict, strict=False)
    if not args.eval_only:
        if args.global_pool:
            assert set(msg.missing_keys) == {
                "head.weight",
                "head.bias",
                "fc_norm.weight",
                "fc_norm.bias",
            }
        else:
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}

    if not (args.task_type == 'SS') and not (args.eval_type == 'kNN') and not args.eval_only:
        trunc_normal_(model.head.weight, std=0.01 if args.eval_type == 'lp' else 2e-5)

    if not args.eval_only:
        model.head = get_task_head(model, args.task_type, args.eval_type, args.nb_classes)

    for _, p in model.named_parameters():
        p.requires_grad = True if (args.eval_type == 'ft' and not args.eval_only) else False
    for _, p in model.head.named_parameters():
        p.requires_grad = True if (not args.eval_type == 'kNN' and not args.eval_only) else False

    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"number of trainable params (M): {(n_trainable_params / 1.e6):.2f}")
    accelerator.print(f"number of total params (M): {(n_params / 1.e6):.2f}")

    global_batch_size = args.eval_batch_size * accelerator.num_processes * args.grad_accum_steps
    batch_ratio = global_batch_size / args.lr_base_batch_size
    args.lr = args.blr * batch_ratio
    
    metric_name, metric_fnc = get_metric_dict(args.eval_dataset, args.eval_type)
    
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

    if args.eval_type == 'kNN':
        model, data_loader_eval_train, data_loader_eval_test = accelerator.prepare(
            model, data_loader_eval_train, data_loader_eval_test
        )
        res = kNN(
            args,
            accelerator,
            model,
            trainloader=data_loader_eval_train,
            testloader=data_loader_eval_test,
            feat_dim=1024 if "large" in args.model else 768
        )
        accelerator.log({metric_name: res}, step=0)
        accelerator.print('{}: {:.2f} %'.format(metric_name, res))
        sys.exit(0)
    elif args.eval_only:
        model, data_loader_eval_test = accelerator.prepare(
            model, data_loader_eval_test
        )
        res = evaluate(
            args,
            accelerator,
            model,
            data_loader_eval_test,
            get_logits_reducer(args.eval_dataset),
            metric_fnc,
            args.nb_classes
        )
        accelerator.log({metric_name: res}, step=0)
        accelerator.print('{}: {:.2f} %'.format(metric_name, res))
        sys.exit(0)
    else:
        label_mixup_fn = None
        label_mixup_active = (
                args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
            ) and (not args.eval_only) and (args.eval_type == 'ft')
        
        if label_mixup_active:
            accelerator.print("Label mixup is activated")
            label_mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.label_smoothing, num_classes=args.nb_classes)

        # set wd as 0 for bias and norm layers
        param_groups = optim_factory.param_groups_layer_decay(
            model, args.weight_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

        if args.eval_type == 'lp':
            lr_scheduler = get_multistep_lr_schedule(
                optimizer=optimizer,
                milestones=[len(data_loader_eval_train)*60, len(data_loader_eval_train)*80]
            )
        elif args.eval_type == 'ft':
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=len(data_loader_eval_train) * args.warmup_epochs,
                num_training_steps=len(data_loader_eval_train) * args.epochs
            )

        model, optimizer, data_loader_eval_train, data_loader_eval_test, lr_scheduler = accelerator.prepare(
            model, optimizer, data_loader_eval_train, data_loader_eval_test, lr_scheduler
        )
        accelerator.register_for_checkpointing(lr_scheduler)
        starting_epoch, global_step, nb_skip_batches = load_training_state(args, accelerator, data_loader_eval_train, auto_check=False)
        if (global_step == 0) and accelerator.is_main_process:
            os.makedirs(run_dir, exist_ok=True)

    accelerator.print(f"Start training for {args.epochs} epochs with {accelerator.num_processes} GPU(s) and {accelerator.mixed_precision} mixed-precision")
    accelerator.print("base lr: %.2e" % args.blr)
    accelerator.print("actual lr: %.2e" % args.lr)
    accelerator.print("effective batch size: %d" % global_batch_size)

    best_m_val = 0.
    for epoch in tqdm(range(starting_epoch, args.epochs), disable=not accelerator.is_local_main_process):
        model.train()
        # New Code #
        if (epoch == starting_epoch):
            # We need to skip steps until we reach the resumed step
            active_dataloader = accelerator.skip_first_batches(data_loader_eval_train, nb_skip_batches)
        else:
            # After the first iteration though, we need to go back to the original dataloader
            active_dataloader = data_loader_eval_train
        for batch in tqdm(active_dataloader, disable=not accelerator.is_local_main_process):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            optimizer.zero_grad(set_to_none=True)
            if label_mixup_active and (not args.task_type == 'SS'):
                batch = apply_label_mixup_fn(batch, label_mixup_fn, args.patch_size)

            with accelerator.autocast():
                outs = model(batch[:-1])
                if (not label_mixup_active) and (args.label_smoothing > 0.) and (args.task_type == 'MLC'):
                    targets = batch[-1] * (1 - args.label_smoothing) + args.label_smoothing / batch[-1].shape[1]
                else:
                    targets = batch[-1]
                loss = calc_downstream_loss(outs, targets, args.eval_dataset, label_mixup_fn=label_mixup_fn, smoothing=args.label_smoothing)
            accelerator.backward(loss)
            optimizer.step()
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step()
            accelerator.log({"train_loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step=global_step)
            if accelerator.is_main_process:
                save_training_state(accelerator, run_dir, epoch, global_step)
            global_step += 1
            
        if (((epoch+1) % args.eval_freq == 0) and not args.skip_eval) or ((epoch + 1) == args.epochs):
            res = evaluate(
                args,
                accelerator,
                model,
                data_loader_eval_test,
                get_logits_reducer(args.eval_dataset),
                metric_fnc,
                args.nb_classes
            )
            accelerator.log({metric_name: res}, step=global_step)
            accelerator.print('{}: {:.2f} %'.format(metric_name, res))

            if res[metric_name] >= best_m_val:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_model(model, accelerator, run_dir, dir='model_weights_best_epoch_{}'.format(epoch+1))
                best_m_val = res[metric_name]
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_model(model, accelerator, run_dir, dir='model_weights_last_epoch_{}'.format(epoch+1))
    accelerator.end_training()

if __name__ == "__main__":
    args = get_args_parser(is_pretrain=False)
    args = args.parse_args()
    assert args.pretrained_model_path is not None, "For eval, pretrained model should be given"
    
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    main(args)

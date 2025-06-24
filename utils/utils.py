import yaml
import os
import torch
from pathlib import Path
from tqdm.auto import tqdm
import shutil
from functools import partial
from sklearn.metrics import average_precision_score, accuracy_score, jaccard_score

def save_model(model, accelerator, run_dir, dir='model_weights_best'):
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save_model(unwrapped_model, save_directory=os.path.join(run_dir, dir))

    dirs = [f for f in os.scandir(Path(run_dir)) if (f.is_dir() and ('model_weights' in f.name))]
    dirs.sort(key=os.path.getctime)

    for i in range(max(len(dirs)-2,0)):
        shutil.rmtree(dirs[i], ignore_errors=True)

def save_training_state(accelerator, run_dir, epoch, global_step):
    out_path = os.path.join(run_dir, 'checkpoint_epoch_{}_step_{}'.format(epoch, global_step))
    accelerator.save_state(output_dir=out_path)

    dirs = [f for f in os.scandir(Path(run_dir)) if (f.is_dir() and ('checkpoint' in f.name))]
    dirs.sort(key=os.path.getctime)

    for i in range(max(len(dirs)-2, 0)):
        shutil.rmtree(dirs[i], ignore_errors=True)

def load_state_from_dir(run_dir_path, args, accelerator, data_loader):
    run_dir = os.path.basename(run_dir_path)
    dirs = [f for f in os.scandir(Path(run_dir_path)) if (f.is_dir() and ('checkpoint' in f.name))]
    if len(dirs) > 0:
        args.wandb_name = run_dir.split('_')[0]
        
        dirs.sort(key=os.path.getctime)
        chkpnt_dir_path = dirs[-1].path  # Sorts folders by date modified, most recent checkpoint is the last

        accelerator.print(f"Resumed from checkpoint: {chkpnt_dir_path}")
        accelerator.load_state(chkpnt_dir_path)
        
        chkpnt_dir = os.path.basename(chkpnt_dir_path)
        if ('step' in chkpnt_dir) and ('epoch' in chkpnt_dir):
            epoch = int(chkpnt_dir.split("checkpoint_epoch_")[-1].split('_step_')[0])
            global_step = int(chkpnt_dir.split("checkpoint_epoch_")[-1].split('_step_')[-1])
            nb_skip_batches = global_step + 1 - (epoch * len(data_loader))
            starting_epoch = epoch+1 if nb_skip_batches == len(data_loader) else epoch
            nb_skip_batches = 0 if nb_skip_batches == len(data_loader) else nb_skip_batches
        elif "step" in chkpnt_dir:
            global_step = int(chkpnt_dir.replace("checkpoint_step_", ""))
            epoch = (global_step) // len(data_loader)
            nb_skip_batches = global_step + 1 - (epoch * len(data_loader))
            starting_epoch = epoch + 1 if nb_skip_batches==0 else epoch
        elif "epoch" in chkpnt_dir:
            epoch = int(chkpnt_dir.replace("checkpoint_epoch_", ""))
            global_step = (len(data_loader) * (epoch+1)) - 1
            starting_epoch = epoch + 1
        starting_step = global_step + 1
    return starting_epoch, starting_step, nb_skip_batches

def load_training_state(args, accelerator, data_loader, auto_check=True):
    starting_epoch=0
    starting_step=0 
    nb_skip_batches=0
    if args.run_dir:
        run_dir = Path(args.run_dir)
        return load_state_from_dir(run_dir, args, accelerator, data_loader)
    return starting_epoch, starting_step, nb_skip_batches    

def set_specs(args):
    with open(args.eval_specs_path) as f:
        eval_ds = yaml.safe_load(f.read())
    args.eval_specs = eval_ds
    with open(args.sensors_specs_path) as f:
        sensors_specs = yaml.safe_load(f.read())
    args.sensors_specs = sensors_specs  # save on args so that it's prop'd to wandb
    with open(args.spectrum_specs_path) as f:
        spectrum_specs = yaml.safe_load(f.read())
    args.spectrum_specs = spectrum_specs

def mIoU(target, pred):
    return jaccard_score(target.ravel(),pred.ravel(),average='macro')

def get_metric_func(metric_name):
    if metric_name == 'mAP':
        return partial(average_precision_score, average='micro')
    elif metric_name == 'acc':
        return accuracy_score
    elif metric_name == 'IoU':
        return mIoU
    else:
        return None

def get_metric_dict(eval_dataset, eval_type):
    if eval_type == 'kNN':
        metric_name = 'kNN@20_acc'
    if 'BigEarthNet' in eval_dataset:
        metric_name = 'mAP'
    elif 'DFC2020' in eval_dataset:
        metric_name = 'IoU'
    else:
        metric_name = 'acc'
    return f'{metric_name}_{eval_dataset}_{eval_type}', get_metric_func(metric_name)

def get_logits_reducer(eval_dataset):
    if 'BigEarthNet' in eval_dataset:
        return torch.sigmoid
    else:
        return partial(torch.argmax, dim=1)

def get_nb_classes(eval_dataset):
    if 'BigEarthNet' in eval_dataset:
        return 19
    elif 'fMoW' in eval_dataset:
        return 62
    elif 'EuroSAT' in eval_dataset:
        return 10
    elif 'WHU-RS19' in eval_dataset:
        return 19
    elif 'RESISC45' in eval_dataset:
        return 45
    elif 'DFC2020' in eval_dataset:
        return 8
    else:
        raise NotImplementedError

def get_task_type(eval_dataset):
    if 'BigEarthNet' in eval_dataset:
        return 'MLC'
    elif 'DFC2020' in eval_dataset:
        return 'SS'
    else:
        return 'SLC'

def get_dtype(mixed_precision):
    if mixed_precision == 'no':
        return torch.float32
    elif mixed_precision == 'bf16':
        return torch.bfloat16
    elif mixed_precision == 'fp16':
        return torch.float16
    else:
        raise NotImplementedError

def get_task_head(model, task_type, eval_type, nb_classes):
    if eval_type == 'kNN':
        return torch.nn.Identity()
    elif task_type == 'SS':
        return torch.nn.Sequential(
            torch.nn.Conv2d(model.head.in_features, nb_classes, kernel_size=1), 
            torch.nn.Upsample(scale_factor=96/14, mode='bilinear', align_corners=False) #TODO: 96 is hardcoded (for DFC2020)
        )
    elif eval_type == 'lp':
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head
        )
    elif eval_type == 'ft':
        return model.head
    
@torch.no_grad()
def kNN(
    args,
    accelerator,
    model,
    trainloader=None,
    testloader=None,
    sigma=0.07,
    feat_dim=768
):
    model.eval()
    # accelerator.print(f"Starting KNN evaluation with K={args.knn}")
    trainFeatures = torch.zeros(
        [feat_dim + 1, len(trainloader) * args.eval_batch_size],
        device=accelerator.device)
    nb_training_samples = 0
    for batch_idx, batch in tqdm(
        enumerate(trainloader), disable=not accelerator.is_local_main_process, desc="eval"
    ):
        batchSize = batch[-1].size(0)
        with accelerator.autocast():
            with torch.no_grad():
                features = model(
                    batch[:-1]
                )
        # breakpoint()
        trainFeatures[
            :-1, nb_training_samples : nb_training_samples + batchSize
        ] = features.T
        trainFeatures[
            -1, nb_training_samples : nb_training_samples + batchSize
        ] = batch[-1]
        nb_training_samples = nb_training_samples + batchSize
    trainFeatures = trainFeatures[:,:nb_training_samples]
    
    if args.num_gpus > 1:
        nb_sample_size = torch.tensor(trainFeatures.shape[1], device=trainFeatures.device)
        gathered_sizes = accelerator.gather(nb_sample_size)
        trainFeatures = accelerator.pad_across_processes(trainFeatures.permute(1, 0).contiguous())
        padded_size = trainFeatures.shape[0]
        trainFeatures = accelerator.gather(trainFeatures)
        trainFeatures = torch.cat(
            [trainFeatures[idx*padded_size:idx*padded_size+size] for idx, size in enumerate(gathered_sizes)], 
            dim=0
        ).permute(1, 0)

    trainLabels = torch.flatten(trainFeatures[-1, :])
    trainFeatures = trainFeatures[:-1, :]
    trainFeatures = torch.nn.functional.normalize(trainFeatures, dim=0)

    top1 = torch.tensor([0.0], device=accelerator.device)
    total = torch.tensor([0.0], device=accelerator.device)
    
    C = int(trainLabels.max() + 1)
    with torch.no_grad():
        retrieval_one_hot = torch.zeros((args.knn, C), device=accelerator.device)
        for batch_idx, batch in  tqdm(
            enumerate(testloader), disable=not accelerator.is_local_main_process, desc=f"kNN@{args.knn}"
        ):
            batchSize = batch[-1].size(0)
            with accelerator.autocast():
                with torch.no_grad():
                    features = model(
                        batch[:-1]
                    )
            features = torch.nn.functional.normalize(features, dim=1)
            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(args.knn, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi).long()

            retrieval_one_hot.resize_(batchSize * args.knn, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batchSize, -1, C),
                    yd_transform.view(batchSize, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)
            # Find which predictions match the target
            correct = predictions.eq(batch[-1].data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            total += batch[-1].size(0)
    top1 = accelerator.reduce(top1)
    total = accelerator.reduce(total)
    top1 = top1.detach().cpu().numpy().item()  # sum
    total = total.detach().cpu().numpy().item()  # sum

    return (top1 / total) * 100

@torch.no_grad()
def evaluate(
    args,
    accelerator,
    model,
    testloader,
    out_reducer,
    metric_fnc,
    nb_classes
):
    model.eval()

    if args.task_type == 'SLC':
        pred_shape = len(testloader) * args.eval_batch_size
    elif args.task_type == 'MLC':
        pred_shape = [len(testloader) * args.eval_batch_size, nb_classes]
    elif args.task_type == 'SS':
        pred_shape = [len(testloader) * args.eval_batch_size, 96, 96] #TODO: 96 is hardcoded (for DFC2020)
    
    predictions = torch.zeros(pred_shape, device=accelerator.device)
    labels = torch.zeros(pred_shape, device=accelerator.device)
    nb_training_samples = 0
    for batch_idx, batch in tqdm(
        enumerate(testloader), disable=not accelerator.is_local_main_process, desc="eval"
    ):
        batchSize = batch[-1].size(0)
        with accelerator.autocast():
            with torch.no_grad():
                outs = model(
                    batch[:-1] #batch[0]
                )
        prediction = out_reducer(outs)
        # breakpoint()
        predictions[nb_training_samples : nb_training_samples + batchSize] = prediction
        labels[nb_training_samples : nb_training_samples + batchSize] = batch[-1]
        nb_training_samples = nb_training_samples + batchSize
    predictions = predictions[:nb_training_samples]
    labels = labels[:nb_training_samples]
    if args.num_gpus > 1:
        nb_sample_size = torch.tensor(predictions.shape[0], device=predictions.device)
        gathered_sizes = accelerator.gather(nb_sample_size)
        predictions = accelerator.pad_across_processes(predictions.contiguous())
        padded_size = predictions.shape[0]
        predictions = accelerator.gather(predictions)
        predictions = torch.cat(
            [predictions[idx*padded_size:idx*padded_size+size] for idx, size in enumerate(gathered_sizes)], 
            dim=0
        )
        labels = accelerator.pad_across_processes(labels)
        labels = accelerator.gather(labels)
        labels = torch.cat(
            [labels[idx*padded_size:idx*padded_size+size] for idx, size in enumerate(gathered_sizes)], 
            dim=0
        )
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return metric_fnc(labels, predictions)*100

def get_multistep_lr_schedule_lambda(current_step, milestones, gamma) -> float:
        if current_step < milestones[0]:
            return 1.
        elif current_step >= milestones[0] and current_step < milestones[1]:
            return float(gamma)
        else:
            return float(gamma ** 2)

def get_multistep_lr_schedule(
    optimizer: torch.optim.Optimizer,
    milestones,
    gamma=0.1,
    last_epoch=-1,
):
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over num_warmup_steps, then decreases to 0.0 on a cosine schedule over
    the remaining num_training_steps-num_warmup_steps (assuming num_cycles = 0.5).

    This is based on the Hugging Face implementation
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to
            schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule. Defaults to 0.5
            (decrease from the max value to 0 following a half-cosine).
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """
    
    lr_lambda = partial(
        get_multistep_lr_schedule_lambda,
        milestones=milestones, 
        gamma=gamma
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
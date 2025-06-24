import argparse

def get_args_parser(is_pretrain=True):
    parser = argparse.ArgumentParser("SMARTIES pre-training and downstream transfer", add_help=False)
    parser.add_argument(
        "--eval_freq",
        default=1,
        type=int,
        help="How often (epochs) to evaluate",
    )
    parser.add_argument(
        "--eval_type",
        default="kNN",
        choices=["lp", "kNN", "ft"],
        type=str,
        help="evaluation type, which can be linear probing (lp), fine-tuning (ft) or k-nearest neighbor (kNN)",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation",
    )
    parser.add_argument(
        "--eval_dataset",
        default="EuroSAT",
        type=str,
        help="name of eval dataset to use",
    )
    parser.add_argument(
        "--eval_scale",
        default=1.0,
        type=float,
        help="scale of evaluation images: should be in (0,1]",
    )
    parser.add_argument("--eval_batch_size", default=32, type=int, help="eval dataset batch size")
    parser.add_argument("--knn", default=20, type=int, help="Number of neighbors to use for KNN")

    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="weight decay"
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=5e-5,
        help="base learning rate",
    )
    parser.add_argument("--epochs", default=300, type=int, help="number of pretraining/fine-tuning/linear-probing epochs")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--nb_workers_per_gpu", default=4, type=int, help="number of workers per GPU")
    parser.add_argument("--pin_mem", action="store_true", dest="pin_mem")
    parser.add_argument(
        "--model",
        default="smarties_vit_base_patch16",
        choices=["smarties_vit_base_patch16", "smarties_vit_large_patch16", "smarties_vit_huge_patch14"],
        type=str,
        help="Name of model to pre-train or to use for downstream transfer",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="The size of the square-shaped input image",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help=(
            "The size of the square-shaped patches across the image. Must be a divisor"
            " of input_size (input_size % patch_size == 0)"
        ),
    )
    parser.add_argument("--wandb_project", default='SMARTIES', type=str, help="wandb project name")
    parser.add_argument("--wandb_entity", default="gencersumbul", type=str, help="wandb username (IMPORTANT: set this to your own wandb username for logging)")
    parser.add_argument("--wandb_name", default=None, type=str, help="Name of wandb entry")
    parser.add_argument("--wandb_notes", default=None, type=str, help="Notes for run logged to wandb")
    parser.add_argument('--wandb_tags', nargs="+",
        default=["pretraining"],
        type=str,
        help="wandb tags")
    
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * #gpus)",
    )
    parser.add_argument(
        "--out_dir", default="weights", help="path where to log and save model files"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Checkpoint project dir path"
    )
    parser.add_argument(
        "--mixed_precision",
        default="fp16",
        choices=["no", "bf16", "fp16", "fp8"],
        type=str,
        help="type of mixed precision"
    )   
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="force the execution on one process on CPU only."
    )

    parser.add_argument("--sensors_specs_path", default="config/pretraining_sensors.yaml", type=str, help="Sensors specifications file")
    parser.add_argument("--spectrum_specs_path", default="config/electromagnetic_spectrum.yaml", type=str, help="Covered electromagnetic spectrum specifications file")
    parser.add_argument("--eval_specs_path", default="config/eval_datasets.yaml", type=str, help="Eval datasets specifications file")
    parser.add_argument("--prefetch_factor", default=2, type=int, help="number of batch prefetches")
    parser.add_argument('--warmup_epochs', default=30, type=int, help="number epochs warmup applied")
    parser.add_argument('--lr_base_batch_size', type=int, default=256,
                   help='base learning rate batch size (divisor, default: 256).')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                   help='The number of steps to accumulate gradients (default: 1)')
    if is_pretrain:
        return get_pretrain_only_args(parser)
    else:
        return get_eval_only_args(parser)

def get_pretrain_only_args(parser):
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.75,
        help="Masking ratio (percentage of removed patches)",
    )
    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    return parser    

def get_eval_only_args(parser):
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Checkpoint path for pretrained model"
    )

    parser.add_argument("--eval_only", action="store_true", help="Only perform evaluation for downstream transfer based on fine-tuned model")
    # fine-tuning params
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
    parser.add_argument('--drop_path', type=float, default=0.2, help='Drop path rate (default: 0.1)')
    # the following is the standard label/image mixup for finetuning, not cross-sensor token mixup of SMARTIES
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument("--global_pool", action="store_true", help='Apply global polling to tokens')
    parser.set_defaults(global_pool=False)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool",
    )
    parser.add_argument("--multi_modal", action="store_true", help='Multi-modal input')

    return parser